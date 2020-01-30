//#include <utility>
//
#include <applications/Basecaller.h>
#include <applications/BazWriter.h>
#include <applications/CudaAllocator.h>
#include <applications/Repacker.h>
#include <applications/TraceDataSource.h>
#include <applications/TraceSaver.h>
#include <dataTypes/BasecallerConfig.h>
#include <dataTypes/MovieConfig.h>
#include <common/MongoConstants.h>
#include <common/cuda/memory/ManagedAllocations.h>
#include <common/graphs/GraphManager.h>
#include <common/graphs/GraphNodeBody.h>
#include <common/graphs/GraphNode.h>

#include <pacbio/primary/SequelTraceFile.h>
#include <pacbio/PBException.h>
#include <pacbio/datasource/DataSource.h>
#include <pacbio/datasource/DataSourceRunner.h>
#include <pacbio/datasource/PacketLayout.h>
#include <pacbio/datasource/SensorPacket.h>
#include <pacbio/datasource/SensorPacketsChunk.h>
#include <pacbio/logging/Logger.h>
#include <pacbio/process/OptionParser.h>
#include <pacbio/process/ProcessBase.h>

using namespace PacBio::Cuda::Memory;
using namespace PacBio::Graphs;
using namespace PacBio::Mongo;
using namespace PacBio::Mongo::Data;

using namespace PacBio::Application;
using namespace PacBio::DataSource;
using namespace PacBio::Process;
using namespace PacBio::Primary;

class SmrtBasecaller : public ThreadedProcessBase
{
public:
    SmrtBasecaller()
    {}

    ~SmrtBasecaller()
    {
        Abort();
        Join();
    }

    void HandleProcessArguments(const std::vector<std::string>& args)
    {
        if (args.size() > 0)
        {
            throw PBException("Positional arguments not supported");
        }
    }

    void HandleProcessOptions(const Values& options)
    {
        ThreadedProcessBase::HandleProcessOptions(options);

        numChunksPreloadInputQueue_ = options.get("numChunksPreload");
        zmwOutputStrideFactor_ = options.get("zmwOutputStrideFactor");

        // TODO validate.  File must be specified
        inputTargetFile_ = options["inputfile"];

        numZmwLanes_ = options.get("numZmwLanes");
        frames_ = options.get("frames");
        // TODO need validation or something, as this is probably a trace file input specific option
        cache_ = options.get("cache");

        // TODO these might need cleanup/moving?  At the least need to be able to set them
        // correctly if not using trace file input
        if (!inputTargetFile_.empty())
        {
            PBLOG_INFO << "Input Target: " << inputTargetFile_;
            MetaDataFromTraceFileSource(inputTargetFile_);
            GroundTruthFromTraceFileSource(inputTargetFile_);
        } else {
            throw PBException("No input file specified");
        }

        if (options.is_set_by_user("outputbazfile"))
        {
            hasBazFile_ = true;
            outputBazFile_ = options["outputbazfile"];
            PBLOG_INFO << "Output Target: " << outputBazFile_;
        }

        PBLOG_INFO << "Number of analysis zmwLanes = " << numZmwLanes_;
        PBLOG_INFO << "Number of analysis chunks = " << static_cast<size_t>(options.get("frames")) / PacBio::Mongo::Data::GetPrimaryConfig().framesPerChunk;
    }

    void Run()
    {
        EnablePerformanceMode();

        RunAnalyzer();
        Join();

        // Need to free up our allocations that are pooled. If that happens
        // during static teardown, we'll likely try to free cuda allocations
        // after the cuda runtime is already gone, which causes a crash
        DisablePerformanceMode();
    }

    SmrtBasecaller& Config(const BasecallerConfig& basecallerConfig)
    {
        basecallerConfig_.Load(basecallerConfig);
        return *this;
    }

    const BasecallerConfig& Config() const
    { return basecallerConfig_; }

private:
    void MetaDataFromTraceFileSource(const std::string& traceFileName)
    {
        const auto& traceFile = SequelTraceFileHDF5(traceFileName);

        traceFile.FrameRate >> movieConfig_.frameRate;
        traceFile.AduGain >> movieConfig_.photoelectronSensitivity;
        traceFile.AnalogRefSnr >> movieConfig_.refSnr;

        // Analog information
        size_t traceNumAnalogs;
        std::string baseMap;
        std::vector<float> relativeAmpl;
        std::vector<float> excessNoiseCV;
        std::vector<float> interPulseDistance;
        std::vector<float> pulseWidth;
        std::vector<float> ipd2SlowStepRatio;
        std::vector<float> pw2SlowStepRatio;

        traceFile.NumAnalog >> traceNumAnalogs;
        if (traceNumAnalogs != numAnalogs)
            throw PBException("Trace file does not have 4 analogs!");

        traceFile.BaseMap >> baseMap;
        traceFile.RelativeAmp >> relativeAmpl;
        traceFile.ExcessNoiseCV >> excessNoiseCV;
        traceFile.IpdMean >> interPulseDistance;
        traceFile.PulseWidthMean >> pulseWidth;
        traceFile.Ipd2SlowStepRatio >> ipd2SlowStepRatio;
        traceFile.Pw2SlowStepRatio >> pw2SlowStepRatio;

        // Check relative amplitude is sorted decreasing.
        if (!std::is_sorted(relativeAmpl.rbegin(), relativeAmpl.rend()))
        {
            throw PBException("Analogs in trace file not sorted by decreasing relative amplitude!");
        }

        for (size_t i = 0; i < numAnalogs; i++)
        {
            auto& analog = movieConfig_.analogs[i];
            analog.baseLabel = baseMap[i];
            analog.relAmplitude = relativeAmpl[i];
            analog.excessNoiseCV = excessNoiseCV[i];
            analog.interPulseDistance = interPulseDistance[i];
            analog.pulseWidth = pulseWidth[i];
            analog.ipd2SlowStepRatio = ipd2SlowStepRatio[i];
            analog.pw2SlowStepRatio = pw2SlowStepRatio[i];
        }
    }

    void GroundTruthFromTraceFileSource(const std::string& traceFileName)
    {
        auto setBlMeanAndCovar = [](const std::string& traceFileName,
                                    ConfigurationObject::ConfigurationPod<float>& blMean,
                                    ConfigurationObject::ConfigurationPod<float>& blCovar,
                                    const std::string& exceptMsg)
        {
            const auto& traceFile = SequelTraceFileHDF5(traceFileName);
            if (traceFile.simulated)
            {
                boost::multi_array<float,2> stateMean;
                boost::multi_array<float,2> stateCovar;
                traceFile.StateMean >> stateMean;
                traceFile.StateCovariance >> stateCovar;
                blMean = stateMean[0][0];
                blCovar = stateCovar[0][0];
            }
            else
            {
                throw PBException(exceptMsg);
            }
        };

        if (basecallerConfig_.algorithm.staticAnalysis == true)
        {
            setBlMeanAndCovar(traceFileName,
                              basecallerConfig_.algorithm.staticDetModelConfig.baselineMean,
                              basecallerConfig_.algorithm.staticDetModelConfig.baselineVariance,
                              "Requested static pipeline analysis but input trace file is not simulated!");
        }
        else if (basecallerConfig_.algorithm.dmeConfig.Method() == BasecallerDmeConfig::MethodName::Fixed &&
                 basecallerConfig_.algorithm.dmeConfig.SimModel.useSimulatedBaselineParams == true)
        {
            setBlMeanAndCovar(traceFileName,
                              basecallerConfig_.algorithm.dmeConfig.SimModel.baselineMean,
                              basecallerConfig_.algorithm.dmeConfig.SimModel.baselineVar,
                              "Requested fixed DME with baseline params but input trace file is not simulated!");
        }
    }

    // TODO support wolverine and potentially other sources
    std::unique_ptr<DataSourceRunner> CreateSource()
    {
        // TODO need to handle sparse as well
        std::array<size_t, 3> layoutDims {
            PacBio::Mongo::Data::GetPrimaryConfig().lanesPerPool,
            PacBio::Mongo::Data::GetPrimaryConfig().framesPerChunk,
            PacBio::Mongo::Data::GetPrimaryConfig().zmwsPerLane
        };
        PacketLayout layout(PacketLayout::BLOCK_LAYOUT_DENSE,
                            PacketLayout::INT16,
                            layoutDims);

        // TODO need a way to let trace file specify numZmw and num frames
        const auto numZmw = numZmwLanes_ * laneSize;
        const auto frameRate = PacBio::Mongo::Data::GetPrimaryConfig().sensorFrameRate;
        DataSourceBase::Configuration config(layout, std::make_unique<CudaAllocator>());

        return std::make_unique<DataSourceRunner>(
            std::make_unique<TraceDataSource>(std::move(config),
                inputTargetFile_,
                frames_,
                numZmw,
                cache_,
                numChunksPreloadInputQueue_));
    }

    std::unique_ptr<MultiTransformBody<SensorPacket, const TraceBatch<int16_t>>> CreateRepacker() const
    {
        BatchDimensions requiredDims;
        requiredDims.lanesPerBatch = PacBio::Mongo::Data::GetPrimaryConfig().lanesPerPool;
        requiredDims.framesPerBatch = PacBio::Mongo::Data::GetPrimaryConfig().framesPerChunk;
        requiredDims.laneWidth = PacBio::Mongo::Data::GetPrimaryConfig().zmwsPerLane;
        return std::make_unique<TrivialRepackerBody>(requiredDims);
    }

    std::unique_ptr<LeafBody<const TraceBatch<int16_t>>> CreateTraceSaver() const
    {
        return std::make_unique<NoopTraceSaverBody>();
    }

    std::unique_ptr<TransformBody<const TraceBatch<int16_t>, BatchResult>> CreateBasecaller(const std::vector<uint32_t>& poolIds) const
    {
        return std::make_unique<BasecallerBody>(poolIds, basecallerConfig_, movieConfig_);
    }

    std::unique_ptr<LeafBody<BatchResult>> CreateBazSaver(const DataSourceRunner& source)
    {
        if (hasBazFile_)
        {
            return std::make_unique<BazWriterBody>(outputBazFile_, source.NumFrames(),
                                                   source.UnitCellIds(), source.UnitCellFeatures(),
                                                   basecallerConfig_, zmwOutputStrideFactor_);
        } else {
            return std::make_unique<NoopBazWriterBody>();

        }
    }

    void RunAnalyzer()
    {
        // Names for the various graph stages
        SMART_ENUM(GraphProfiler, REPACKER, SAVE_TRACE, ANALYSIS, BAZWRITER);

        auto source = CreateSource();

        GraphManager<GraphProfiler> graph(Config().init.numWorkerThreads);
        auto * repacker = graph.AddNode(CreateRepacker(), GraphProfiler::REPACKER);
        repacker->AddNode(CreateTraceSaver(), GraphProfiler::SAVE_TRACE);
        auto * analyzer = repacker->AddNode(CreateBasecaller(source->PoolIds()), GraphProfiler::ANALYSIS);
        analyzer->AddNode(CreateBazSaver(*source), GraphProfiler::BAZWRITER);

        size_t numChunksAnalyzed = 0;
        PacBio::Dev::QuietAutoTimer timer(0);

        const double chunkDurationMS = PacBio::Mongo::Data::GetPrimaryConfig().framesPerChunk
                                     / PacBio::Mongo::Data::GetPrimaryConfig().sensorFrameRate
                                     * 1e3;
        while(source->IsActive())
        {
            SensorPacketsChunk chunk;
            if (source->PopData(chunk))
            {
                PBLOG_INFO << "Analyzing chunk frames = ["
                    + std::to_string(chunk.StartFrame()) + ","
                    + std::to_string(chunk.EndFrame()) + ")";
                for (auto& batch : chunk)
                    repacker->ProcessInput(std::move(batch));
                const auto& reports = graph.FlushAndReport(chunkDurationMS);

                std::stringstream ss;
                ss << "Chunk finished: Duty Cycle%, Avg Occupancy:\n";

                for (auto& report: reports)
                {
                    if (!report.realtime)
                    {
                        PBLOG_WARN << report.stage.toString() << " is not currently slower than budgeted:  Duty Cycle%, Duration MS, Idle %, Occupancy -- "
                                   << report.dutyCycle * 100 << "%, "
                                   << 1e3 / report.avgDuration << "ms, "
                                   << report.idlePercent << "%, "
                                   << report.avgOccupancy;
                    }
                    ss << "\t\t" << report.stage.toString() << ": "
                       << report.dutyCycle * 100 << "%, "
                       << report.avgOccupancy << "\n";
                }
                PacBio::Logging::LogStream(PacBio::Logging::LogLevel::INFO) << ss.str();

                numChunksAnalyzed++;
            }
            if (this->ExitRequested())
            {
                source->RequestExit();
                break;
            }
        }
        graph.Flush();

        PBLOG_INFO << "All chunks analyzed.";
        PBLOG_INFO << "Total frames analyzed = "
                   << source->NumFrames();

        timer.SetCount(numChunksAnalyzed);
        double chunkAnalyzeRate = timer.GetRate();
        PBLOG_INFO << "Analyzed " << numChunksAnalyzed
                   << " chunks at " << chunkAnalyzeRate << " chunks/sec"
                   << " (" << (source->NumZmw() * chunkAnalyzeRate)
                   << " zmws/sec)";
    }

private:
    // Configuration objects
    BasecallerConfig basecallerConfig_;
    MovieConfig movieConfig_;

    std::string inputTargetFile_;
    std::string outputBazFile_;
    bool hasBazFile_ = false;
    size_t zmwOutputStrideFactor_ = 1;
    size_t numChunksPreloadInputQueue_ = 0;
    size_t numZmwLanes_ = 0;
    size_t frames_ = 0;
    bool cache_ = false;
};

int main(int argc, char* argv[])
{
    try
    {
        auto parser = ProcessBase::OptionParserFactory();
        parser.description("Prototype to demonstrate mongo basecaller");
        parser.version("0.1");

        parser.epilog("");

        parser.add_option("--config").action_append().help("Loads JSON configuration file, JSON string or Boost ptree value");
        parser.add_option("--strict").action_store_true().help("Strictly check all configuration options. Do not allow unrecognized configuration options");
        parser.add_option("--showconfig").action_store_true().help("Shows the entire configuration namespace and exits");

        parser.add_option("--inputfile").set_default("").help("input file (must be *.trc.h5)");
        parser.add_option("--outputbazfile").set_default("").help("BAZ output file");
        parser.add_option("--numChunksPreload").type_int().set_default(0).help("Number of chunks to preload (Default: %default)");
        parser.add_option("--cache").action_store_true().help("Cache trace file to avoid disk I/O");
        parser.add_option("--numWorkerThreads").type_int().set_default(0).help("Number of compute threads to use.  ");

        auto group1 = OptionGroup(parser, "Data Selection/Tiling Options",
                                  "Controls data selection/tiling options for simulation and testing");
        group1.add_option("--numZmwLanes").type_int().set_default(131072).help("Specifies number of zmw lanes to analyze");
        group1.add_option("--frames").type_int().set_default(10000).help("Specifies number of frames to run");
        parser.add_option_group(group1);

        auto group2 = OptionGroup(parser, "Data Output Throttling Options",
                                  "Controls data throttling for BAZ file writing");
        group2.add_option("--zmwOutputStrideFactor").type_int().set_default(1).help("Throttle zmw writing data output");
        parser.add_option_group(group2);

        auto options = parser.parse_args(argc, (const char* const*) argv);
        ThreadedProcessBase::HandleGlobalOptions(options);

        ConfigMux mux;
        BasecallerConfig basecallerConfig;
        mux.Add("basecaller", basecallerConfig);
        mux.Add("common", PacBio::Mongo::Data::GetPrimaryConfig());
        mux.SetStrict(options.get("strict"));
        mux.ProcessCommandLine(options.all("config"));

        if (options.get("showconfig"))
        {
            std::cout << mux.ToJson() << std::endl;
            return 0;
        }

        if (options.is_set_by_user("numWorkerThreads"))
        {
            basecallerConfig.init.numWorkerThreads = options.get("numWorkerThreads");
        }

        auto bc = std::unique_ptr<SmrtBasecaller>(new SmrtBasecaller());

        bc->HandleProcessArguments(parser.args());
        bc->Config(basecallerConfig);

        {
            PacBio::Logging::LogStream ls;
            ls << "\"common\" : " << PacBio::Mongo::Data::GetPrimaryConfig().RenderJSON() << "\n";
            ls << "\"basecaller\" : " << bc->Config();
        }

        bc->HandleProcessOptions(options);

        bc->Run();

    } catch (std::exception &ex) {
        PBLOG_ERROR << "Exception caught: " << ex.what();
        return 1;
    }

    return 0;
}
