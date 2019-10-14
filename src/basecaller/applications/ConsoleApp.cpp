
#include <basecaller/analyzer/ITraceAnalyzer.h>
#include <dataTypes/BasecallerConfig.h>
#include <dataTypes/BatchMetadata.h>
#include <dataTypes/MovieConfig.h>
#include <dataTypes/Pulse.h>
#include <dataTypes/PulseBatch.h>
#include <common/DataGenerators/BatchGenerator.h>
#include <common/MongoConstants.h>

#include <pacbio/primary/BazWriter.h>
#include <pacbio/primary/FileHeaderBuilder.h>
#include <pacbio/primary/SequelTraceFile.h>
#include <pacbio/primary/ZmwResultBuffer.h>

#include <pacbio/PBException.h>
#include <pacbio/dev/profile/ScopedProfilerChain.h>
#include <pacbio/logging/Logger.h>
#include <pacbio/process/OptionParser.h>
#include <pacbio/process/ProcessBase.h>
#include <pacbio/smrtdata/NucleotideLabel.h>

using namespace PacBio::Cuda::Data;
using namespace PacBio::Cuda::Memory;
using namespace PacBio::Mongo;
using namespace PacBio::Mongo::Data;
using namespace PacBio::Mongo::Basecaller;

using namespace PacBio::Process;
using namespace PacBio::Primary;

class MongoBasecallerConsole : public ThreadedProcessBase
{
    SMART_ENUM(
        ChunkProfiler,
        ANALYZE_CHUNK,
        WRITE_CHUNK,
        READ_CHUNK
    );

    using BazWriter = PacBio::Primary::BazWriter<SpiderMetricBlock>;
    using Profiler = PacBio::Dev::Profile::ScopedProfilerChain<ChunkProfiler>;
public:
    MongoBasecallerConsole()
    {}

    ~MongoBasecallerConsole()
    {
        Abort();
        Join();

        PBLOG_INFO << readThroughputStats_.str();
        PBLOG_INFO << analyzeThroughputStats_.str();
        PBLOG_INFO << writeThroughputStats_.str();
        Profiler::FinalReport();
    }

    void HandleProcessArguments(const std::vector<std::string>& args)
    {
        if (args.size() == 1)
        {
            inputTargetFile_ = args[0];
        }
    }

    void HandleProcessOptions(const Values& options)
    {
        ThreadedProcessBase::HandleProcessOptions(options);

        numChunksPreloadInputQueue_ = options.get("numChunksPreload");
        zmwOutputStrideFactor_ = options.get("zmwOutputStrideFactor");

        batchGenerator_.reset(new BatchGenerator(PacBio::Mongo::Data::GetPrimaryConfig().framesPerChunk,
                                                 PacBio::Mongo::Data::GetPrimaryConfig().zmwsPerLane,
                                                 PacBio::Mongo::Data::GetPrimaryConfig().lanesPerPool,
                                                 options.get("frames"),
                                                 options.get("numZmwLanes")));

        if (inputTargetFile_.size() == 0)
        {
            inputTargetFile_ = options["inputfile"];
        }

        if (!inputTargetFile_.empty())
        {
            PBLOG_INFO << "Input Target: " << inputTargetFile_;
            batchGenerator_->SetTraceFileSource(inputTargetFile_, options.get("cache"));
            MetaDataFromTraceFileSource(inputTargetFile_);
            GroundTruthFromTraceFileSource(inputTargetFile_);
        }

        if (options.is_set_by_user("outputbazfile"))
        {
            hasBazFile_ = true;
            outputBazFile_ = options["outputbazfile"];
            PBLOG_INFO << "Output Target: " << outputBazFile_;
        }

        PBLOG_INFO << "Number of analysis zmwLanes = " << batchGenerator_->NumZmwLanes();
        PBLOG_INFO << "Number of analysis chunks = " << batchGenerator_->NumChunks();
    }

    void Run()
    {
        EnablePerformanceMode();

        Setup();
        RunAnalyzer();
        Join();
        Teardown();

        // Need to free up our allocations that are pooled. If that happens
        // during static teardown, we'll likely try to free cuda allocations
        // after the cuda runtime is already gone, which causes a crash
        DisablePerformanceMode();
    }

    MongoBasecallerConsole& Config(const BasecallerConfig& basecallerConfig)
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

    void RunAnalyzer()
    {
        size_t numChunksAnalyzed = 0;
        PacBio::Dev::QuietAutoTimer timer(0);
        while (!ExitRequested())
        {
            std::vector<TraceBatch<int16_t>> chunk;
            if (inputDataQueue_.Pop(chunk, std::chrono::milliseconds(100)))
            {
                PBLOG_INFO << "Analyzing chunk frames = ["
                              + std::to_string(chunk.front().GetMeta().FirstFrame()) + ","
                              + std::to_string(chunk.front().GetMeta().LastFrame()) + ")";
                Profiler::Mode mode = Profiler::Mode::REPORT;
                if (numChunksAnalyzed < 15) mode = Profiler::Mode::OBSERVE;
                if (numChunksAnalyzed < 3) mode = Profiler::Mode::IGNORE;
                Profiler profiler(mode, 3.0f, std::numeric_limits<float>::max());
                auto analyzeChunkProfile = profiler.CreateScopedProfiler(ChunkProfiler::ANALYZE_CHUNK);
                (void)analyzeChunkProfile;
                auto output = (*analyzer_)(std::move(chunk));
                numChunksAnalyzed++;
                outputDataQueue_.Push(std::move(output));
            }
            else
            {
                if (numChunksWritten_ >= batchGenerator_->NumChunks())
                {
                    PBLOG_INFO << "All chunks analyzed.";
                    PBLOG_INFO << "Total frames analyzed = "
                               << batchGenerator_->NumChunks() * PacBio::Mongo::Data::GetPrimaryConfig().framesPerChunk;
                    break;
                }
            }
        }

        timer.SetCount(numChunksAnalyzed);
        double chunkAnalyzeRate = timer.GetRate();
        analyzeThroughputStats_ << "Analyzed " << numChunksAnalyzed
            << " chunks at " << chunkAnalyzeRate << " chunks/sec"
            << " (" << (batchGenerator_->NumZmwLanes() *
                        PacBio::Mongo::Data::GetPrimaryConfig().zmwsPerLane *
                        chunkAnalyzeRate)
            << " zmws/sec)";
    }

    void Teardown()
    {
        // Perform any necessary teardown.
    }

    void Setup()
    {
        if (batchGenerator_->NumBatches() == 0)
            throw PBException("Number of batches cannot be 0");

        PBLOG_INFO << "MongoBasecallerConsole::Setup() - Creating analyzer with num pools = "
                   << batchGenerator_->NumBatches();

        analyzer_ = ITraceAnalyzer::Create(batchGenerator_->NumBatches(), basecallerConfig_, movieConfig_);

        PreloadInputQueue();

        PBLOG_INFO << "MongoBasecallerConsole::Setup() - Creating reader thread";
        CreateThread("reader", [this]() { this->RunReader(); });
        PBLOG_INFO << "MongoBasecallerConsole::Setup() - Creating writer thread";
        CreateThread("writer", [this]() { this->RunWriter(); });
    }

    void OpenBazFile()
    {
        if (hasBazFile_)
        {
            std::vector<uint32_t> zmwNumbers = batchGenerator_->UnitCellIds();
            std::vector<uint32_t> zmwFeatures = batchGenerator_->UnitCellFeatures();

            PBLOG_INFO << "Opening BAZ file for writing: " << outputBazFile_ << " zmws: " << zmwNumbers.size();

            uint64_t movieLengthInFrames = batchGenerator_->NumFrames();
            FileHeaderBuilder fh(outputBazFile_,
                              100.0f,
                              movieLengthInFrames,
                              Readout::BASES,
                              MetricsVerbosity::MINIMAL,
                              "",
                              basecallerConfig_.RenderJSON(),
                              zmwNumbers,
                              zmwFeatures,
                              PacBio::Mongo::Data::GetPrimaryConfig().framesPerChunk,
                              PacBio::Mongo::Data::GetPrimaryConfig().framesPerChunk,
                              PacBio::Mongo::Data::GetPrimaryConfig().framesPerChunk,
                              ChipClass::Spider,
                              false,
                              true,
                              true,
                              false);
            fh.BaseCallerVersion("0.1");

            bazWriter_.reset(new BazWriter(outputBazFile_, fh, BazIOConfig{}, ReadBuffer::MaxNumSamplesPerBuffer()));
        }
    }

    void CloseBazFile()
    {
        if (hasBazFile_)
        {
            if (currentZmwIndex_) bazWriter_->Flush();
            PBLOG_INFO << "Closing BAZ file: " << outputBazFile_;
            bazWriter_->WaitForTermination();
            bazWriter_.reset();
        }
    }

    void ConvertMetric(const std::unique_ptr<BatchResult::MetricsT>& metricsPtr,
                       SpiderMetricBlock& sm,
                       size_t laneIndex,
                       size_t zmwIndex)
    {
        if (metricsPtr)
        {
            const auto& metrics = metricsPtr->GetHostView()[laneIndex];
            sm.numBasesA_ = metrics.numBasesByAnalog[0][zmwIndex];
            sm.numBasesC_ = metrics.numBasesByAnalog[1][zmwIndex];
            sm.numBasesG_ = metrics.numBasesByAnalog[2][zmwIndex];
            sm.numBasesT_ = metrics.numBasesByAnalog[3][zmwIndex];

            sm.numPulses_ = metrics.numBases[zmwIndex];
        }
    }

    auto ConvertMongoPulsesToSequelBasecalls(const Pulse* pulses, uint32_t numPulses)
    {
        auto LabelConv = [](Data::Pulse::NucleotideLabel label) {
            switch (label)
            {
                case Data::Pulse::NucleotideLabel::A:
                    return PacBio::SmrtData::NucleotideLabel::A;
                case Data::Pulse::NucleotideLabel::C:
                    return PacBio::SmrtData::NucleotideLabel::C;
                case Data::Pulse::NucleotideLabel::G:
                    return PacBio::SmrtData::NucleotideLabel::G;
                case Data::Pulse::NucleotideLabel::T:
                    return PacBio::SmrtData::NucleotideLabel::T;
                case Data::Pulse::NucleotideLabel::N:
                    return PacBio::SmrtData::NucleotideLabel::N;
                default:
                    assert(label == Data::Pulse::NucleotideLabel::NONE);
                    return PacBio::SmrtData::NucleotideLabel::NONE;
            }
        };

        std::vector<Basecall> baseCalls(numPulses);
        for (size_t pulseNum = 0; pulseNum < numPulses; ++pulseNum)
        {
            const auto& pulse = pulses[pulseNum];
            auto& bc = baseCalls[pulseNum];

            static constexpr int8_t qvDefault_ = 0;

            auto label = LabelConv(pulse.Label());

            // Populate pulse data
            bc.GetPulse().Start(pulse.Start()).Width(pulse.Width());
            bc.GetPulse().MeanSignal(pulse.MeanSignal()).MidSignal(pulse.MidSignal()).MaxSignal(pulse.MaxSignal());
            bc.GetPulse().Label(label).LabelQV(qvDefault_);
            bc.GetPulse().AltLabel(label).AltLabelQV(qvDefault_);
            bc.GetPulse().MergeQV(qvDefault_);

            // Populate base data.
            bc.Base(label).InsertionQV(qvDefault_);
            bc.DeletionTag(PacBio::SmrtData::NucleotideLabel::N).DeletionQV(qvDefault_);
            bc.SubstitutionTag(PacBio::SmrtData::NucleotideLabel::N).SubstitutionQV(qvDefault_);
        }
        return baseCalls;
    }

    void WriteOutputChunk(const std::vector<std::unique_ptr<BatchAnalyzer::OutputType>>& outputChunk)
    {
        static const std::string bazWriterError = "BazWriter has failed. Last error message was ";

        if (!bazWriter_) return; // NoOp if we're not doing output

        for (const auto& outputBatchPtr : outputChunk)
        {
            const auto& pulseBatch = outputBatchPtr->pulses;
            const auto& metricsPtr = outputBatchPtr->metrics;
            for (uint32_t lane = 0; lane < pulseBatch.Dims().lanesPerBatch; ++lane)
            {
                const auto& lanePulses = pulseBatch.Pulses().LaneView(lane);

                for (uint32_t zmw = 0; zmw < laneSize; zmw++)
                {
                    if (currentZmwIndex_ % zmwOutputStrideFactor_ == 0)
                    {
                        const auto& baseCalls = ConvertMongoPulsesToSequelBasecalls(lanePulses.ZmwData(zmw),
                                                                                    lanePulses.size(zmw));
                        if (!bazWriter_->AddZmwSlice(baseCalls.data(),
                                                     baseCalls.size(),
                                                     [&](MemoryBufferView<SpiderMetricBlock>& dest)
                                                     {
                                                        for (size_t i = 0; i < dest.size(); i++)
                                                        {
                                                            ConvertMetric(metricsPtr, dest[i], lane, zmw);
                                                        }
                                                     },
                                                     1,
                                                     currentZmwIndex_))
                        {
                            throw PBException(bazWriterError + bazWriter_->ErrorMessage());
                        }
                    }
                    else
                    {
                        if (!bazWriter_->AddZmwSlice(NULL, 0, [](auto&){}, 0, currentZmwIndex_))
                        {
                            throw PBException(bazWriterError + bazWriter_->ErrorMessage());
                        }
                    }
                    currentZmwIndex_++;
                }
            }
        }
        bazWriter_->Flush();
    }

    void RunWriter()
    {
        OpenBazFile();

        PacBio::Dev::QuietAutoTimer timer(0);
        while (!ExitRequested())
        {
            std::vector<std::unique_ptr<BatchAnalyzer::OutputType>> outputChunk;
            if (outputDataQueue_.TryPop(outputChunk))
            {
                currentZmwIndex_ = 0;
                Profiler::Mode mode = Profiler::Mode::REPORT;
                if (numChunksWritten_ < 15) mode = Profiler::Mode::OBSERVE;
                Profiler profiler(mode, 3.0f, std::numeric_limits<float>::max());
                auto writeChunkProfile = profiler.CreateScopedProfiler(ChunkProfiler::WRITE_CHUNK);
                (void)writeChunkProfile;
                WriteOutputChunk(outputChunk);
                numChunksWritten_++;
            }
            else
            {
                if (numChunksWritten_ >= batchGenerator_->NumChunks())
                {
                    PBLOG_INFO << "All chunks written out.";
                    break;
                }
            }
        }

        CloseBazFile();

        timer.SetCount(numChunksWritten_);
        double chunkWriteRate = timer.GetRate();
        writeThroughputStats_ << "Wrote " << numChunksWritten_
            << " chunks at " << chunkWriteRate << " chunks/sec"
            << " (" << (batchGenerator_->NumZmwLanes() *
                        PacBio::Mongo::Data::GetPrimaryConfig().zmwsPerLane *
                        chunkWriteRate)
                        / zmwOutputStrideFactor_
            << " zmws/sec)";
    }

    void PreloadInputQueue()
    {
        size_t numPreload = std::min(numChunksPreloadInputQueue_, batchGenerator_->NumChunks());

        if (numPreload > 0)
        {
            PBLOG_INFO << "Preloading input data queue with " + std::to_string(numPreload) + " chunks";
            for (size_t numChunk = 0; numChunk < numPreload; numChunk++)
            {
                auto chunk = batchGenerator_->PopulateChunk();
                PBLOG_INFO << "Read in chunk frames = ["
                              + std::to_string(chunk.front().GetMeta().FirstFrame()) + ","
                              + std::to_string(chunk.front().GetMeta().LastFrame()) + ")";
                inputDataQueue_.Push(std::move(chunk));
            }
            PBLOG_INFO << "Done preloading input queue.";
        }
    }

    void RunReader()
    {
        PacBio::Dev::QuietAutoTimer timer(0);
        size_t numChunksRead = 0;
        while (!ExitRequested())
        {
            if (!batchGenerator_->Finished())
            {
                auto readChunk = [&numChunksRead, this]() {
                    Profiler::Mode mode = Profiler::Mode::REPORT;
                    if (numChunksRead < 15) mode = Profiler::Mode::OBSERVE;
                    Profiler profiler(mode, 3.0f, std::numeric_limits<float>::max());
                    auto readChunkProfile = profiler.CreateScopedProfiler(ChunkProfiler::READ_CHUNK);
                    (void)readChunkProfile;
                    auto chunk = batchGenerator_->PopulateChunk();
                    PBLOG_INFO << "Read in chunk frames = ["
                                  + std::to_string(chunk.front().GetMeta().FirstFrame()) + ","
                                  + std::to_string(chunk.front().GetMeta().LastFrame()) + ")";
                    return chunk;
                };
                auto chunk = readChunk();
                numChunksRead++;
                while (inputDataQueue_.Size() > numChunksPreloadInputQueue_
                        && !ExitRequested())
                {
                    usleep(1000);
                }
                inputDataQueue_.Push(std::move(chunk));
            }
            else
            {
                PBLOG_INFO << "All chunks read in.";
                break;
            }
        }
        timer.SetCount(numChunksRead);
        double chunkReadRate = timer.GetRate();
        readThroughputStats_ << "Read " << numChunksRead
            << " chunks at " << chunkReadRate << " chunks/sec"
            << " (" << batchGenerator_->NumZmwLanes() *
                       PacBio::Mongo::Data::GetPrimaryConfig().zmwsPerLane *
                       chunkReadRate
            << " zmws/sec)";
    }

private:
    // Configuration objects
    BasecallerConfig basecallerConfig_;
    MovieConfig movieConfig_;

    // Main analyzer
    std::unique_ptr<ITraceAnalyzer> analyzer_;
    std::unique_ptr<BazWriter> bazWriter_;

    // Data generator, input and output queues
    std::unique_ptr<BatchGenerator> batchGenerator_;
    PacBio::ThreadSafeQueue<std::vector<TraceBatch<int16_t>>> inputDataQueue_;
    PacBio::ThreadSafeQueue<std::vector<std::unique_ptr<BatchAnalyzer::OutputType>>> outputDataQueue_;

    std::string inputTargetFile_;
    std::string outputBazFile_;
    std::ostringstream readThroughputStats_;
    std::ostringstream analyzeThroughputStats_;
    std::ostringstream writeThroughputStats_;
    size_t numChunksPreloadInputQueue_ = 0;
    size_t numChunksWritten_ = 0;
    bool hasBazFile_ = false;
    uint64_t numZmwsSoFar_ = 0;
    size_t zmwOutputStrideFactor_ = 1;
    uint32_t currentZmwIndex_ = 0;
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

        auto bc = std::unique_ptr<MongoBasecallerConsole>(new MongoBasecallerConsole());

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

