
#include <basecaller/analyzer/ITraceAnalyzer.h>
#include <dataTypes/BasecallerConfig.h>
#include <dataTypes/BatchMetadata.h>
#include <dataTypes/MovieConfig.h>
#include <common/DataGenerators/BatchGenerator.h>

#include <pacbio/primary/BazWriter.h>
#include <pacbio/primary/FileHeaderBuilder.h>
#include <pacbio/primary/ZmwResultBuffer.h>

#include <pacbio/PBException.h>
#include <pacbio/dev/profile/ScopedProfilerChain.h>
#include <pacbio/logging/Logger.h>
#include <pacbio/process/OptionParser.h>
#include <pacbio/process/ProcessBase.h>

using namespace PacBio::Cuda::Data;
using namespace PacBio::Cuda::Memory;
using namespace PacBio::Mongo::Data;
using namespace PacBio::Mongo::Basecaller;

using namespace PacBio::Process;
using namespace PacBio::Primary;

class MongoBasecallerConsole : public ThreadedProcessBase
{
    SMART_ENUM(
        ProfileSpots,
        ANALYZE_CHUNK,
        WRITE_CHUNK
    );

    using BazWriter = PacBio::Primary::BazWriter<SpiderMetricBlock>;
    using Profiler = PacBio::Dev::Profile::ScopedProfilerChain<ProfileSpots>;
public:
    MongoBasecallerConsole()
    {}

    ~MongoBasecallerConsole()
    {
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
        Setup();
        RunAnalyzer();
        Join();
        Teardown();
    }

    MongoBasecallerConsole& BasecallerConfig(const BasecallerAlgorithmConfig& basecallerConfig)
    {
        basecallerConfig_.Load(basecallerConfig);
        return *this;
    }

    const BasecallerAlgorithmConfig& BasecallerConfig() const
    { return basecallerConfig_; }

private:
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
                Profiler profiler(mode, 3.0f, std::numeric_limits<float>::max());
                auto analyzeChunkProfile = profiler.CreateScopedProfiler(ProfileSpots::ANALYZE_CHUNK);
                (void)analyzeChunkProfile;
                auto baseCalls = (*analyzer_)(std::move(chunk));
                numChunksAnalyzed++;
                outputDataQueue_.Push(std::move(baseCalls));
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

        MovieConfig movConfig;
        analyzer_ = ITraceAnalyzer::Create(batchGenerator_->NumBatches(), basecallerConfig_, movConfig);

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
                              true);
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

    void ConvertMetric(const PacBio::Mongo::Data::BasecallingMetrics& bm, SpiderMetricBlock& sm)
    {
        sm.numBasesA_ = bm.NumBasesByAnalog()[0];
        sm.numBasesC_ = bm.NumBasesByAnalog()[1];
        sm.numBasesG_ = bm.NumBasesByAnalog()[2];
        sm.numBasesT_ = bm.NumBasesByAnalog()[3];

        sm.numPulses_ = bm.NumPulses();
    }

    void WriteBasecallsChunk(const std::vector<BasecallBatch>& basecallChunk)
    {
        static const std::string bazWriterError = "BazWriter has failed. Last error message was ";

        for (const auto& basecallBatch : basecallChunk)
        {
            for (uint32_t z = 0; z < basecallBatch.Dims().zmwsPerBatch(); z++)
            {
                if (currentZmwIndex_ % zmwOutputStrideFactor_ == 0)
                {
                    if (!bazWriter_->AddZmwSlice(basecallBatch.Basecalls().data() + basecallBatch.zmwOffset(z),
                                                 basecallBatch.SeqLengths()[z],
                                                 [&](MemoryBufferView<SpiderMetricBlock>& dest)
                                                 {
                                                    for (size_t i = 0; i < dest.size(); i++)
                                                    {
                                                        ConvertMetric(basecallBatch.Metrics()[z], dest[i]);
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

    void RunWriter()
    {
        OpenBazFile();

        PacBio::Dev::QuietAutoTimer timer(0);
        while (!ExitRequested())
        {
            std::vector<BasecallBatch> basecallChunk;
            if (outputDataQueue_.TryPop(basecallChunk))
            {
                currentZmwIndex_ = 0;
                Profiler::Mode mode = Profiler::Mode::REPORT;
                if (numChunksWritten_ < 15) mode = Profiler::Mode::OBSERVE;
                Profiler profiler(mode, 3.0f, std::numeric_limits<float>::max());
                auto writeChunkProfile = profiler.CreateScopedProfiler(ProfileSpots::WRITE_CHUNK);
                (void)writeChunkProfile;
                WriteBasecallsChunk(basecallChunk);
                bazWriter_->Flush();
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
        size_t numPreload = std::min(numChunksPreloadInputQueue_, batchGenerator_->NumTraceChunks());

        if (numPreload > 0)
        {
            PBLOG_INFO << "Preloading input data queue with " + std::to_string(numPreload) + " chunks";
            for (size_t numChunk = 0; numChunk < numPreload; numChunk++)
            {
                PBLOG_INFO << "Preloaded chunk = " << numChunk;
                inputDataQueue_.Push(std::move(batchGenerator_->PopulateChunk()));
            }
            PBLOG_INFO << "Done preloading input queue.";
        }
    }

    void RunReader()
    {
        size_t numChunksRead = 0;
        PacBio::Dev::QuietAutoTimer timer(0);
        while (!ExitRequested())
        {
            if (!batchGenerator_->Finished())
            {

                auto chunk = batchGenerator_->PopulateChunk();
                numChunksRead++;
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
    BasecallerAlgorithmConfig basecallerConfig_;

    // Main analyzer
    std::unique_ptr<ITraceAnalyzer> analyzer_;
    std::unique_ptr<BazWriter> bazWriter_;

    // Data generator, input and output queues
    std::unique_ptr<BatchGenerator> batchGenerator_;
    PacBio::ThreadSafeQueue<std::vector<TraceBatch<int16_t>>> inputDataQueue_;
    PacBio::ThreadSafeQueue<std::vector<BasecallBatch>> outputDataQueue_;

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

        auto group1 = OptionGroup(parser, "Data Selection/Tiling Options",
                                  "Controls data selection/tiling options for simulation and testing");
        group1.add_option("--numZmwLanes").type_int().set_default(0).help("Specifies number of zmw lanes to analyze");
        group1.add_option("--frames").type_int().set_default(0).help("Specifies number of frames to run");
        parser.add_option_group(group1);

        auto group2 = OptionGroup(parser, "Data Output Throttling Options",
                                  "Controls data throttling for BAZ file writing");
        group2.add_option("--zmwOutputStrideFactor").type_int().set_default(1).help("Throttle zmw writing data output");
        parser.add_option_group(group2);

        auto options = parser.parse_args(argc, (const char* const*) argv);
        ThreadedProcessBase::HandleGlobalOptions(options);

        ConfigMux mux;
        BasecallerAlgorithmConfig basecallerConfig;
        mux.Add("basecaller", basecallerConfig);
        mux.Add("common", PacBio::Mongo::Data::GetPrimaryConfig());
        mux.SetStrict(options.get("strict"));
        mux.ProcessCommandLine(options.all("config"));

        if (options.get("showconfig"))
        {
            std::cout << mux.ToJson() << std::endl;
            return 0;
        }

        auto bc = std::unique_ptr<MongoBasecallerConsole>(new MongoBasecallerConsole());

        bc->HandleProcessArguments(parser.args());
        bc->BasecallerConfig(basecallerConfig);

        {
            PacBio::Logging::LogStream ls;
            ls << "\"common\" : " << PacBio::Mongo::Data::GetPrimaryConfig().RenderJSON() << "\n";
            ls << "\"basecaller\" : " << bc->BasecallerConfig();
        }

        bc->HandleProcessOptions(options);

        bc->Run();

    } catch (std::exception &ex) {
        PBLOG_ERROR << "Exception caught: " << ex.what();
        return 1;
    }

    return 0;
}

