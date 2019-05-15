
#include <basecaller/analyzer/ITraceAnalyzer.h>
#include <dataTypes/BasecallerConfig.h>
#include <dataTypes/BatchMetadata.h>
#include <dataTypes/MovieConfig.h>
#include <common/DataGenerators/TraceFileGenerator.h>

#include <pacbio/PBException.h>
#include <pacbio/logging/Logger.h>
#include <pacbio/process/OptionParser.h>
#include <pacbio/process/ProcessBase.h>

using namespace PacBio::Cuda::Data;
using namespace PacBio::Mongo::Data;
using namespace PacBio::Mongo::Basecaller;

class MongoBasecallerConsole : public PacBio::Process::ThreadedProcessBase
{
public:
    MongoBasecallerConsole()
    : numChunksWritten_(0)
    {}

    ~MongoBasecallerConsole()
    {}

    void HandleProcessArguments(const std::vector<std::string>& args)
    {
        if (args.size() == 1)
        {
            inputTargetFile_ = args[0];
        }
    }

    void HandleProcessOptions(const PacBio::Process::Values& options)
    {

        PacBio::Process::ThreadedProcessBase::HandleProcessOptions(options);

        if (inputTargetFile_.size() == 0)
            inputTargetFile_ = options["inputfile"];

        numChunksPreloadInputQueue_ = options.get("numChunksPreload");

        // TODO: Determine data selection.

        PBLOG_INFO << "Input Target: " << inputTargetFile_;
        PBLOG_INFO << "Number of chunks to preload = " << numChunksPreloadInputQueue_;

        uint32_t tileBatches = options.get("tileBatches");
        uint32_t tileChunks = options.get("tileChunks");

        if (options.is_set_by_user("tileBatches"))
            PBLOG_INFO << "Requested batches to tile = " << tileBatches;
        if (options.is_set_by_user("tileChunks"))
            PBLOG_INFO << "Requested chunks to tile = " << tileChunks;

        inputTraceFile_.reset(new TraceFileGenerator(inputTargetFile_,
                                                     primaryConfig.zmwsPerLane,
                                                     primaryConfig.lanesPerPool,
                                                     primaryConfig.framesPerChunk,
                                                     tileBatches, tileChunks, options.get("cache")));
    }

    void Run()
    {
        Setup();
        RunAnalyzer();
        Join();
        Teardown();
    }

    void RunAnalyzer()
    {
        while (!ExitRequested())
        {
            std::vector<TraceBatch<int16_t>> chunk;
            if (inputDataQueue_.Pop(chunk, std::chrono::milliseconds(100)))
            {
                PBLOG_INFO << "Analyzing chunk frames = ["
                    + std::to_string(chunk.front().GetMeta().FirstFrame()) + ","
                    + std::to_string(chunk.front().GetMeta().LastFrame()) + ")";
                const auto baseCalls = (*analyzer_)(std::move(chunk));
                outputDataQueue_.Push(&baseCalls);
            }
            else
            {
                if (numChunksWritten_ >= inputTraceFile_->GetNumChunks())
                {
                    PBLOG_INFO << "All chunks analyzed.";
                    break;
                }
            }
        }
    }

    void Teardown()
    {
        // Perform any necessary teardown.
    }

    BasecallerAlgorithmConfig& BasecallerConfig()
    { return basecallerConfig_; };

private:
    void Setup()
    {
        MovieConfig movConfig;
        analyzer_ = ITraceAnalyzer::Create(inputTraceFile_->GetNumBatches(), basecallerConfig_, movConfig);

        PreloadInputQueue();

        PBLOG_INFO << "MongoBasecallerConsole::Setup() - Creating reader thread";
        CreateThread("reader", [this]() { this->RunReader(); });
        PBLOG_INFO << "MongoBasecallerConsole::Setup() - Creating writer thread";
        CreateThread("writer", [this]() { this->RunWriter(); });
    }

    void RunWriter()
    {
        while (!ExitRequested())
        {
            const std::vector<BasecallBatch>* baseCalls;
            if (outputDataQueue_.TryPop(baseCalls))
            {
                // Write out data here.
                numChunksWritten_++;
            }
            else
            {
                if (numChunksWritten_ >= inputTraceFile_->GetNumChunks())
                {
                    PBLOG_INFO << "All chunks written out.";
                    break;
                }
            }
        }
    }

    void PreloadInputQueue()
    {
        size_t numPreload = std::min(numChunksPreloadInputQueue_, inputTraceFile_->GetNumChunks());

        PBLOG_INFO << "Preloading input data queue with " + std::to_string(numPreload) + " chunks";
        for (size_t numChunk = 0; numChunk < numPreload; numChunk++)
        {
            std::vector<TraceBatch<int16_t>> chunk;
            if (inputTraceFile_->PopulateChunk(chunk))
            {
                PBLOG_INFO << "Preloaded chunk = " << numChunk;
                inputDataQueue_.Push(std::move(chunk));
            }
        }
        PBLOG_INFO << "Done preloading input queue.";
    }

    void RunReader()
    {
        while (!ExitRequested())
        {
            std::vector<TraceBatch<int16_t>> chunk;
            if (inputTraceFile_->PopulateChunk(chunk))
            {
                inputDataQueue_.Push(std::move(chunk));
            }
            else
            {
                if (inputTraceFile_->Finished())
                {
                    PBLOG_INFO << "All chunks read in.";
                    break;
                }
            }
        }
    }

private:
    BasecallerAlgorithmConfig basecallerConfig_;
    std::unique_ptr<ITraceAnalyzer> analyzer_;
    std::unique_ptr<TraceFileGenerator> inputTraceFile_;
    PacBio::ThreadSafeQueue<std::vector<TraceBatch<int16_t>>> inputDataQueue_;
    PacBio::ThreadSafeQueue<const std::vector<BasecallBatch>*> outputDataQueue_;
    std::string inputTargetFile_;
    size_t numChunksPreloadInputQueue_;
    size_t numChunksWritten_;
};

int main(int argc, char* argv[])
{
    try
    {
        auto parser = PacBio::Process::ProcessBase::OptionParserFactory();
        parser.description("Prototype to demonstrate mongo basecaller");
        parser.version("0.1");

        parser.epilog("");

        parser.add_option("--config").action_append().help("Loads JSON configuration file, JSON string or Boost ptree value");
        parser.add_option("--strict").action_store_true().help("Strictly check all configuration options. Do not allow unrecognized configuration options");
        parser.add_option("--showconfig").action_store_true().help("Shows the entire configuration namespace and exits");

        parser.add_option("--inputfile").set_default("").help("input file (must be *.trc.h5)");
        parser.add_option("--outputbazfile").set_default("").help("BAZ output file");
        parser.add_option("--numChunksPreload").type_int().set_default(3).help("Number of chunks to preload (Default: %default)");
        parser.add_option("--cache").action_store_true().help("Cache trace file to avoid disk I/O");

        auto group1 = PacBio::Process::OptionGroup(parser, "Data Selection Options",
                                                   "Controls data selection from input data");
        group1.add_option("--zmwHoleNumbers").type_string().help("");
        group1.add_option("--startFrame").type_int().help("Starting frame to begin analysis");
        group1.add_option("--numFrames").type_int().help(
                "Number of frames, defaults to number of frames in input file");
        parser.add_option_group(group1);

        auto group2 = PacBio::Process::OptionGroup(parser, "Data Tiling Options",
                                                   "Controls data tiling options for simulation and testing");
        group2.add_option("--tileBatches").type_int().set_default(0).help("Specifies number of batches to tile out");
        group2.add_option("--tileChunks").type_int().set_default(0).help("Specifies number of chunks to tile out");
        parser.add_option_group(group2);

        auto options = parser.parse_args(argc, (const char* const*) argv);
        PacBio::Process::ThreadedProcessBase::HandleGlobalOptions(options);

        PacBio::Process::ConfigMux mux;
        BasecallerAlgorithmConfig basecallerConfig;
        mux.Add("basecaller", basecallerConfig);
        mux.Add("common", primaryConfig);
        mux.ProcessCommandLine(options.all("config"));

        if (options.get("showconfig"))
        {
            std::cout << mux.ToJson() << std::endl;
            return 0;
        }

        auto bc = std::unique_ptr<MongoBasecallerConsole>(new MongoBasecallerConsole());

        bc->HandleProcessArguments(parser.args());

        bc->BasecallerConfig().SetStrict(options.get("strict"));
        bc->BasecallerConfig().Load(basecallerConfig);

        {
            PacBio::Logging::LogStream ls;
            ls << "\"common\" : " << primaryConfig.RenderJSON() << "\n";
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

