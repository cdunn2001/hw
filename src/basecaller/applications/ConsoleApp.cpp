
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
using namespace PacBio::Cuda::Memory;
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
        lanesPerPool_ = primaryConfig.lanesPerPool;
        zmwsPerLane_ = primaryConfig.zmwsPerLane;
        framesPerChunk_ = primaryConfig.framesPerChunk;

        PBLOG_INFO << "Input Target: " << inputTargetFile_;

        inputTraceFile_.reset(new TraceFileGenerator(inputTargetFile_,
                                                     lanesPerPool_,
                                                     zmwsPerLane_,
                                                     framesPerChunk_,
                                                     options.get("cache")));

        if (options.is_set_by_user("numZmwLanes"))
            inputTraceFile_->NumReqZmwLanes(options.get("numZmwLanes"));

        if (options.is_set_by_user("frames"))
            inputTraceFile_->NumFrames(options.get("frames"));

        PBLOG_INFO << "Number of trace file zmwLanes = " << inputTraceFile_->NumTraceZmwLanes();
        PBLOG_INFO << "Number of trace file chunks = " << inputTraceFile_->NumTraceChunks();
        PBLOG_INFO << "Requested number of zmwLanes = " << inputTraceFile_->NumReqZmwLanes();
        PBLOG_INFO << "Requested number of analysis chunks = " << inputTraceFile_->NumChunks();

        const size_t count = primaryConfig.lanesPerPool * primaryConfig.zmwsPerLane * primaryConfig.framesPerChunk;
        tracePool_ = std::make_shared<GpuAllocationPool<int16_t>>(count);

        batchDims_.lanesPerBatch = lanesPerPool_;
        batchDims_.laneWidth = primaryConfig.zmwsPerLane;
        batchDims_.framesPerBatch = primaryConfig.framesPerChunk;
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
                DeallocateTraceChunk(chunk);
            }
            else
            {
                if (numChunksWritten_ >= inputTraceFile_->NumChunks())
                {
                    PBLOG_INFO << "All chunks analyzed.";
                    PBLOG_INFO << "Total frames analyzed = " << inputTraceFile_->NumChunks() * primaryConfig.framesPerChunk;
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
        PBLOG_INFO << "MongoBasecallerConsole::Setup() - Creating analyzer with num pools = "
                   << inputTraceFile_->NumBatches();
        analyzer_ = ITraceAnalyzer::Create(inputTraceFile_->NumBatches(), basecallerConfig_, movConfig);

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
                if (numChunksWritten_ >= inputTraceFile_->NumChunks())
                {
                    PBLOG_INFO << "All chunks written out.";
                    break;
                }
            }
        }
    }

    std::vector<TraceBatch<int16_t>> AllocateTraceChunk()
    {
        std::vector<TraceBatch<int16_t>> chunk;

        size_t chunkIndex = inputTraceFile_->ChunkIndex();
        for (size_t b = 0; b < inputTraceFile_->NumBatches(); b++)
        {
            chunk.emplace_back(BatchMetadata(b, chunkIndex * framesPerChunk_,
                               (chunkIndex * framesPerChunk_) + framesPerChunk_),
                               batchDims_,
                               SyncDirection::HostWriteDeviceRead,
                               tracePool_);
        }

        return chunk;
    }

    void DeallocateTraceChunk(std::vector<TraceBatch<int16_t>>& chunk)
    {
       for (auto& batch : chunk)
           batch.DeactivateGpuMem();
    }

    void PreloadInputQueue()
    {
        size_t numPreload = std::min(numChunksPreloadInputQueue_, inputTraceFile_->NumTraceChunks());

        PBLOG_INFO << "Preloading input data queue with " + std::to_string(numPreload) + " chunks";
        for (size_t numChunk = 0; numChunk < numPreload; numChunk++)
        {
            std::vector<TraceBatch<int16_t>> chunk = AllocateTraceChunk();
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
            if (!inputTraceFile_->Finished())
            {
                std::vector<TraceBatch<int16_t>> chunk = AllocateTraceChunk();
                if (inputTraceFile_->PopulateChunk(chunk))
                    inputDataQueue_.Push(std::move(chunk));
            }
            else
            {
                PBLOG_INFO << "All chunks read in.";
                break;
            }
        }
    }

private:
    std::shared_ptr<GpuAllocationPool<int16_t>> tracePool_;
    BasecallerAlgorithmConfig basecallerConfig_;
    std::unique_ptr<ITraceAnalyzer> analyzer_;
    std::unique_ptr<TraceFileGenerator> inputTraceFile_;
    PacBio::ThreadSafeQueue<std::vector<TraceBatch<int16_t>>> inputDataQueue_;
    PacBio::ThreadSafeQueue<const std::vector<BasecallBatch>*> outputDataQueue_;
    std::string inputTargetFile_;
    size_t numChunksPreloadInputQueue_;
    size_t numChunksWritten_;
    uint32_t lanesPerPool_;
    uint32_t zmwsPerLane_;
    uint32_t framesPerChunk_;
    BatchDimensions batchDims_;
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
        
        auto group1 = PacBio::Process::OptionGroup(parser, "Data Selection/Tiling Options",
                                                   "Controls data selection/tiling options for simulation and testing");
        group1.add_option("--numZmwLanes").type_int().set_default(0).help("Specifies number of zmw lanes to analyze");
        group1.add_option("--frames").type_int().set_default(0).help("Specifies number of frames to run");
        parser.add_option_group(group1);

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

