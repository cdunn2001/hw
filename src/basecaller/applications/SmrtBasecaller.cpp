// Copyright (c) 2020, Pacific Biosciences of California, Inc.
//
// All rights reserved.
//
// THIS SOFTWARE CONSTITUTES AND EMBODIES PACIFIC BIOSCIENCES' CONFIDENTIAL
// AND PROPRIETARY INFORMATION.
//
// Disclosure, redistribution and use of this software is subject to the
// terms and conditions of the applicable written agreement(s) between you
// and Pacific Biosciences, where "you" refers to you or your company or
// organization, as applicable.  Any other disclosure, redistribution or
// use is prohibited.
//
// THIS SOFTWARE IS PROVIDED BY PACIFIC BIOSCIENCES AND ITS CONTRIBUTORS "AS
// IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
// THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL PACIFIC BIOSCIENCES OR ITS
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
// OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
// WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
// OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
// ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <appModules/Basecaller.h>
#include <appModules/BazWriter.h>
#include <appModules/BlockRepacker.h>
#include <appModules/TrivialRepacker.h>
#include <appModules/TraceFileDataSource.h>
#include <appModules/TraceSaver.h>
#include <dataTypes/configs/SmrtBasecallerConfig.h>
#include <dataTypes/configs/MovieConfig.h>
#include <common/MongoConstants.h>
#include <common/cuda/memory/ManagedAllocations.h>
#include <common/graphs/GraphManager.h>
#include <common/graphs/GraphNodeBody.h>
#include <common/graphs/GraphNode.h>

#include <pacbio/PBException.h>
#include <pacbio/configuration/MergeConfigs.h>
#include <pacbio/datasource/DataSourceBase.h>
#include <pacbio/datasource/DataSourceRunner.h>
#include <pacbio/datasource/PacketLayout.h>
#include <pacbio/datasource/SensorPacket.h>
#include <pacbio/datasource/SensorPacketsChunk.h>
#include <pacbio/ipc/JSON.h>
#include <pacbio/logging/Logger.h>
#include <pacbio/POSIX.h>
#include <pacbio/process/OptionParser.h>
#include <pacbio/process/ProcessBase.h>
#include <pacbio/sensor/SparseROI.h>
#include <mongo/datasource/WXDataSource.h>

using namespace PacBio::Cuda::Memory;
using namespace PacBio::Graphs;
using namespace PacBio::Mongo;
using namespace PacBio::Mongo::Data;
using namespace PacBio::Mongo::DataSource;
using namespace PacBio::Sensor;

using namespace PacBio::Application;
using namespace PacBio::Configuration;
using namespace PacBio::DataSource;
using namespace PacBio::Process;
using namespace PacBio::Primary;
using namespace PacBio::TraceFile;

////////////
// vvvv TODO This could be cleaned up.

// Need a global to support CtrlC handling
std::atomic<bool> globalHalt{false};


// ^^^^
///////

class SmrtBasecaller : public ThreadedProcessBase
{
public:
    SmrtBasecaller(const SmrtBasecallerConfig& config)
        : config_(config)
    {}

    ~SmrtBasecaller()
    {
        Abort();
        Join();
    }


    void HandleProcessArguments(const std::vector <std::string>& args)
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
        nop_ = options.get("nop");

        if (nop_ == 1 && inputTargetFile_ != "constant/123")
        {
            throw PBException("--nop=1 must be used with --inputfile=constant/123 to get correct validation pattern.");
        }

        // TODO these might need cleanup/moving?  At the least need to be able to set them
        // correctly if not using trace file input
        switch (config_.source.sourceType)
        {
            case Source_t::TRACE_FILE:
                if (inputTargetFile_.empty())
                {
                    throw PBException("TRACE_FILE requires an --inputfile argument");
                }
                PBLOG_INFO << "Input Target: " << inputTargetFile_;
                MetaDataFromTraceFileSource(inputTargetFile_);
                GroundTruthFromTraceFileSource(inputTargetFile_);
                break;

            case Source_t::WX2:
                // FIXME An API to load metadata from ICS configuration is not finalized yet.
                // FIXME In the interest of rapid development, we are just loading a precanned metadata snippet.
                // TODO replace this call with the official command line API for loading metadata (JSON) from ICS.
                // for example, LoadMetaDataFromSequelFormat(json value of metadata);
                movieConfig_ = PacBio::Mongo::Data::MockMovieConfig();
                break;

            default:
                PBLOG_INFO << "Meta data not initialized from source " << config_.source.sourceType;
                break;
        }

        if (options.is_set_by_user("outputbazfile"))
        {
            hasBazFile_ = true;
            outputBazFile_ = options["outputbazfile"];
            PBLOG_INFO << "Output Target: " << outputBazFile_;
        }

        if (options["outputtrcfile"] != "")
        {
            outputTrcFileName_ = options["outputtrcfile"];
        }

        PBLOG_INFO << "Number of analysis zmwLanes = " << numZmwLanes_;
        PBLOG_INFO << "Number of analysis chunks = " << static_cast<size_t>(options.get("frames")) /
                                                        config_.layout.framesPerChunk;

        auto devices = PacBio::Cuda::CudaAllGpuDevices();
        if(devices.size() == 0)
        {
            throw PBException("No CUDA devices available on this computer");
        }
        PBLOG_INFO << "Found " << devices.size() << " CUDA devices";
        int idevice = 0;
        for(const auto& d : devices)
        {
            const auto& dd(d.deviceProperties);
            PBLOG_INFO << " CUDA GPU device: " << idevice;
            PBLOG_INFO << "  Device Name:" << dd.name;
            PBLOG_INFO << "  Global memory:" << dd.totalGlobalMem;
            // PBLOG_INFO << "  Constant memory:" << dd.totalConstMem;
            // PBLOG_INFO << "  Warp size:" << dd.warpSize;
            PBLOG_INFO << "  sharedMemPerBlock:" << dd.sharedMemPerBlock;
            PBLOG_INFO << "  major/minog:" << dd.major << "/" << dd.minor;
            PBLOG_INFO << "  Error Message:" << d.errorMessage;
            idevice++;
        }
    }

    void Run()
    {
        SetGlobalAllocationMode(CachingMode::ENABLED, AllocatorMode::CUDA);

        RunAnalyzer();
        Join();

        // Go ahead and free up all our allocation pools, though we
        // don't strictly need to do this as they can clean up after
        // themselves.  Still, there are currently some outstanding
        // static lifetime issues that can affect *other* allocations
        // that live past the end of main, so for now this is just
        // a way to be explicit and encourage the practice of manually
        // getting rid of any static lifetime allocations before main
        // ends
        DisableAllCaching();
    }

    const SmrtBasecallerConfig& Config() const
    { return config_; }

private:
    void MetaDataFromTraceFileSource(const std::string& traceFileName)
    {
        const TraceFile traceFile(traceFileName);
        const auto& acqParams = traceFile.Scan().AcqParams();
        const auto& chipInfo = traceFile.Scan().ChipInfo();
        const auto& dyeSet = traceFile.Scan().DyeSet();

        movieConfig_.frameRate = acqParams.frameRate;
        movieConfig_.photoelectronSensitivity = acqParams.aduGain;
        movieConfig_.refSnr = chipInfo.analogRefSnr;

        // Analog information
        std::string baseMap;
        std::vector<float> relativeAmpl;
        std::vector<float> excessNoiseCV;
        std::vector<float> interPulseDistance;
        std::vector<float> pulseWidth;
        std::vector<float> ipd2SlowStepRatio;
        std::vector<float> pw2SlowStepRatio;

        const constexpr size_t traceNumAnalogs = TraceFile::DefaultNumAnalogs;
        static_assert (traceNumAnalogs == numAnalogs, "Trace file does not have 4 analogs!");

        baseMap = dyeSet.baseMap;
        relativeAmpl = dyeSet.relativeAmp;
        excessNoiseCV = dyeSet.excessNoiseCV;
        interPulseDistance = dyeSet.ipdMean;
        pulseWidth = dyeSet.pulseWidthMean;
        ipd2SlowStepRatio = dyeSet.ipd2SlowStepRatio;
        pw2SlowStepRatio = dyeSet.pw2SlowStepRatio;

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
                                    float& blMean,
                                    float& blCovar,
                                    const std::string& exceptMsg)
        {
            const TraceFile traceFile{traceFileName};
            if (traceFile.IsSimulated())
            {
                const auto groundTruth = traceFile.GroundTruth();
                blMean = groundTruth.stateMean[0][0];
                blCovar = groundTruth.stateCovariance[0][0];
            } else
            {
                throw PBException(exceptMsg);
            }
        };

        if (config_.algorithm.modelEstimationMode == BasecallerAlgorithmConfig::ModelEstimationMode::FixedEstimations)
        {
            setBlMeanAndCovar(traceFileName,
                              config_.algorithm.staticDetModelConfig.baselineMean,
                              config_.algorithm.staticDetModelConfig.baselineVariance,
                              "Requested static pipeline analysis but input trace file is not simulated!");
        }
        else if (config_.algorithm.dmeConfig.Method == BasecallerDmeConfig::MethodName::Fixed &&
                 config_.algorithm.dmeConfig.SimModel.useSimulatedBaselineParams == true)
        {
            setBlMeanAndCovar(traceFileName,
                              config_.algorithm.dmeConfig.SimModel.baselineMean,
                              config_.algorithm.dmeConfig.SimModel.baselineVar,
                              "Requested fixed DME with baseline params but input trace file is not simulated!");
        }
    }

    /// In the interim of SequelOnKestrel, we may use metadata that was structured for Spider in JSON format.
    /// This helper function loads that JSON in to the internal data format.
    /// TODO: using this as a starting point, support Kestrel runmeta data.
    void LoadMetaDataFromSequelFormat(const Json::Value& metadata)
    {
        movieConfig_.frameRate = metadata["expectedFrameRate"].asFloat();
        movieConfig_.photoelectronSensitivity = metadata["photoelectronSensitivity"].asFloat();
        movieConfig_.refSnr = metadata["refDwsSnr"].asFloat();

        auto baseMap = metadata["baseMap"].asString();

        for (unsigned int i = 0; i < numAnalogs; i++)
        {
            const auto& analogJson = metadata["analogs"][i];
            auto& analog = movieConfig_.analogs[i];

            analog.baseLabel = analogJson["base"].asString()[0];
            analog.relAmplitude = analogJson["relativeAmplitude"].asFloat();
            analog.excessNoiseCV = analogJson["intraPulseXsnCV"].asFloat();
            analog.interPulseDistance = analogJson["ipdMeanSeconds"].asFloat();
            analog.pulseWidth = analogJson["pulseWidthMeanSeconds"].asFloat();
            analog.ipd2SlowStepRatio = 0.15;
            analog.pw2SlowStepRatio = 0.15;

            if (analog.baseLabel != baseMap[i])
            {
                throw PBException("basemap in wrong order to analogs array");
            }
        }
    }

    // TODO fix this up. It is too specialized for WX2 or TraceFile datasources.
    std::unique_ptr<DataSourceRunner> CreateSource()
    {

        std::array<size_t, 3> layoutDims;
        layoutDims[0] = config_.layout.lanesPerPool;
        layoutDims[1] = config_.layout.framesPerChunk;
        layoutDims[2] = config_.layout.zmwsPerLane;

        PacketLayout layout(PacketLayout::BLOCK_LAYOUT_DENSE,
                            PacketLayout::INT16,
                            layoutDims);

        // TODO need a way to let trace file specify numZmw and num frames
        const auto numZmw = numZmwLanes_ * laneSize;

        /// MTL/BB hack.begin
        /// We manually loaded the allocator with allocations. This greatly improves handling of lots of Packets.
        /// TODO: clean up this ugliness.
        // TODO need to handle sparse as well
        size_t numPreallocatedPackets = 0;
        switch(config_.source.sourceType)
        {
            case Source_t::TRACE_FILE:
                numPreallocatedPackets = 0;
                break;
            case Source_t::WX2:
                numPreallocatedPackets = 800000 /
                    config_.source.wx2SourceConfig.wxlayout.lanesPerPacket; // FIXME. this should depend on the number of tiles on the chip, which is hardwired to Spider size right now...
                break;
            default:
                numPreallocatedPackets = 0;
                break;
        }
        auto allo = CreateAllocator(AllocatorMode::CUDA, AllocationMarker(config_.source.sourceType.toString()));
        if (numPreallocatedPackets>0)
        {
            std::vector<PacBio::Memory::SmartAllocation> allocations;
            allocations.reserve(numPreallocatedPackets);
            for(uint32_t i=0;i<numPreallocatedPackets;i++)
            {
                allocations.emplace_back(allo->GetAllocation(sizeof(Tile)));
            }
            for(uint32_t i=0;i<numPreallocatedPackets;i++)
            {
                allo->ReturnHostAllocation(std::move(allocations[i]));
            }
        }        
        /// MTL/BB hack.end

        DataSourceBase::Configuration datasourceConfig(layout, std::move(allo));
        datasourceConfig.numFrames = frames_;

        std::unique_ptr<DataSourceBase> dataSource;
        switch(config_.source.sourceType)
        {
            case Source_t::TRACE_FILE:
                dataSource = std::make_unique<TraceFileDataSource>(std::move(datasourceConfig),
                                                              inputTargetFile_,
                                                              frames_,
                                                              numZmw,
                                                              cache_,
                                                              numChunksPreloadInputQueue_);
                break;
            case Source_t::WX2:
            {

                // TODO this glue code is messy. It is gluing untyped strings to the strongly typed
                // enums of WX2, but this was on purpose to avoid entangling the config with configs from WX2.
                // I am not sure what the best next step is.  This is getting me going, so I am
                // going to leave it. MTL
                WXDataSourceConfig wxconfig;
                wxconfig.dataPath = DataPath_t(config_.source.wx2SourceConfig.dataPath);
                wxconfig.platform = Platform(config_.source.wx2SourceConfig.platform);
                wxconfig.sleepDebug = config_.source.wx2SourceConfig.sleepDebug;
                wxconfig.maxPopLoops = config_.source.wx2SourceConfig.maxPopLoops;
                wxconfig.tilePoolFactor = config_.source.wx2SourceConfig.tilePoolFactor;
                wxconfig.chipLayoutName = "Spider_1p0_NTO"; // FIXME this needs to be a command line parameter supplied by ICS.
                wxconfig.layoutDims[0] = config_.source.wx2SourceConfig.wxlayout.lanesPerPacket;
                wxconfig.layoutDims[1] = config_.source.wx2SourceConfig.wxlayout.framesPerPacket;
                wxconfig.layoutDims[2] = config_.source.wx2SourceConfig.wxlayout.zmwsPerLane;
                dataSource = std::make_unique<WXDataSource>(std::move(datasourceConfig), wxconfig);
            }
                break;
            default:
                throw PBException("Data source " + config_.source.sourceType.toString() + " not supported");
        }
        return std::make_unique<DataSourceRunner>(std::move(dataSource));
    }

    std::unique_ptr <MultiTransformBody<SensorPacket, const TraceBatch <int16_t>>>
    CreateRepacker(PacketLayout inputLayout) const
    {
        BatchDimensions requiredDims;
        requiredDims.lanesPerBatch = config_.layout.lanesPerPool;
        requiredDims.framesPerBatch = config_.layout.framesPerChunk;
        requiredDims.laneWidth = config_.layout.zmwsPerLane;

        if (inputLayout.Encoding() != PacketLayout::INT16)
        {
            throw PBException("Only 16 bit input is supported so far");
        }

        if (inputLayout.Type() != PacketLayout::BLOCK_LAYOUT_DENSE)
        {
            throw PBException("Only dense block layouts are supported so far");
        }

        // check if the trivial repacker is a good fit first, as it's always preferred
        {
            bool trivial = true;
            if (requiredDims.lanesPerBatch != inputLayout.NumBlocks()) trivial = false;
            if (requiredDims.laneWidth != inputLayout.BlockWidth()) trivial = false;
            if (requiredDims.framesPerBatch != inputLayout.NumFrames()) trivial = false;

            if (trivial)
            {
                PBLOG_INFO << "Instantiating TrivialRepacker";
                return std::make_unique<TrivialRepackerBody>(requiredDims);
            }
        }

        // Now check if the BlockRepacker is a valid fit
        {
            bool valid = true;
            if (inputLayout.BlockWidth() % 32 != 0) valid = false;
            if (requiredDims.framesPerBatch % inputLayout.NumFrames() != 0) valid = false;
            if (valid)
            {
                PBLOG_INFO << "Instantiating BlockRepacker";
                const size_t numZmw = numZmwLanes_ * laneSize;
                const size_t numThreads = 6;
                return std::make_unique<BlockRepacker>(inputLayout, requiredDims, numZmw, numThreads);
            }
        }

        throw PBException("No repacker exists that can handle this PacketLayout:"
                          " numBlocks, numFrames, blockWidth -- " + std::to_string(inputLayout.NumBlocks())
                          + ", " + std::to_string(inputLayout.NumFrames()) + ", "
                          + std::to_string(inputLayout.BlockWidth()));
    }


    std::unique_ptr <LeafBody<const TraceBatch <int16_t>>> CreateTraceSaver(DataSourceBase& dataSource)
    {
        if (outputTrcFileName_ != "")
        {
            auto sourceLaneOffsets = dataSource.SelectedLanesWithinROI(config_.traceROI.roi);
            const auto sourceLaneWidth = dataSource.Layout().BlockWidth();
            const size_t numZmws = sourceLaneOffsets.size() * sourceLaneWidth;

            // conversion of source lanes (DataSource) into destination lanes (For the TraceFile).
            // The lane widths may be different.
            std::vector<DataSourceBase::UnitCellProperties> roiFeatures(numZmws);
            std::unordered_set<DataSourceBase::LaneIndex> destLaneSeen;
            std::vector<DataSourceBase::LaneIndex> destLanes;
            {
                // currate the features of the ROI ZMWs
                const auto allFeatures = dataSource.GetUnitCellProperties();
                size_t k=0;
                for(const auto laneOffset : sourceLaneOffsets)
                {
                    const uint64_t zmwIndex = laneOffset * sourceLaneWidth;
                    for(uint32_t j =0; j < sourceLaneWidth; j++)
                    {
                        roiFeatures[k] = allFeatures[zmwIndex+j];
                        k++;
                    }

                    const DataSourceBase::LaneIndex destLaneIndex = zmwIndex / laneSize;
                    if (destLaneSeen.find(destLaneIndex) == destLaneSeen.end())
                    {
                        destLanes.push_back(destLaneIndex);
                        destLaneSeen.insert(destLaneIndex);
                    }
                }
                if (k != numZmws)
                {
                    throw PBException("ROI selection algorithm is buggy");
                }
            }
            if (destLanes.size() != numZmws / laneSize)
            {
                PBLOG_WARN << "The lane calcuations are not as predicted: lanes:" << destLanes.size() 
                    << " numZmws/laneSize:" << numZmws/laneSize;
                PBLOG_WARN << "This can happen if the ROI is modulo hardware tile sizes but not modulo lane sizes";
            }
            DataSourceBase::LaneSelector blocks(destLanes);
            PBLOG_INFO << "Opening TraceSaver with output file " << outputTrcFileName_ << ", " << numZmws << " ZMWS.";
            outputTrcFile_ = std::make_unique<TraceFile>(outputTrcFileName_,
                                                         numZmws,
                                                         frames_);

            return std::make_unique<TraceSaverBody>(std::move(outputTrcFile_), roiFeatures, std::move(blocks));
        }
        else
        {
            return std::make_unique<NoopTraceSaverBody>();
        }
    }

    std::unique_ptr <TransformBody<const TraceBatch <int16_t>, BatchResult>>
    CreateBasecaller(const std::map<uint32_t, Data::BatchDimensions>& poolDims) const
    {
        return std::make_unique<BasecallerBody>(poolDims,
                                                config_.algorithm,
                                                movieConfig_,
                                                config_.system.maxPermGpuDataMB);
    }

    std::unique_ptr <LeafBody<BatchResult>> CreateBazSaver(const DataSourceRunner& source)
    {
        if (hasBazFile_)
        {
            auto features1 = source.GetUnitCellProperties();
            std::vector<uint32_t> features2;            
            transform(features1.begin(), features1.end(), back_inserter(features2), [](DataSourceBase::UnitCellProperties x){return x.flags;});

            return std::make_unique<BazWriterBody>(outputBazFile_,
                                                   source.NumFrames(),
                                                   source.UnitCellIds(),
                                                   features2,
                                                   config_,
                                                   zmwOutputStrideFactor_);
        } else
        {
            return std::make_unique<NoopBazWriterBody>();

        }
    }

    void RunAnalyzer()
    {
        // Names for the various graph stages
        SMART_ENUM(GraphProfiler, REPACKER, SAVE_TRACE, ANALYSIS, BAZWRITER);

        auto source = CreateSource();
#define POOLID_FIX
#ifdef POOLID_FIX
        std::vector<uint32_t> poolIds;
        uint32_t zmwsPerPool = config_.layout.zmwsPerLane * config_.layout.lanesPerPool;
        uint32_t numPools = source->NumZmw()/zmwsPerPool;
        for(uint32_t i=0; i<numPools;i++)
        {
            poolIds.push_back(i);
        }
        uint32_t runts = (source->NumZmw() % zmwsPerPool);
        if (runts)
        {
            poolIds.push_back(numPools);

        }
#else
        const auto& poolIds = source->PoolIds();
#endif
        try
        { 
          // this try block is to catch problems before `source` is destroyed. The destruction of WXDataSource is expensive
          // and not reliable. So better to catch and report exceptions here before they percolate to the top of the call stack...

            // TODO Some negotation between the source and repacker needs to happen,
            // which results in the creation of this map.  For now, hard code all
            // batches to be uniform dimensions.  In reality, the number of lanes
            // per pool may vary
            std::map<uint32_t, Data::BatchDimensions> poolDims;
            Data::BatchDimensions dims;
            dims.framesPerBatch = config_.layout.framesPerChunk;
            dims.laneWidth = config_.layout.zmwsPerLane;
            dims.lanesPerBatch = config_.layout.lanesPerPool;
            for (auto id : poolIds) poolDims[id] = dims;
#ifdef POOLID_FIX
            if (runts)
            {
                poolDims[numPools].lanesPerBatch = runts/dims.laneWidth;
            }
#endif
            GraphManager <GraphProfiler> graph(config_.system.numWorkerThreads);
            auto* repacker = graph.AddNode(CreateRepacker(source->GetDataSource().Layout()), GraphProfiler::REPACKER);
            repacker->AddNode(CreateTraceSaver(source->GetDataSource()), GraphProfiler::SAVE_TRACE);
            if (nop_ != 2)
            {
                auto* analyzer = repacker->AddNode(CreateBasecaller(poolDims), GraphProfiler::ANALYSIS);
                analyzer->AddNode(CreateBazSaver(*source), GraphProfiler::BAZWRITER);
            }

            size_t numChunksAnalyzed = 0;
            PacBio::Dev::QuietAutoTimer timer(0);

            const double chunkDurationMS = config_.layout.framesPerChunk
                                        / movieConfig_.frameRate
                                        * 1e3;

            // This snippet starts up a simulation on the WX2, if the WX2 is selected and the data path is one of the
            // loopback paths (anything but Normal).
            auto dsr = dynamic_cast<WXDataSource*>(&source->GetDataSource());
            if (dsr)
            {
                // wait for wxshim
                uint32_t waitCount = 0;
                while(! dsr->WXShimReady())
                {
                    if (waitCount++ > 100) throw PBException("WXShim was not ready");
                    PBLOG_NOTICE << "WX Shim not ready, waiting ...";
                    PacBio::POSIX::Sleep(1.0);
                }
            }
            if (dsr && dsr->GetDataPath() != DataPath_t::Normal)
            {
                TransmitConfig config(dsr->GetPlatform());
                config.condensed = true;
                config.frames = frames_; // total number of frames to transmit
                config.limitFileFrames = 512; // loop in coproc memory.
                config.hdf5input = inputTargetFile_ == "" ? "constant/123" : inputTargetFile_; // "alpha", "random";
                if (PacBio::POSIX::IsFile(config.hdf5input))
                {
                    config.mode = TransmitConfig::Mode::File;
                    PBLOG_NOTICE << "Trying to generate simulated loopback movie with file " << config.hdf5input;
                }
                else
                {
                    config.mode = TransmitConfig::Mode::Generated;
                    PBLOG_NOTICE << "Trying to generate simulated loopback movie with pattern " << config.hdf5input;
                }
                config.rate = config_.source.wx2SourceConfig.simulatedFrameRate;
                dsr->TransmitMovieSim(config);
            }

            uint64_t nopSuccesses = 0;
            uint64_t framesAnalyzed = 0;

            while (source->IsActive())
            {
                SensorPacketsChunk chunk;
                if (source->PopChunk(chunk, std::chrono::milliseconds{10}))
                {
                    PBLOG_INFO << "Analyzing chunk frames = ["
                        + std::to_string(chunk.StartFrame()) + ","
                        + std::to_string(chunk.StopFrame()) + ")";
                    if (nop_ == 1)
                    {
                        int16_t expectedPixel = 123;
                        for(const auto& batch : chunk)
                        {
                            for(int iblock = 0; iblock < 1; iblock ++)
                            {
                                int16_t actualPixel = batch.BlockData(iblock).Data()[0];
                                if (expectedPixel != actualPixel)
                                {
                                    PBLOG_ERROR << "Mismatched pixels, expected:" <<
                                        expectedPixel << " != actual:" << actualPixel <<
                                        " at " << batch.ToString();
                                }
                                else
                                {
                                    nopSuccesses++;
                                }
                            }
                        }
                    }
                    else
                    {
                        for (auto& batch : chunk)
                            repacker->ProcessInput(std::move(batch));
                        const auto& reports = graph.FlushAndReport(chunkDurationMS);

                        std::stringstream ss;
                        ss << "Chunk finished: Duty Cycle%, Avg Occupancy:\n";

                        for (auto& report: reports)
                        {
                            if (!report.realtime)
                            {
                                PBLOG_WARN << report.stage.toString()
                                        << " is currently slower than budgeted:  Duty Cycle%, Duration MS, Idle %, Occupancy -- "
                                        << report.dutyCycle * 100 << "%, "
                                        << report.avgDuration << "ms, "
                                        << report.idlePercent << "%, "
                                        << report.avgOccupancy;
                            }
                            ss << "\t\t" << report.stage.toString() << ": "
                            << report.dutyCycle * 100 << "%, "
                            << report.avgOccupancy << "\n";
                        }
                        PacBio::Logging::LogStream(PacBio::Logging::LogLevel::INFO) << ss.str();
                    }
                    PacBio::Cuda::Memory::ReportAllMemoryStats();

                    framesAnalyzed += chunk.NumFrames();
                    for(auto&& packet : chunk)
                    {
                        PacBio::Cuda::Memory::GetGlobalAllocator().ReturnHostAllocation(std::move(packet).RelinquishAllocation());
                    }
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
            PBLOG_INFO << "Total frames analyzed = " << framesAnalyzed
                    << " out of " << source->NumFrames() << " requested from source. ("
                    << (source->NumFrames() ? (100.0 * framesAnalyzed / source->NumFrames()) : -1) << "%)";
            if (nop_ == 1)
            {        
                PBLOG_INFO << "NOP pixel comparison successes = " << nopSuccesses;
            }
            timer.SetCount(numChunksAnalyzed);
            double chunkAnalyzeRate = timer.GetRate();
            PBLOG_INFO << "Analyzed " << numChunksAnalyzed
                    << " chunks at " << chunkAnalyzeRate << " chunks/sec"
                    << " (" << (source->NumZmw() * chunkAnalyzeRate)
                    << " zmws/sec)";
        }
        catch(const std::exception& ex)
        {
            PBLOG_ERROR << "Exception caught during graphmanager setup:" << ex.what();
            throw;
        }
    }

private:
    // Configuration objects
    SmrtBasecallerConfig config_;
    MovieConfig movieConfig_;

    std::string inputTargetFile_;
    std::string outputBazFile_;
    bool hasBazFile_ = false;
    size_t zmwOutputStrideFactor_ = 1;
    size_t numChunksPreloadInputQueue_ = 0;
    size_t numZmwLanes_ = 0;
    size_t frames_ = 0;
    bool cache_ = false;
    int nop_ = 0; ///< 0 = normal. 1 = don't process any SensorPackets at all. 2 =dont instantiate the basecaller, but allow repacker and tracesaver
    std::string outputTrcFileName_;
    std::unique_ptr<TraceFile> outputTrcFile_;
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
        parser.add_option("--outputtrcfile").help("Trace file output file (trc.h5). Optional");
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

        auto group3 = OptionGroup(parser, "Developer options",
                                  "For use by developers only");
        group3.add_option("--nop").type_int().set_default(0).help("Ways of making the analyzer do less");
        parser.add_option_group(group3);

        auto options = parser.parse_args(argc, (const char* const*) argv);
        ThreadedProcessBase::HandleGlobalOptions(options);

        Json::Value json = MergeConfigs(options.all("config"));
        PBLOG_DEBUG << json; // this does NOT work with --showconfig
        SmrtBasecallerConfig configs(json);
        auto validation = configs.Validate();
        if (validation.ErrorCount() > 0)
        {
            validation.PrintErrors();
            throw PBException("Json validation failed");
        }

        if (options.get("showconfig"))
        {
            std::cout << configs.Serialize() << std::endl;
            return 0;
        }

        if (options.is_set_by_user("numWorkerThreads"))
        {
            configs.system.numWorkerThreads = options.get("numWorkerThreads");
        }

        auto bc = std::make_unique<SmrtBasecaller>(configs);
        bc->HandleProcessArguments(parser.args());

        {
            PacBio::Logging::LogStream ls;
            ls << configs.Serialize();
        }

        bc->HandleProcessOptions(options);

        bc->Run();

    } catch (std::exception& ex) {
        PBLOG_ERROR << "Exception caught: " << ex.what();
        return 1;
    }

    return 0;
}
