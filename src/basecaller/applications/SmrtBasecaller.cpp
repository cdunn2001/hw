// Copyright (c) 2020-2021, Pacific Biosciences of California, Inc.
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
#include <appModules/BazWriterBody.h>
#include <appModules/BlockRepacker.h>
#include <appModules/PrelimHQFilter.h>
#include <appModules/TrivialRepacker.h>
#include <appModules/TraceFileDataSource.h>
#include <appModules/TraceSaver.h>
#include <basecaller/traceAnalysis/AnalysisProfiler.h>
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
#include <pacbio/text/String.h>
#define USE_WXIPCDATASOURCE
#ifdef USE_WXIPCDATASOURCE
#include <acquisition/wxipcdatasource/WXIPCDataSource.h>
#include <pacbio/datasource/SharedMemoryAllocator.h>
#else
#include <acquisition/datasource/WXDataSource.h>
#endif

#include <git-rev.h>

using namespace PacBio::Cuda::Memory;
using namespace PacBio::Graphs;
using namespace PacBio::Mongo;
using namespace PacBio::Mongo::Data;
using namespace PacBio::Acquisition::DataSource;
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

        frames_ = options.get("maxFrames");
        // TODO need validation or something, as this is probably a trace file input specific option
        nop_ = options.get("nop");

        if (nop_ == 1)
        {
#ifdef USE_WXIPCDATASOURCE
            throw PBException("--nop=1 must be used with the WX2DataSource and simulatedInputFile=constant/123 to get correct validation pattern.");
#else
            try {
                const auto& wxDataSource = boost::get<WX2SourceConfig>(config_.source.data());
                if (wxDataSource.simulatedInputFile!= "constant/123")
                    throw PBException("Dummy Exception");
            } catch (...) {
                throw PBException("--nop=1 must be used with the WX2DataSource and simulatedInputFile=constant/123 to get correct validation pattern.");
            }
#endif
        }

        // TODO these might need cleanup/moving?  At the least need to be able to set them
        // correctly if not using trace file input
        config_.source.Visit(
            [&](const auto& traceConfig) {
                const std::string& traceFile = traceConfig.traceFile;
                PBLOG_INFO << "Input Target: " << traceFile;
                MetaDataFromTraceFileSource(traceFile);
                GroundTruthFromTraceFileSource(traceFile);
            },
            [&](const WX2SourceConfig& wxConfig) {
                // FIXME An API to load metadata from ICS configuration is not finalized yet.
                // FIXME In the interest of rapid development, we are just loading a precanned metadata snippet.
                // TODO replace this call with the official command line API for loading metadata (JSON) from ICS.
                // for example, LoadMetaDataFromSequelFormat(json value of metadata);
                movieConfig_ = PacBio::Mongo::Data::MockMovieConfig();
            }
        );

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
            PBLOG_INFO << "  major/minor:" << dd.major << "/" << dd.minor;
            if (d.errorMessage != "") PBLOG_ERROR << "  Error Message:" << d.errorMessage;
            else PBLOG_INFO << "  No message";
            idevice++;
        }
    }

    void Run()
    {
        SetGlobalAllocationMode(CachingMode::ENABLED, AllocatorMode::CUDA);
        EnableHostCaching(AllocatorMode::MALLOC);

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

#if 0 // def USE_WXIPCDATASOURCE
        // TODO. Use curl localhost:23602/shared_memory/0
        //  baseAddress, key, numaNodes, segsz
        PacBio::DataSource::SharedMemoryAllocator::SharedMemoryAllocatorConfig memConfig;
        memConfig.baseAddress = 17'179'869'184;
        memConfig.size = 25'769'803'776;
        memConfig.numaBinding = 1;
        memConfig.removeSharedSegmentsOnDestruction = false;
        auto allo = std::make_unique<PacBio::DataSource::SharedMemoryAllocator>(memConfig);
#else
        const auto mode = AllocatorMode::SHARED_MEMORY;
        auto allo = CreateAllocator(mode, AllocationMarker(config_.source.GetEnum().toString()));
#endif
        DataSourceBase::Configuration datasourceConfig(layout, std::move(allo));
        datasourceConfig.numFrames = frames_;

        auto dataSource = config_.source.Visit(
            [&](const TraceReanalysis& config) -> std::unique_ptr<DataSourceBase>
            {
                return std::make_unique<TraceFileDataSource>(std::move(datasourceConfig), config);
            },
            [&](const TraceReplication& config) -> std::unique_ptr<DataSourceBase>
            {
                return std::make_unique<TraceFileDataSource>(std::move(datasourceConfig), config);
            },
            [&](const WX2SourceConfig& wx2SourceConfig) -> std::unique_ptr<DataSourceBase>
            {
#ifdef USE_WXIPCDATASOURCE
                WXIPCDataSourceConfig wxconfig;
                wxconfig.dataPath = DataPath_t(wx2SourceConfig.dataPath);
                if ( wxconfig.dataPath == DataPath_t::SimGen || wxconfig.dataPath == DataPath_t::SimLoop)
                {
                    throw PBException("wxconfig.simconfig needs to be set, not implemented yet.");
                }
                wxconfig.sleepDebug = wx2SourceConfig.sleepDebug;
                wxconfig.simulatedFrameRate = wx2SourceConfig.simulatedFrameRate;
                wxconfig.simulatedInputFile = wx2SourceConfig.simulatedInputFile;
                wxconfig.maxPopLoops = wx2SourceConfig.maxPopLoops;
                wxconfig.tilePoolFactor = wx2SourceConfig.tilePoolFactor;
                wxconfig.chipLayoutName = "Spider_1p0_NTO"; // FIXME this needs to be a command line parameter supplied by ICS.
                wxconfig.layoutDims[0] = wx2SourceConfig.wxlayout.lanesPerPacket;
                wxconfig.layoutDims[1] = wx2SourceConfig.wxlayout.framesPerPacket;
                wxconfig.layoutDims[2] = wx2SourceConfig.wxlayout.zmwsPerLane;
                wxconfig.verbosity = 100;
                return std::make_unique<WXIPCDataSource>(std::move(datasourceConfig), wxconfig);
#else
                // TODO this glue code is messy. It is gluing untyped strings to the strongly typed
                // enums of WX2, but this was on purpose to avoid entangling the config with configs from WX2.
                // I am not sure what the best next step is.  This is getting me going, so I am
                // going to leave it. MTL
                WXDataSourceConfig wxconfig;
                wxconfig.dataPath = DataPath_t(wx2SourceConfig.dataPath);
                wxconfig.platform = Platform(wx2SourceConfig.platform);
                wxconfig.sleepDebug = wx2SourceConfig.sleepDebug;
                wxconfig.simulatedFrameRate = wx2SourceConfig.simulatedFrameRate;
                wxconfig.simulatedInputFile = wx2SourceConfig.simulatedInputFile;
                wxconfig.maxPopLoops = wx2SourceConfig.maxPopLoops;
                wxconfig.tilePoolFactor = wx2SourceConfig.tilePoolFactor;
                wxconfig.chipLayoutName = "Spider_1p0_NTO"; // FIXME this needs to be a command line parameter supplied by ICS.
                wxconfig.layoutDims[0] = wx2SourceConfig.wxlayout.lanesPerPacket;
                wxconfig.layoutDims[1] = wx2SourceConfig.wxlayout.framesPerPacket;
                wxconfig.layoutDims[2] = wx2SourceConfig.wxlayout.zmwsPerLane;
                return std::make_unique<WXDataSource>(std::move(datasourceConfig), wxconfig);
#endif                
            }
        );
        return std::make_unique<DataSourceRunner>(std::move(dataSource));
    }

    std::unique_ptr<RepackerBody>
    CreateRepacker(const std::map<uint32_t, PacketLayout>& inputLayouts, size_t numZmw) const
    {
        BatchDimensions requiredDims;
        requiredDims.lanesPerBatch = config_.layout.lanesPerPool;
        requiredDims.framesPerBatch = config_.layout.framesPerChunk;
        requiredDims.laneWidth = config_.layout.zmwsPerLane;

        if (inputLayouts.empty())
            throw PBException("Received empty layout map!");

        const auto& layout1 = inputLayouts.begin()->second;
        const auto encoding = layout1.Encoding();
        const auto type = layout1.Type();
        const auto numFrames = layout1.NumFrames();
        const auto blockWidth = layout1.BlockWidth();
        bool sameBlockWidth = true;
        for (const auto& kv : inputLayouts)
        {
            const auto& layout = kv.second;
            if (layout.Encoding() != encoding)
                throw PBException("Inconsistent packet encodings");
            if (layout.Type() != type)
                throw PBException("Inconsistent packet types");
            if (layout.BlockWidth() % 32 != 0)
                throw PBException("Found packet with unsupported block width");
            if (layout.NumFrames() != numFrames)
                throw PBException("Found packets with different frame counts");
            if (layout.BlockWidth() != blockWidth)
                sameBlockWidth = false;
        }

        if (type == PacketLayout::FRAME_LAYOUT)
        {
            throw PBException("Frame Layouts not supported");
        }

        // check if the trivial repacker is a good fit first, as it's always preferred
        {
            bool trivial = true;
            if (numFrames != requiredDims.framesPerBatch) trivial = false;
            if (blockWidth != laneSize || !sameBlockWidth) trivial = false;

            // "sparse" reanalysis gets a free pass for this particular check.
            // "dense" setups need to be "close enough
            //
            // Note: This process could definitely use some tuning.  Just putting
            //       something here for now, we can revisit when we actually start
            //       having a surprising/undesirable decision made.
            if (type == PacketLayout::BLOCK_LAYOUT_DENSE)
            {
                double score = 0.0f;
                for (const auto& kv : inputLayouts)
                {
                    const auto& layout = kv.second;
                    const uint32_t b1 = layout.NumBlocks();
                    const uint32_t b2 = requiredDims.lanesPerBatch;
                    const uint32_t dist = std::max(b1,b2) - std::min(b1,b2);
                    const auto frac = static_cast<double>(dist) / b2;
                    if (frac < .1) score += 1;
                    else if (frac < .7) score += .5;
                    else score -= 1;
                }
                score /= inputLayouts.size();
                if (score < .9) trivial = false;
            }

            if (trivial)
            {
                PBLOG_INFO << "Instantiating TrivialRepacker";
                return std::make_unique<TrivialRepackerBody>(inputLayouts);
            }
        }

        // Now check if the BlockRepacker is a valid fit
        {
            bool valid = true;
            if (requiredDims.framesPerBatch % numFrames != 0) valid = false;
            if (valid)
            {
                PBLOG_INFO << "Instantiating BlockRepacker";
                const size_t numThreads = 6;
                return std::make_unique<BlockRepacker>(inputLayouts, requiredDims, numZmw, numThreads);
            }
        }

        throw PBException("No repacker exists that can handle the provided PacketLayouts");
    }


    std::unique_ptr <LeafBody<const TraceBatchVariant>> CreateTraceSaver(const DataSourceBase& dataSource)
    {
        if (outputTrcFileName_ != "")
        {
            auto sourceLaneOffsets = dataSource.SelectedLanesWithinROI(config_.traceSaver.roi);
            const auto sampleLayout = dataSource.PacketLayouts().begin()->second;
            const auto sourceLaneWidth = sampleLayout.BlockWidth();
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
                PBLOG_WARN << "The lane calculations are not as predicted: lanes:" << destLanes.size()
                    << " numZmws/laneSize:" << numZmws/laneSize;
                PBLOG_WARN << "This can happen if the ROI is modulo hardware tile sizes but not modulo lane sizes";
            }
            DataSourceBase::LaneSelector blocks(destLanes);
            PBLOG_INFO << "Opening TraceSaver with output file " << outputTrcFileName_ << ", " << numZmws << " ZMWS.";
            TraceDataType outputType = TraceDataType::INT16;
            if (config_.traceSaver.outFormat == TraceSaverConfig::OutFormat::UINT8)
            {
                outputType = TraceDataType::UINT8;
            } else if (config_.traceSaver.outFormat == TraceSaverConfig::OutFormat::Natural)
            {
                if (sampleLayout.Encoding() == PacketLayout::UINT8)
                {
                    outputType = TraceDataType::UINT8;
                }
            }
            outputTrcFile_ = std::make_unique<TraceFile>(outputTrcFileName_,
                                                         outputType,
                                                         numZmws,
                                                         dataSource.NumFrames());
            outputTrcFile_->Traces().Pedestal(dataSource.Pedestal());

            return std::make_unique<TraceSaverBody>(std::move(outputTrcFile_), roiFeatures, std::move(blocks));
        }
        else
        {
            return std::make_unique<NoopTraceSaverBody>();
        }
    }

    std::unique_ptr <TransformBody<const TraceBatchVariant, BatchResult>>
    CreateBasecaller(const std::map<uint32_t, Data::BatchDimensions>& poolDims) const
    {
        return std::make_unique<BasecallerBody>(poolDims,
                                                config_.algorithm,
                                                movieConfig_,
                                                config_.system);
    }

    std::unique_ptr<MultiTransformBody<BatchResult, std::unique_ptr<PacBio::BazIO::BazBuffer>>>
    CreatePrelimHQFilter(size_t numZmw, const std::map<uint32_t, Data::BatchDimensions>& poolDims)
    {
        return std::make_unique<PrelimHQFilterBody>(numZmw, poolDims, config_);
    }

    std::unique_ptr <LeafBody<std::unique_ptr<PacBio::BazIO::BazBuffer>>>
    CreateBazSaver(const DataSourceRunner& source, const std::map<uint32_t, Data::BatchDimensions>& poolDims)
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
                                                   poolDims,
                                                   config_,
                                                   movieConfig_);
        } else
        {
            return std::make_unique<NoopBazWriterBody>();

        }
    }

    void RunAnalyzer()
    {
        // Names for the various graph stages
        SMART_ENUM(GraphProfiler, REPACKER, SAVE_TRACE, ANALYSIS, PRE_HQ, BAZWRITER);

        auto source = CreateSource();

        PBLOG_INFO << "Number of analysis zmwLanes = " << source->NumZmw() / laneSize;
        PBLOG_INFO << "Number of analysis chunks = " << source->NumFrames() /
                                                        config_.layout.framesPerChunk;

        try
        {
            // this try block is to catch problems before `source` is destroyed. The destruction of WXDataSource is expensive
            // and not reliable. So better to catch and report exceptions here before they percolate to the top of the call stack...

            auto repacker = CreateRepacker(source->PacketLayouts(), source->NumZmw());
            auto poolDims = repacker->BatchLayouts();

            GraphManager<GraphProfiler> graph(config_.system.numWorkerThreads);
            auto* inputNode = graph.AddNode(std::move(repacker), GraphProfiler::REPACKER);
            inputNode->AddNode(CreateTraceSaver(source->GetDataSource()), GraphProfiler::SAVE_TRACE);
            if (nop_ != 2)
            {
                auto* analyzer = inputNode->AddNode(CreateBasecaller(poolDims), GraphProfiler::ANALYSIS);
                auto* preHQ = analyzer->AddNode(CreatePrelimHQFilter(source->NumZmw(), poolDims), GraphProfiler::PRE_HQ);
                preHQ->AddNode(CreateBazSaver(*source, poolDims), GraphProfiler::BAZWRITER);
            }

            size_t numChunksAnalyzed = 0;
            PacBio::Dev::QuietAutoTimer timer(0);

            const double chunkDurationMS = config_.layout.framesPerChunk
                                        / movieConfig_.frameRate
                                        * 1e3;

            uint64_t nopSuccesses = 0;
            uint64_t framesAnalyzed = 0;
            uint64_t framesSinceBigReports = 0;

            source->Start();
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
                        PacBio::Dev::QuietAutoTimer t;
                        for (auto& batch : chunk)
                            inputNode->ProcessInput(std::move(batch));
                        const auto& reports = graph.SynchronizeAndReport(chunkDurationMS);

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
                        PBLOG_INFO << t.GetElapsedMilliseconds() / 1000 << " seconds to process chunk";
                    }
                    numChunksAnalyzed++;
                    framesSinceBigReports += config_.layout.framesPerChunk;
                    framesAnalyzed += chunk.NumFrames();

                    if (framesSinceBigReports >= config_.monitoringReportInterval)
                    {
                        PacBio::Cuda::Memory::ReportAllMemoryStats();
                        Basecaller::AnalysisProfiler::IntermediateReport();
                        if (config_.system.analyzerHardware != Basecaller::ComputeDevices::Host)
                            Basecaller::IOProfiler::IntermediateReport();
                        framesSinceBigReports = 0;
                    }
                }

                if (this->ExitRequested())
                {
                    source->RequestExit();
                    break;
                }
            }
            inputNode->FlushNode();

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

    std::string outputBazFile_;
    bool hasBazFile_ = false;
    size_t frames_ = 0;
    int nop_ = 0; ///< 0 = normal. 1 = don't process any SensorPackets at all. 2 =dont instantiate the basecaller, but allow repacker and tracesaver
    std::string outputTrcFileName_;
    std::unique_ptr<TraceFile> outputTrcFile_;
};

int main(int argc, char* argv[])
{
    try
    {
        std::ostringstream cliArgs;
        for (int i = 0; i < argc; ++i)
        {
            cliArgs << argv[i] << " ";
        }
        PBLOG_INFO << cliArgs.str();

        auto parser = ProcessBase::OptionParserFactory();
        std::stringstream ss;
        ss << "Prototype to demonstrate mongo basecaller"
           << "\n git branch: " << cmakeGitBranch()
           << "\n git hash: " << cmakeGitHash()
           << "\n git commit date: " << cmakeGitCommitDate();
        parser.description(ss.str());

        const std::string version = "0.1";
        parser.version(version);

        parser.epilog("");

        parser.add_option("--config").action_append().help("Loads JSON configuration file, JSON string or Boost ptree value");
        parser.add_option("--showconfig").action_store_true().help("Shows the entire configuration namespace and exits (before validation)");
        parser.add_option("--validateconfig").action_store_true().help("Validates the supplied configuration settings and exits.");

        parser.add_option("--outputbazfile").set_default("").help("BAZ output file");
        parser.add_option("--outputtrcfile").help("Trace file output file (trc.h5). Optional");
        parser.add_option("--numWorkerThreads").type_int().set_default(0).help("Number of compute threads to use.  ");
        parser.add_option("--maxFrames").type_int().set_default(0).help("Specifies maximum number of frames to run. 0 means unlimited");

        auto group1 = OptionGroup(parser, "Developer options",
                                  "For use by developers only");
        group1.add_option("--nop").type_int().set_default(0).help("Ways of making the analyzer do less");
        parser.add_option_group(group1);

        auto options = parser.parse_args(argc, (const char* const*) argv);
        auto unusedArgs = parser.args();
        if (unusedArgs.size() > 0)
        {
            throw PBException("There were unrecognized arguments on the command line: " 
                + PacBio::Text::String::Join(unusedArgs.begin(), unusedArgs.end(), ' ')
                + ". Did you forget '--' before an option?");
        }

        ThreadedProcessBase::HandleGlobalOptions(options);

        Json::Value json = MergeConfigs(options.all("config"));
        PBLOG_DEBUG << json; // this does NOT work with --showconfig
        SmrtBasecallerConfig configs(json);
        if (options.get("showconfig"))
        {
            std::cout << configs.Serialize() << std::endl;
            return 0;
        }

        auto validation = configs.Validate();
        if (validation.ErrorCount() > 0)
        {
            validation.PrintErrors();
            throw PBException("Json validation failed");
        }
        if (options.get("validateconfig"))
        {
            return 0;
        }

        PBLOG_INFO << "Version " << version;
        PBLOG_INFO << "git branch: " << cmakeGitBranch();
        PBLOG_INFO << "git commit hash: " << cmakeGitHash();
        PBLOG_INFO << "git commit date: " << cmakeGitCommitDate();

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
