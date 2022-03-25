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
#include <appModules/RealTimeMetrics.h>
#include <appModules/TrivialRepacker.h>
#include <appModules/TraceFileDataSource.h>
#include <appModules/TraceSaver.h>
#include <basecaller/traceAnalysis/AnalysisProfiler.h>
#include <bazio/file/ZmwInfo.h>
#include <dataTypes/configs/SmrtBasecallerConfig.h>
#include <dataTypes/configs/AnalysisConfig.h>
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
#include <pacbio/datasource/ZmwFeatures.h>
#include <pacbio/ipc/JSON.h>
#include <pacbio/logging/Logger.h>
#include <pacbio/POSIX.h>
#include <pacbio/process/OptionParser.h>
#include <pacbio/process/ProcessBase.h>
#include <pacbio/sensor/SparseROI.h>
#include <pacbio/System.h>
#include <pacbio/text/String.h>
#include <acquisition/wxipcdatasource/WXIPCDataSource.h>
#include <pacbio/datasource/SharedMemoryAllocator.h>

#include <app-common/ProgressMessage.h>

#include <boost/iostreams/device/file_descriptor.hpp>
#include <boost/iostreams/stream.hpp>

#include <git-rev.h>

using namespace PacBio::Cuda::Memory;
using namespace PacBio::Graphs;
using namespace PacBio::Mongo;
using namespace PacBio::Mongo::Data;
using namespace PacBio::Acquisition::DataSource;
using namespace PacBio::Sensor;
using namespace PacBio::BazIO;

using namespace PacBio::Application;
using namespace PacBio::Configuration;
using namespace PacBio::DataSource;
using namespace PacBio::Process;
using namespace PacBio::Primary;
using namespace PacBio::File;
using namespace PacBio::IPC;

// ^^^^
///////

SMART_ENUM(SmrtBasecallerStages,
           StartUp,
           Analyze,
           Shutdown);

using SmrtBasecallerProgressMessage = ProgressMessage<SmrtBasecallerStages>;
using SmrtBasecallerStageReporter = SmrtBasecallerProgressMessage::StageReporter;

SmrtBasecallerProgressMessage::Table stages = {
        { "StartUp",    { false, 0, 10 } },
        { "Analyze",    {  true, 1, 80 } },
        { "Shutdown",   { false, 2, 10 } }
};

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
        const auto framesPerChunk = config_.layout.framesPerChunk;
        if (frames_ % framesPerChunk != 0)
        {
            frames_ = (frames_ + framesPerChunk - 1) / framesPerChunk * framesPerChunk;
            PBLOG_INFO << "Rounding up maxFrames to " << frames_ << ", to be "
                       << "an even multiple of the chunk size (" << framesPerChunk << ")";
        }
        // TODO need validation or something, as this is probably a trace file input specific option
        nop_ = options.get("nop");
        statusFileDescriptor_ = options.get("statusfd");
        progressMessage_ = std::make_unique<SmrtBasecallerProgressMessage>(stages,
                                                                           "PA_BASECALLER_STATUS", statusFileDescriptor_);

        if (nop_ == 1)
        {
            throw PBException("--nop=1 must be used with the WX2DataSource and simulatedInputFile=constant/123 to get correct validation pattern.");
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

        auto devices = PacBio::Cuda::CudaAllGpuDevices();
        bool usesGpu = config_.algorithm.ComputingMode() != BasecallerAlgorithmConfig::ComputeMode::PureHost;
        if(devices.size() == 0 && usesGpu)
        {
            throw PBException("No CUDA devices available on this computer. "
                              "Did you mean to use --config=system.analyzerHardware=Host?");
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
        const bool pureHost = config_.algorithm.ComputingMode() == BasecallerAlgorithmConfig::ComputeMode::PureHost;
        // resetMem is a `Finally` object, that will return the memory settings to the default
        // at the end of scope (which will prevent potential issues during static teardown)
        auto resetMem = SetGlobalAllocationMode(CacheMode::GLOBAL_CACHE,
                                                pureHost
                                                ? AllocatorMode::MALLOC
                                                : AllocatorMode::CUDA);

        RunAnalyzer();
        Join();
    }

    const SmrtBasecallerConfig& Config() const
    { return config_; }

private:
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

        auto allo = config_.source.Visit(
            [&](const auto& config)
            {
                using Config_t = std::decay_t<decltype(config)>;
                static_assert(std::is_same_v<Config_t, TraceReanalysis>
                              || std::is_same_v<Config_t, TraceReplication>);

                return (config_.algorithm.ComputingMode() == BasecallerAlgorithmConfig::ComputeMode::PureHost)
                    ? CreateMallocAllocator(config_.source.GetEnum().toString(),
                                            CacheMode::GLOBAL_CACHE)
                    : CreatePinnedAllocator(config_.source.GetEnum().toString(),
                                            CacheMode::GLOBAL_CACHE);
            },
            [&](const WXIPCDataSourceConfig& wx2SourceConfig)
            {
                WXIPCDataSource::WaitForDaemon(wx2SourceConfig);
                return CreateSharedHugePinnedAllocator(config_.source.GetEnum().toString(),
                                                       WXIPCDataSource::CreateAllocator(wx2SourceConfig),
                                                       CacheMode::PRIVATE_CACHE);
            }
        );

        DataSourceBase::Configuration datasourceConfig(layout, std::move(allo));
        datasourceConfig.numFrames = frames_;

        auto dataSource = config_.source.Visit(
            [&](const TraceReanalysis& config) -> std::unique_ptr<DataSourceBase>
            {
                auto ds = std::make_unique<TraceFileDataSource>(std::move(datasourceConfig), config);
                ds->LoadGroundTruth(config_.algorithm);
                return ds;
            },
            [&](const TraceReplication& config) -> std::unique_ptr<DataSourceBase>
            {
                return std::make_unique<TraceFileDataSource>(std::move(datasourceConfig), config);
            },
            [&](const WXIPCDataSourceConfig& config) -> std::unique_ptr<DataSourceBase>
            {
                if (config.dataPath == DataPath_t::SimGen || config.dataPath == DataPath_t::SimLoop)
                {
                    throw PBException("config.simConfig needs to be set, not implemented yet.");
                }
                return std::make_unique<WXIPCDataSource>(std::move(datasourceConfig), config);
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

    ScanData::Data CreateExperimentMetadata(const DataSourceRunner& dataSource, const AnalysisConfig& analysisConfig)
    {
        ScanData::Data expMetadata;

        auto& runInfo = expMetadata.runInfo;
        runInfo.platformId = ScanData::RunInfoData::ToPlatformId(dataSource.Platform());
        runInfo.instrumentName = dataSource.InstrumentName();
        // FIXME: Stub in a default HQRFMethod for now. This should eventually go away
        // or be renamed and moved elsewhere.
        runInfo.hqrfMethod = "N2";

        auto& acqParams = expMetadata.acqParams;
        acqParams.aduGain = analysisConfig.movieInfo.photoelectronSensitivity;
        acqParams.frameRate = analysisConfig.movieInfo.frameRate;
        acqParams.numFrames = dataSource.NumFrames();

        // FIXME: For running on Sequel, the chip layout name is needed here.
        // We want to eventually move to storing directly into the trace file the
        // chip layout information.
        constexpr std::string_view defaultLayoutName = "KestrelPOCRTO3";

        auto& chipInfo = expMetadata.chipInfo;
        chipInfo.layoutName = defaultLayoutName;
        chipInfo.analogRefSnr = analysisConfig.movieInfo.refSnr;
        const auto& imagePsf = dataSource.ImagePsfMatrix();
        chipInfo.imagePsf.resize(boost::extents[imagePsf.shape()[0]][imagePsf.shape()[1]]);
        chipInfo.imagePsf = imagePsf;
        const auto& crossTalk = dataSource.CrosstalkFilterMatrix();
        chipInfo.xtalkCorrection.resize(boost::extents[crossTalk.shape()[0]][crossTalk.shape()[1]]);
        chipInfo.xtalkCorrection = crossTalk;

        auto& dyeSet = expMetadata.dyeSet;
        const size_t numAnalogs = analysisConfig.movieInfo.analogs.size();
        dyeSet.numAnalog = static_cast<uint16_t>(numAnalogs);
        dyeSet.relativeAmp.resize(numAnalogs);
        dyeSet.excessNoiseCV.resize(numAnalogs);
        dyeSet.ipdMean.resize(numAnalogs);
        dyeSet.pulseWidthMean.resize(numAnalogs);
        dyeSet.pw2SlowStepRatio.resize(numAnalogs);
        dyeSet.ipd2SlowStepRatio.resize(numAnalogs);
        dyeSet.baseMap = "";
        for (size_t i = 0; i < numAnalogs; i++)
        {
            const auto& am = analysisConfig.movieInfo.analogs[i];
            dyeSet.relativeAmp[i] = am.relAmplitude;
            dyeSet.excessNoiseCV[i] = am.excessNoiseCV;
            dyeSet.ipdMean[i] = am.interPulseDistance;
            dyeSet.pulseWidthMean[i] = am.pulseWidth;
            dyeSet.pw2SlowStepRatio[i] = am.pw2SlowStepRatio;
            dyeSet.ipd2SlowStepRatio[i] = am.ipd2SlowStepRatio;
            dyeSet.baseMap += am.baseLabel;
        }

        expMetadata.acquisitionXML = "ADD_ME";

        return expMetadata;
    }

    std::unique_ptr<LeafBody<const TraceBatchVariant>> CreateTraceSaver(const DataSourceRunner& dataSource,
                                                                        const std::map<uint32_t, Data::BatchDimensions>& poolDims,
                                                                        const AnalysisConfig& analysisConfig,
                                                                        const ScanData::Data& experimentMetadata)
    {
        if (outputTrcFileName_ != "")
        {
            auto selection = SelectedBasecallerLanesWithinROI(dataSource, config_.traceSaver.roi, laneSize);
            const auto actualZmw = selection.size() * laneSize;

            std::vector<uint32_t> fullBatchIds;
            fullBatchIds.reserve(dataSource.NumZmw());
            PBLOG_DEBUG << "poolDims.size:" << poolDims.size();
            for (const auto& kv : poolDims)
            {
                PBLOG_DEBUG << "Pooldims:" << kv.first << " " << kv.second.ZmwsPerBatch()
                            << " " << kv.second.laneWidth << "," << kv.second.lanesPerBatch
                            << "," << kv.second.framesPerBatch;
                fullBatchIds.insert(fullBatchIds.end(), kv.second.ZmwsPerBatch(), kv.first);
            }
            if(fullBatchIds.size() != dataSource.NumZmw())
            {
                throw PBException("fullBatchIds.size():" + std::to_string(fullBatchIds.size()) 
                    + " != dataSource.NumZmw():" + std::to_string(dataSource.NumZmw()));
            }

            const auto& fullHoleIds = dataSource.UnitCellIds();
            const auto& fullProperties = dataSource.GetUnitCellProperties();

            std::vector<uint32_t> holeNumbers(actualZmw);
            std::vector<DataSourceBase::UnitCellProperties> properties(actualZmw);
            std::vector<uint32_t> batchIds(actualZmw);

            size_t idx = 0;
            for (const auto& lane : selection)
            {
                size_t currZmw = lane*laneSize;
                for (size_t i = 0; i < laneSize; ++i)
                {
                    assert(idx < actualZmw);
                    holeNumbers[idx] = fullHoleIds[currZmw];
                    properties[idx] = fullProperties[currZmw];
                    batchIds[idx] = fullBatchIds[currZmw];

                    currZmw++;
                    idx++;
                }
            }
            assert(idx == actualZmw);

            const auto sampleLayout = dataSource.PacketLayouts().begin()->second;
            auto dataType = TraceDataType::INT16;
            if (config_.traceSaver.outFormat == TraceSaverConfig::OutFormat::UINT8)
            {
                dataType = TraceDataType::UINT8;
            } else if (config_.traceSaver.outFormat == TraceSaverConfig::OutFormat::Natural)
            {
                if (sampleLayout.Encoding() == PacketLayout::UINT8)
                {
                    dataType = TraceDataType::UINT8;
                }
            }

            return std::make_unique<TraceSaverBody>(outputTrcFileName_,
                                                    dataSource.NumFrames(),
                                                    std::move(selection),
                                                    sampleLayout.NumFrames(),
                                                    sampleLayout.BlockWidth(),
                                                    dataType,
                                                    holeNumbers,
                                                    properties,
                                                    batchIds,
                                                    experimentMetadata,
                                                    analysisConfig);
        }
        else
        {
            return std::make_unique<NoopTraceSaverBody>();
        }
    }

    std::unique_ptr<TransformBody<const TraceBatchVariant, BatchResult>>
    CreateBasecaller(const std::map<uint32_t, Data::BatchDimensions>& poolDims, const AnalysisConfig& analysisConfig) const
    {
        return std::make_unique<BasecallerBody>(poolDims,
                                                config_.algorithm,
                                                analysisConfig,
                                                config_.system);
    }

    std::unique_ptr<MultiTransformBody<BatchResult, std::unique_ptr<PacBio::BazIO::BazBuffer>>>
    CreatePrelimHQFilter(size_t numZmw, const std::map<uint32_t, Data::BatchDimensions>& poolDims)
    {
        return std::make_unique<PrelimHQFilterBody>(numZmw, poolDims, config_);
    }

    std::unique_ptr <LeafBody<std::unique_ptr<PacBio::BazIO::BazBuffer>>>
    CreateBazSaver(const DataSourceRunner& source, const std::map<uint32_t, Data::BatchDimensions>& poolDims,
                   const ScanData::Data& experimentMetadata)
    {
        if (hasBazFile_)
        {
            auto props = source.GetUnitCellProperties();

            // NOTE: UnitCellProperties currently defines x,y as int32_t.
            std::vector<uint16_t> unitX;
            std::vector<uint16_t> unitY;
            std::vector<uint8_t> unitTypes;
            std::vector<uint32_t> unitFeatures;
            transform(props.begin(), props.end(), back_inserter(unitX),
                      [](DataSourceBase::UnitCellProperties x){ return static_cast<uint16_t>(x.x); });
            transform(props.begin(), props.end(), back_inserter(unitY),
                      [](DataSourceBase::UnitCellProperties x){ return static_cast<uint16_t>(x.y); });
            transform(props.begin(), props.end(), back_inserter(unitTypes),
                      [](DataSourceBase::UnitCellProperties x){ return static_cast<uint8_t>(x.type); });
            transform(props.begin(), props.end(), back_inserter(unitFeatures),
                      [](DataSourceBase::UnitCellProperties x){ return static_cast<uint32_t>(x.flags); });

            ZmwInfo zmwInfo(ZmwInfo::Data(source.UnitCellIds(), unitTypes, unitX, unitY, unitFeatures));

            return std::make_unique<BazWriterBody>(outputBazFile_,
                                                   source.NumFrames(),
                                                   zmwInfo,
                                                   poolDims,
                                                   config_,
                                                   experimentMetadata);
        } else
        {
            return std::make_unique<NoopBazWriterBody>();

        }
    }

    std::unique_ptr<TransformBody<BatchResult, BatchResult>>
    CreateRealTimeMetrics(const DataSourceRunner& dataSource, const std::map<uint32_t, Data::BatchDimensions>& poolDims)
    {
        if (!config_.realTimeMetrics.regions.empty())
        {
            std::vector<DataSourceBase::LaneSelector> selections;
            std::vector<std::vector<uint32_t>> properties;
            for (const auto& region : config_.realTimeMetrics.regions)
            {
                auto selection = SelectedBasecallerLanesWithinROI(dataSource, region.roi, laneSize);

                const auto actualZmw = selection.size() * laneSize;

                const auto& fullProperties = dataSource.GetUnitCellProperties();
                std::vector<uint32_t> selectionFeatures(actualZmw);

                size_t idx = 0;
                for (const auto& lane: selection)
                {
                    size_t currZmw = lane * laneSize;
                    for (size_t i = 0; i < laneSize; ++i)
                    {
                        assert(idx < actualZmw);
                        selectionFeatures[idx] = fullProperties[currZmw].flags;
                        currZmw++;
                        idx++;
                    }
                }
                assert(idx == actualZmw);
                selections.emplace_back(std::move(selection));
                properties.emplace_back(selectionFeatures);
            }

            return std::make_unique<RealTimeMetrics>(config_.algorithm.Metrics.framesPerHFMetricBlock,
                                                     poolDims.size(),
                                                     std::move(config_.realTimeMetrics.regions),
                                                     std::move(selections), properties);
        }
        else
        {
            return std::make_unique<NoopRealTimeMetrics>();
        }
    }

    void RunAnalyzer()
    {
        SmrtBasecallerStageReporter startUpRpt(progressMessage_.get(), SmrtBasecallerStages::StartUp, 300);

        // Names for the various graph stages
        SMART_ENUM(GraphProfiler, REPACKER, SAVE_TRACE, ANALYSIS, PRE_HQ, BAZWRITER, RT_METRICS);

        auto source = CreateSource();
        AnalysisConfig analysisConfig;
        analysisConfig.movieInfo = source->MovieInformation();
        analysisConfig.encoding = source->PacketLayouts().begin()->second.Encoding();
        analysisConfig.pedestal = source->Pedestal();

        PBLOG_INFO << "Number of analysis zmwLanes = " << source->NumZmw() / laneSize;
        PBLOG_INFO << "Number of analysis chunks = " << source->NumFrames() /
                                                        config_.layout.framesPerChunk;



        try
        {
            // this try block is to catch problems before `source` is destroyed. The destruction of WXDataSource is expensive
            // and not reliable. So better to catch and report exceptions here before they percolate to the top of the call stack...

            auto repacker = CreateRepacker(source->PacketLayouts(), source->NumZmw());
            auto poolDims = repacker->BatchLayouts();

            auto experimentData = CreateExperimentMetadata(*source, analysisConfig);

            GraphManager<GraphProfiler> graph(config_.system.numWorkerThreads);
            auto* inputNode = graph.AddNode(std::move(repacker), GraphProfiler::REPACKER);
            inputNode->AddNode(CreateTraceSaver(*source, poolDims, analysisConfig, experimentData), GraphProfiler::SAVE_TRACE);
            if (nop_ != 2)
            {
                auto* analyzer = inputNode->AddNode(CreateBasecaller(poolDims, analysisConfig), GraphProfiler::ANALYSIS);
                auto* rtMetrics = analyzer->AddNode(CreateRealTimeMetrics(*source, poolDims), GraphProfiler::RT_METRICS);
                auto* preHQ = rtMetrics->AddNode(CreatePrelimHQFilter(source->NumZmw(), poolDims), GraphProfiler::PRE_HQ);
                preHQ->AddNode(CreateBazSaver(*source, poolDims, experimentData), GraphProfiler::BAZWRITER);
            }

            size_t numChunksAnalyzed = 0;
            PacBio::Dev::QuietAutoTimer timer(0);

            const double chunkDurationMS = config_.layout.framesPerChunk
                                        / analysisConfig.movieInfo.frameRate
                                        * 1e3;

            uint64_t nopSuccesses = 0;
            uint64_t framesAnalyzed = 0;
            uint64_t framesSinceBigReports = 0;

            source->Start();
            startUpRpt.Update(1);

            SmrtBasecallerStageReporter analyzeStageRpt(progressMessage_.get(), SmrtBasecallerStages::Analyze, frames_, 60);
            while (source->IsActive())
            {
                SensorPacketsChunk chunk;
                if (source->PopChunk(chunk, std::chrono::milliseconds{10}))
                {
                    PBLOG_INFO << "Analyzing chunk frames = ["
                        + std::to_string(chunk.StartFrame()) + ","
                        + std::to_string(chunk.StopFrame()) + ")";
                    PacBio::Dev::QuietAutoTimer t;
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
                        {
#if 0
// uncomment this to get verbose dumps of the raw data
                            auto dv = batch.BlockData(0);
                            std::ostringstream oss;
                            for(uint32_t x = 0; x < 32 && x < dv.Count(); x++)
                            {
                                oss << std::hex << (int)dv.Data()[x] << " ";
                            }
                            PBLOG_INFO << "DUMP:" << (void*)dv.Data() << ":" << oss.str();

                            static bool first = true;
                            if (first)
                            {
                                first = false;
                                std::stringstream ss;
                                ss << "/home/UNIXHOME/mlakata/git/pa-common/build_mongo/x86_64/Release_gcc/tests/pacbio/memory/pbishmtool -m 0x50420041 -a ";
                                ss << (void*)(dv.Data()) << "-" << (void*)(dv.Data() + 32);
                                PBLOG_NOTICE << PacBio::System::Run(ss.str());
                            }
#endif
                            inputNode->ProcessInput(std::move(batch));
                        }
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
                    }
                    PBLOG_INFO << t.GetElapsedMilliseconds() / 1000 << " seconds to process chunk";
                    numChunksAnalyzed++;
                    framesSinceBigReports += config_.layout.framesPerChunk;
                    framesAnalyzed += chunk.NumFrames();

                    analyzeStageRpt.Update(framesAnalyzed);

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

            SmrtBasecallerStageReporter shutdownRpt(progressMessage_.get(), SmrtBasecallerStages::Shutdown, 300);
            inputNode->FlushNode();

            PBLOG_INFO << "Exited chunk analysis loop.";
            const double framePercentage =  (source->NumFrames() ? (100.0 * framesAnalyzed / source->NumFrames()) : -1);
            // As this is all integer math, I'm not worried about round off error to 99.99 or something silly like that.
            // Note that the framesAnalyzed *might* exceed the requested frames if the last chunk is not truncated correctly.
            if (framePercentage < 100.0)
            {
                PBLOG_WARN << "Not all Frames analyzed = " << framesAnalyzed
                        << " out of " << source->NumFrames() << " requested from source. ("
                        << framePercentage << "%)";
            }
            else if (framePercentage > 100.0)
            {
                PBLOG_NOTICE << "Slightly concerned that the number of framesAnalyzed (" << framesAnalyzed << ") exceeded the requested number ("
                    << source->NumFrames() << ") but not concerned enough to do something about it. Just passively-aggressively pointing it out."
                    << " You do realize that is not supposed to happen, right?";
            }
            else
            {
                PBLOG_INFO << "All frames analyzed = " << framesAnalyzed;
            }
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


            shutdownRpt.Update(1);
        }
        catch(const std::exception& ex)
        {
            PBLOG_ERROR << "Exception caught during graphmanager setup:" << ex.what();
            progressMessage_->Exception(ex.what());
            throw;
        }
    }

    DataSourceBase::LaneSelector SelectedBasecallerLanesWithinROI(const DataSourceRunner& dataSource, const std::vector<std::vector<int>>& roi, uint laneSize)
    {
        auto sourceLaneOffsets = dataSource.SelectedLanesWithinROI(roi);
        const auto sampleLayout = dataSource.PacketLayouts().begin()->second;
        const auto sourceLaneWidth = sampleLayout.BlockWidth();

        if (!(sourceLaneWidth % laneSize == 0
              || laneSize % sourceLaneWidth == 0))
        {
            throw PBException("Cannot handle incoming sensor lane width of " + std::to_string(sourceLaneWidth) + ". "
                              "It is neither a multiple nor a even divisor of the analysis lane width of "
                              + std::to_string(laneSize));
        }

        // Conversion of source lanes (DataSource) into lanes with given size.
        // The lane widths may be different.  Depending on the incoming lane width, the
        // resulting ROI may have more ZMW, as selecting one ZMW from a lane gives you the
        // entire zmw lane.
        std::set<DataSourceBase::LaneIndex> destLaneSeen;
        {
            for(const auto laneOffset : sourceLaneOffsets)
            {
                const uint64_t zmwIndex = laneOffset * sourceLaneWidth;
                for (size_t i = zmwIndex; i < zmwIndex + sourceLaneWidth; i += laneSize)
                {
                    destLaneSeen.insert(i / laneSize);
                }
            }
        }

        const auto requestedZmw = sourceLaneOffsets.size() * sourceLaneWidth;
        const auto actualZmw = destLaneSeen.size() * laneSize;
        if (actualZmw < requestedZmw)
            throw PBException("Error handling the roi lane selection");
        else if (actualZmw > requestedZmw)
        {
            PBLOG_WARN << "Saving " << actualZmw << " ZMW when only " << requestedZmw << " were requested.";
            PBLOG_WARN << "This can happen if the ROI is modulo hardware tile sizes but not modulo lane sizes";
        }

        std::vector<DataSourceBase::LaneIndex> destLanes(destLaneSeen.begin(), destLaneSeen.end());
        DataSourceBase::LaneSelector selection(destLanes);
        return selection;
    }

private:
    // Configuration objects
    SmrtBasecallerConfig config_;

    std::string outputBazFile_;
    bool hasBazFile_ = false;
    size_t frames_ = 0;
    int nop_ = 0; ///< 0 = normal. 1 = don't process any SensorPackets at all. 2 =dont instantiate the basecaller, but allow repacker and tracesaver
    std::string outputTrcFileName_;
    std::unique_ptr<TraceFile> outputTrcFile_;
    int statusFileDescriptor_ = 1;
    std::unique_ptr<SmrtBasecallerProgressMessage> progressMessage_;
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

        auto parser = ProcessBase::OptionParserFactory();
        std::stringstream ss;
        ss << "Prototype to demonstrate mongo basecaller"
           << "\n git branch: " << cmakeGitBranch()
           << "\n git hash: " << cmakeGitHash()
           << "\n git commit date: " << cmakeGitCommitDate();
        parser.description(ss.str());

        const std::string version = STRINGIFIED_SOFTWARE_VERSION;
        parser.version(version);

        parser.epilog("");

        parser.add_option("--config").action_append().help("Loads JSON configuration file, JSON string or Boost ptree value");
        parser.add_option("--showconfig").action_store_true().help("Shows the entire configuration namespace and exits (before validation)");
        parser.add_option("--validateconfig").action_store_true().help("Validates the supplied configuration settings and exits.");

        parser.add_option("--outputbazfile").set_default("").help("BAZ output file");
        parser.add_option("--outputtrcfile").help("Trace file output file (trc.h5). Optional");
        parser.add_option("--numWorkerThreads").type_int().set_default(0).help("Number of compute threads to use.  ");
        parser.add_option("--maxFrames").type_int().set_default(0).help("Specifies maximum number of frames to run. 0 means unlimited");
        parser.add_option("--statusfd").type_int().set_default(-1).help("Write status messages to this file description. Default -1 (null)");

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
        SmrtBasecallerConfig configs(json);
        if (options.get("showconfig"))
        {
            std::cout << configs.Serialize() << std::endl;
            return 0;
        }
        PBLOG_INFO << cliArgs.str();

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

    } catch (const std::exception& ex) {
        std::cerr << "Exception caught: " << ex.what() << std::endl;
        PBLOG_ERROR << "Exception caught: " << ex.what();
        return 1;
    }

    return 0;
}
