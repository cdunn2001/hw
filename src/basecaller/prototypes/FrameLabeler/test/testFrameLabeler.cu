#include <array>
#include <cassert>

#include <gtest/gtest.h>

#include <pacbio/datasource/DataSourceRunner.h>
#include <pacbio/primary/HDFMultiArrayIO.h>

#include <appModules/TraceFileDataSource.h>

#include <common/cuda/memory/DeviceOnlyObject.cuh>
#include <common/ZmwDataManager.h>
#include <common/cuda/memory/ManagedAllocations.h>
#include <common/cuda/utility/CudaArray.h>
#include <common/DataGenerators/SignalGenerator.h>

#include <dataTypes/configs/MovieConfig.h>

#include <SubframeScorer.cuh>
#include <FrameLabelerKernels.cuh>

using namespace PacBio::Application;
using namespace PacBio::Cuda;
using namespace PacBio::Cuda::Data;
using namespace PacBio::Cuda::Utility;
using namespace PacBio::Cuda::Memory;
using namespace PacBio::DataSource;
using namespace PacBio::Mongo::Data;
using namespace PacBio::Primary;

TEST(FrameLabelerTest, CompareVsGroundTruth)
{
    static constexpr size_t lanesPerPool = 2;
    static constexpr size_t poolsPerChip = 2;
    static constexpr size_t numBlocks = 64;
    static constexpr size_t blockLen = 128;

    const std::string traceFile = "/pbi/dept/primary/sim/mongo/test2_mongo_SNR-40.trc.h5";

    const auto& GroundTruth = [&](){
        HDFMultiArrayIO io(traceFile, HDFMultiArrayIO::ReadOnly);
        return io.Read<int, 2>("GroundTruth/FrameLabels");
    }();

    PacketLayout layout(PacketLayout::BLOCK_LAYOUT_DENSE, PacketLayout::INT16, {lanesPerPool, blockLen, laneSize});
    DataSourceBase::Configuration cfg(layout, CreateAllocator(AllocatorMode::CUDA, SOURCE_MARKER()));

    auto source = std::make_unique<TraceFileDataSource>(
            std::move(cfg),
            traceFile,
            numBlocks * blockLen,
            layout.NumZmw()*poolsPerChip,
            true);

    // Hard code our models to match this specific trace file
    // Beware that we're ignoring some values like relAmp in our
    // analogs, which FrameLabeler does not currently need to know
    LaneModelParameters<PBHalf, laneSize> refModel;
    MovieConfig movieConfig;
    movieConfig.frameRate = 100;

    movieConfig.analogs[0].ipd2SlowStepRatio = 0;
    movieConfig.analogs[1].ipd2SlowStepRatio = 0;
    movieConfig.analogs[2].ipd2SlowStepRatio = 0;
    movieConfig.analogs[3].ipd2SlowStepRatio = 0;

    movieConfig.analogs[0].interPulseDistance = .308f;
    movieConfig.analogs[1].interPulseDistance = .234f;
    movieConfig.analogs[2].interPulseDistance = .234f;
    movieConfig.analogs[3].interPulseDistance = .188f;

    movieConfig.analogs[0].pulseWidth = .232f;
    movieConfig.analogs[1].pulseWidth = .185f;
    movieConfig.analogs[2].pulseWidth = .181f;
    movieConfig.analogs[3].pulseWidth = .214f;

    movieConfig.analogs[0].pw2SlowStepRatio = 3.2f;
    movieConfig.analogs[1].pw2SlowStepRatio = 3.2f;
    movieConfig.analogs[2].pw2SlowStepRatio = 3.2f;
    movieConfig.analogs[3].pw2SlowStepRatio = 3.2f;

    refModel.AnalogMode(0).SetAllMeans(227.13f);
    refModel.AnalogMode(1).SetAllMeans(154.45f);
    refModel.AnalogMode(2).SetAllMeans(97.67f);
    refModel.AnalogMode(3).SetAllMeans(61.32f);

    refModel.AnalogMode(0).SetAllVars(776);
    refModel.AnalogMode(1).SetAllVars(426);
    refModel.AnalogMode(2).SetAllVars(226);
    refModel.AnalogMode(3).SetAllVars(132);

    refModel.BaselineMode().SetAllMeans(0);
    refModel.BaselineMode().SetAllVars(33);

    std::vector<UnifiedCudaArray<LaneModelParameters<PBHalf, laneSize>>> models;
    FrameLabeler::Configure(movieConfig.analogs, movieConfig.frameRate);
    std::vector<FrameLabeler> frameLabelers;

    BatchDimensions latBatchDims;
    latBatchDims.framesPerBatch = blockLen;
    latBatchDims.laneWidth = laneSize;
    latBatchDims.lanesPerBatch = lanesPerPool;
    std::vector<BatchData<int16_t>> latTrace;

    models.reserve(poolsPerChip);
    frameLabelers.reserve(poolsPerChip);
    for (uint32_t i = 0; i < poolsPerChip; ++i)
    {
        frameLabelers.emplace_back(lanesPerPool);
        models.emplace_back(lanesPerPool,SyncDirection::Symmetric, SOURCE_MARKER());
        auto hostModels = models.back().GetHostView();
        for (uint32_t j = 0; j < lanesPerPool; ++j)
        {
            hostModels[j] = refModel;
        }

        latTrace.emplace_back(latBatchDims, SyncDirection::HostReadDeviceWrite, SOURCE_MARKER());
    }

    std::vector<FrameLabelerMetrics> frameLabelerMetrics;
    for (size_t i = 0; i < poolsPerChip; ++i)
    {
        frameLabelerMetrics.emplace_back(
                latBatchDims, SyncDirection::HostReadDeviceWrite, SOURCE_MARKER());
    }

    int mismatches = 0;
    int matches = 0;
    int subframeMiss = 0;
    auto LabelValidator = [&](int16_t actual, int16_t expected) {
        if (actual == expected) matches++;
        else if (actual > 4 || expected > 4) subframeMiss++;
        else mismatches++;
    };

    DataSourceRunner runner(std::move(source));
    runner.Start();
    while (runner.IsActive())
    {
        SensorPacketsChunk currChunk;
        if(runner.PopChunk(currChunk, std::chrono::milliseconds{10}))
        {
            for (auto& packet : currChunk)
            {
                auto firstFrame = packet.StartFrame();
                auto batchIdx = packet.PacketID();
                BatchMetadata meta(packet.PacketID(), packet.StartFrame(), packet.StartFrame() + packet.NumFrames(), packet.StartZmw());
                BatchDimensions dims;
                dims.lanesPerBatch = packet.Layout().NumBlocks();
                dims.framesPerBatch = packet.Layout().NumFrames();
                dims.laneWidth = packet.Layout().BlockWidth();
                TraceBatch<int16_t> in(std::move(packet),
                                       meta,
                                       dims,
                                       SyncDirection::HostWriteDeviceRead,
                                       SOURCE_MARKER());
                TraceBatch<int16_t> out(meta, dims, SyncDirection::HostReadDeviceWrite, SOURCE_MARKER());
                frameLabelers[batchIdx].ProcessBatch(models[batchIdx], in, latTrace[batchIdx], out,
                                                     frameLabelerMetrics[batchIdx]);

                for (size_t i = 0; i < out.LanesPerBatch(); ++i)
                {
                    const auto& block = out.GetBlockView(i);
                    for (size_t j = 0; j < block.NumFrames(); ++j)
                    {
                        for (size_t k = 0; k < block.LaneWidth(); ++k)
                        {
                            int zmw = batchIdx * laneSize * lanesPerPool + i * laneSize + k;
                            int frame = firstFrame + j - ViterbiStitchLookback;
                            if (frame < 0) continue;
                            LabelValidator(block(j,k), GroundTruth[zmw][frame]);
                        }

                    }
                }

            }
        }
    }

    float total = static_cast<float>(matches + subframeMiss + mismatches);
    EXPECT_GT(matches / total, 0.97);
    EXPECT_LT(subframeMiss / total, .020);
    EXPECT_LT(mismatches / total, .01);
    std::cerr << "Matches/SubframeConfusion/Mismatches: " << matches << "/" << subframeMiss << "/" << mismatches << std::endl;

    FrameLabeler::Finalize();
}
