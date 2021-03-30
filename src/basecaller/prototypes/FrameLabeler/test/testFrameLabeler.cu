#include <array>
#include <cassert>

#include <gtest/gtest.h>

#include <pacbio/primary/HDFMultiArrayIO.h>

#include <common/cuda/memory/DeviceOnlyObject.cuh>
#include <common/ZmwDataManager.h>
#include <common/cuda/utility/CudaArray.h>
#include <common/DataGenerators/SignalGenerator.h>

#include <SubframeScorer.cuh>
#include <FrameLabelerKernels.cuh>

using namespace PacBio::Cuda;
using namespace PacBio::Cuda::Data;
using namespace PacBio::Cuda::Utility;
using namespace PacBio::Cuda::Memory;
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

    auto dataParams = DataManagerParams()
            .LaneWidth(laneSize)
            .ImmediateCopy(false)
            .FrameRate(10000)
            .NumZmwLanes(lanesPerPool*poolsPerChip)
            .KernelLanes(lanesPerPool)
            .NumBlocks(numBlocks)
            .BlockLength(blockLen);

    auto traceParams = TraceFileParams().TraceFileName(traceFile);

    ZmwDataManager<int16_t, int16_t> manager(dataParams,
                                    std::make_unique<SignalGenerator>(dataParams, traceParams),
                                    true);

    LaneModelParameters<PBHalf, laneSize> refModel;
    std::array<AnalogMode, 4> analogs{};
    double frameRate = 100;

    // Hard code our models to match this specific trace file
    // Beware that we're ignoring some values like relAmp in our
    // analogs, which FrameLabeler does not currently need to know
    analogs[0].ipd2SlowStepRatio = 0;
    analogs[1].ipd2SlowStepRatio = 0;
    analogs[2].ipd2SlowStepRatio = 0;
    analogs[3].ipd2SlowStepRatio = 0;

    analogs[0].interPulseDistance = .308f;
    analogs[1].interPulseDistance = .234f;
    analogs[2].interPulseDistance = .234f;
    analogs[3].interPulseDistance = .188f;

    analogs[0].pulseWidth = .232f;
    analogs[1].pulseWidth = .185f;
    analogs[2].pulseWidth = .181f;
    analogs[3].pulseWidth = .214f;

    analogs[0].pw2SlowStepRatio = 3.2f;
    analogs[1].pw2SlowStepRatio = 3.2f;
    analogs[2].pw2SlowStepRatio = 3.2f;
    analogs[3].pw2SlowStepRatio = 3.2f;

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
    FrameLabeler::Configure(analogs, frameRate);
    std::vector<FrameLabeler> frameLabelers;

    BatchDimensions latBatchDims;
    latBatchDims.framesPerBatch = dataParams.blockLength;
    latBatchDims.laneWidth = laneSize;
    latBatchDims.lanesPerBatch = dataParams.kernelLanes;
    std::vector<BatchData<int16_t>> latTrace;

    models.reserve(poolsPerChip);
    frameLabelers.reserve(poolsPerChip);
    for (uint32_t i = 0; i < poolsPerChip; ++i)
    {
        frameLabelers.emplace_back(dataParams.kernelLanes);
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

    while (manager.MoreData())
    {
        auto data = manager.NextBatch();
        auto firstFrame = data.FirstFrame();
        auto batchIdx = data.Batch();
        const auto& in = data.KernelInput();
        auto& out = data.KernelOutput();
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

        manager.ReturnBatch(std::move(data));
    }

    float total = static_cast<float>(matches + subframeMiss + mismatches);
    EXPECT_GT(matches / total, 0.97);
    EXPECT_LT(subframeMiss / total, .020);
    EXPECT_LT(mismatches / total, .01);
    std::cerr << "Matches/SubframeConfusion/Mismatches: " << matches << "/" << subframeMiss << "/" << mismatches << std::endl;

    FrameLabeler::Finalize();
}
