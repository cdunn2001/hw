#include <array>
#include <cassert>

#include <gtest/gtest.h>

#include <pacbio/primary/HDFMultiArrayIO.h>

#include <common/cuda/memory/DeviceOnlyObject.cuh>
#include <common/ZmwDataManager.h>
#include <common/cuda/utility/CudaArray.cuh>
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
    static constexpr size_t laneWidth = 64;
    static constexpr size_t gpuBlockThreads = laneWidth/2;
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
            .LaneWidth(laneWidth)
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

    std::array<Subframe::AnalogMeta, 4> meta{};
    Subframe::AnalogMeta baselineMeta{};

    // Hard code our models to match this specific trace file
    {
        int frameRate = 100;
        meta[0].ipdSSRatio = 0;
        meta[1].ipdSSRatio = 0;
        meta[2].ipdSSRatio = 0;
        meta[3].ipdSSRatio = 0;

        meta[0].ipd = frameRate * .308;
        meta[1].ipd = frameRate * .234;
        meta[2].ipd = frameRate * .234;
        meta[3].ipd = frameRate * .188;

        meta[0].pw = frameRate * .232;
        meta[1].pw = frameRate * .185;
        meta[2].pw = frameRate * .181;
        meta[3].pw = frameRate * .214;

        meta[0].pwSSRatio = 3.2;
        meta[1].pwSSRatio = 3.2;
        meta[2].pwSSRatio = 3.2;
        meta[3].pwSSRatio = 3.2;

        meta[0].mean = 227.13;
        meta[1].mean = 154.45;
        meta[2].mean = 97.67;
        meta[3].mean = 61.32;

        meta[0].var = 776;
        meta[1].var = 426;
        meta[2].var = 226;
        meta[3].var = 132;

        baselineMeta.mean = 0;
        baselineMeta.var = 33;
    }

    LaneModelParameters<gpuBlockThreads> refModel;
    refModel.BaselineMode().SetAllMeans(baselineMeta.mean).SetAllVars(baselineMeta.var);
    for (int i = 0; i < 4; ++i)
    {
        refModel.AnalogMode(i).SetAllMeans(meta[i].mean).SetAllVars(meta[i].var);
    }

    std::vector<UnifiedCudaArray<LaneModelParameters<gpuBlockThreads>>> models;
    FrameLabeler::Configure(meta, dataParams.kernelLanes, dataParams.blockLength);
    std::vector<FrameLabeler> frameLabelers(poolsPerChip);

    models.reserve(poolsPerChip);
    for (int i = 0; i < poolsPerChip; ++i)
    {
        models.emplace_back(lanesPerPool,SyncDirection::Symmetric);
        auto hostModels = models.back().GetHostView();
        for (int j = 0; j < lanesPerPool; ++j)
        {
            hostModels[j] = refModel;
        }
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
        auto& in = data.KernelInput();
        auto& out = data.KernelOutput();
        frameLabelers[batchIdx].ProcessBatch(models[batchIdx], in, out);

        for (size_t i = 0; i < out.LanesPerBatch(); ++i)
        {
            auto block = out.GetBlockView(i);
            for (size_t j = 0; j < block.NumFrames(); ++j)
            {
                for (size_t k = 0; k < block.LaneWidth(); ++k)
                {
                    int zmw = batchIdx * laneWidth * lanesPerPool + i * laneWidth + k;
                    int frame = firstFrame + j - Viterbi::lookbackDist;
                    if (frame < 0) continue;
                    LabelValidator(block(j,k), GroundTruth[zmw][frame]);
                }

            }
        }

        manager.ReturnBatch(std::move(data));
    }

    float total = matches + subframeMiss + mismatches;
    EXPECT_GT(matches / total, 0.98);
    EXPECT_LT(subframeMiss / total, .020);
    EXPECT_LT(mismatches / total, .001);
    std::cerr << "Matches/SubframeConfusion/Mismatches: " << matches << "/" << subframeMiss << "/" << mismatches << std::endl;

    FrameLabeler::Finalize();
}
