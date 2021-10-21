#include <array>
#include <cassert>

#include <gtest/gtest.h>

#include <pacbio/datasource/DataSourceRunner.h>
#include <pacbio/primary/HDFMultiArrayIO.h>

#include <appModules/TraceFileDataSource.h>

#include <common/cuda/memory/DeviceOnlyObject.cuh>
#include <common/cuda/memory/ManagedAllocations.h>
#include <common/cuda/utility/CudaArray.h>

#include <dataTypes/configs/MovieConfig.h>
#include <dataTypes/configs/BasecallerFrameLabelerConfig.h>

#include <basecaller/traceAnalysis/FrameLabelerDevice.h>
#include <basecaller/traceAnalysis/FrameLabelerHost.h>

using namespace PacBio::Application;
using namespace PacBio::Cuda::Utility;
using namespace PacBio::Cuda::Memory;
using namespace PacBio::DataSource;
using namespace PacBio::Mongo;
using namespace PacBio::Mongo::Data;
using namespace PacBio::Mongo::Basecaller;
using namespace PacBio::Primary;

namespace {

template <typename Labeler>
std::vector<std::unique_ptr<FrameLabeler>> CreateAndConfigure(const MovieConfig& config,
                                                              size_t lanesPerPool,
                                                              size_t numPools)
{
    Labeler::Configure(config);
    std::vector<std::unique_ptr<FrameLabeler>> ret;
    for (size_t pool = 0; pool < numPools; ++pool)
    {
        ret.emplace_back(std::make_unique<Labeler>(pool, lanesPerPool));
    }
    return ret;
}

std::vector<std::unique_ptr<FrameLabeler>> CreateAndConfigure(BasecallerFrameLabelerConfig::MethodName method,
                                                              const MovieConfig& config,
                                                              size_t lanesPerPool,
                                                              size_t numPools)
{
    switch(method)
    {
    case BasecallerFrameLabelerConfig::MethodName::SubFrameGaussCapsDevice:
        return CreateAndConfigure<FrameLabelerDevice>(config, lanesPerPool, numPools);
    case BasecallerFrameLabelerConfig::MethodName::SubFrameGaussCapsHost:
        return CreateAndConfigure<FrameLabelerHost>(config, lanesPerPool, numPools);
    default:
        throw PBException("Test does not support this FrameLabeler type");
    }
}

}

class FrameLabelerTest : public testing::TestWithParam<BasecallerFrameLabelerConfig::MethodName::RawEnum>
{};

TEST_P(FrameLabelerTest, CompareVsGroundTruth)
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

    TraceReplication trcConfig;
    trcConfig.traceFile = traceFile;
    trcConfig.numFrames = numBlocks * blockLen;
    trcConfig.numZmwLanes = layout.NumBlocks() * poolsPerChip;
    trcConfig.cache = true;
    auto source = std::make_unique<TraceFileDataSource>(
            std::move(cfg),
            trcConfig);

    // Hard code our models to match this specific trace file
    // Beware that we're ignoring some values like relAmp in our
    // analogs, which FrameLabeler does not currently need to know
    LaneModelParameters<PacBio::Cuda::PBHalf, laneSize> refModel;
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

    std::vector<UnifiedCudaArray<LaneModelParameters<PacBio::Cuda::PBHalf, laneSize>>> models;
    auto frameLabelers = CreateAndConfigure(GetParam(),
                                            movieConfig,
                                            lanesPerPool,
                                            poolsPerChip);

    models.reserve(poolsPerChip);
    for (uint32_t i = 0; i < poolsPerChip; ++i)
    {
        models.emplace_back(lanesPerPool,SyncDirection::Symmetric, SOURCE_MARKER());
        auto hostModels = models.back().GetHostView();
        for (uint32_t j = 0; j < lanesPerPool; ++j)
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

                auto result = (*frameLabelers[batchIdx])(std::move(in), models[batchIdx]);
                const auto& labels = result.first;
                auto viterbiLookback = labels.LatentTrace().NumFrames();

                for (size_t i = 0; i < labels.LanesPerBatch(); ++i)
                {
                    const auto& block = labels.GetBlockView(i);
                    for (size_t j = 0; j < block.NumFrames(); ++j)
                    {
                        for (size_t k = 0; k < block.LaneWidth(); ++k)
                        {
                            int zmw = batchIdx * laneSize * lanesPerPool + i * laneSize + k;
                            int frame = firstFrame + j - viterbiLookback;
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

INSTANTIATE_TEST_SUITE_P(somthing, FrameLabelerTest,
                         testing::Values(BasecallerFrameLabelerConfig::MethodName::SubFrameGaussCapsDevice,
                                         BasecallerFrameLabelerConfig::MethodName::SubFrameGaussCapsHost));
