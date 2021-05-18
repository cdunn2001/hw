"
// Copyright (c) 2019, Pacific Biosciences of California, Inc.
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
//
//  Description:
//  Defines unit tests for the strategies for estimation and subtraction of
//  baseline and estimation of associated statistics.

#include <pacbio/datasource/DataSourceRunner.h>

#include <appModules/SimulatedDataSource.h>

#include <basecaller/traceAnalysis/HostNoOpBaseliner.h>
#include <basecaller/traceAnalysis/HostMultiScaleBaseliner.h>

#include <common/cuda/memory/ManagedAllocations.h>
#include <common/DataGenerators/BatchGenerator.h>
#include <common/DataGenerators/PicketFenceGenerator.h>
#include <common/ZmwDataManager.h>

#include <dataTypes/configs/BasecallerBaselinerConfig.h>
#include <dataTypes/configs/MovieConfig.h>
#include <dataTypes/configs/BatchLayoutConfig.h>

#include <gtest/gtest.h>

using namespace PacBio::Application;
using namespace PacBio::Cuda::Memory;
using namespace PacBio::DataSource;
using namespace PacBio::Mongo::Data;

namespace PacBio {
namespace Mongo {
namespace Basecaller {

namespace {

struct TestConfig : public Configuration::PBConfig<TestConfig>
{
PB_CONFIG(TestConfig);

    PB_CONFIG_OBJECT(Data::BasecallerBaselinerConfig, baselineConfig);

    PB_CONFIG_PARAM(ComputeDevices, analyzerHardware, ComputeDevices::Host);

    static BasecallerBaselinerConfig BaselinerConfig(BasecallerBaselinerConfig::MethodName method)
    {
        Json::Value json;
        json["baselineConfig"]["Method"] = method.toString();
        TestConfig cfg{json};

        return cfg.baselineConfig;
    }
};

}

TEST(TestNoOpBaseliner, Run)
{
    Data::MovieConfig movConfig;
    const auto baselinerConfig = TestConfig::BaselinerConfig(BasecallerBaselinerConfig::MethodName::NoOp);
    HostNoOpBaseliner::Configure(baselinerConfig, movConfig);

    const uint32_t numZmwLanes = 4;
    const uint32_t numPools = 2;
    const uint32_t lanesPerPool = numZmwLanes / numPools;
    const size_t numFrames = 8192;

    Data::BatchLayoutConfig batchConfig;
    batchConfig.lanesPerPool = lanesPerPool;
    std::vector<std::unique_ptr<HostNoOpBaseliner>> baseliners;

    for (size_t poolId = 0; poolId < numPools; poolId++)
    {
        baseliners.emplace_back(std::make_unique<HostNoOpBaseliner>(poolId));
    }

    auto generator = std::make_unique<ConstantGenerator>();
    PacketLayout layout(PacketLayout::BLOCK_LAYOUT_DENSE,
                        PacketLayout::INT16,
                        {lanesPerPool, batchConfig.framesPerChunk, laneSize});
    DataSourceBase::Configuration sourceConfig(layout, CreateAllocator(AllocatorMode::CUDA, SOURCE_MARKER()));
    SimulatedDataSource::SimConfig simConfig(laneSize, numFrames);
    sourceConfig.numFrames = simConfig.NumFrames();

    auto source = std::make_unique<SimulatedDataSource>(
            baseliners.size() * sourceConfig.requestedLayout.NumZmw(),
            simConfig,
            std::move(sourceConfig),
            std::move(generator));

    DataSourceRunner runner(std::move(source));
    runner.Start();
    while (runner.IsActive())
    {
        SensorPacketsChunk currChunk;
        if(runner.PopChunk(currChunk, std::chrono::milliseconds{10}))
        {
            for (auto& packet : currChunk)
            {
                auto batchIdx = packet.PacketID();
                BatchMetadata meta(packet.PacketID(), packet.StartFrame(), packet.StartFrame() + packet.NumFrames(),
                                   packet.StartZmw());
                BatchDimensions dims;
                dims.lanesPerBatch = packet.Layout().NumBlocks();
                dims.framesPerBatch = packet.Layout().NumFrames();
                dims.laneWidth = packet.Layout().BlockWidth();
                TraceBatch<int16_t> in(std::move(packet),
                                       meta,
                                       dims,
                                       SyncDirection::HostWriteDeviceRead,
                                       SOURCE_MARKER());

                auto cameraBatch = (*baseliners[batchIdx])(std::move(in));
                auto traces = std::move(cameraBatch.first);
                auto stats = std::move(cameraBatch.second);
                for (size_t laneIdx = 0; laneIdx < traces.LanesPerBatch(); laneIdx++)
                {
                    const auto& cameraBlock = traces.GetBlockView(laneIdx);
                    for (auto it = cameraBlock.CBegin(); it != cameraBlock.CEnd(); it++)
                    {
                        for (unsigned i = 0; i < laneSize; i++)
                            EXPECT_EQ(MakeUnion(it.Extract())[i], i);
                    }

                    const auto& baselineStats = stats.baselinerStats.GetHostView()[laneIdx].baselineStats;
                    EXPECT_TRUE(std::all_of(baselineStats.moment0.data(),
                                            baselineStats.moment0.data() + laneSize,
                                            [](float v) { return v == 0; }));
                    EXPECT_TRUE(std::all_of(baselineStats.moment1.data(),
                                            baselineStats.moment1.data() + laneSize,
                                            [](float v) { return v == 0; }));
                    EXPECT_TRUE(std::all_of(baselineStats.moment2.data(),
                                            baselineStats.moment2.data() + laneSize,
                                            [](float v) { return v == 0; }));
                }
            }
        }
    }

    HostNoOpBaseliner::Finalize();
}

TEST(TestHostMultiScaleBaseliner, AllBaselineFrames)
{
    Data::MovieConfig movConfig;
    const auto baselinerConfig = TestConfig::BaselinerConfig(BasecallerBaselinerConfig::MethodName::MultiScaleLarge);

    const uint32_t numZmwLanes = 4;
    const uint32_t numPools = 2;
    const uint32_t lanesPerPool = numZmwLanes / numPools;
    const size_t numBlocks = 256;

    Data::BatchLayoutConfig batchConfig;
    batchConfig.lanesPerPool = lanesPerPool;
    const size_t numFrames = numBlocks * batchConfig.framesPerChunk;
    HostMultiScaleBaseliner::Configure(baselinerConfig, movConfig);
    std::vector<std::unique_ptr<HostMultiScaleBaseliner>> baseliners;

    for (size_t poolId = 0; poolId < numPools; poolId++)
    {
        baseliners.emplace_back(std::make_unique<HostMultiScaleBaseliner>(poolId, 1.0f,
                                                                          FilterParamsLookup(baselinerConfig.Method),
                                                                          batchConfig.lanesPerPool));
    }

    PicketFenceGenerator::Config pfConfig;
    pfConfig.generatePoisson = false;
    pfConfig.pulseWidth = 0;    // No pulses.
    pfConfig.pulseIpd = static_cast<uint16_t>(batchConfig.framesPerChunk);
    auto generator = std::make_unique<PicketFenceGenerator>(pfConfig);
    PacketLayout layout(PacketLayout::BLOCK_LAYOUT_DENSE,
                        PacketLayout::INT16,
                        {lanesPerPool, batchConfig.framesPerChunk, laneSize});
    DataSourceBase::Configuration sourceConfig(layout, CreateAllocator(AllocatorMode::CUDA, SOURCE_MARKER()));
    SimulatedDataSource::SimConfig simConfig(laneSize, numFrames);
    sourceConfig.numFrames = simConfig.NumFrames();

    auto source = std::make_unique<SimulatedDataSource>(
            baseliners.size() * sourceConfig.requestedLayout.NumZmw(),
            simConfig,
            std::move(sourceConfig),
            std::move(generator));

    const size_t burnIn = 10;
    const size_t burnInFrames = burnIn * batchConfig.framesPerChunk;
    DataSourceRunner runner(std::move(source));
    runner.Start();
    while (runner.IsActive())
    {
        SensorPacketsChunk currChunk;
        if(runner.PopChunk(currChunk, std::chrono::milliseconds{10}))
        {
            if (burnInFrames <= currChunk.StartFrame())
            {
                for (auto& packet : currChunk)
                {
                    auto batchIdx = packet.PacketID();
                    BatchMetadata meta(packet.PacketID(), packet.StartFrame(), packet.StartFrame() + packet.NumFrames(),
                                       packet.StartZmw());
                    BatchDimensions dims;
                    dims.lanesPerBatch = packet.Layout().NumBlocks();
                    dims.framesPerBatch = packet.Layout().NumFrames();
                    dims.laneWidth = packet.Layout().BlockWidth();
                    TraceBatch<int16_t> in(std::move(packet),
                                           meta,
                                           dims,
                                           SyncDirection::HostWriteDeviceRead,
                                           SOURCE_MARKER());

                    auto cameraBatch = (*baseliners[batchIdx])(std::move(in));
                    auto traces = std::move(cameraBatch.first);
                    auto stats = std::move(cameraBatch.second);
                    for (size_t laneIdx = 0; laneIdx < traces.LanesPerBatch(); laneIdx++)
                    {
                        const auto& baselinerStatAccumState = stats.baselinerStats.GetHostView()[laneIdx];
                        const auto& baselineStats = baselinerStatAccumState.baselineStats;

                        const auto count = baselineStats.moment0[0];
                        const auto mean = baselineStats.moment1[0] / baselineStats.moment0[0];
                        auto var = baselineStats.moment1[0] * baselineStats.moment1[0] / baselineStats.moment0[0];
                        var = (baselineStats.moment2[0] - var) / (baselineStats.moment0[0] - 1.0f);
                        const auto rawMean = baselinerStatAccumState.rawBaselineSum[0] / baselineStats.moment0[0];

                        EXPECT_NEAR(count, (batchConfig.framesPerChunk / (pfConfig.pulseIpd + pfConfig.pulseWidth)) * pfConfig.pulseIpd, 20);
                        EXPECT_NEAR(mean, 0, pfConfig.baselineSigma / std::sqrt(count));
                        EXPECT_NEAR(var, pfConfig.baselineSigma * pfConfig.baselineSigma, 2 * pfConfig.baselineSigma);
                        EXPECT_NEAR(rawMean, pfConfig.baselineSignalLevel, pfConfig.baselineSigma / std::sqrt(count));
                    }
                }
            }
        }
    }

    HostMultiScaleBaseliner::Finalize();
}


TEST(TestHostMultiScaleBaseliner, OneSignalLevel)
{
    Data::MovieConfig movConfig;
    const auto baselinerConfig = TestConfig::BaselinerConfig(BasecallerBaselinerConfig::MethodName::MultiScaleLarge);

    const uint32_t numZmwLanes = 4;
    const uint32_t numPools = 2;
    const uint32_t lanesPerPool = numZmwLanes / numPools;
    const size_t numBlocks = 256;

    Data::BatchLayoutConfig batchConfig;
    batchConfig.lanesPerPool = lanesPerPool;
    const size_t numFrames = numBlocks * batchConfig.framesPerChunk;
    HostMultiScaleBaseliner::Configure(baselinerConfig, movConfig);
    std::vector<std::unique_ptr<HostMultiScaleBaseliner>> baseliners;

    for (size_t poolId = 0; poolId < numPools; poolId++)
    {
        baseliners.emplace_back(std::make_unique<HostMultiScaleBaseliner>(poolId, 1.0f,
                                                                          FilterParamsLookup(baselinerConfig.Method),
                                                                          batchConfig.lanesPerPool));
    }

    PicketFenceGenerator::Config pfConfig;
    pfConfig.pulseSignalLevels = { 600 };
    pfConfig.pulseWidthRate = 0.04167f;
    pfConfig.pulseIpdRate = 0.05f;
    auto generator = std::make_unique<PicketFenceGenerator>(pfConfig);
    PacketLayout layout(PacketLayout::BLOCK_LAYOUT_DENSE,
                        PacketLayout::INT16,
                        {lanesPerPool, batchConfig.framesPerChunk, laneSize});
    DataSourceBase::Configuration sourceConfig(layout, CreateAllocator(AllocatorMode::CUDA, SOURCE_MARKER()));
    SimulatedDataSource::SimConfig simConfig(laneSize, numFrames);
    sourceConfig.numFrames = simConfig.NumFrames();

    auto source = std::make_unique<SimulatedDataSource>(
            baseliners.size() * sourceConfig.requestedLayout.NumZmw(),
            simConfig,
            std::move(sourceConfig),
            std::move(generator));

    const size_t burnIn = 10;
    const size_t burnInFrames = burnIn * batchConfig.framesPerChunk;
    DataSourceRunner runner(std::move(source));
    runner.Start();
    while (runner.IsActive())
    {
        SensorPacketsChunk currChunk;
        if(runner.PopChunk(currChunk, std::chrono::milliseconds{10}))
        {
            if (burnInFrames <= currChunk.StartFrame())
            {
                for (auto& packet : currChunk)
                {
                    auto batchIdx = packet.PacketID();
                    BatchMetadata meta(packet.PacketID(), packet.StartFrame(), packet.StartFrame() + packet.NumFrames(),
                                       packet.StartZmw());
                    BatchDimensions dims;
                    dims.lanesPerBatch = packet.Layout().NumBlocks();
                    dims.framesPerBatch = packet.Layout().NumFrames();
                    dims.laneWidth = packet.Layout().BlockWidth();
                    TraceBatch<int16_t> in(std::move(packet),
                                           meta,
                                           dims,
                                           SyncDirection::HostWriteDeviceRead,
                                           SOURCE_MARKER());

                    auto cameraBatch = (*baseliners[batchIdx])(std::move(in));
                    auto traces = std::move(cameraBatch.first);
                    auto stats = std::move(cameraBatch.second);
                    for (size_t laneIdx = 0; laneIdx < traces.LanesPerBatch(); laneIdx++)
                    {
                        const auto& baselinerStatAccumState = stats.baselinerStats.GetHostView()[laneIdx];
                        const auto& baselineStats = baselinerStatAccumState.baselineStats;

                        const auto count = baselineStats.moment0[0];
                        const auto mean = baselineStats.moment1[0] / baselineStats.moment0[0];
                        auto var = baselineStats.moment1[0] * baselineStats.moment1[0] / baselineStats.moment0[0];
                        var = (baselineStats.moment2[0] - var) / (baselineStats.moment0[0] - 1.0f);
                        const auto rawMean = baselinerStatAccumState.rawBaselineSum[0] / baselineStats.moment0[0];

                        EXPECT_NEAR(count, (batchConfig.framesPerChunk / (pfConfig.pulseIpd + pfConfig.pulseWidth)) * pfConfig.pulseIpd, 20);
                        EXPECT_NEAR(mean, 0, pfConfig.baselineSigma / std::sqrt(count));
                        EXPECT_NEAR(var, pfConfig.baselineSigma * pfConfig.baselineSigma, 2 * pfConfig.baselineSigma);
                        EXPECT_NEAR(rawMean, pfConfig.baselineSignalLevel, pfConfig.baselineSigma / std::sqrt(count));
                    }
                }
            }
        }
    }

    HostMultiScaleBaseliner::Finalize();
}

}}} // PacBio::Mongo::Basecaller"