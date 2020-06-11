
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

#include <basecaller/traceAnalysis/HostNoOpBaseliner.h>
#include <basecaller/traceAnalysis/HostMultiScaleBaseliner.h>
//#include <basecaller/traceAnalysis/DeviceMultiScaleBaseliner.h>

#include <common/DataGenerators/BatchGenerator.h>
#include <common/DataGenerators/PicketFenceGenerator.h>
#include <common/ZmwDataManager.h>

#include <dataTypes/configs/BasecallerBaselinerConfig.h>
#include <dataTypes/configs/MovieConfig.h>
#include <dataTypes/configs/BatchLayoutConfig.h>

#include <gtest/gtest.h>

namespace PacBio {
namespace Mongo {
namespace Basecaller {

TEST(TestNoOpBaseliner, Run)
{
    Data::MovieConfig movConfig;
    Data::BasecallerBaselinerConfig baselinerConfig;

    const uint32_t numZmwLanes = 4;
    const uint32_t numPools = 2;
    const uint32_t lanesPerPool = numZmwLanes / numPools;

    Data::BatchLayoutConfig batchConfig;
    batchConfig.lanesPerPool = lanesPerPool;
    HostNoOpBaseliner::Configure(baselinerConfig, movConfig);
    std::vector<std::unique_ptr<HostNoOpBaseliner>> baseliners;

    for (size_t poolId = 0; poolId < numPools; poolId++)
    {
        baseliners.emplace_back(std::make_unique<HostNoOpBaseliner>(poolId));
    }

    Cuda::Data::BatchGenerator batchGenerator(batchConfig.framesPerChunk,
                                              batchConfig.zmwsPerLane,
                                              batchConfig.lanesPerPool,
                                              8192,
                                              numZmwLanes);
    while (!batchGenerator.Finished())
    {
        auto chunk = batchGenerator.PopulateChunk();

        for (size_t batchNum = 0; batchNum < chunk.size(); batchNum++)
        {
            auto& traceBatch = chunk[batchNum];
            for (size_t laneIdx = 0; laneIdx < traceBatch.LanesPerBatch(); laneIdx++)
            {
                auto block = traceBatch.GetBlockView(laneIdx);
                std::fill(block.Data(), block.Data() + block.Size(), laneIdx * batchNum);
            }
        }

        for (size_t batchNum = 0; batchNum < chunk.size(); batchNum++)
        {
            auto cameraBatch = (*baseliners[batchNum])(std::move(chunk[batchNum]));
            auto traces = std::move(cameraBatch.first);
            auto stats = std::move(cameraBatch.second);

            for (size_t laneIdx = 0; laneIdx < traces.LanesPerBatch(); laneIdx++)
            {
                const auto& cameraBlock = traces.GetBlockView(laneIdx);
                EXPECT_TRUE(std::all_of(cameraBlock.Data(),
                                        cameraBlock.Data() + cameraBlock.Size(),
                                        [&laneIdx, &batchNum](short v) { return v == static_cast<short>(laneIdx * batchNum); }));

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

    HostNoOpBaseliner::Finalize();
}

TEST(TestHostMultiScaleBaseliner, Zeros)
{
    Data::MovieConfig movConfig;
    Data::BasecallerBaselinerConfig baselinerConfig;
    baselinerConfig.Method = Data::BasecallerBaselinerConfig::MethodName::MultiScaleLarge;

    const uint32_t numZmwLanes = 4;
    const uint32_t numPools = 2;
    const uint32_t lanesPerPool = numZmwLanes / numPools;

    Data::BatchLayoutConfig batchConfig;
    batchConfig.lanesPerPool = lanesPerPool;
    HostMultiScaleBaseliner::Configure(baselinerConfig, movConfig);
    std::vector<std::unique_ptr<HostMultiScaleBaseliner>> baseliners;

    for (size_t poolId = 0; poolId < numPools; poolId++)
    {
        baseliners.emplace_back(std::make_unique<HostMultiScaleBaseliner>(poolId, 1.0f,
                                                                          FilterParamsLookup(baselinerConfig.Method),
                                                                          batchConfig.lanesPerPool));
    }

    const auto framesPerChunk = batchConfig.framesPerChunk;
    const auto zmwsPerLane = batchConfig.zmwsPerLane;
    Cuda::Data::BatchGenerator batchGenerator(framesPerChunk, zmwsPerLane,
                                              lanesPerPool, 128, numZmwLanes);
    while (!batchGenerator.Finished())
    {
        auto chunk = batchGenerator.PopulateChunk();
        for (size_t batchNum = 0; batchNum < chunk.size(); batchNum++)
        {
            // This test was relying on a BatchGenerator with no trace file returning a
            // batch of all zeros.  I'm unsure if this was (or even should be) an
            // intentional feature of BatchGenerator, but even if it's intentional it
            // only works when the raw allocation itself comes pre-zeroed out, which
            // isn't necessarily true.
            std::memset(chunk[batchNum].GetBlockView(0).Data(), 0,
                        framesPerChunk * zmwsPerLane * lanesPerPool * sizeof(int16_t));
            auto cameraBatch = (*baseliners[batchNum])(std::move(chunk[batchNum]));
            auto stats = std::move(cameraBatch.second);
            const auto& baselineStats = stats.baselinerStats.GetHostView()[0].baselineStats;
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

    HostMultiScaleBaseliner::Finalize();
}

TEST(TestHostMultiScaleBaseliner, DISABLED_AllBaselineFrames)
{
    Data::MovieConfig movConfig;
    Data::BasecallerBaselinerConfig baselinerConfig;
    baselinerConfig.Method = Data::BasecallerBaselinerConfig::MethodName::MultiScaleLarge;

    const uint32_t numZmwLanes = 4;
    const uint32_t numPools = 2;
    const uint32_t lanesPerPool = numZmwLanes / numPools;
    const size_t numBlocks = 256;

    Data::BatchLayoutConfig batchConfig;
    batchConfig.lanesPerPool = lanesPerPool;
    HostMultiScaleBaseliner::Configure(baselinerConfig, movConfig);
    std::vector<std::unique_ptr<HostMultiScaleBaseliner>> baseliners;

    for (size_t poolId = 0; poolId < numPools; poolId++)
    {
        baseliners.emplace_back(std::make_unique<HostMultiScaleBaseliner>(poolId, 1.0f,
                                                                          FilterParamsLookup(baselinerConfig.Method),
                                                                          batchConfig.lanesPerPool));
    }

    auto dmParams = Cuda::Data::DataManagerParams()
            .LaneWidth(batchConfig.zmwsPerLane)
            .NumZmwLanes(numZmwLanes)
            .KernelLanes(batchConfig.lanesPerPool)
            .BlockLength(batchConfig.framesPerChunk)
            .NumBlocks(numBlocks);

    // Generate all baseline frames, these should be normally
    // distributed with given mean and sigma below.
    const short baselineLevel = 250;
    const short baselineSigma = 30;
    const uint16_t pulseWidth = 0;    // No pulses.
    const uint16_t pulseIpd = batchConfig.framesPerChunk;
    auto pfParams = Cuda::Data::PicketFenceParams()
            .PulseWidth(pulseWidth)
            .PulseIpd(batchConfig.framesPerChunk)
            .BaselineSignalLevel(baselineLevel)
            .BaselineSigma(baselineSigma);

    Cuda::Data::PicketFenceGenerator pfGenerator(dmParams, pfParams);

    Cuda::Data::BatchGenerator batchGenerator(batchConfig.framesPerChunk,
                                              batchConfig.zmwsPerLane,
                                              batchConfig.lanesPerPool,
                                              numBlocks * batchConfig.framesPerChunk,
                                              numZmwLanes);
    const size_t burnIn = 10;
    size_t blockNum = 0;
    while (!batchGenerator.Finished())
    {
        auto chunk = batchGenerator.PopulateChunk();
        for (auto& batch : chunk)
        {
            // Fill batch with simulated data.
            for (size_t lane = 0; lane < batchConfig.lanesPerPool; lane++)
            {
                auto block = batch.GetBlockView(lane);
                pfGenerator.Fill(lane, blockNum, block);
            }
        }

        for (size_t batchNum = 0; batchNum < chunk.size(); batchNum++)
        {
            auto cameraBatch = (*baseliners[batchNum])(std::move(chunk[batchNum]));
            auto stats = std::move(cameraBatch.second);
            const auto& baselinerStatAccumState = stats.baselinerStats.GetHostView()[0];
            const auto& baselineStats = baselinerStatAccumState.baselineStats;

            // Wait for baseliner to warm up before testing.
            if (blockNum > burnIn)
            {
                const auto count = baselineStats.moment0[0];
                const auto mean = baselineStats.moment1[0] / baselineStats.moment0[0];
                auto var = baselineStats.moment1[0] * baselineStats.moment1[0] / baselineStats.moment0[0];
                var = (baselineStats.moment2[0] - var) / (baselineStats.moment0[0] - 1.0f);
                const auto rawMean = baselinerStatAccumState.rawBaselineSum[0] / baselineStats.moment0[0];

                EXPECT_NEAR(count, (batchConfig.framesPerChunk / (pulseIpd + pulseWidth)) * pulseIpd, 20);
                EXPECT_NEAR(mean, 0, baselineSigma / sqrt(count));
                EXPECT_NEAR(var, baselineSigma * baselineSigma, 2 * baselineSigma);
                EXPECT_NEAR(rawMean, baselineLevel, baselineSigma / sqrt(count));
            }
        }
        blockNum++;
    }

    HostMultiScaleBaseliner::Finalize();
}


TEST(TestHostMultiScaleBaseliner, DISABLED_OneSignalLevel)
{
    Data::MovieConfig movConfig;
    Data::BasecallerBaselinerConfig baselinerConfig;
    baselinerConfig.Method = Data::BasecallerBaselinerConfig::MethodName::MultiScaleLarge;

    const uint32_t numZmwLanes = 4;
    const uint32_t numPools = 2;
    const uint32_t lanesPerPool = numZmwLanes / numPools;
    const size_t numBlocks = 256;

    Data::BatchLayoutConfig batchConfig;
    batchConfig.lanesPerPool = lanesPerPool;
    HostMultiScaleBaseliner::Configure(baselinerConfig, movConfig);
    std::vector<std::unique_ptr<HostMultiScaleBaseliner>> baseliners;

    for (size_t poolId = 0; poolId < numPools; poolId++)
    {
        baseliners.emplace_back(std::make_unique<HostMultiScaleBaseliner>(poolId, 1.0f,
                                                                          FilterParamsLookup(baselinerConfig.Method),
                                                                          batchConfig.lanesPerPool));
    }

    auto dmParams = Cuda::Data::DataManagerParams()
            .LaneWidth(batchConfig.zmwsPerLane)
            .NumZmwLanes(numZmwLanes)
            .KernelLanes(batchConfig.lanesPerPool)
            .BlockLength(batchConfig.framesPerChunk)
            .NumBlocks(numBlocks);

    // Generate baseline and single signal level.
    const short baselineLevel = 250;
    const short baselineSigma = 30;
    const short signalLevel = 600;
    const uint16_t pulseWidth = 24;
    const uint16_t pulseIpd = 20;
    auto pfParams = Cuda::Data::PicketFenceParams()
            .NumSignals(1)
            .PulseSignalLevels({ std::to_string(signalLevel) })
            .PulseWidth(24)    
            .PulseIpd(20)
            .BaselineSignalLevel(baselineLevel)
            .BaselineSigma(baselineSigma);

    Cuda::Data::PicketFenceGenerator pfGenerator(dmParams, pfParams);

    Cuda::Data::BatchGenerator batchGenerator(batchConfig.framesPerChunk,
                                              batchConfig.zmwsPerLane,
                                              batchConfig.lanesPerPool,
                                              numBlocks * batchConfig.framesPerChunk,
                                              numZmwLanes);
    const size_t burnIn = 10;
    size_t blockNum = 0;
    while (!batchGenerator.Finished())
    {
        auto chunk = batchGenerator.PopulateChunk();
        for (auto& batch : chunk)
        {
            // Fill batch with simulated data.
            for (size_t lane = 0; lane < batchConfig.lanesPerPool; lane++)
            {
                auto block = batch.GetBlockView(lane);
                pfGenerator.Fill(lane, blockNum, block);
            }
        }

        for (size_t batchNum = 0; batchNum < chunk.size(); batchNum++)
        {
            auto cameraBatch = (*baseliners[batchNum])(std::move(chunk[batchNum]));
            const auto& baselinerStatAccumState = cameraBatch.second.baselinerStats.GetHostView()[0];
            const auto& baselineStats = baselinerStatAccumState.baselineStats;

            // Wait for baseliner to warm up.
            if (blockNum > burnIn)
            {
                const auto count = baselineStats.moment0[0];
                const auto mean = baselineStats.moment1[0] / baselineStats.moment0[0];
                auto var = baselineStats.moment1[0] * baselineStats.moment1[0] / baselineStats.moment0[0];
                var = (baselineStats.moment2[0] - var) / (baselineStats.moment0[0] - 1.0f);
                const auto rawMean = baselinerStatAccumState.rawBaselineSum[0] / baselineStats.moment0[0];

                EXPECT_NEAR(count, (batchConfig.framesPerChunk / (pulseIpd + pulseWidth)) * pulseIpd, 20);
                EXPECT_NEAR(mean, 0, baselineSigma / sqrt(count));
                EXPECT_NEAR(var, baselineSigma * baselineSigma, 2 * baselineSigma);
                EXPECT_NEAR(rawMean, baselineLevel, baselineSigma / sqrt(count));
            }
        }
        blockNum++;
    }

    HostMultiScaleBaseliner::Finalize();
}

}}} // PacBio::Mongo::Basecaller
