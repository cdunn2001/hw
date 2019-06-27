
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
#include <basecaller/traceAnalysis/DeviceMultiScaleBaseliner.h>

#include <common/DataGenerators/BatchGenerator.h>
#include <common/DataGenerators/PicketFenceGenerator.h>
#include <common/ZmwDataManager.h>

#include <dataTypes/BasecallerConfig.h>
#include <dataTypes/MovieConfig.h>
#include <dataTypes/PrimaryConfig.h>

#include <gtest/gtest.h>

namespace PacBio {
namespace Mongo {
namespace Basecaller {

TEST(TestNoOpBaseliner, Run)
{
    Data::MovieConfig movConfig;
    Data::BasecallerBaselinerConfig baselinerConfig;

    Data::GetPrimaryConfig().lanesPerPool = 1;

    HostNoOpBaseliner::Configure(baselinerConfig, movConfig);
    HostNoOpBaseliner baseliner{0};

    Cuda::Data::BatchGenerator batchGenerator(Data::GetPrimaryConfig().framesPerChunk,
                                              Data::GetPrimaryConfig().zmwsPerLane,
                                              Data::GetPrimaryConfig().lanesPerPool,
                                              8192,
                                              Data::GetPrimaryConfig().lanesPerPool);

    while (!batchGenerator.Finished())
    {
        auto chunk = batchGenerator.PopulateChunk();
        Data::CameraTraceBatch cameraBatch = baseliner(std::move(chunk.front()));
        const auto& baselineStats = cameraBatch.Stats(0);
        EXPECT_TRUE(std::all_of(baselineStats.m0_.data(),
                                baselineStats.m0_.data()+laneSize,
                                [](float v) { return v == 0; }));
        EXPECT_TRUE(std::all_of(baselineStats.m1_.data(),
                                baselineStats.m1_.data()+laneSize,
                                [](float v) { return v == 0; }));
        EXPECT_TRUE(std::all_of(baselineStats.m2_.data(),
                                baselineStats.m2_.data()+laneSize,
                                [](float v) { return v == 0; }));
    }

    HostNoOpBaseliner::Finalize();
}

TEST(TestHostMultiScaleBaseliner, Zeros)
{
    Data::MovieConfig movConfig;
    Data::BasecallerBaselinerConfig baselinerConfig;
    baselinerConfig.Method = Data::BasecallerBaselinerConfig::MethodName::MultiScaleLarge;

    Data::GetPrimaryConfig().lanesPerPool = 1;

    HostMultiScaleBaseliner::Configure(baselinerConfig, movConfig);

    HostMultiScaleBaseliner baseliner(0, 1.0f,
                                      FilterParamsLookup(baselinerConfig.Method),
                                      Data::GetPrimaryConfig().lanesPerPool);

    Cuda::Data::BatchGenerator batchGenerator(Data::GetPrimaryConfig().framesPerChunk,
                                              Data::GetPrimaryConfig().zmwsPerLane,
                                              Data::GetPrimaryConfig().lanesPerPool,
                                              128,
                                              Data::GetPrimaryConfig().lanesPerPool);
    while (!batchGenerator.Finished())
    {
        auto chunk = batchGenerator.PopulateChunk();
        Data::CameraTraceBatch cameraBatch = baseliner(std::move(chunk.front()));
        const auto& baselineStats = cameraBatch.Stats(0);
        EXPECT_TRUE(std::all_of(baselineStats.m0_.data(),
                                baselineStats.m0_.data()+laneSize,
                                [](float v) { return v == 0; }));
        EXPECT_TRUE(std::all_of(baselineStats.m1_.data(),
                                baselineStats.m1_.data()+laneSize,
                                [](float v) { return v == 0; }));
        EXPECT_TRUE(std::all_of(baselineStats.m2_.data(),
                                baselineStats.m2_.data()+laneSize,
                                [](float v) { return v == 0; }));
    }

    HostMultiScaleBaseliner::Finalize();
}

TEST(TestHostMultiScaleBaseliner, AllBaselineFrames)
{
    Data::MovieConfig movConfig;
    Data::BasecallerBaselinerConfig baselinerConfig;
    baselinerConfig.Method = Data::BasecallerBaselinerConfig::MethodName::MultiScaleLarge;

    // Simulate a single lane of data.
    Data::GetPrimaryConfig().lanesPerPool = 1;
    size_t numFrames = 32768;
    size_t numBlocks = numFrames / Data::GetPrimaryConfig().framesPerChunk;

    HostMultiScaleBaseliner::Configure(baselinerConfig, movConfig);

    HostMultiScaleBaseliner baseliner(0, 1.0f,
                                      FilterParamsLookup(baselinerConfig.Method),
                                      Data::GetPrimaryConfig().lanesPerPool);

    auto dmParams = Cuda::Data::DataManagerParams()
            .LaneWidth(Data::GetPrimaryConfig().zmwsPerLane)
            .NumZmwLanes(Data::GetPrimaryConfig().lanesPerPool)
            .KernelLanes(Data::GetPrimaryConfig().lanesPerPool)
            .BlockLength(Data::GetPrimaryConfig().framesPerChunk)
            .NumBlocks(numBlocks);

    // Generate all baseline frames, these should be normally
    // distributed with givem mean and sigma below.
    short baselineLevel = 250;
    short baselineSigma = 30;
    uint16_t pulseWidth = 0;    // No pulses.
    uint16_t pulseIpd = Data::GetPrimaryConfig().framesPerChunk;
    auto pfParams = Cuda::Data::PicketFenceParams()
            .PulseWidth(pulseWidth)
            .PulseIpd(Data::GetPrimaryConfig().framesPerChunk)
            .BaselineSignalLevel(baselineLevel)
            .BaselineSigma(baselineSigma);

    Cuda::Data::PicketFenceGenerator pfGenerator(dmParams, pfParams);

    Cuda::Data::BatchGenerator batchGenerator(Data::GetPrimaryConfig().framesPerChunk,
                                              Data::GetPrimaryConfig().zmwsPerLane,
                                              Data::GetPrimaryConfig().lanesPerPool,
                                              numBlocks * Data::GetPrimaryConfig().framesPerChunk,
                                              Data::GetPrimaryConfig().lanesPerPool);
    size_t blockNum = 0;
    while (!batchGenerator.Finished())
    {
        auto chunk = batchGenerator.PopulateChunk();
        auto& batch = chunk.front();

        // Fill batch with simulated data.
        for (size_t lane = 0; lane < Data::GetPrimaryConfig().lanesPerPool; lane++)
        {
            auto block = batch.GetBlockView(lane);
            pfGenerator.Fill(lane, blockNum, block);
        }

        Data::CameraTraceBatch cameraBatch = baseliner(std::move(batch));
        const auto& cameraBlock = cameraBatch.GetBlockView(0);
        const auto& baselineStats = cameraBatch.Stats(0);

        // Wait for baseliner to warm up before testing.
        if (blockNum > 50)
        {
            auto count = baselineStats.m0_[0];
            auto mean = baselineStats.m1_[0] / baselineStats.m0_[0];
            auto var = baselineStats.m1_[0] * baselineStats.m1_[0] / baselineStats.m0_[0];
            var = (baselineStats.m2_[0] - var) / (baselineStats.m0_[0] - 1.0f);
            auto rawMean = baselineStats.rawBaselineSum_[0] / baselineStats.m0_[0];

            EXPECT_NEAR(count, (Data::GetPrimaryConfig().framesPerChunk / (pulseIpd + pulseWidth)) * pulseIpd, 20);
            EXPECT_NEAR(mean, 0, 2*baselineSigma);
            EXPECT_NEAR(var, baselineSigma*baselineSigma, 4*baselineSigma);
            EXPECT_NEAR(rawMean, baselineLevel, 2*baselineSigma);
        }

        blockNum++;
    }

    HostMultiScaleBaseliner::Finalize();
}


TEST(TestHostMultiScaleBaseliner, OneSignalLevel)
{
    Data::MovieConfig movConfig;
    Data::BasecallerBaselinerConfig baselinerConfig;
    baselinerConfig.Method = Data::BasecallerBaselinerConfig::MethodName::MultiScaleLarge;

    Data::GetPrimaryConfig().lanesPerPool = 1;
    size_t numFrames = 32768;
    size_t numBlocks = numFrames / Data::GetPrimaryConfig().framesPerChunk;

    HostMultiScaleBaseliner::Configure(baselinerConfig, movConfig);

    HostMultiScaleBaseliner baseliner(0, 1.0f,
                                      FilterParamsLookup(baselinerConfig.Method),
                                      Data::GetPrimaryConfig().lanesPerPool);

    auto dmParams = Cuda::Data::DataManagerParams()
            .LaneWidth(Data::GetPrimaryConfig().zmwsPerLane)
            .NumZmwLanes(Data::GetPrimaryConfig().lanesPerPool)
            .KernelLanes(Data::GetPrimaryConfig().lanesPerPool)
            .BlockLength(Data::GetPrimaryConfig().framesPerChunk)
            .NumBlocks(numBlocks);

    // Generate baseline and single signal level.
    short baselineLevel = 250;
    short baselineSigma = 30;
    short signalLevel = 600;
    uint16_t pulseWidth = 24;
    uint16_t pulseIpd = 20;
    auto pfParams = Cuda::Data::PicketFenceParams()
            .NumSignals(1)
            .PulseSignalLevels({ std::to_string(signalLevel) })
            .PulseWidth(24)    
            .PulseIpd(20)
            .BaselineSignalLevel(baselineLevel)
            .BaselineSigma(baselineSigma);

    Cuda::Data::PicketFenceGenerator pfGenerator(dmParams, pfParams);

    Cuda::Data::BatchGenerator batchGenerator(Data::GetPrimaryConfig().framesPerChunk,
                                              Data::GetPrimaryConfig().zmwsPerLane,
                                              Data::GetPrimaryConfig().lanesPerPool,
                                              numBlocks * Data::GetPrimaryConfig().framesPerChunk,
                                              Data::GetPrimaryConfig().lanesPerPool);
    size_t blockNum = 0;
    while (!batchGenerator.Finished())
    {
        auto chunk = batchGenerator.PopulateChunk();
        auto& batch = chunk.front();

        // Fill batch with simulated data.
        for (size_t lane = 0; lane < Data::GetPrimaryConfig().lanesPerPool; lane++)
        {
            auto block = batch.GetBlockView(lane);
            pfGenerator.Fill(lane, blockNum, block);
        }

        Data::CameraTraceBatch cameraBatch = baseliner(std::move(batch));
        const auto& cameraBlock = cameraBatch.GetBlockView(0);
        const auto& baselineStats = cameraBatch.Stats(0);

        if (blockNum > 50)
        {
            auto count = baselineStats.m0_[0];
            auto mean = baselineStats.m1_[0] / baselineStats.m0_[0];
            auto var = baselineStats.m1_[0] * baselineStats.m1_[0] / baselineStats.m0_[0];
            var = (baselineStats.m2_[0] - var) / (baselineStats.m0_[0] - 1.0f);
            auto rawMean = baselineStats.rawBaselineSum_[0] / baselineStats.m0_[0];

            EXPECT_NEAR(count, (Data::GetPrimaryConfig().framesPerChunk / (pulseIpd + pulseWidth)) * pulseIpd, 20);
            EXPECT_NEAR(mean, 0, 2*baselineSigma);
            EXPECT_NEAR(var, baselineSigma*baselineSigma, 4*baselineSigma);
            EXPECT_NEAR(rawMean, baselineLevel, 2*baselineSigma);
        }

        blockNum++;
    }

    HostMultiScaleBaseliner::Finalize();
}
    



}}} // PacBio::Mongo::Basecaller
