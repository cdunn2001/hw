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

#include <basecaller/traceAnalysis/DeviceMultiScaleBaseliner.h>

#include <common/cuda/memory/ManagedAllocations.h>

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

BatchDimensions Layout2Dims(const PacBio::DataSource::PacketLayout& layout);

namespace {

struct TestConfig : public Configuration::PBConfig<TestConfig>
{
    PB_CONFIG(TestConfig);

    PB_CONFIG_OBJECT(Data::BasecallerBaselinerConfig, baselineConfig);

    PB_CONFIG_PARAM(ComputeDevices, analyzerHardware, ComputeDevices::V100);

    static BasecallerBaselinerConfig BaselinerConfig(BasecallerBaselinerConfig::FilterTypes type)
    {
        Json::Value json;
        json["baselineConfig"]["Method"] = "DeviceMultiScale";
        json["baselineConfig"]["Filter"] = type.toString();
        TestConfig cfg{json};

        return cfg.baselineConfig;
    }
};

}

struct TestingParams
{
    uint16_t pfg_pulseIpd   = -1;
    uint16_t pfg_pulseWidth = -1;
    int16_t  pfg_baseSignalLevel = -1;
    std::vector<short> pfg_pulseSignalLevels;
};

struct DeviceMultiScaleBaselinerTest : public ::testing::TestWithParam<TestingParams>
{
    const uint32_t numZmwLanes = 4;
    const uint32_t numPools = 2;
    const uint32_t lanesPerPool = numZmwLanes / numPools;
    const size_t numBlocks = 16;

    float scaler_ = 1.0f;

    Data::BatchLayoutConfig batchConfig;
    std::vector<std::unique_ptr<DeviceMultiScaleBaseliner>> baseliners;

    std::unique_ptr<SimulatedDataSource> source;
    PicketFenceGenerator::Config pfConfig;

    const size_t burnIn = 10;
    size_t burnInFrames;

    void SetUp() override
    {
        auto params = GetParam();
        pfConfig.generatePoisson = false;
        pfConfig.pulseIpd            = (params.pfg_pulseIpd         != -1 ? params.pfg_pulseIpd          : pfConfig.pulseIpd); 
        pfConfig.pulseWidth          = (params.pfg_pulseWidth       != -1 ? params.pfg_pulseWidth        : pfConfig.pulseWidth);
        pfConfig.baselineSignalLevel = (params.pfg_baseSignalLevel  != -1 ? params.pfg_baseSignalLevel   : pfConfig.baselineSignalLevel);
        pfConfig.pulseSignalLevels   = (!params.pfg_pulseSignalLevels.empty() ? params.pfg_pulseSignalLevels : pfConfig.pulseSignalLevels);

        Data::MovieConfig movConfig;
        const auto baselinerConfig = TestConfig::BaselinerConfig(BasecallerBaselinerConfig::FilterTypes::TwoScaleMedium);
        DeviceMultiScaleBaseliner::Configure(baselinerConfig, movConfig);

        batchConfig.lanesPerPool = lanesPerPool;

        for (size_t poolId = 0; poolId < numPools; poolId++)
        {
            baseliners.emplace_back(std::make_unique<DeviceMultiScaleBaseliner>(poolId, scaler_,
                                                            FilterParamsLookup(baselinerConfig.Filter),
                                                            lanesPerPool));
        }

        burnInFrames = burnIn * batchConfig.framesPerChunk;

        PacketLayout layout(PacketLayout::BLOCK_LAYOUT_DENSE,
                            PacketLayout::INT16,
                            {lanesPerPool, batchConfig.framesPerChunk, laneSize});
        size_t numFrames = numBlocks * batchConfig.framesPerChunk;
        SimulatedDataSource::SimConfig simConfig(laneSize, numFrames);
        DataSourceBase::Configuration sourceConfig(layout, CreateAllocator(AllocatorMode::CUDA, SOURCE_MARKER()));
        sourceConfig.numFrames = numFrames;
        auto generator = std::make_unique<PicketFenceGenerator>(pfConfig);
        source = std::make_unique<SimulatedDataSource>(
                baseliners.size() * sourceConfig.requestedLayout.NumZmw(),
                simConfig,
                std::move(sourceConfig),
                std::move(generator));
    }

    // Conversion between DN and e-
    float Scale() const { return scaler_; }

    void TearDown() override
    {
        DeviceMultiScaleBaseliner::Finalize();
    }
};

class DeviceMultiScaleBaselinerChunk : public DeviceMultiScaleBaselinerTest {};

TEST_P(DeviceMultiScaleBaselinerChunk, Chunk)
{
    while (source->IsRunning())
    {
        SensorPacketsChunk currChunk;
        source->ContinueProcessing();
        if (source->PopChunk(currChunk, std::chrono::milliseconds(10)))
        {
            for (auto& packet : currChunk)
            {
                auto batchIdx = packet.PacketID();
                auto startFrame = packet.StartFrame(); auto endFrame = packet.StartFrame() + packet.NumFrames();
                BatchMetadata meta(batchIdx, startFrame, endFrame, packet.StartZmw());

                TraceBatch<int16_t> in(std::move(packet),
                                       meta,
                                       Layout2Dims(packet.Layout()),
                                       SyncDirection::HostWriteDeviceRead,
                                       SOURCE_MARKER());

                // ACTION
                auto cameraBatch = baseliners[batchIdx]->FilterBaseline(in);

                if (currChunk.StartFrame() < burnInFrames) continue;

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

                    EXPECT_NEAR(count,
                                (batchConfig.framesPerChunk / (pfConfig.pulseIpd + pfConfig.pulseWidth)) *
                                pfConfig.pulseIpd,
                                20)
                                << "poolId=" << meta.PoolId() << " zmw=" << meta.FirstZmw()
                                << " laneIdx=" << laneIdx << " startframe=" << meta.FirstFrame() << std::endl;
                    EXPECT_NEAR(mean, 0, 
                                6 * (pfConfig.baselineSigma / std::sqrt(count)) + ((0.5f) * pfConfig.baselineSigma))
                                << "poolId=" << meta.PoolId() << " zmw=" << meta.FirstZmw()
                                << " laneIdx=" << laneIdx << " startframe=" << meta.FirstFrame() << std::endl;
                    EXPECT_NEAR(var, pfConfig.baselineSigma * pfConfig.baselineSigma,
                                6 * pfConfig.baselineSigma)
                                << "poolId=" << meta.PoolId() << " zmw=" << meta.FirstZmw()
                                << " laneIdx=" << laneIdx << " startframe=" << meta.FirstFrame() << std::endl;
                    EXPECT_NEAR(rawMean, pfConfig.baselineSignalLevel,
                                6 * (pfConfig.baselineSigma / std::sqrt(count)))
                                << "poolId=" << meta.PoolId() << " zmw=" << meta.FirstZmw()
                                << " laneIdx=" << laneIdx << " startframe=" << meta.FirstFrame() << std::endl;
                }

                break;
            }
        }
    }
}

//-----------------------------------------Testing parameters---------------------------//

INSTANTIATE_TEST_SUITE_P(DeviceMultiScaleBaselinerGroup1,
                        DeviceMultiScaleBaselinerChunk,
                        testing::Values(
                            TestingParams{
                                512,  /* pulseIpd          */
                                0,    /* pulseWidth        */  // no pulses
                            },
                            TestingParams{
                                20,     /* pulseIpd          */
                                24,     /* pulseWidth        */  // pulse period doesn't fit 512
                                200,    /* baseSignalLevel   */
                                { 600 } /* pulseSignalLevels */
                            }
                            ));


}}} // PacBio::Mongo::Basecaller
