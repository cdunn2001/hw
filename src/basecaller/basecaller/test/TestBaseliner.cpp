// Copyright (c) 2019-2021, Pacific Biosciences of California, Inc.
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

#include <cmath>

#include <pacbio/datasource/DataSourceRunner.h>
#include <pacbio/PBException.h>

#include <appModules/SimulatedDataSource.h>

#include <basecaller/traceAnalysis/HostNoOpBaseliner.h>
#include <basecaller/traceAnalysis/HostMultiScaleBaseliner.h>
#include <basecaller/traceAnalysis/DeviceMultiScaleBaseliner.h>

#include <common/cuda/memory/ManagedAllocations.h>

#include <dataTypes/configs/MovieConfig.h>
#include <dataTypes/configs/BatchLayoutConfig.h>
#include <dataTypes/configs/BasecallerBaselinerConfig.h>

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

    static BasecallerBaselinerConfig BaselinerConfig(BasecallerBaselinerConfig::MethodName method,
                                                     BasecallerBaselinerConfig::FilterTypes type)
    {
        Json::Value json;
        json["baselineConfig"]["Method"] = method.toString();
        json["baselineConfig"]["Filter"] = type.toString();
        TestConfig cfg{json};

        return cfg.baselineConfig;
    }
};

}

TEST(TestHostNoOpBaseliner, Run)
{
    Data::MovieConfig movConfig;
    auto baselinerConfig = TestConfig::BaselinerConfig(BasecallerBaselinerConfig::MethodName::NoOp,
                                                       BasecallerBaselinerConfig::FilterTypes::TwoScaleMedium);
    HostNoOpBaseliner::Configure(baselinerConfig, movConfig);

    const uint32_t numZmwLanes = 4;
    const uint32_t numPools = 2;
    const uint32_t lanesPerPool = numZmwLanes / numPools;
    const size_t numFrames = 8192;

    Data::BatchLayoutConfig batchConfig;
    batchConfig.lanesPerPool = lanesPerPool;
    std::vector<HostNoOpBaseliner> baseliners;

    TraceInputProperties traceInfo;
    traceInfo.pedestal = 0;
    for (size_t poolId = 0; poolId < numPools; poolId++)
    {
        baseliners.emplace_back(HostNoOpBaseliner(poolId, traceInfo));
    }

    auto generator = std::make_unique<ConstantGenerator>();
    PacketLayout layout(PacketLayout::BLOCK_LAYOUT_DENSE,
                        PacketLayout::INT16,
                        {lanesPerPool, batchConfig.framesPerChunk, laneSize});
    DataSourceBase::Configuration sourceConfig(layout, CreateAllocator(AllocatorMode::CUDA, SOURCE_MARKER()));
    SimulatedDataSource::SimConfig simConfig(laneSize, numFrames);
    sourceConfig.numFrames = simConfig.NumFrames();

    auto source = SimulatedDataSource(
            baseliners.size() * sourceConfig.requestedLayout.NumZmw(),
            simConfig,
            std::move(sourceConfig),
            std::move(generator));

    for (auto& in : source.AllBatches())
    {
        auto cameraBatch = baseliners[in.Metadata().PoolId()](in);
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

    HostNoOpBaseliner::Finalize();
}


TEST(TestHostMultiScaleBaseliner, Configure)
{
    PacBio::Logging::LogSeverityContext logLevelRaii {PacBio::Logging::LogLevel::NOTICE};

    BasecallerBaselinerConfig bbcConfig
        = TestConfig::BaselinerConfig(BasecallerBaselinerConfig::MethodName::HostMultiScale,
                                      BasecallerBaselinerConfig::FilterTypes::TwoScaleMedium);
    const auto movConfig = MockMovieConfig();

    // Test a few valid settings for SigmaEmaScaleStrides.
    for (const auto sess : {0.0f, 0.5f, 1.618f, 42.0f, 512.0f, 1.0e+6f, INFINITY})
    {
        bbcConfig.SigmaEmaScaleStrides = sess;
        ASSERT_TRUE(bbcConfig.Validate()) << "  SigmaEmaScaleStrides = " << sess << '.';
        HostMultiScaleBaseliner::Configure(bbcConfig, movConfig);
        EXPECT_FLOAT_EQ(std::pow(0.5f, 1.0f / sess), HostMultiScaleBaseliner::SigmaEmaAlpha())
            << "  SigmaEmaScaleStrides = " << sess << '.';
    }

    // Test a few invalid settings.
    for (const auto sess : {-0.0f, -1.0f, -INFINITY, NAN})
    {
        bbcConfig.SigmaEmaScaleStrides = sess;
        EXPECT_FALSE(bbcConfig.Validate()) << "  SigmaEmaScaleStrides = " << sess << '.';
    }
}


struct TestingParams
{
    // NoOp is an invalid setting for the tests.  It's the default here
    // to make sure this value is set when constructing a test suite
    BasecallerBaselinerConfig::MethodName method =
        BasecallerBaselinerConfig::MethodName::NoOp;
    uint16_t pfg_pulseIpd   = -1;
    uint16_t pfg_pulseWidth = -1;
    int16_t  pfg_baseSignalLevel = -1;
    std::vector<short> pfg_pulseSignalLevels;
};

struct MultiScaleBaseliner : public ::testing::TestWithParam<TestingParams>
{
    const uint32_t numZmwLanes = 4;
    const uint32_t numPools = 2;
    const uint32_t lanesPerPool = numZmwLanes / numPools;
    const size_t numBlocks = 16;

    float scaler = 3.46410f; // sqrt(12)

    Data::BatchLayoutConfig batchConfig;
    std::vector<std::unique_ptr<Baseliner>> baseliners;

    std::unique_ptr<SimulatedDataSource> source;
    PicketFenceGenerator::Config pfConfig;

    const size_t burnIn = 10;
    size_t burnInFrames;

    PacBio::Logging::LogSeverityContext logLevelRaii {PacBio::Logging::LogLevel::NOTICE};

    void SetUp() override
    {
        auto params = GetParam();
        pfConfig.generatePoisson = false;
        pfConfig.pulseIpd            = (params.pfg_pulseIpd         != static_cast<uint16_t>(-1) ?
                            params.pfg_pulseIpd          : pfConfig.pulseIpd);
        pfConfig.pulseWidth          = (params.pfg_pulseWidth       != static_cast<uint16_t>(-1) ?
                            params.pfg_pulseWidth        : pfConfig.pulseWidth);
        pfConfig.baselineSignalLevel = (params.pfg_baseSignalLevel  != -1 ?
                            params.pfg_baseSignalLevel   : pfConfig.baselineSignalLevel);
        pfConfig.pulseSignalLevels   = (!params.pfg_pulseSignalLevels.empty() ?
                            params.pfg_pulseSignalLevels : pfConfig.pulseSignalLevels);

        Data::MovieConfig movConfig;
        movConfig.photoelectronSensitivity = scaler;
        const auto baselinerConfig = TestConfig::BaselinerConfig(
                            params.method,
                            BasecallerBaselinerConfig::FilterTypes::TwoScaleMedium);
        if (params.method == BasecallerBaselinerConfig::MethodName::HostMultiScale)
            HostMultiScaleBaseliner::Configure(baselinerConfig, movConfig);
        else if (params.method == BasecallerBaselinerConfig::MethodName::DeviceMultiScale)
            DeviceMultiScaleBaseliner::Configure(baselinerConfig, movConfig);
        else
            throw PBException("Unexpected baseline filter in TestBaeliner");

        batchConfig.lanesPerPool = lanesPerPool;

        TraceInputProperties traceInfo;
        traceInfo.pedestal = 0;
        for (size_t poolId = 0; poolId < numPools; poolId++)
        {
            if (params.method == BasecallerBaselinerConfig::MethodName::HostMultiScale)
            {
                auto ptr = std::make_unique<HostMultiScaleBaseliner>(poolId,
                                                                     FilterParamsLookup(baselinerConfig.Filter),
                                                                     lanesPerPool,
                                                                     traceInfo);
                baseliners.push_back(std::move(ptr));
            }
            else if (params.method == BasecallerBaselinerConfig::MethodName::DeviceMultiScale)
            {
                auto ptr = std::make_unique<DeviceMultiScaleBaseliner>(poolId,
                                                                       FilterParamsLookup(baselinerConfig.Filter),
                                                                       lanesPerPool,
                                                                       traceInfo);
                baseliners.push_back(std::move(ptr));
            }
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

    void TearDown() override
    {
        HostMultiScaleBaseliner::Finalize();
    }
};

TEST_P(MultiScaleBaseliner, StatsAndSubtraction)
{
     for (auto& batch : source->AllBatches())
     {
         // ACTION
         const auto& meta = batch.Metadata();
         auto& baseliner = *baseliners[meta.PoolId()];
         auto cameraBatch = baseliner(batch);

         if (batch.Metadata().FirstFrame() < burnInFrames) continue;

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
             EXPECT_NEAR(mean/scaler, 0,
                         6 * (pfConfig.baselineSigma / std::sqrt(count)) + ((0.5f) * pfConfig.baselineSigma))
                         << "poolId=" << meta.PoolId() << " zmw=" << meta.FirstZmw()
                         << " laneIdx=" << laneIdx << " startframe=" << meta.FirstFrame() << std::endl;
             EXPECT_NEAR(var/scaler/scaler, pfConfig.baselineSigma * pfConfig.baselineSigma,
                         6 * pfConfig.baselineSigma)
                         << "poolId=" << meta.PoolId() << " zmw=" << meta.FirstZmw()
                         << " laneIdx=" << laneIdx << " startframe=" << meta.FirstFrame() << std::endl;
             EXPECT_NEAR(rawMean/scaler, pfConfig.baselineSignalLevel,
                         6 * (pfConfig.baselineSigma / std::sqrt(count)))
                         << "poolId=" << meta.PoolId() << " zmw=" << meta.FirstZmw()
                         << " laneIdx=" << laneIdx << " startframe=" << meta.FirstFrame() << std::endl;

             std::visit([&](const auto& origTrace)
             {
                 float diffSum = 0;
                 auto inBlock = origTrace.GetBlockView(laneIdx);
                 auto outBlock = traces.GetBlockView(laneIdx);
                 auto itrIn = inBlock.Begin();
                 auto itrOut = outBlock.Begin();
                 for ( ; itrIn != inBlock.End(); ++itrIn, ++itrOut)
                 {
                     auto diff = itrIn.Extract() * scaler - itrOut.Extract();
                     diffSum += MakeUnion(diff)[0];
                 }
                 auto observedBaseline = diffSum / traces.NumFrames();
                 // The observed baseline is a stat taken from all frames, determined just
                 // by measuring the shift between the original and subtracted traces.  rawMean
                 // however, is a stat taken only off frames determined to be actually be baseline.
                 // Point being I wouldn't expect these two estimates to agree super perfectly,
                 // so I'm putting a relative loose 7% tolerance on this check
                 EXPECT_NEAR(observedBaseline, rawMean,
                             0.07 * rawMean)
                             << "poolId=" << meta.PoolId() << " zmw=" << meta.FirstZmw()
                             << " laneIdx=" << laneIdx << " startframe=" << meta.FirstFrame() << std::endl;
             }, batch.Data());
         }
     }
}

//-----------------------------------------Testing parameters---------------------------//

INSTANTIATE_TEST_SUITE_P(,
                         MultiScaleBaseliner,
                         testing::Values(
                             TestingParams {
                                 BasecallerBaselinerConfig::MethodName::HostMultiScale,
                                 512, /* pulseIpd          */
                                 0,
                                 /* pulseWidth        */  // no pulses
                             },
                             TestingParams {
                                 BasecallerBaselinerConfig::MethodName::HostMultiScale,
                                 20,   /* pulseIpd          */
                                 24,   /* pulseWidth        */
                                 200,  /* baseSignalLevel   */
                                 {600} /* pulseSignalLevels */
                             },
                             TestingParams {
                                 BasecallerBaselinerConfig::MethodName::DeviceMultiScale,
                                 512, /* pulseIpd          */
                                 0,
                                 /* pulseWidth        */  // no pulses
                             },
                             TestingParams {
                                 BasecallerBaselinerConfig::MethodName::DeviceMultiScale,
                                 20,   /* pulseIpd          */
                                 24,   /* pulseWidth        */
                                 200,  /* baseSignalLevel   */
                                 {600} /* pulseSignalLevels */
                             }),
                         [](const testing::TestParamInfo<TestingParams>& info)
                         {
                             std::stringstream name;
                             if (info.param.method == BasecallerBaselinerConfig::MethodName::HostMultiScale)
                                 name << "HostMultiScale";
                             else
                                 name << "DeviceMultiScale";
                             if (info.param.pfg_pulseIpd == 512)
                                 name << "_AllBaseline";
                             else
                                 name << "_WithPulses";
                             return name.str();
                         });
template <typename T>
struct MultiScaleBaselinerSmallBatch : ::testing::Test {};

using MyTypes = ::testing::Types<HostMultiScaleBaseliner, DeviceMultiScaleBaseliner>;
TYPED_TEST_SUITE(MultiScaleBaselinerSmallBatch, MyTypes);

TYPED_TEST(MultiScaleBaselinerSmallBatch, OneBatch)
{
    using Baseliner = TypeParam;
    constexpr float scaler = 3.46410f; // sqrt(12)

    Data::MovieConfig movConfig;
    movConfig.photoelectronSensitivity = scaler;
    const auto baselinerConfig = TestConfig::BaselinerConfig(
            BasecallerBaselinerConfig::MethodName::HostMultiScale, /*ignored*/
            BasecallerBaselinerConfig::FilterTypes::TwoScaleMedium);
    Baseliner::Configure(baselinerConfig, movConfig);

    // BaselinerParams is taken from FilterParamsLookup(baselinerConfig.Method) for
    // BasecallerBaselinerConfig::MethodName::TwoScaleMedium
    BaselinerParams blp({2, 8}, {9, 31}, 2.44f, 0.50f); // strides, widths, sigma, mean

    uint32_t lanesPerPool_ = 1;
    TraceInputProperties traceInfo;
    traceInfo.pedestal = 0;
    Baseliner baseliner(0, blp, lanesPerPool_, traceInfo);

    uint32_t framesPerBlock = 512;

    size_t signalIdx = 0;
    PicketFenceGenerator::Config pfConfig;
    pfConfig.pulseIpd = framesPerBlock;
    pfConfig.pulseWidth = 0;
    PicketFenceGenerator generator(pfConfig);
    boost::multi_array<int16_t, 2> dataBuf(boost::extents[framesPerBlock][laneSize]);

    for(size_t zmwIdx = 0; zmwIdx < laneSize; ++zmwIdx)
    {
        size_t frame = 0;
        auto signal = generator.GenerateSignal(framesPerBlock, signalIdx++);

        for (size_t frameIdx = 0; frameIdx < framesPerBlock; ++frameIdx, ++frame)
        {
            dataBuf[frameIdx][zmwIdx] = signal[frame];
        }
    }

    BatchMetadata meta(0, 0, framesPerBlock, 0);
                        TraceBatch<int16_t> inRaw(meta, {lanesPerPool_, framesPerBlock, laneSize},
                        SyncDirection::HostWriteDeviceRead, SOURCE_MARKER());

    auto li = 0 /* laneIdx */;
    std::memcpy(inRaw.GetBlockView(li).Data(), dataBuf.origin(), framesPerBlock*laneSize*sizeof(int16_t));
    TraceBatchVariant in{std::move(inRaw)};

    // ACTION
    std::vector<std::pair<TraceBatch<int16_t>, BaselinerMetrics>> cameraOutput;
    cameraOutput.push_back(baseliner(in));
    cameraOutput.push_back(baseliner(in));
    cameraOutput.push_back(baseliner(in));

    std::vector<BlockView<int16_t>> traces; std::vector<StatAccumState> blStats;
    for (auto& e : cameraOutput)
    {
        traces.push_back(e.first.GetBlockView(li));
        blStats.push_back(e.second.baselinerStats.GetHostView()[li].baselineStats);
    }

    // Assert statistics
    auto zi = 22, fi = 0;
    std::vector<float> trCount; std::vector<float> mfMean; std::vector<float> trVar;
    for (auto &stat : blStats) {
        trCount.push_back(stat.moment0[zi]);
        mfMean.push_back(stat.moment1[zi]/stat.moment0[zi]);
        auto _v = stat.moment1[zi]*stat.moment1[zi] / stat.moment0[zi];
        trVar.push_back((stat.moment2[zi] - _v) / (stat.moment0[zi] - 1));
    }

    auto laneIdx = li, pi = 1;
    auto count = trCount[pi]; auto mean = mfMean[pi]; auto var = trVar[pi];
    EXPECT_NEAR(count,
                (framesPerBlock / (pfConfig.pulseIpd + pfConfig.pulseWidth)) * pfConfig.pulseIpd,
                40)
                << "poolId=" << meta.PoolId() << " zmw=" << meta.FirstZmw()
                << " laneIdx=" << laneIdx << " startframe=" << meta.FirstFrame() << std::endl;
    EXPECT_NEAR(mean/scaler,
                0,
                6 * (pfConfig.baselineSigma / std::sqrt(count)) + ((0.5f) * pfConfig.baselineSigma))
                << "poolId=" << meta.PoolId() << " zmw=" << meta.FirstZmw()
                << " laneIdx=" << laneIdx << " startframe=" << meta.FirstFrame() << std::endl;
    EXPECT_NEAR(var/scaler/scaler,
                pfConfig.baselineSigma * pfConfig.baselineSigma,
                6 * pfConfig.baselineSigma)
                << "poolId=" << meta.PoolId() << " zmw=" << meta.FirstZmw()
                << " laneIdx=" << laneIdx << " startframe=" << meta.FirstFrame() << std::endl;

    // Assert baseline converted from DN to e-
    EXPECT_NEAR(traces[0][fi*laneSize+zi], dataBuf[fi][zi] * scaler, 1);   // Within a rounding error
    // Assert baseline filtered
    EXPECT_LE(traces[1][fi*laneSize+zi], traces[0][fi*laneSize+zi]);

    Baseliner::Finalize();
}

}}} // PacBio::Mongo::Basecaller
