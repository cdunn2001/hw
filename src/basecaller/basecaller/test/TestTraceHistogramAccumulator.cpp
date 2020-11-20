// Copyright (c) 2020, Pacific Biosciences of California, Inc.
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

#include <gtest/gtest.h>

#include <pacbio/dev/profile/ScopedProfilerChain.h>

#include <appModules/SimulatedDataSource.h>

#include <basecaller/traceAnalysis/TraceHistogramAccumHost.h>
#include <basecaller/traceAnalysis/DeviceTraceHistogramAccum.h>
#include <common/cuda/memory/DeviceAllocationStash.h>
#include <common/cuda/memory/ManagedAllocations.h>
#include <dataTypes/configs/BasecallerTraceHistogramConfig.h>

using namespace PacBio::Application;
using namespace PacBio::Cuda::Memory;
using namespace PacBio::Cuda;
using namespace PacBio::Dev::Profile;
using namespace PacBio::DataSource;
using namespace PacBio::Mongo;
using namespace PacBio::Mongo::Basecaller;
using namespace PacBio::Mongo::Data;

namespace {

static constexpr size_t numBins = LaneHistogram<float, int16_t>::numBins;

}

struct TestParameters
{
    size_t lanesPerPool;
    size_t numPools;
    size_t framesPerBlock;
    size_t numFrames;
    LaneHistBounds bounds;
};

template <typename Hist>
class Histogram : public testing::Test
{
public:
    Histogram()
    {
        // TODO keep?  Only do for perf tests?
        Memory::SetGlobalAllocationMode(CachingMode::ENABLED, AllocatorMode::CUDA);
    }
    ~Histogram()
    {
        Memory::SetGlobalAllocationMode(CachingMode::DISABLED, AllocatorMode::CUDA);
        Memory::SetGlobalAllocationMode(CachingMode::DISABLED, AllocatorMode::MALLOC);
    }
    using Hist_t = Hist;

    static void BinData(std::vector<std::unique_ptr<TraceHistogramAccumulator>>& hists,
                        DeviceAllocationStash& stash,
                        DataSourceBase::Configuration sourceConfig,
                        const SimulatedDataSource::SimConfig& simConfig,
                        std::unique_ptr<SignalGenerator> gen)
    {
        BatchDimensions dims;
        dims.laneWidth = laneSize;
        dims.lanesPerBatch = sourceConfig.layout.NumBlocks();
        dims.framesPerBatch = sourceConfig.layout.NumFrames();

        SimulatedDataSource source(hists.size() * sourceConfig.layout.NumZmw(),
                                   simConfig,
                                   std::move(sourceConfig),
                                   std::move(gen));

        ASSERT_EQ(source.NumBatches(), hists.size());

        SMART_ENUM(Profiles, TRACE_UPLOAD, HIST_UPLOAD, HIST_DOWNLOAD, BINNING);
        using Profiler = ScopedProfilerChain<Profiles>;

        SensorPacketsChunk chunk;
        while (source.IsRunning())
        {
            source.ContinueProcessing();
            if (source.PopChunk(chunk, std::chrono::milliseconds{10}))
            {
                ASSERT_EQ(chunk.NumPackets(), hists.size());
                for (auto&& packet : chunk)
                {
                    auto poolId = packet.StartZmw() / packet.NumZmw();
                    ASSERT_LT(poolId, hists.size());
                    BatchMetadata meta(poolId,
                                       packet.StartFrame(),
                                       packet.StopFrame(),
                                       packet.StartZmw());
                    TraceBatch<int16_t> batch(std::move(packet), meta, dims,
                                              SyncDirection::HostWriteDeviceRead, SOURCE_MARKER());
                    Profiler profiler(Profiler::Mode::OBSERVE, 1000, 1000);

                    auto bcopy = profiler.CreateScopedProfiler(Profiles::TRACE_UPLOAD);
                    (void)bcopy;
                    batch.CopyToDevice();
                    CudaSynchronizeDefaultStream();

                    auto hcopy = profiler.CreateScopedProfiler(Profiles::HIST_UPLOAD);
                    (void)hcopy;
                    stash.RetrievePool(poolId);
                    CudaSynchronizeDefaultStream();

                    auto binning = profiler.CreateScopedProfiler(Profiles::BINNING);
                    (void)binning;
                    hists[poolId]->AddBatch(batch);
                    CudaSynchronizeDefaultStream();

                    auto download = profiler.CreateScopedProfiler(Profiles::HIST_DOWNLOAD);
                    (void)download;
                    stash.StashPool(poolId);
                    CudaSynchronizeDefaultStream();
                }
            }
        }
        Profiler::FinalReport();
    }

    static auto RunTest(const TestParameters& params,
                        const SimulatedDataSource::SimConfig& simConfig,
                        std::unique_ptr<SignalGenerator> generator)
    {
        DeviceAllocationStash stash;

        std::vector<std::unique_ptr<TraceHistogramAccumulator>> hists;
        for (size_t i = 0; i < params.numPools; ++i)
        {
            StashableAllocRegistrar registrar(i, stash);
            hists.emplace_back(std::make_unique<Hist>(i, params.lanesPerPool, &registrar));

            UnifiedCudaArray<LaneHistBounds> poolBounds(params.lanesPerPool,
                                                        SyncDirection::HostWriteDeviceRead,
                                                        SOURCE_MARKER());
            auto boundsView = poolBounds.GetHostView();
            for (size_t i = 0; i < boundsView.Size(); ++i)
            {
                boundsView[i] = params.bounds;
            }
            hists.back()->Reset(std::move(poolBounds));
        }

        RunTest(hists, stash, params, simConfig, std::move(generator));

        return hists;
    }

    static void RunTest(std::vector<std::unique_ptr<TraceHistogramAccumulator>>& hists,
                        DeviceAllocationStash& stash,
                        const TestParameters& params,
                        const SimulatedDataSource::SimConfig& simConfig,
                        std::unique_ptr<SignalGenerator> generator)
    {

        PacketLayout layout(PacketLayout::BLOCK_LAYOUT_DENSE,
                            PacketLayout::INT16,
                            {params.lanesPerPool, params.framesPerBlock, laneSize});
        DataSourceBase::Configuration sourceConfig(layout, CreateAllocator(AllocatorMode::CUDA, SOURCE_MARKER()));
        sourceConfig.numFrames = params.numFrames;

        BinData(hists, stash, std::move(sourceConfig), simConfig, std::move(generator));
    }
};

using HistTypes = ::testing::Types<TraceHistogramAccumHost,
                                   DeviceTraceHistogramAccum<DeviceHistogramTypes::GlobalInterleaved>,
                                   DeviceTraceHistogramAccum<DeviceHistogramTypes::GlobalContig>,
                                   DeviceTraceHistogramAccum<DeviceHistogramTypes::GlobalContigAtomic>>;
TYPED_TEST_SUITE(Histogram, HistTypes);

TYPED_TEST(Histogram, ResetFromStats)
{
    Data::BasecallerTraceHistogramConfig config;
    TestFixture::Hist_t::Configure(config);

    const uint32_t numLanes = 2;
    Data::BaselinerMetrics metrics(numLanes,
                                   SyncDirection::HostWriteDeviceRead,
                                   SOURCE_MARKER());

    auto mView = metrics.baselinerStats.GetHostView();
    for (size_t lane = 0; lane < numLanes; ++lane)
    {
        for (size_t zmw = 0; zmw < laneSize; ++zmw)
        {
            // We're going to set up a silly little signal.
            // The baseline frame count will go up monotonically from
            // 2 by one (to trigger the min baseline check and sigma
            // fallback).  We're also going to set up a mean that has
            // the same scaling and a variance that is the square of that,
            // mostly because that gives us zmw dependant data that is
            // easy to generate.
            float tmp = zmw + lane*laneSize + 2;
            mView[lane].baselineStats.offset[zmw] = 0;
            mView[lane].baselineStats.moment0[zmw] = tmp;
            mView[lane].baselineStats.moment1[zmw] = tmp*tmp;
            mView[lane].baselineStats.moment2[zmw] = 2*tmp*tmp*tmp;
        }
    }

    typename TestFixture::Hist_t hAccum(1, numLanes, nullptr);
    hAccum.Reset(std::move(metrics));

    auto hist = hAccum.Histogram();
    auto hView = hist.data.GetHostView();
    for (size_t lane = 0; lane < numLanes; ++lane)
    {
        for (size_t zmw = 0; zmw < laneSize; ++zmw)
        {
            const float baselineFrames = zmw + lane*laneSize + 2;
            const float baselineMean = baselineFrames;
            const float baselineSig = (baselineFrames >= config.BaselineStatMinFrameCount)
                ? std::sqrt(baselineMean * baselineMean * baselineMean / (baselineMean - 1.f))
                : config.FallBackBaselineSigma;

            EXPECT_FLOAT_EQ(hView[lane].binSize[zmw], config.BinSizeCoeff * baselineSig) << baselineFrames;
            // The current implementation is actuall 4 sigma, but I don't want to code against
            // that too tightly.
            EXPECT_LT(hView[lane].lowBound[zmw], baselineMean - 2*baselineSig);
        }
    }
}

TYPED_TEST(Histogram, SingleLaneConstant)
{
    TestParameters params;
    params.lanesPerPool = 1;
    params.numPools = 1;
    params.framesPerBlock = 512;
    params.numFrames = 2048;
    params.bounds.lowerBounds = 0;
    params.bounds.upperBounds = numBins;

    SimulatedDataSource::SimConfig simConfig(laneSize, params.numFrames);
    auto generator = std::make_unique<ConstantGenerator>();

    auto hists = TestFixture::RunTest(params,
                                      simConfig,
                                      std::move(generator));

    EXPECT_EQ(hists[0]->FramesAdded(), params.numFrames);

    const auto& histdata = hists[0]->Histogram();
    const auto& lanedata = histdata.data.GetHostView()[0];
    EXPECT_TRUE(all(LaneArray<uint16_t>(lanedata.outlierCountHigh) == 0u));
    EXPECT_TRUE(all(LaneArray<uint16_t>(lanedata.outlierCountLow) == 0u));

    ArrayUnion<LaneArray<uint16_t>> expected;
    for (size_t i = 0; i < numBins; ++i)
    {
        auto counts = LaneArray<uint16_t>(lanedata.binCount[i]);
        expected.Simd() = 0;
        if (i < 64)
        {
            expected[i] = params.numFrames;
        }
        EXPECT_TRUE(all(counts == expected));
    }
}

TYPED_TEST(Histogram, MultiLaneConstant)
{
    TestParameters params;
    params.lanesPerPool = 4;
    params.numPools = 1;
    params.framesPerBlock = 512;
    params.numFrames = 2048;
    params.bounds.lowerBounds = 0;
    params.bounds.upperBounds = numBins;

    SimulatedDataSource::SimConfig simConfig(params.lanesPerPool*laneSize, params.numFrames);
    auto generator = std::make_unique<ConstantGenerator>();

    auto hists = TestFixture::RunTest(params,
                                      simConfig,
                                      std::move(generator));

    EXPECT_EQ(hists[0]->FramesAdded(), params.numFrames);

    const auto& histdata = hists[0]->Histogram();
    for (size_t lane = 0; lane < histdata.data.Size(); ++lane)
    {
        const auto& lanedata = histdata.data.GetHostView()[lane];
        EXPECT_TRUE(all(LaneArray<uint16_t>(lanedata.outlierCountHigh) == 0u));
        EXPECT_TRUE(all(LaneArray<uint16_t>(lanedata.outlierCountLow) == 0u));

        ArrayUnion<LaneArray<uint16_t>> expected;
        for (size_t i = 0; i < numBins; ++i)
        {
            auto counts = LaneArray<uint16_t>(lanedata.binCount[i]);
            expected.Simd() = 0;
            if (i >= lane * laneSize && i < (lane+1) * laneSize)
            {
                expected[i%laneSize] = params.numFrames;
            }
            EXPECT_TRUE(all(counts == expected)) << lane << " " << i;
        }
    }
}

TYPED_TEST(Histogram, MultiPoolConstant)
{
    TestParameters params;
    params.lanesPerPool = 2;
    params.numPools = 2;
    params.framesPerBlock = 512;
    params.numFrames = 2048;
    params.bounds.lowerBounds = 0;
    params.bounds.upperBounds = numBins;

    SimulatedDataSource::SimConfig simConfig(params.numPools*params.lanesPerPool*laneSize, params.numFrames);
    auto generator = std::make_unique<ConstantGenerator>();

    auto hists = TestFixture::RunTest(params,
                                      simConfig,
                                      std::move(generator));

    for (size_t pool = 0; pool < params.numPools; ++pool)
    {
        EXPECT_EQ(hists[pool]->FramesAdded(), params.numFrames);

        const auto& histdata = hists[pool]->Histogram();
        for (size_t lane = 0; lane < histdata.data.Size(); ++lane)
        {
            const auto& lanedata = histdata.data.GetHostView()[lane];
            EXPECT_TRUE(all(LaneArray<uint16_t>(lanedata.outlierCountHigh) == 0u));
            EXPECT_TRUE(all(LaneArray<uint16_t>(lanedata.outlierCountLow) == 0u));

            ArrayUnion<LaneArray<uint16_t>> expected;
            for (size_t i = 0; i < numBins; ++i)
            {
                auto counts = LaneArray<uint16_t>(lanedata.binCount[i]);
                expected.Simd() = 0;
                auto blockZmw = (pool * params.lanesPerPool + lane) * laneSize;
                if (i >= blockZmw && i < blockZmw + laneSize)
                {
                    expected[i%laneSize] = params.numFrames;
                }
                EXPECT_TRUE(all(counts == expected));
            }
        }
    }
}

TYPED_TEST(Histogram, SawtoothSimpleUniform)
{
    TestParameters params;
    params.lanesPerPool = 2;
    params.numPools = 2;
    params.framesPerBlock = 512;
    params.numFrames = 2048;
    params.bounds.lowerBounds = 0;
    params.bounds.upperBounds = numBins;

    SimulatedDataSource::SimConfig simConfig(laneSize, params.numFrames);
    SawtoothGenerator::Config sawConfig;
    sawConfig.minAmp = 0;
    sawConfig.maxAmp = numBins;
    sawConfig.periodFrames = numBins;
    sawConfig.startFrameStagger = 0;
    auto generator = std::make_unique<SawtoothGenerator>(sawConfig);

    auto hists = TestFixture::RunTest(params,
                                      simConfig,
                                      std::move(generator));

    for (size_t pool = 0; pool < params.numPools; ++pool)
    {
        EXPECT_EQ(hists[pool]->FramesAdded(), params.numFrames);

        const auto& histdata = hists[pool]->Histogram();
        for (size_t lane = 0; lane < histdata.data.Size(); ++lane)
        {
            const auto& lanedata = histdata.data.GetHostView()[lane];
            EXPECT_TRUE(all(LaneArray<uint16_t>(lanedata.outlierCountHigh) == 0u));
            EXPECT_TRUE(all(LaneArray<uint16_t>(lanedata.outlierCountLow) == 0u));

            ArrayUnion<LaneArray<uint16_t>> expected;
            for (size_t i = 0; i < numBins; ++i)
            {
                auto counts = LaneArray<uint16_t>(lanedata.binCount[i]);
                EXPECT_TRUE(all(counts == 6u | counts == 7u));
            }
        }
    }
}

TYPED_TEST(Histogram, SawtoothSkipUniform)
{
    TestParameters params;
    params.lanesPerPool = 2;
    params.numPools = 2;
    params.framesPerBlock = 512;
    params.numFrames = 2048;
    params.bounds.lowerBounds = 0;
    params.bounds.upperBounds = numBins;

    SimulatedDataSource::SimConfig simConfig(laneSize, params.numFrames);
    SawtoothGenerator::Config sawConfig;
    sawConfig.minAmp = 0;
    sawConfig.maxAmp = numBins;
    sawConfig.periodFrames = numBins/2;
    sawConfig.startFrameStagger = 0;
    auto generator = std::make_unique<SawtoothGenerator>(sawConfig);

    auto hists = TestFixture::RunTest(params,
                                      simConfig,
                                      std::move(generator));

    const uint16_t expect = params.numFrames / sawConfig.periodFrames;
    for (size_t pool = 0; pool < params.numPools; ++pool)
    {
        EXPECT_EQ(hists[pool]->FramesAdded(), params.numFrames);

        const auto& histdata = hists[pool]->Histogram();
        for (size_t lane = 0; lane < histdata.data.Size(); ++lane)
        {
            const auto& lanedata = histdata.data.GetHostView()[lane];
            EXPECT_TRUE(all(LaneArray<uint16_t>(lanedata.outlierCountHigh) == 0u));
            EXPECT_TRUE(all(LaneArray<uint16_t>(lanedata.outlierCountLow) == 0u));

            ArrayUnion<LaneArray<uint16_t>> expected;
            for (size_t i = 0; i < numBins; ++i)
            {
                auto counts = LaneArray<uint16_t>(lanedata.binCount[i]);
                if (i % 2 == 0)
                    EXPECT_TRUE(all(counts == expect | counts == expect+1u));
                else
                    EXPECT_TRUE(all(counts == 0u));
            }
        }
    }
}

TYPED_TEST(Histogram, SawtoothOutliersUniform)
{
    TestParameters params;
    params.lanesPerPool = 2;
    params.numPools = 2;
    params.framesPerBlock = 512;
    params.numFrames = 2048;
    params.bounds.lowerBounds = 100;
    params.bounds.upperBounds = 100+numBins;

    SimulatedDataSource::SimConfig simConfig(laneSize, params.numFrames);
    SawtoothGenerator::Config sawConfig;
    sawConfig.minAmp = 0;
    sawConfig.maxAmp = 512;
    sawConfig.periodFrames = 512;
    sawConfig.startFrameStagger = 0;
    auto generator = std::make_unique<SawtoothGenerator>(sawConfig);

    auto hists = TestFixture::RunTest(params,
                                      simConfig,
                                      std::move(generator));

    for (size_t pool = 0; pool < params.numPools; ++pool)
    {
        EXPECT_EQ(hists[pool]->FramesAdded(), params.numFrames);

        const auto& histdata = hists[pool]->Histogram();
        for (size_t lane = 0; lane < histdata.data.Size(); ++lane)
        {
            const auto& lanedata = histdata.data.GetHostView()[lane];
            EXPECT_TRUE(all(LaneArray<uint16_t>(lanedata.outlierCountHigh) == (sawConfig.maxAmp - params.bounds.upperBounds[0])*4u));
            EXPECT_TRUE(all(LaneArray<uint16_t>(lanedata.outlierCountLow) == (params.bounds.lowerBounds[0] - sawConfig.minAmp)*4u));

            ArrayUnion<LaneArray<uint16_t>> expected;
            for (size_t i = 0; i < numBins; ++i)
            {
                auto counts = LaneArray<uint16_t>(lanedata.binCount[i]);
                EXPECT_TRUE(all(counts == 4u));
            }
        }
    }
}

TYPED_TEST(Histogram, SawtoothOutliersStagger)
{
    TestParameters params;
    params.lanesPerPool = 2;
    params.numPools = 2;
    params.framesPerBlock = 512;
    params.numFrames = 2048;
    params.bounds.lowerBounds = 100;
    params.bounds.upperBounds = 100+numBins;

    SimulatedDataSource::SimConfig simConfig(laneSize, params.numFrames);
    SawtoothGenerator::Config sawConfig;
    sawConfig.minAmp = 0;
    sawConfig.maxAmp = 512;
    sawConfig.periodFrames = 512;
    sawConfig.startFrameStagger = 2;
    auto generator = std::make_unique<SawtoothGenerator>(sawConfig);

    auto hists = TestFixture::RunTest(params,
                                      simConfig,
                                      std::move(generator));

    for (size_t pool = 0; pool < params.numPools; ++pool)
    {
        EXPECT_EQ(hists[pool]->FramesAdded(), params.numFrames);

        const auto& histdata = hists[pool]->Histogram();
        for (size_t lane = 0; lane < histdata.data.Size(); ++lane)
        {
            const auto& lanedata = histdata.data.GetHostView()[lane];
            EXPECT_TRUE(all(LaneArray<uint16_t>(lanedata.outlierCountHigh) == (sawConfig.maxAmp - params.bounds.upperBounds[0])*4u));
            EXPECT_TRUE(all(LaneArray<uint16_t>(lanedata.outlierCountLow) == (params.bounds.lowerBounds[0] - sawConfig.minAmp)*4u));

            ArrayUnion<LaneArray<uint16_t>> expected;
            for (size_t i = 0; i < numBins; ++i)
            {
                auto counts = LaneArray<uint16_t>(lanedata.binCount[i]);
                EXPECT_TRUE(all(counts == 4u));
            }
        }
    }
}

TYPED_TEST(Histogram, PerfPulseManySignals)
{
    TestParameters params;
    params.lanesPerPool = 8192;
    params.numPools = 30;
    params.framesPerBlock = 512;
    params.numFrames = 512;
    params.bounds.lowerBounds = 100;
    params.bounds.upperBounds = 400;

    SimulatedDataSource::SimConfig simConfig(laneSize, params.numFrames);
    PicketFenceGenerator::Config picketConfig;
    picketConfig.baselineSignalLevel = 120;
    picketConfig.baselineSigma = 10;
    picketConfig.pulseIpdRate = .2;
    picketConfig.pulseWidthRate = 0.1;
    picketConfig.pulseSignalLevels = {180, 250, 320, 380};
    picketConfig.generatePoisson = true;
    auto generator = std::make_unique<PicketFenceGenerator>(picketConfig);

    auto hists = TestFixture::RunTest(params,
                                      simConfig,
                                      std::move(generator));

}

TYPED_TEST(Histogram, PerfPulseOneSignal)
{
    TestParameters params;
    params.lanesPerPool = 8192;
    params.numPools = 30;
    params.framesPerBlock = 512;
    params.numFrames = 512;
    params.bounds.lowerBounds = 100;
    params.bounds.upperBounds = 400;

    SimulatedDataSource::SimConfig simConfig(laneSize, params.numFrames);
    PicketFenceGenerator::Config picketConfig;
    picketConfig.baselineSignalLevel = 120;
    picketConfig.baselineSigma = 10;
    picketConfig.pulseIpdRate = .2;
    picketConfig.pulseWidthRate = 0.1;
    picketConfig.pulseSignalLevels = {180, 250, 320, 380};
    picketConfig.generatePoisson = true;
    picketConfig.seedFunc = [](size_t) { return 0; };
    auto generator = std::make_unique<PicketFenceGenerator>(picketConfig);

    auto hists = TestFixture::RunTest(params,
                                      simConfig,
                                      std::move(generator));

}

TYPED_TEST(Histogram, PerfSortedPulseManySignals)
{
    TestParameters params;
    params.lanesPerPool = 8192;
    params.numPools = 30;
    params.framesPerBlock = 512;
    params.numFrames = 512;
    params.bounds.lowerBounds = 100;
    params.bounds.upperBounds = 400;

    SimulatedDataSource::SimConfig simConfig(laneSize, params.numFrames);
    PicketFenceGenerator::Config picketConfig;
    picketConfig.baselineSignalLevel = 120;
    picketConfig.baselineSigma = 10;
    picketConfig.pulseIpdRate = .2;
    picketConfig.pulseWidthRate = 0.1;
    picketConfig.pulseSignalLevels = {180, 250, 320, 380};
    picketConfig.generatePoisson = true;
    auto generator = SortedGenerator::Create<PicketFenceGenerator>(picketConfig);

    auto hists = TestFixture::RunTest(params,
                                      simConfig,
                                      std::move(generator));

}

TYPED_TEST(Histogram, PerfSortedPulseOneSignal)
{
    TestParameters params;
    params.lanesPerPool = 8192;
    params.numPools = 30;
    params.framesPerBlock = 512;
    params.numFrames = 512;
    params.bounds.lowerBounds = 100;
    params.bounds.upperBounds = 400;

    SimulatedDataSource::SimConfig simConfig(laneSize, params.numFrames);
    PicketFenceGenerator::Config picketConfig;
    picketConfig.baselineSignalLevel = 120;
    picketConfig.baselineSigma = 10;
    picketConfig.pulseIpdRate = .2;
    picketConfig.pulseWidthRate = 0.1;
    picketConfig.pulseSignalLevels = {180, 250, 320, 380};
    picketConfig.generatePoisson = true;
    picketConfig.seedFunc = [](size_t) { return 0; };
    auto generator = SortedGenerator::Create<PicketFenceGenerator>(picketConfig);

    auto hists = TestFixture::RunTest(params,
                                      simConfig,
                                      std::move(generator));

}

TYPED_TEST(Histogram, PerfRandomPulseManySignals)
{
    TestParameters params;
    params.lanesPerPool = 8192;
    params.numPools = 30;
    params.framesPerBlock = 512;
    params.numFrames = 512;
    params.bounds.lowerBounds = 100;
    params.bounds.upperBounds = 400;

    SimulatedDataSource::SimConfig simConfig(laneSize, params.numFrames);
    PicketFenceGenerator::Config picketConfig;
    picketConfig.baselineSignalLevel = 120;
    picketConfig.baselineSigma = 10;
    picketConfig.pulseIpdRate = .2;
    picketConfig.pulseWidthRate = 0.1;
    picketConfig.pulseSignalLevels = {180, 250, 320, 380};
    picketConfig.generatePoisson = true;
    auto generator = RandomizedGenerator::Create<PicketFenceGenerator>(RandomizedGenerator::Config{}, picketConfig);

    auto hists = TestFixture::RunTest(params,
                                      simConfig,
                                      std::move(generator));

}

TYPED_TEST(Histogram, PerfRandomPulseOneSignal)
{
    TestParameters params;
    params.lanesPerPool = 8192;
    params.numPools = 30;
    params.framesPerBlock = 512;
    params.numFrames = 512;
    params.bounds.lowerBounds = 100;
    params.bounds.upperBounds = 400;

    SimulatedDataSource::SimConfig simConfig(laneSize, params.numFrames);
    PicketFenceGenerator::Config picketConfig;
    picketConfig.baselineSignalLevel = 120;
    picketConfig.baselineSigma = 10;
    picketConfig.pulseIpdRate = .2;
    picketConfig.pulseWidthRate = 0.1;
    picketConfig.pulseSignalLevels = {180, 250, 320, 380};
    picketConfig.generatePoisson = true;
    picketConfig.seedFunc = [](size_t) { return 0; };

    RandomizedGenerator::Config randConfig;
    randConfig.seedFunc = [](size_t) { return 0; };
    auto generator = RandomizedGenerator::Create<PicketFenceGenerator>(randConfig, picketConfig);

    auto hists = TestFixture::RunTest(params,
                                      simConfig,
                                      std::move(generator));

}
