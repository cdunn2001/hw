// Copyright (c) 2020-2021, Pacific Biosciences of California, Inc.
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

#include <algorithm>
#include <numeric>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>
#include <gtest/gtest.h>

#include <pacbio/dev/profile/ScopedProfilerChain.h>
#include <pacbio/utilities/Finally.h>

#include <appModules/SimulatedDataSource.h>

#include <basecaller/traceAnalysis/TraceHistogramAccumHost.h>
#include <basecaller/traceAnalysis/DeviceTraceHistogramAccum.h>
#include <common/cuda/memory/DeviceAllocationStash.h>
#include <common/cuda/memory/ManagedAllocations.h>
#include <common/cuda/utility/CudaArray.h>
#include <common/MongoConstants.h>
#include <dataTypes/configs/BasecallerTraceHistogramConfig.h>

#include "SpeedTestToggle.h"

using namespace PacBio::Application;
using namespace PacBio::Configuration;
using namespace PacBio::Cuda::Memory;
using namespace PacBio::Cuda;
using namespace PacBio::Dev::Profile;
using namespace PacBio::DataSource;
using namespace PacBio::Mongo;
using namespace PacBio::Mongo::Basecaller;
using namespace PacBio::Mongo::Data;

using boost::numeric_cast;

namespace {

// Just extracting this here to make it easier to type elsewhere
static constexpr size_t numBins = LaneHistogram<float, int16_t>::numBins;

template <typename T>
LaneArray<T> iotaLaneArray(const T start = T(0))
{
    CudaArray<T, laneSize> a;
    std::iota(a.begin(), a.end(), start);
    return LaneArray<T>::FromArray(a);
}

}

struct TestParameters
{
    size_t lanesPerPool;
    size_t numPools;
    size_t framesPerBlock;
    size_t numFrames;
    LaneHistBounds bounds;

    size_t NumBlocks() const
    { return (numFrames + framesPerBlock - 1u) / framesPerBlock; }
};

struct TestConfig : public PBConfig<TestConfig>
{
    PB_CONFIG(TestConfig);

    PB_CONFIG_OBJECT(Data::BasecallerTraceHistogramConfig, histConfig);

    PB_CONFIG_PARAM(ComputeDevices, analyzerHardware, ComputeDevices::Host);
};

enum class TestTypes
{
    TraceHistogramAccumHost,
    DeviceGlobalInterleaved,
    DeviceGlobalContig,
    DeviceGlobalContigCoopWarps,
    DeviceSharedContigCoopWarps,
    DeviceSharedContig2DBlock,
    DeviceSharedInterleaved2DBlock
};

// Factory method to unify histogram construction, converting
// from an enum value to a trace histogram implementation
std::unique_ptr<TraceHistogramAccumulator> HistFactory(TestTypes type,
                                                       uint32_t poolId,
                                                       uint32_t lanesPerPool,
                                                       StashableAllocRegistrar* registrar)
{
    switch (type)
    {
    case TestTypes::TraceHistogramAccumHost:
        return std::make_unique<TraceHistogramAccumHost>(poolId, lanesPerPool);
    case TestTypes::DeviceGlobalInterleaved:
        return std::make_unique<DeviceTraceHistogramAccum>(poolId, lanesPerPool, registrar, DeviceHistogramTypes::GlobalInterleaved);
    case TestTypes::DeviceGlobalContig:
        return std::make_unique<DeviceTraceHistogramAccum>(poolId, lanesPerPool, registrar, DeviceHistogramTypes::GlobalContig);
    case TestTypes::DeviceGlobalContigCoopWarps:
        return std::make_unique<DeviceTraceHistogramAccum>(poolId, lanesPerPool, registrar, DeviceHistogramTypes::GlobalContigCoopWarps);
    case TestTypes::DeviceSharedContigCoopWarps:
        return std::make_unique<DeviceTraceHistogramAccum>(poolId, lanesPerPool, registrar, DeviceHistogramTypes::SharedContigCoopWarps);
    case TestTypes::DeviceSharedContig2DBlock:
        return std::make_unique<DeviceTraceHistogramAccum>(poolId, lanesPerPool, registrar, DeviceHistogramTypes::SharedContig2DBlock);
    case TestTypes::DeviceSharedInterleaved2DBlock:
        return std::make_unique<DeviceTraceHistogramAccum>(poolId, lanesPerPool, registrar, DeviceHistogramTypes::SharedInterleaved2DBlock);
    }
    throw PBException("Not a valid test type");
}

// Test fixture, which will also handle the logic for generating data and
// getting it placed into the histograms
class Histogram : public testing::TestWithParam<TestTypes>
{
    SMART_ENUM(Profiles, TRACE_UPLOAD, HIST_UPLOAD, HIST_DOWNLOAD, BINNING);
    using Profiler = ScopedProfilerChain<Profiles>;
public:

    static void ConfigureHists(const Data::BasecallerTraceHistogramConfig& config)
    {
        switch (GetParam())
        {
        case TestTypes::TraceHistogramAccumHost:
            TraceHistogramAccumHost::Configure(config);
            break;
        case TestTypes::DeviceGlobalInterleaved:
        case TestTypes::DeviceGlobalContig:
        case TestTypes::DeviceGlobalContigCoopWarps:
        case TestTypes::DeviceSharedContigCoopWarps:
        case TestTypes::DeviceSharedContig2DBlock:
        case TestTypes::DeviceSharedInterleaved2DBlock:
            DeviceTraceHistogramAccum::Configure(config);
            break;
        }
    }

    static bool PerfTestsEnabled()
    { return SpeedTestToggle::Enabled(); }

    // Sets up things for performance monitoring, and creates an RAII functor
    // to tear it down again at the end of the test.
    PacBio::Utilities::Finally SetupPerfMonitoring()
    {
        Memory::SetGlobalAllocationMode(CachingMode::ENABLED, AllocatorMode::CUDA);
        monitorPerf_ = true;

        return PacBio::Utilities::Finally([](){
            Memory::SetGlobalAllocationMode(CachingMode::DISABLED, AllocatorMode::CUDA);
            Memory::SetGlobalAllocationMode(CachingMode::DISABLED, AllocatorMode::MALLOC);
            Profiler::FinalReport();
        });
    }

    // Creates histograms to span a chunk, and populates them with simulated data
    std::vector<std::unique_ptr<TraceHistogramAccumulator>>
    RunTest(const TestParameters& params,
            const SimulatedDataSource::SimConfig& simConfig,
            std::unique_ptr<SignalGenerator> generator)
    {
        DeviceAllocationStash stash;

        std::vector<std::unique_ptr<TraceHistogramAccumulator>> hists;
        for (size_t pool = 0; pool < params.numPools; ++pool)
        {
            StashableAllocRegistrar registrar(pool, stash);
            hists.emplace_back(HistFactory(GetParam(), pool, params.lanesPerPool, &registrar));

            UnifiedCudaArray<LaneHistBounds> poolBounds(params.lanesPerPool,
                                                        SyncDirection::HostWriteDeviceRead,
                                                        SOURCE_MARKER());
            auto boundsView = poolBounds.GetHostView();
            for (size_t lane = 0; lane < boundsView.Size(); ++lane)
            {
                boundsView[lane] = params.bounds;
            }
            hists.back()->Reset(std::move(poolBounds));
        }

        GenerateAndBinData(hists, stash, params, simConfig, std::move(generator));

        return hists;
    }

private:

    // Generates simulated data and feeds it to the histograms passed in.
    void BinData(std::vector<std::unique_ptr<TraceHistogramAccumulator>>& hists,
                 DeviceAllocationStash& stash,
                 DataSourceBase::Configuration sourceConfig,
                 const SimulatedDataSource::SimConfig& simConfig,
                 std::unique_ptr<SignalGenerator> gen)
    {
        BatchDimensions dims;
        dims.laneWidth = laneSize;
        dims.lanesPerBatch = sourceConfig.requestedLayout.NumBlocks();
        dims.framesPerBatch = sourceConfig.requestedLayout.NumFrames();

        SimulatedDataSource source(hists.size() * sourceConfig.requestedLayout.NumZmw(),
                                   simConfig,
                                   std::move(sourceConfig),
                                   std::move(gen));

        ASSERT_EQ(source.PacketLayouts().size(), hists.size());

        // Set up detection models for the pool.
        TraceHistogramAccumulator::PoolDetModel pdm {dims.lanesPerBatch,
                                                     SyncDirection::Symmetric,
                                                     SOURCE_MARKER()};
        {
            const auto laneDetModel = MockLaneDetectionModel<PBHalf>();
            auto pdmv = pdm.GetHostView();
            std::fill(pdmv.begin(), pdmv.end(), laneDetModel);
        }

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

                    // Profiler will only do something if we're configured to enable profiling.
                    typename Profiler::Mode mode = monitorPerf_ ? Profiler::Mode::OBSERVE : Profiler::Mode::IGNORE;
                    Profiler profiler(mode, 1000, 1000);

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
                    hists[poolId]->AddBatch(batch, pdm);
                    CudaSynchronizeDefaultStream();

                    auto download = profiler.CreateScopedProfiler(Profiles::HIST_DOWNLOAD);
                    (void)download;
                    stash.StashPool(poolId);
                    CudaSynchronizeDefaultStream();
                }
            }
        }
    }

    // Sets up the SimulatedDataSource and feeds it's data into the histograms
    void GenerateAndBinData(std::vector<std::unique_ptr<TraceHistogramAccumulator>>& hists,
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

    // Is the current test monitoring performance?
    bool monitorPerf_ = false;
};


TEST_P(Histogram, ResetFromStats)
{
    PacBio::Logging::LogSeverityContext ls(PacBio::Logging::LogLevel::ERROR);

    TestConfig testConfig;
    const auto& histConfig = testConfig.histConfig;
    ConfigureHists(histConfig);

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
            auto tmp = static_cast<float>(zmw + lane*laneSize + 2);
            mView[lane].baselineStats.offset[zmw] = 0;
            mView[lane].baselineStats.moment0[zmw] = tmp;
            mView[lane].baselineStats.moment1[zmw] = tmp*tmp;
            mView[lane].baselineStats.moment2[zmw] = 2*tmp*tmp*tmp;
        }
    }

    auto hAccum = HistFactory(GetParam(), 1, numLanes, nullptr);
    hAccum->Reset(std::move(metrics));

    auto hist = hAccum->Histogram();
    auto hView = hist.data.GetHostView();
    for (size_t lane = 0; lane < numLanes; ++lane)
    {
        for (size_t zmw = 0; zmw < laneSize; ++zmw)
        {
            const float baselineFrames = static_cast<float>(zmw + lane*laneSize + 2);
            const float baselineMean = baselineFrames;
            const float baselineSig = (baselineFrames >= histConfig.BaselineStatMinFrameCount)
                ? std::sqrt(baselineMean * baselineMean * baselineMean / (baselineMean - 1.f))
                : histConfig.FallBackBaselineSigma;

            EXPECT_FLOAT_EQ(hView[lane].binSize[zmw], histConfig.BinSizeCoeff * baselineSig) << baselineFrames;
            // The current implementation is actuall 4 sigma, but I don't want to code against
            // that too tightly.
            EXPECT_LT(hView[lane].lowBound[zmw], baselineMean - 2*baselineSig);
        }
    }
}

TEST_P(Histogram, SingleLaneConstant)
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

    auto hists = RunTest(params,
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

            // In host implementation, edge-frame filter excludes first and last
            // frame.
            if (GetParam() == TestTypes::TraceHistogramAccumHost)
            {
                expected[i] -= 2;
            }
        }
        EXPECT_TRUE(all(counts == expected));
    }
}

TEST_P(Histogram, MultiLaneConstant)
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

    auto hists = RunTest(params,
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
                const auto j = i % laneSize;
                expected[j] = params.numFrames;

                // In host implementation, edge-frame filter excludes first and
                // last.
                if (GetParam() == TestTypes::TraceHistogramAccumHost)
                {
                    expected[j] -= 2;
                }
            }
            EXPECT_TRUE(all(counts == expected)) << lane << " " << i;
        }
    }
}

TEST_P(Histogram, MultiPoolConstant)
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

    auto hists = RunTest(params,
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
                    const auto j = i % laneSize;
                    expected[j] = params.numFrames;

                    // In host implementation, edge-frame filter excludes first and
                    // last frame.
                    if (GetParam() == TestTypes::TraceHistogramAccumHost)
                    {
                        expected[j] -= 2;
                    }
                }
                EXPECT_TRUE(all(counts == expected));
            }
        }
    }
}

TEST_P(Histogram, SawtoothSimpleUniform)
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

    auto hists = RunTest(params,
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

            // The histogram range is [0.0, numBins].
            // The histogram bin size is 1.0.
            // The value of the first frame generated is 0.
            // The sawtooth slope is 1.
            // Baseline sigma is 1.0.  So the threshold used in the edge-frame
            // filter (EFF) is 2.0.

            // The range of bin 0 is [0.0f, 1.0f).  Since the trace data are
            // integers, the only value in this range is 0.  The neighboring
            // frame values are always numBins and 1.  So the EFF will exclude
            // all the frames that would fall into this bin.

            // Similarly, 1 is the only value in the range of bin 1. Neighboring
            // frame values are always 0 and 2.  2 is _barely_ "above" the
            // threshold (2 >= 2).  So all of the frames that would fall into
            // this bin are also excluded.

            // 2 is the only value in the range of bin 2.  Neighboring frame
            // values are always 1 and 3.  So all of the frames that would fall
            // into this bin are also excluded.

            // For bins b = 3, ... numbins-1, the neighboring frame values are
            // all above the threshold.  So all of these frames will pass the
            // EFF, except for the first and last frame of each block.

            // numBins -1 is the only value in the range of the last bin.  The
            // neighboring frame values are always numBins - 2 and 0.  So the
            // EFF will exclude all the frames that would fall into this bin.
            const std::vector<size_t> effBins {0u, 1u, 2u, numBins - 1u};
            for (size_t i = 0; i < numBins; ++i)
            {
                auto counts = LaneArray<uint16_t>(lanedata.binCount[i]);
                const auto binSize = lanedata.binSize.front();
                const auto binStart = lanedata.lowBound.front() + i * binSize;
                const auto binStop = lanedata.lowBound.front() + (i+1) * binSize;
                if (GetParam() == TestTypes::TraceHistogramAccumHost &&
                    std::count(effBins.cbegin(), effBins.cend(), i) != 0)
                {
                    EXPECT_TRUE(all(counts == 0u))
                        << "    bin " << i << ','
                        << "  counts = " << counts.ToArray().front()
                        << "\n    start = " << binStart
                        << "\n    stop  = " << binStop;
                }
                else
                {
                    EXPECT_TRUE(all((counts == 6u) | (counts == 7u)))
                        << "    bin " << i << ','
                        << "  counts = " << counts.ToArray().front()
                        << "\n    start = " << binStart
                        << "\n    stop  = " << binStop;
                }
            }
        }
    }
}

TEST_P(Histogram, SawtoothSkipUniform)
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

    // Bin size is 1.
    // Signal is repeating ramps of {0, 2, 4, ..., numBins - 2?}.
    // Only 2 and (numBins - 2) are excluded as edge frames by CPU
    // implementation.

    auto hists = RunTest(params,
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
                const auto counts = LaneArray<uint16_t>(lanedata.binCount[i]);
                if (GetParam() == TestTypes::TraceHistogramAccumHost
                    && (i == 2 || i == numBins - 2))
                {
                    EXPECT_TRUE(all(counts == 0u))
                        << "  i is " << i
                        << ", count is " << lanedata.binCount[i][0];
                }
                else if (i % 2 == 0)
                {
                    EXPECT_TRUE(all((counts == expect) | (counts == expect+1u)))
                        << "  i is " << i
                        << ", count is " << lanedata.binCount[i][0];
                }
                else
                {
                    EXPECT_TRUE(all(counts == 0u))
                        << "  i is " << i
                        << ", count is " << lanedata.binCount[i][0];
                }
            }
        }
    }
}

TEST_P(Histogram, SawtoothOutliersUniform)
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

    // Bin size = 1.
    // Histogram range = [100, 400].
    // Signal is four repeats of {0, 1, 2, ..., 511}.
    // Values that are marked as edge frames are {0, 1, 2, 511}.
    // The the first and last frames for each ZMW are also defined as edge frames.
    // 511 is a high outlier.  The others are low outliers.

    auto hists = RunTest(params,
                         simConfig,
                         std::move(generator));

    const bool isHostImpl = (GetParam() == TestTypes::TraceHistogramAccumHost);
    for (size_t pool = 0; pool < params.numPools; ++pool)
    {
        EXPECT_EQ(hists[pool]->FramesAdded(), params.numFrames);

        const auto& histdata = hists[pool]->Histogram();
        for (size_t lane = 0; lane < histdata.data.Size(); ++lane)
        {
            const auto& lanedata = histdata.data.GetHostView()[lane];

            const auto expectHigh = 4 * (sawConfig.maxAmp
                                         - params.bounds.upperBounds[0]
                                         - (isHostImpl ? 1 : 0));
            EXPECT_TRUE(all(LaneArray<uint16_t>(lanedata.outlierCountHigh) == expectHigh))
                << "  outlierCountHigh is " << lanedata.outlierCountHigh[0]
                << ",  expectHigh is " << expectHigh;

            const auto expectLow = 4 * (params.bounds.lowerBounds[0]
                                        - sawConfig.minAmp
                                        - (isHostImpl ? 3 : 0));
            EXPECT_TRUE(all(LaneArray<uint16_t>(lanedata.outlierCountLow) == expectLow))
                << "  outlierCountLow is " << lanedata.outlierCountLow[0]
                << ",  expectLow is " << expectLow;

            ArrayUnion<LaneArray<uint16_t>> expected;
            for (size_t i = 0; i < numBins; ++i)
            {
                auto counts = LaneArray<uint16_t>(lanedata.binCount[i]);
                EXPECT_TRUE(all(counts == 4u))
                    << "  i is " << i
                    << ", count is " << lanedata.binCount[i][0];
            }
        }
    }
}

TEST_P(Histogram, SawtoothOutliersStagger)
{
    constexpr unsigned int numPeriods = 4u;
    constexpr unsigned int period = 512u;

    TestParameters params;
    params.lanesPerPool = 2;
    params.numPools = 2;
    params.framesPerBlock = period;
    params.numFrames = period * numPeriods;
    params.bounds.lowerBounds = 100;
    params.bounds.upperBounds = 100+numBins;

    SimulatedDataSource::SimConfig simConfig(laneSize, params.numFrames);
    SawtoothGenerator::Config sawConfig;
    sawConfig.minAmp = 0;
    sawConfig.maxAmp = period;
    sawConfig.periodFrames = period;
    sawConfig.startFrameStagger = 2;
    auto generator = std::make_unique<SawtoothGenerator>(sawConfig);

    // Bin size = 1.
    // Histogram range = [100, 400].
    // Signal is four repeats of {0, 1, 2, ..., maxAmp - 1} for ZMW 0.
    // ZMW z is the same except with a shift of 2*z.  0 <= z < 64.
    // ZMWs are number sequentially over the sequence of lanes.
    // Values that are marked as edge frames are {0, 1, 2, 511}.
    // 511 is a high outlier.  The others are low outliers.

    auto hists = RunTest(params,
                         simConfig,
                         std::move(generator));

    const bool isHostImpl = (GetParam() == TestTypes::TraceHistogramAccumHost);
    const std::vector<size_t> effVals {0u, 1u, 2u, sawConfig.maxAmp - 1u};

    using IntArray = LaneArray<int32_t>;

    // Currently assume that upper and lower histogram bounds are uniform over the lane.

    // Returns the expected number high outliers
    const auto expectHigh = [&](const IntArray& firstVal, const IntArray& lastVal)
    {
        IntArray r = sawConfig.maxAmp - params.bounds.upperBounds[0];
        r *= numeric_cast<int>(numPeriods);
        if (isHostImpl)
        {
            // Count "edge frames" over the histogram range
            // We should have 4 copies of maxAmp - 1.
            IntArray edgeCount = numeric_cast<int>(numPeriods);

            // Is the first frame < maxAmp - 1 and >= histogram range?
            const auto firstValHigh = (firstVal < sawConfig.maxAmp - 1)
                                      & (firstVal >= params.bounds.upperBounds[0]);
            edgeCount += Blend(firstValHigh, IntArray(1), IntArray(0));

            // Is the last frame < maxAmp - 1 and >= histogram range?
            const auto lastValHigh = (lastVal < sawConfig.maxAmp - 1)
                                     & (lastVal >= params.bounds.upperBounds[0]);
            edgeCount += Blend(lastValHigh, IntArray(1), IntArray(0));

            r -= edgeCount;
        }
        return r;
    };

    // Returns the expected number of low outliers.
    const auto expectLow = [&](const IntArray& firstVal, const IntArray& lastVal)
    {
        IntArray r = params.bounds.lowerBounds[0] - sawConfig.minAmp;
        r *= numeric_cast<int>(numPeriods);

        if (isHostImpl)
        {
            // Count "edge frames" over the histogram range
            // We should have 4 copies of {0, 1, 2}.
            IntArray edgeCount = 3 * numeric_cast<int>(numPeriods);

            // Is the first frame > 2 and < histogram range?
            const auto firstValLow = (firstVal > 2)
                                     & (firstVal < params.bounds.lowerBounds[0]);
            edgeCount += Blend(firstValLow, IntArray(1), IntArray(0));

            // Is the last frame > 2 and < histogram range?
            const auto lastValLow = (lastVal > 2)
                                    & (lastVal < params.bounds.lowerBounds[0]);
            edgeCount += Blend(lastValLow, IntArray(1), IntArray(0));

            r -= edgeCount;
        }
        return r;
    };


    for (size_t pool = 0; pool < params.numPools; ++pool)
    {
        EXPECT_EQ(hists[pool]->FramesAdded(), params.numFrames);

        const auto& histdata = hists[pool]->Histogram();
        for (size_t lane = 0; lane < histdata.data.Size(); ++lane)
        {
            // Values of first and last frames.
            const auto z = iotaLaneArray<int32_t>();
            const IntArray firstVal = z * sawConfig.startFrameStagger;
            const IntArray lastVal = (firstVal + numeric_cast<int32_t>(params.numFrames) - 1)
                                     % sawConfig.maxAmp;

            const auto& lanedata = histdata.data.GetHostView()[lane];
            EXPECT_TRUE(all(IntArray(lanedata.outlierCountHigh) == expectHigh(firstVal, lastVal)));
            EXPECT_TRUE(all(IntArray(lanedata.outlierCountLow) == expectLow(firstVal, lastVal)));

            ArrayUnion<LaneArray<uint16_t>> expected;
            for (size_t i = 0; i < numBins; ++i)
            {
                using UShortArray = LaneArray<uint16_t>;
                UShortArray expect = 4u;
                if (isHostImpl)
                {
                    const auto binVal = numeric_cast<int32_t>(i + params.bounds.lowerBounds[0]);
                    expect -= Blend(firstVal == binVal, UShortArray(1), UShortArray(0));
                    expect -= Blend(lastVal == binVal, UShortArray(1), UShortArray(0));
                }

                const auto counts = LaneArray<uint16_t>(lanedata.binCount[i]);
                EXPECT_TRUE(all(counts == expect))
                    << "  i is " << i
                    << ", count of ZMW 0 is " << lanedata.binCount[i][0];
            }
        }
    }
}

TEST_P(Histogram, Reset)
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

    auto hists = RunTest(params,
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

    // Now reset all the hists to make sure they really are empty
    for (auto& hist : hists)
    {
        UnifiedCudaArray<LaneHistBounds> bounds(params.numPools, SyncDirection::HostWriteDeviceRead, SOURCE_MARKER());
        auto bview = bounds.GetHostView();
        for (size_t i = 0; i < bview.Size(); ++i)
        {
            bview[i].lowerBounds = 5;
            bview[i].upperBounds = 10;
        }
        hist->Reset(std::move(bounds));
    }
    for (size_t pool = 0; pool < params.numPools; ++pool)
    {
        EXPECT_EQ(hists[pool]->FramesAdded(), 0);

        const auto& histdata = hists[pool]->Histogram();
        for (size_t lane = 0; lane < histdata.data.Size(); ++lane)
        {
            const auto& lanedata = histdata.data.GetHostView()[lane];
            EXPECT_TRUE(all(LaneArray<uint16_t>(lanedata.outlierCountHigh) == 0u));
            EXPECT_TRUE(all(LaneArray<uint16_t>(lanedata.outlierCountLow) == 0u));

            EXPECT_TRUE(all(LaneArray<float>(lanedata.lowBound) == 5.f));
            EXPECT_TRUE(all(LaneArray<float>(lanedata.binSize) == 5.f/numBins));

            ArrayUnion<LaneArray<uint16_t>> expected;
            for (size_t i = 0; i < numBins; ++i)
            {
                auto counts = LaneArray<uint16_t>(lanedata.binCount[i]);
                EXPECT_TRUE(all(counts == 0u));
            }
        }
    }
}

TEST_P(Histogram, PerfPulseManySignals)
{
    if (!PerfTestsEnabled()) GTEST_SKIP();
    auto finally = this->SetupPerfMonitoring();

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

    auto hists = RunTest(params,
                         simConfig,
                         std::move(generator));

}

TEST_P(Histogram, PerfPulseOneSignal)
{
    if (!PerfTestsEnabled()) GTEST_SKIP();
    auto finally = this->SetupPerfMonitoring();

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

    auto hists = RunTest(params,
                         simConfig,
                         std::move(generator));

}

TEST_P(Histogram, PerfSortedPulseManySignals)
{
    if (!PerfTestsEnabled()) GTEST_SKIP();
    auto finally = this->SetupPerfMonitoring();

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

    auto hists = RunTest(params,
                         simConfig,
                         std::move(generator));

}

TEST_P(Histogram, PerfSortedPulseOneSignal)
{
    if (!PerfTestsEnabled()) GTEST_SKIP();
    auto finally = this->SetupPerfMonitoring();

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

    auto hists = RunTest(params,
                         simConfig,
                         std::move(generator));

}

TEST_P(Histogram, PerfRandomPulseManySignals)
{
    if (!PerfTestsEnabled()) GTEST_SKIP();
    auto finally = this->SetupPerfMonitoring();

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

    auto hists = RunTest(params,
                         simConfig,
                         std::move(generator));

}

TEST_P(Histogram, PerfRandomPulseOneSignal)
{
    if (!PerfTestsEnabled()) GTEST_SKIP();
    auto finally = this->SetupPerfMonitoring();

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

    auto hists = RunTest(params,
                         simConfig,
                         std::move(generator));

}

INSTANTIATE_TEST_SUITE_P(TraceHistogramAccumHost,
                         Histogram,
                         testing::Values(TestTypes::TraceHistogramAccumHost));

INSTANTIATE_TEST_SUITE_P(DeviceGlobalInterleaved,
                         Histogram,
                         testing::Values(TestTypes::DeviceGlobalInterleaved));

INSTANTIATE_TEST_SUITE_P(DeviceGlobalContig,
                         Histogram,
                         testing::Values(TestTypes::DeviceGlobalContig));

INSTANTIATE_TEST_SUITE_P(DeviceGlobalContigCoopWarps,
                         Histogram,
                         testing::Values(TestTypes::DeviceGlobalContigCoopWarps));

INSTANTIATE_TEST_SUITE_P(DeviceSharedContigCoopWarps,
                         Histogram,
                         testing::Values(TestTypes::DeviceSharedContigCoopWarps));

INSTANTIATE_TEST_SUITE_P(DeviceSharedContig2DBlock,
                         Histogram,
                         testing::Values(TestTypes::DeviceSharedContig2DBlock));

INSTANTIATE_TEST_SUITE_P(DeviceSharedInterleaved2DBlock,
                         Histogram,
                         testing::Values(TestTypes::DeviceSharedInterleaved2DBlock));
