// Copyright (c) 2019-2021 Pacific Biosciences of California, Inc.
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
//  Defines unit tests for class BaselineStatsAggregator

#include <basecaller/traceAnalysis/BaselineStatsAggregatorHost.h>
#include <basecaller/traceAnalysis/BaselineStatsAggregatorDevice.h>

#include <algorithm>
#include <map>
#include <vector>

#include <pacbio/logging/Logger.h>

#include <gtest/gtest.h>

namespace PacBio {
namespace Mongo {
namespace Basecaller {

template <typename T>
struct TestBaselineStatsAggregator : public ::testing::Test
{
    using TraceElementType = Data::BaselinedTraceElement;

    static constexpr unsigned int chunkSize = 64;  // frames per chunk
    static constexpr unsigned int poolSize = 2;    // lanes per pool

    PacBio::Logging::LogSeverityContext logContext {PacBio::Logging::LogLevel::WARN};

    // Produces a baseliner stats defined by blMean and blVar.
    // laneAddition and zmwAddition are optional modifiers that
    // can be used to produce a non-uniform data signal.  The
    // mean and variance used will both be:
    //    inputValue + lane * laneAddition + zmwInLane * zmwAddition
    Data::BaselinerMetrics GenerateStats(float blMean, float blVar,
                                         int laneAddition = 0, int zmwAddition = 0)
    {
        Data::BaselinerMetrics stats(poolSize, Cuda::Memory::SyncDirection::HostWriteDeviceRead, SOURCE_MARKER());
        stats.frameInterval = {0, chunkSize};
        ArrayUnion<LaneArray<float>> addVec;
        const auto lag = AutocorrAccumState::lag;
        for (uint32_t i = 0; i < laneSize; ++i)
        {
            addVec[i] = static_cast<float>(i*zmwAddition);
        }

        for (unsigned int l = 0; l < poolSize; ++l)
        {
            Data::BaselinerStatAccumulator<TraceElementType> bsa;
            Data::BaselinerStatAccumState& bls = stats.baselinerStats.GetHostView()[l];
            bls = bsa.GetState();

            const auto n0 = chunkSize/2 + l * laneAddition + addVec;  // Number of mock baseline frames.
            const auto meanVec = blMean + l * laneAddition + addVec;
            const auto varVec  = blVar  + l * laneAddition + addVec;
            
            // Model the baseline statistics
            bls.baselineStats.moment0 = n0;
            bls.baselineStats.moment1 = n0 * meanVec;
            bls.baselineStats.moment2 = (n0 - 1)*varVec + n0*pow2(meanVec);

            // Model autocor values too
            bls.fullAutocorrState.moment2      = lag * meanVec * 0.0f                // left part
                                               + (n0 - 2*lag)  * pow2(meanVec)       // main part
                                               + lag * (meanVec + varVec) * meanVec; // right part
            for (auto k = 0u; k < lag; ++k) bls.fullAutocorrState.fBuf[k] = meanVec - varVec;   // left values
            for (auto k = 0u; k < lag; ++k) bls.fullAutocorrState.bBuf[k] = meanVec + varVec;   // right values

            auto cnt = uint32_t(n0.ToArray()[0]);
            bls.fullAutocorrState.bIdx[0] = std::min(cnt, lag);
            bls.fullAutocorrState.bIdx[1] = cnt % lag;

            // Cheat a bit and duplicate the baseline stats into the autocorr stats
            bls.fullAutocorrState.basicStats = bls.baselineStats;

            // Set the min/max (rather arbitrarily) based off the mean and variance
            bls.traceMin = meanVec - 2*varVec;
            bls.traceMax = meanVec * 10 + 5*varVec;

            bls.rawBaselineSum = n0 * meanVec;
        }

        return stats;
    }
};

using TestTypes = ::testing::Types<BaselineStatsAggregatorHost,
                                   BaselineStatsAggregatorDevice>;
TYPED_TEST_SUITE(TestBaselineStatsAggregator, TestTypes);

template <typename T>
std::unique_ptr<T> CreateAggregator(uint32_t poolId, uint32_t numLanes);

template <>
std::unique_ptr<BaselineStatsAggregatorHost> CreateAggregator<BaselineStatsAggregatorHost>(uint32_t poolId, uint32_t numLanes)
{
    return std::make_unique<BaselineStatsAggregatorHost>(poolId, numLanes);
}
template <>
std::unique_ptr<BaselineStatsAggregatorDevice> CreateAggregator<BaselineStatsAggregatorDevice>(uint32_t poolId, uint32_t numLanes)
{
    return std::make_unique<BaselineStatsAggregatorDevice>(poolId, numLanes, nullptr);
}

TYPED_TEST(TestBaselineStatsAggregator, PoolMeta)
{
    std::unique_ptr<BaselineStatsAggregator> bsa = CreateAggregator<TypeParam>(7, TestFixture::poolSize);
    EXPECT_EQ(bsa->PoolId(), 7);
    EXPECT_EQ(bsa->PoolSize(), uint32_t{TestFixture::poolSize});
    EXPECT_TRUE(bsa->FrameInterval().Empty());
}

// Make sure the aggregator has a sensible initial "empty" state
TYPED_TEST(TestBaselineStatsAggregator, EmptyAggregator)
{
    auto bsa = CreateAggregator<TypeParam>(7, TestFixture::poolSize);
    using LaneArr = LaneArray<float>;
    Data::BaselinerMetrics metrics = bsa->TraceStats();
    EXPECT_TRUE(metrics.frameInterval.Empty());
    for(size_t i = 0; i < metrics.baselinerStats.Size(); ++i)
    {
        const auto& actual = metrics.baselinerStats.GetHostView()[i];
        EXPECT_TRUE(all(LaneArr(actual.baselineStats.moment0) == 0));
        EXPECT_TRUE(all(LaneArr(actual.baselineStats.moment1) == 0));
        EXPECT_TRUE(all(LaneArr(actual.baselineStats.moment2) == 0));
        EXPECT_TRUE(all(LaneArr(actual.baselineStats.offset) == 0));

        EXPECT_TRUE(all(LaneArr(actual.traceMax) == std::numeric_limits<int16_t>::lowest()));
        EXPECT_TRUE(all(LaneArr(actual.traceMin) == std::numeric_limits<int16_t>::max()));
        EXPECT_TRUE(all(LaneArr(actual.rawBaselineSum) == 0));

        EXPECT_TRUE(all(LaneArr(actual.fullAutocorrState.basicStats.moment0) == 0));
        EXPECT_TRUE(all(LaneArr(actual.fullAutocorrState.basicStats.moment1) == 0));
        EXPECT_TRUE(all(LaneArr(actual.fullAutocorrState.basicStats.moment2) == 0));
        EXPECT_TRUE(all(LaneArr(actual.fullAutocorrState.basicStats.offset) == 0));

        EXPECT_TRUE(all(LaneArr(actual.fullAutocorrState.moment2) == 0));
    }
}

// The aggregate of a single entry should be identical to that entry.
// Calling the reset function should restore us to our original
// pristine "empty" state
TYPED_TEST(TestBaselineStatsAggregator, OneAndReset)
{
    auto lag = AutocorrAccumState::lag;
    auto bsa = CreateAggregator<TypeParam>(7, TestFixture::poolSize);
    auto generatedStats = TestFixture::GenerateStats(4, 8);
    bsa->AddMetrics(generatedStats);
    using LaneArr = LaneArray<float>;
    Data::BaselinerMetrics metrics = bsa->TraceStats();
    EXPECT_EQ(generatedStats.frameInterval, metrics.frameInterval);
    for(size_t i = 0; i < metrics.baselinerStats.Size(); ++i)
    {
        const auto& actual = metrics.baselinerStats.GetHostView()[i];
        const auto& expected = generatedStats.baselinerStats.GetHostView()[i];
        EXPECT_TRUE(all(LaneArr(actual.baselineStats.moment0) == LaneArr(expected.baselineStats.moment0)));
        EXPECT_TRUE(all(LaneArr(actual.baselineStats.moment1) == LaneArr(expected.baselineStats.moment1)));
        EXPECT_TRUE(all(LaneArr(actual.baselineStats.moment2) == LaneArr(expected.baselineStats.moment2)));
        EXPECT_TRUE(all(LaneArr(actual.baselineStats.offset) == LaneArr(expected.baselineStats.offset)));

        EXPECT_TRUE(all(LaneArr(actual.traceMax) == LaneArr(expected.traceMax)));
        EXPECT_TRUE(all(LaneArr(actual.traceMin) == LaneArr(expected.traceMin)));
        EXPECT_TRUE(all(LaneArr(actual.rawBaselineSum) == LaneArr(expected.rawBaselineSum)));

        EXPECT_TRUE(all(LaneArr(actual.fullAutocorrState.basicStats.moment0) == LaneArr(expected.fullAutocorrState.basicStats.moment0)));
        EXPECT_TRUE(all(LaneArr(actual.fullAutocorrState.basicStats.moment1) == LaneArr(expected.fullAutocorrState.basicStats.moment1)));
        EXPECT_TRUE(all(LaneArr(actual.fullAutocorrState.basicStats.moment2) == LaneArr(expected.fullAutocorrState.basicStats.moment2)));
        EXPECT_TRUE(all(LaneArr(actual.fullAutocorrState.basicStats.offset) == LaneArr(expected.fullAutocorrState.basicStats.offset)));

        EXPECT_TRUE(all(LaneArr(actual.fullAutocorrState.moment2) == LaneArr(expected.fullAutocorrState.moment2)));
    }

    Data::BaselinerMetrics metrics2 = bsa->TraceStats();
    EXPECT_EQ(metrics.frameInterval, metrics2.frameInterval);
    for(size_t i = 0; i < metrics.baselinerStats.Size(); ++i)
    {
        const auto& actual   =  metrics.baselinerStats.GetHostView()[i];
        const auto& expected = metrics2.baselinerStats.GetHostView()[i];
        EXPECT_TRUE(all(LaneArr(actual.baselineStats.moment0) == LaneArr(expected.baselineStats.moment0)));
        EXPECT_TRUE(all(LaneArr(actual.baselineStats.moment1) == LaneArr(expected.baselineStats.moment1)));
        EXPECT_TRUE(all(LaneArr(actual.baselineStats.moment2) == LaneArr(expected.baselineStats.moment2)));
        EXPECT_TRUE(all(LaneArr(actual.baselineStats.offset) == LaneArr(expected.baselineStats.offset)));

        EXPECT_TRUE(all(LaneArr(actual.traceMax) == LaneArr(expected.traceMax)));
        EXPECT_TRUE(all(LaneArr(actual.traceMin) == LaneArr(expected.traceMin)));
        EXPECT_TRUE(all(LaneArr(actual.rawBaselineSum) == LaneArr(expected.rawBaselineSum)));

        EXPECT_TRUE(all(LaneArr(actual.fullAutocorrState.basicStats.moment0) == LaneArr(expected.fullAutocorrState.basicStats.moment0)));
        EXPECT_TRUE(all(LaneArr(actual.fullAutocorrState.basicStats.moment1) == LaneArr(expected.fullAutocorrState.basicStats.moment1)));
        EXPECT_TRUE(all(LaneArr(actual.fullAutocorrState.basicStats.moment2) == LaneArr(expected.fullAutocorrState.basicStats.moment2)));
        EXPECT_TRUE(all(LaneArr(actual.fullAutocorrState.basicStats.offset) == LaneArr(expected.fullAutocorrState.basicStats.offset)));

        EXPECT_TRUE(all(LaneArr(actual.fullAutocorrState.moment2) == LaneArr(expected.fullAutocorrState.moment2)));
    }

    bsa->Reset();
    Data::BaselinerMetrics metrics3 = bsa->TraceStats();
    EXPECT_TRUE(metrics3.frameInterval.Empty());
    for(size_t j = 0; j < metrics.baselinerStats.Size(); ++j)
    {
        const auto& actual = metrics3.baselinerStats.GetHostView()[j];
        EXPECT_TRUE(all(LaneArr(actual.baselineStats.moment0) == 0));
        EXPECT_TRUE(all(LaneArr(actual.baselineStats.moment1) == 0));
        EXPECT_TRUE(all(LaneArr(actual.baselineStats.moment2) == 0));
        EXPECT_TRUE(all(LaneArr(actual.baselineStats.offset) == 0));

        EXPECT_TRUE(all(LaneArr(actual.traceMax) == std::numeric_limits<int16_t>::lowest()));
        EXPECT_TRUE(all(LaneArr(actual.traceMin) == std::numeric_limits<int16_t>::max()));
        EXPECT_TRUE(all(LaneArr(actual.rawBaselineSum) == 0));

        EXPECT_TRUE(all(LaneArr(actual.fullAutocorrState.basicStats.moment0) == 0));
        EXPECT_TRUE(all(LaneArr(actual.fullAutocorrState.basicStats.moment1) == 0));
        EXPECT_TRUE(all(LaneArr(actual.fullAutocorrState.basicStats.moment2) == 0));
        EXPECT_TRUE(all(LaneArr(actual.fullAutocorrState.basicStats.offset) == 0));

        EXPECT_TRUE(all(LaneArr(actual.fullAutocorrState.moment2) == 0));

        for (auto k = 0u; k < lag; ++k)
        { 
            EXPECT_TRUE(all(LaneArr(actual.fullAutocorrState.fBuf[k]) == 0));
            EXPECT_TRUE(all(LaneArr(actual.fullAutocorrState.bBuf[k]) == 0));
        }

        for (auto k = 0u; k < actual.fullAutocorrState.bIdx.size(); ++k)
        { 
            EXPECT_TRUE(all(LaneArr(actual.fullAutocorrState.bIdx[k]) == 0));
        }
    }
}


// This block is a legacy test which I'm loath to delete, but doesn't actually
// test the full functionality of the BaselineStatsAggregator, just the baseline
// mean/variance themselves as reported by the host BaselinerStatAccumulator code.
// It's the product of the code evolving and being refactored, but the subsequent
// tests below will more fully check all the values the BaselineStatsAggregator is
// actually responsible for.
TYPED_TEST(TestBaselineStatsAggregator, UniformSimple)
{
    auto bsa = CreateAggregator<TypeParam>(7, TestFixture::poolSize);

    const std::vector<float> mPar {0.0f, 1.0f, 4.0f, 1.0f};
    const std::vector<float> s2Par {2.0f, 3.0f, 6.0f, 3.1f};
    const uint32_t nChunks = mPar.size();
    ASSERT_EQ(nChunks, s2Par.size()) << "Test is broken.";

    // Feed mock data to BaselineStatsAggregator under test.
    for (unsigned int i = 0; i < nChunks; ++i)
    {
        auto stats = TestFixture::GenerateStats(mPar[i], s2Par[i]);
        stats.frameInterval += i * this->chunkSize;
        bsa->AddMetrics(stats);
    }

    // Expected accumulated baseline statistics.
    const auto n0 = TestFixture::chunkSize/2;
    const float mExpect = std::accumulate(mPar.begin(), mPar.end(), 0.0f) / nChunks;
    float s2Expect = (n0-1) * std::accumulate(s2Par.begin(), s2Par.end(), 0.0f);
    for (unsigned int i = 0; i < nChunks; ++i)
    {
        s2Expect += n0 * pow2(mPar[i] - mExpect);
    }
    s2Expect /= (nChunks*n0 - 1);

    // Check the accumulated baseline statistics.
    const auto& tsPool = bsa->TraceStats();
    for (const auto& tsLane : tsPool.baselinerStats.GetHostView())
    {
        auto bls = Data::BaselinerStatAccumulator<uint16_t>(tsLane).BaselineFramesStats();
        const auto n = bls.Count().ToArray();
        const auto m = bls.Mean().ToArray();
        const auto s2 = bls.Variance().ToArray();
        for (unsigned int i = 0; i < laneSize; ++i)
        {
            EXPECT_EQ(nChunks*n0, n[i]);
            EXPECT_FLOAT_EQ(mExpect, m[i]);
            EXPECT_FLOAT_EQ(s2Expect, s2[i]);
        }
    }

    // Check frame interval.
    using FrameIntervalT = BaselineStatsAggregator::FrameIntervalType;
    EXPECT_EQ(FrameIntervalT(0, nChunks * this->chunkSize), tsPool.frameInterval);
}

// Similar to the past test, but now we'll make sure we actually
// have data variability, both within and across lanes.  We'll also
// check the aggregation directly (which for most fields just means
// summing them), rather than via a proxy that also relies on the
// mean/variance computation from moments.
TYPED_TEST(TestBaselineStatsAggregator, VariedData)
{
    auto bsa = CreateAggregator<TypeParam>(7, TestFixture::poolSize);

    const std::vector<float> mPar {0.0f, 1.0f, 4.0f, 1.0f};
    const std::vector<float> s2Par {2.0f, 3.0f, 6.0f, 3.1f};
    ASSERT_EQ(mPar.size(), s2Par.size()) << "Number of means and variance should be equal";
    const Data::FrameIndexType frameOffset = 42;

    // Feed mock data to BaselineStatsAggregator under test.
    std::vector<Data::BaselinerMetrics> simulatedStats;
    for (unsigned int i = 0; i < mPar.size(); ++i)
    {
        // couple parameters to force non-constant data across pool/lane
        int laneMultiplier = 2;
        int zmwAddition = 1;
        auto stats = TestFixture::GenerateStats(mPar[i], s2Par[i], laneMultiplier, zmwAddition);
        stats.frameInterval += i * this->chunkSize + frameOffset;
        bsa->AddMetrics(stats);
        simulatedStats.push_back(std::move(stats));
    }

    using LaneArr = LaneArray<float>;
    using FrameIntervalT = BaselineStatsAggregator::FrameIntervalType;
    auto lag = AutocorrAccumState::lag;
    Data::BaselinerMetrics metrics = bsa->TraceStats();
    EXPECT_EQ(FrameIntervalT(0, mPar.size() * this->chunkSize) + frameOffset,
              metrics.frameInterval);
    for(size_t lane = 0; lane < metrics.baselinerStats.Size(); ++lane)
    {
        Data::BaselinerStatAccumState expected{};
        for (const auto& stat : simulatedStats)
        {
            const auto& laneStat = stat.baselinerStats.GetHostView()[lane];
            for (size_t zmw = 0; zmw < laneSize; ++zmw)
            {
                expected.baselineStats.moment0[zmw] += laneStat.baselineStats.moment0[zmw];
                expected.baselineStats.moment1[zmw] += laneStat.baselineStats.moment1[zmw];
                expected.baselineStats.moment2[zmw] += laneStat.baselineStats.moment2[zmw];
                expected.baselineStats.offset[zmw] += laneStat.baselineStats.offset[zmw];

                expected.rawBaselineSum[zmw] += laneStat.rawBaselineSum[zmw];
                expected.traceMax[zmw] = std::max(expected.traceMax[zmw], laneStat.traceMax[zmw]);
                expected.traceMin[zmw] = std::min(expected.traceMin[zmw], laneStat.traceMin[zmw]);

                expected.fullAutocorrState.basicStats.moment0[zmw] += laneStat.fullAutocorrState.basicStats.moment0[zmw];
                expected.fullAutocorrState.basicStats.moment1[zmw] += laneStat.fullAutocorrState.basicStats.moment1[zmw];
                expected.fullAutocorrState.basicStats.moment2[zmw] += laneStat.fullAutocorrState.basicStats.moment2[zmw];
                expected.fullAutocorrState.basicStats.offset[zmw] += laneStat.fullAutocorrState.basicStats.offset[zmw];

                expected.fullAutocorrState.moment2[zmw] += laneStat.fullAutocorrState.moment2[zmw];

                for (auto k = 0u; k < lag; ++k)
                {
                    expected.fullAutocorrState.moment2[zmw] +=
                        expected.fullAutocorrState.bBuf[k][zmw] * laneStat.fullAutocorrState.fBuf[k][zmw];

                    expected.fullAutocorrState.bBuf[k][zmw] = laneStat.fullAutocorrState.bBuf[k][zmw];
                }
            }
        }

        const auto& actual = metrics.baselinerStats.GetHostView()[lane];
        EXPECT_TRUE(all(LaneArr(actual.baselineStats.moment0) == LaneArr(expected.baselineStats.moment0)));
        EXPECT_TRUE(all(LaneArr(actual.baselineStats.moment1) == LaneArr(expected.baselineStats.moment1)));
        EXPECT_TRUE(all(LaneArr(actual.baselineStats.moment2) == LaneArr(expected.baselineStats.moment2)));
        EXPECT_TRUE(all(LaneArr(actual.baselineStats.offset) == LaneArr(expected.baselineStats.offset)));

        EXPECT_TRUE(all(LaneArr(actual.traceMax) == LaneArr(expected.traceMax)));
        EXPECT_TRUE(all(LaneArr(actual.traceMin) == LaneArr(expected.traceMin)));
        EXPECT_TRUE(all(LaneArr(actual.rawBaselineSum) == LaneArr(expected.rawBaselineSum)));

        EXPECT_TRUE(all(LaneArr(actual.fullAutocorrState.basicStats.moment0) == LaneArr(expected.fullAutocorrState.basicStats.moment0)));
        EXPECT_TRUE(all(LaneArr(actual.fullAutocorrState.basicStats.moment1) == LaneArr(expected.fullAutocorrState.basicStats.moment1)));
        EXPECT_TRUE(all(LaneArr(actual.fullAutocorrState.basicStats.moment2) == LaneArr(expected.fullAutocorrState.basicStats.moment2)));
        EXPECT_TRUE(all(LaneArr(actual.fullAutocorrState.basicStats.offset) == LaneArr(expected.fullAutocorrState.basicStats.offset)));

        EXPECT_TRUE(all(LaneArr(actual.fullAutocorrState.moment2) == LaneArr(expected.fullAutocorrState.moment2)));
    }
}

}}}     // namespace PacBio::Mongo::Basecaller
