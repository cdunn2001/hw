
// Copyright (c) 2019,2020 Pacific Biosciences of California, Inc.
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
//  Defines unit tests for class BaselineStatsAggregatorHost

#include <basecaller/traceAnalysis/BaselineStatsAggregatorHost.h>

#include <algorithm>
#include <map>
#include <vector>

#include <pacbio/logging/Logger.h>

#include <dataTypes/configs/BasecallerBaselineStatsAggregatorConfig.h>

#include <gtest/gtest.h>

namespace PacBio {
namespace Mongo {
namespace Basecaller {

struct TestBaselineStatsAggregatorHost : public ::testing::Test
{
    using TraceElementType = Data::BaselinedTraceElement;

    const unsigned int chunkSize = 64;  // frames per chunk
    const unsigned int poolSize = 6;    // lanes per pool

    Data::BasecallerBaselineStatsAggregatorConfig sigConfig;
    PacBio::Logging::LogSeverityContext logContext {PacBio::Logging::LogLevel::WARN};

    // Produces a baseliner stats defined by blMean and blVar.
    Data::BaselinerMetrics GenerateStats(float blMean, float blVar)
    {
        Data::BaselinerMetrics stats(poolSize, Cuda::Memory::SyncDirection::HostWriteDeviceRead, SOURCE_MARKER());

        const auto n0 = chunkSize/2;  // Number of mock baseline frames.
        for (unsigned int l = 0; l < poolSize; ++l)
        {
            Data::BaselinerStatAccumulator<TraceElementType> bsa;
            Data::BaselinerStatAccumState& bls = stats.baselinerStats.GetHostView()[l];
            bls = bsa.GetState();

            // Hack the baseline statistics.
            bls.baselineStats.moment0 = n0;
            bls.baselineStats.moment1 = n0 * blMean;
            bls.baselineStats.moment2 = (n0 - 1)*blVar + n0*pow2(blMean);
        }

        return stats;
    }
};


TEST_F(TestBaselineStatsAggregatorHost, UniformSimple)
{
    BaselineStatsAggregatorHost bsa (7, poolSize);

    const std::vector<float> mPar {0.0f, 1.0f, 4.0f, 1.0f};
    const std::vector<float> s2Par {2.0f, 3.0f, 6.0f, 3.1f};
    const uint32_t nChunks = mPar.size();
    ASSERT_EQ(nChunks, s2Par.size()) << "Test is broken.";

    // Feed mock data to BaselineStatsAggregator under test.
    for (unsigned int i = 0; i < nChunks; ++i)
    {
        auto stats = GenerateStats(mPar[i], s2Par[i]);
        bsa.AddMetrics(stats);
    }

    // Expected accumulated baseline statistics.
    const auto n0 = chunkSize/2;
    const float mExpect = std::accumulate(mPar.begin(), mPar.end(), 0.0f) / nChunks;
    float s2Expect = (n0-1) * std::accumulate(s2Par.begin(), s2Par.end(), 0.0f);
    for (unsigned int i = 0; i < nChunks; ++i)
    {
        s2Expect += n0 * pow2(mPar[i] - mExpect);
    }
    s2Expect /= (nChunks*n0 - 1);

    // Check the accumulated baseline statistics.
    const auto& tsPool = bsa.TraceStatsHost();
    for (const auto& tsLane : tsPool)
    {
        const auto& bls = tsLane.BaselineFramesStats();
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
}

}}}     // namespace PacBio::Mongo::Basecaller
