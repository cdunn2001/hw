
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
//  Defines unit tests for class TraceHistogramAccumHost.

#include <basecaller/traceAnalysis/TraceHistogramAccumHost.h>

#include <algorithm>
#include <map>
#include <vector>
#include <boost/numeric/conversion/cast.hpp>

#include <pacbio/logging/Logger.h>

#include <dataTypes/configs/BasecallerTraceHistogramConfig.h>
#include <dataTypes/configs/MovieConfig.h>
#include <dataTypes/CameraTraceBatch.h>

#include <gtest/gtest.h>

using std::numeric_limits;
using boost::numeric_cast;

namespace PacBio {
namespace Mongo {
namespace Basecaller {

struct TestTraceHistogramAccumHost : public ::testing::Test
{
    using TraceElementType = Data::BaselinedTraceElement;

    const unsigned int chunkSize = 64;  // frames per chunk
    const unsigned int poolSize = 6;    // lanes per pool

    Data::BatchMetadata bmd {7, 0, chunkSize, 7*poolSize*laneSize};
    Data::BasecallerTraceHistogramConfig histConfig;
    Data::MovieConfig movConfig;
    Data::CameraBatchFactory ctbFactory {Cuda::Memory::SyncDirection::Symmetric};
    PacBio::Logging::LogSeverityContext logContext {PacBio::Logging::LogLevel::WARN};

    void SetUp()
    {
        histConfig.NumFramesPreAccumStats = 100;
        movConfig = Data::MockMovieConfig();
        TraceHistogramAccumulator::Configure(histConfig, movConfig);
    }

    // Produces a trace batch with all trace frames set to x
    // and baseliner stats defined by blMean and blVar.
    std::pair<Data::TraceBatch<TraceElementType>,
              Data::BaselinerMetrics>
    GenerateCamTraceBatch(TraceElementType x, float blMean, float blVar)
    {
        Data::BatchDimensions dims;
        dims.framesPerBatch = chunkSize;
        dims.laneWidth = laneSize;
        dims.lanesPerBatch = poolSize;
        auto ctb = ctbFactory.NewBatch(bmd, dims);
        auto& traces = ctb.first;
        auto& stats = ctb.second;

        const auto n0 = chunkSize/2;  // Number of mock baseline frames.
        for (unsigned int l = 0; l < poolSize; ++l)
        {
            Data::BaselinerStatAccumulator<TraceElementType> bsa;

            // Fill in the trace data.
            auto bvl = traces.GetBlockView(l);
            for (auto lfi = bvl.Begin(); lfi != bvl.End(); ++lfi)
            {
                *lfi = x;
                bsa.AddSample(*lfi, *lfi, true);
            }

            Data::BaselinerStatAccumState& bls = stats.baselinerStats.GetHostView()[l];
            bls = bsa.GetState();

            // Hack the baseline statistics.
            LaneArrayRef<float>(bls.baselineStats.moment0) = n0;
            LaneArrayRef<float>(bls.baselineStats.moment1) = n0 * blMean;
            LaneArrayRef<float>(bls.baselineStats.moment2) = (n0 - 1)*blVar + n0*pow2(blMean);
        }

        // Prepare for the next call.
        AdvanceFrameInterval(chunkSize);

        return ctb;
    }

    void AdvanceFrameInterval(uint32_t n)
    {
        auto t0 = bmd.FirstFrame();
        auto t1 = bmd.LastFrame();
        assert(t0 < t1);
        t0 += n;
        t1 += n;
        assert(t0 < t1);
        bmd = Data::BatchMetadata(bmd.PoolId(), t0, t1, bmd.FirstZmw());
    }
};


TEST_F(TestTraceHistogramAccumHost, UniformSimple)
{
    TraceHistogramAccumHost tha (bmd.PoolId(), poolSize);
    ASSERT_EQ(0, tha.FramesAdded());
    ASSERT_EQ(0, tha.HistogramFrameCount());

    const std::vector<float> mPar {0.0f, 1.0f, 4.0f, 1.0f};
    const std::vector<float> s2Par {2.0f, 3.0f, 6.0f, 3.1f};
    const auto nChunks = mPar.size();
    ASSERT_EQ(nChunks, s2Par.size()) << "Test is broken.";

    // Count repeats. Skip first value because of NumFramesPreAccumStats logic.
    std::map<float, unsigned int> nRepeat;

    // Feed mock data to histogram accumulator under test.
    for (unsigned int i = 0; i < nChunks; ++i)
    {
        const auto x = round_cast<TraceElementType>(mPar[i]);
        if (i > 0) nRepeat[numeric_cast<float>(x)] += chunkSize;
        auto baselinedTracesAndStats = GenerateCamTraceBatch(x, mPar[i], s2Par[i]);
        tha.AddBatch(baselinedTracesAndStats.first,
                     baselinedTracesAndStats.second.baselinerStats);
        ASSERT_EQ((i+1)*chunkSize, tha.FramesAdded());
        ASSERT_EQ(i*chunkSize, tha.HistogramFrameCount());
    }

    // Expected accumulated baseline statistics.
    const auto n0 = chunkSize/2;
    const float mExpect = std::accumulate(mPar.begin(), mPar.end(), 0.0f) / nChunks;
    float s2Expect = (n0 - 1) * std::accumulate(s2Par.begin(), s2Par.end(), 0.0f);
    for (unsigned int i = 0; i < nChunks; ++i)
    {
        s2Expect += n0 * pow2(mPar[i] - mExpect);
    }
    s2Expect /= (nChunks*n0 - 1);

    // Check the accumulated baseline statistics.
    const auto& tsPool = tha.TraceStatsHost();
    for (const auto& tsLane : tsPool)
    {
        const auto& bls = tsLane.BaselineFramesStats();
        const auto n = bls.Count();
        const auto m = bls.Mean();
        const auto s2 = bls.Variance();
        for (unsigned int i = 0; i < laneSize; ++i)
        {
            EXPECT_EQ(nChunks*n0, n[i]);
            EXPECT_FLOAT_EQ(mExpect, m[i]);
            EXPECT_FLOAT_EQ(s2Expect, s2[i]);
        }
    }

    // Check the histogram bin counts.
    const auto& hPool = tha.HistogramHost();
    for (const auto& hLane : hPool)
    {
        ASSERT_GE(hLane.NumBins(), 0);
        const unsigned int nBins = hLane.NumBins();
        const auto& irc = hLane.InRangeCount();
        for (const unsigned int n : irc) EXPECT_EQ((nChunks-1)*chunkSize, n);
        for (unsigned int b = 0; b < nBins; ++b)
        {
            const auto& bStart = hLane.BinStart(b);
            const auto& bStop = hLane.BinStart(b+1);
            for (unsigned int z = 0; z < laneSize; ++z)
            {
                const auto i0 = nRepeat.lower_bound(bStart[z]);
                const auto i1 = nRepeat.lower_bound(bStop[z]);
                const unsigned int nExpect
                        = std::accumulate(i0, i1, 0u,
                                          [](unsigned int s, auto p){return s + p.second;});
                EXPECT_EQ(nExpect, hLane.BinCount(b)[z]);
            }
        }
    }
}

}}}     // namespace PacBio::Mongo::Basecaller
