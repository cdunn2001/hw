
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
#include <dataTypes/BasecallerConfig.h>
#include <dataTypes/MovieConfig.h>

#include <gtest/gtest.h>

using std::numeric_limits;

namespace PacBio {
namespace Mongo {
namespace Basecaller {

struct TestTraceHistogramAccumHost : public ::testing::Test
{
    using TraceElementType = Data::CameraTraceBatch::ElementType;

    const float blMean = 0.42f;         // baseline mean
    const float blVar = 4.0f;           // baseline variance
    const unsigned int chunkSize = 64;  // frames per chunk
    const unsigned int poolSize = 6;    // lanes per pool
    const bool ucaPinnedAlloc = false;

    Data::BatchMetadata bmd {7, 0, chunkSize};
    Data::BasecallerTraceHistogramConfig histConfig;
    Data::MovieConfig movConfig;
    Data::CameraBatchFactory ctbFactory {chunkSize, poolSize,
                                         Cuda::Memory::SyncDirection::Symmetric,
                                         ucaPinnedAlloc};

    void SetUp()
    {
        movConfig = Data::MockMovieConfig();
        TraceHistogramAccumulator::Configure(histConfig, movConfig);
    }

    // Produces a trace batch with fixed baseliner stats and all trace frames
    // set to x.
    Data::CameraTraceBatch GenerateCamTraceBatch(TraceElementType x)
    {
        auto ctb = ctbFactory.NewBatch(bmd);

        const auto n0 = chunkSize/2;  // Number of mock baseline frames.
        for (unsigned int l = 0; l < poolSize; ++l)
        {
            // Mock up some baseliner statistics.
            Data::BaselinerStatAccumState& bls = ctb.Stats().GetHostView()[l];
            LaneArrayRef<float>(bls.fullAutocorrState.moment2) = 0;
            bls.fullAutocorrState.moment1First = bls.fullAutocorrState.moment1Last = bls.fullAutocorrState.moment2;
            LaneArrayRef<float>(bls.baselineStats.moment0) = n0;
            LaneArrayRef<float>(bls.baselineStats.moment1) = n0 * blMean;
            LaneArrayRef<float>(bls.baselineStats.moment2) = (n0 - 1)*blVar + n0*pow2(blMean);

            // Fill in the trace data.
            auto bvl = ctb.GetBlockView(l);
            for (auto lfi = bvl.Begin(); lfi != bvl.End(); ++lfi)
            {
                *lfi = x;
            }
            LaneArrayRef<TraceElementType>(bls.traceMin) = x;
            LaneArrayRef<TraceElementType>(bls.traceMax) = x;
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
        bmd = Data::BatchMetadata(bmd.PoolId(), t0, t1);
    }
};


TEST_F(TestTraceHistogramAccumHost, DISABLED_WIP_One)
{
    const bool pinnedAlloc = false;
    TraceHistogramAccumHost tha (bmd.PoolId(), poolSize, pinnedAlloc);
    EXPECT_EQ(0, tha.FramesAdded());
    EXPECT_EQ(0, tha.HistogramFrameCount());

    // TODO: Blocked by incompleteness of BaselineStats. See BEN-896.

    tha.AddBatch(GenerateCamTraceBatch(blMean));
    EXPECT_EQ(chunkSize, tha.FramesAdded());
    EXPECT_EQ(0, tha.HistogramFrameCount());


//    const auto& h = tha.Histogram();
    FAIL() << "Test under construction.";
}

}}}     // namespace PacBio::Mongo::Basecaller
