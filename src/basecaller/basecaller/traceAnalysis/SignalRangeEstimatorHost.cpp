// Copyright (c) 2020 Pacific Biosciences of California, Inc.
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
//  Defines some members of class SignalRangeEstimatorHost.

#include "SignalRangeEstimatorHost.h"

#include <algorithm>

#include <tbb/parallel_for.h>

namespace PacBio {
namespace Mongo {
namespace Basecaller {

void SignalRangeEstimatorHost::AddMetricsImpl(const Data::BaselinerMetrics& metrics)
{
    assert(stats_.size() == metrics.baselinerStats.Size());

    const auto& statsView = metrics.baselinerStats.GetHostView();
    tbb::parallel_for((size_t){0}, stats_.size(), [&](size_t lane)
    {
        // Accumulate baseliner stats.
        stats_[lane].Merge(Data::BaselinerStatAccumulator<Data::RawTraceElement>(statsView[lane]));
    });
}

const Data::BaselinerMetrics& SignalRangeEstimatorHost::TraceStatsImpl() const
{
    auto ptsv = poolTraceStats_.baselinerStats.GetHostView();
    for (unsigned int lane = 0; lane < PoolSize(); ++lane)
    {
        ptsv[lane] = stats_[lane].GetState();
    }
    return poolTraceStats_;
}

Cuda::Memory::UnifiedCudaArray<LaneHistBounds> SignalRangeEstimatorHost::EstimateRangeAndResetImpl()
{
    // Number of bins in the histogram is a compile-time constant!
    using LaneHistType = Data::LaneHistogram<float, uint16_t>;
    constexpr unsigned int numBins = LaneHistType::numBins;

    Cuda::Memory::UnifiedCudaArray<LaneHistBounds> ret(stats_.size(),
                                                       Cuda::Memory::SyncDirection::HostWriteDeviceRead,
                                                       SOURCE_MARKER());
    auto view = ret.GetHostView();
    for (size_t lane = 0; lane < view.Size(); ++lane)
    {
        // Determine histogram parameters.
        const auto& laneBlStats = stats_[lane].BaselineFramesStats();

        const auto& blCount = laneBlStats.Count();
        const auto& sufficientData = blCount >= BaselineStatMinFrameCount();
        const auto& blMean = Blend(sufficientData,
                                   laneBlStats.Mean(),
                                   LaneArray<float>(0.0f));
        const auto& blSigma = Blend(sufficientData,
                                    sqrt(laneBlStats.Variance()),
                                    LaneArray<float>(FallBackBaselineSigma()));

        const auto binSize = BinSizeCoeff() * blSigma;
        auto lower = blMean - 4.0f*blSigma;
        view[lane].lowerBounds = lower;
        view[lane].upperBounds = lower + float(numBins)*binSize;

    }

    // Reset for the next aggregation
    stats_.clear();
    stats_.resize(PoolSize());

    return ret;
}

}}}     // namespace PacBio::Mongo::Basecaller
