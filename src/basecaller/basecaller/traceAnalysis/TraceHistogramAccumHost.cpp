
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
//  Defines some members of class TraceHistogramAccumHost.

#include <algorithm>
#include "TraceHistogramAccumHost.h"

namespace PacBio {
namespace Mongo {
namespace Basecaller {

TraceHistogramAccumHost::TraceHistogramAccumHost(unsigned int poolId, unsigned int poolSize)
    : TraceHistogramAccumulator(poolId, poolSize)
{
    // TODO
}


void TraceHistogramAccumHost::AddBatchImpl(const Data::CameraTraceBatch& ctb)
{
    // TODO: Can some of this logic be lifted into the base class's AddBatch
    // method (template method pattern)?

    const auto numLanes = ctb.Dimensions().lanesPerBatch;

    if (FramesAdded() == ctb.Dimensions().framesPerBatch)
    {
        // This is the first trace batch.

        // Reset all the histograms.
        hist_.clear();
        hist_.reserve(ctb.Dimensions().lanesPerBatch);
        isHistInitialized_ = false;

        // Reset baseline stat accumulators.
        InitStats(numLanes);
    }

    const bool doInitHist = !isHistInitialized_
            && FramesAdded() >= NumFramesPreAccumStats();

    // For each lane/block in the batch ...
    for (unsigned int lane = 0; lane < numLanes; ++lane)
    {
        // Accumulate baseliner stats.
        stats_[lane].Merge(Data::BaselinerStatAccumulator<Data::RawTraceElement>(ctb.Stats(lane)));

        if (doInitHist)
        {
            // Define histogram parameters and construct empty histogram.
            InitHistogram(lane);
        }

        // TODO: Map trace data to LaneArray and feed to UHistogramSimd.
        AddBlock(ctb, lane);
    }
}


void TraceHistogramAccumHost::InitHistogram(unsigned int lane)
{
    assert (hist_.size() == lane);

    // Number of bins in the histogram is a compile-time constant!
    constexpr unsigned int numBins = LaneHistType::numBins;

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
    const auto lowerBound = blMean - 4.0f*blSigma;
    const auto upperBound = lowerBound + float(numBins)*binSize;

    // TODO: Should we do anything if upperBound < stats_[lane].TraceMax()?

    hist_.emplace_back(numBins, lowerBound, upperBound);
}


void TraceHistogramAccumHost::AddBlock(const Data::CameraTraceBatch& ctb,
                                       unsigned int lane)
{
    assert(lane < hist_.size());

    auto& h = hist_[lane];

    // Get view to the trace data of lane i.
    const auto traceBlock = ctb.GetBlockView(lane);

    // Iterate over lane-frames.
    for (auto lfi = traceBlock.CBegin(); lfi != traceBlock.CEnd(); ++lfi)
    {
        // TODO: Filter edge frames.
        // TODO: Would be nice to avoid copying to the temporary, x.
        // Note that there is a possible type conversion here.
        const LaneArray<HistDataType> x {*lfi};
        h.AddDatum(x);
    }
}


void TraceHistogramAccumHost::InitStats(unsigned int numLanes)
{
    stats_.clear();
    stats_.resize(numLanes);
}


TraceHistogramAccumHost::PoolHistType
TraceHistogramAccumHost::HistogramImpl() const
{
    using std::copy;

    assert(hist_.size() == PoolSize());
    PoolHistType ph (PoolId(), PoolSize());
    auto phv = ph.data.GetHostView();

    using CArray = Cuda::Utility::CudaArray<HistCountType, laneSize>;

    for (unsigned int lane = 0; lane < PoolSize(); ++lane)
    {
        LaneHistType& phvl = phv[lane];
        const auto& histl = hist_[lane];

        // TODO: Should define a conversion operator that converts from ConstLaneArrayRef to CudaArray.

        const auto& lb = histl.LowerBound();
        copy(lb.begin(), lb.end(), phvl.lowBound.begin());

        const auto& bs = histl.BinSize();
        copy(bs.begin(), bs.end(), phvl.binSize.begin());

        const auto& ocLow = histl.LowOutlierCount();
        copy(ocLow.begin(), ocLow.end(), phvl.outlierCountLow.begin());

        const auto& ocHigh = histl.HighOutlierCount();
        copy(ocHigh.begin(), ocHigh.end(), phvl.outlierCountHigh.begin());

        assert(histl.NumBins() == phvl.numBins);
        for (unsigned int bin = 0; bin < phvl.numBins; ++bin)
        {
            const auto& bc = histl.BinCount(bin);
            copy(bc.begin(), bc.end(), phvl.binCount[bin].begin());
        }
    }

    return ph;
}


TraceHistogramAccumHost::PoolTraceStatsType
TraceHistogramAccumHost::TraceStatsImpl() const
{
    PoolTraceStatsType pts (PoolSize(), Cuda::Memory::SyncDirection::Symmetric);
    auto ptsv = pts.GetHostView();
    for (unsigned int lane = 0; lane < PoolSize(); ++lane)
    {
        ptsv[lane] = stats_[lane].ToBaselineStats();
    }
    return pts;
}

}}}     // namespace PacBio::Mongo::Basecaller
