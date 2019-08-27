
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

TraceHistogramAccumHost::TraceHistogramAccumHost(unsigned int poolId,
                                                 unsigned int poolSize,
                                                 bool pinnedAlloc)
    : TraceHistogramAccumulator(poolId, poolSize)
    , poolHist_ (poolId, poolSize, pinnedAlloc)
    , poolTraceStats_ (poolSize, Cuda::Memory::SyncDirection::Symmetric, pinnedAlloc)
{ }


void TraceHistogramAccumHost::AddBatchImpl(
        const Data::TraceBatch<TraceElementType>& traces,
        const Cuda::Memory::UnifiedCudaArray<Data::BaselinerStatAccumState>& stats)
{
    // TODO: Can some of this logic be lifted into the base class's AddBatch
    // method (template method pattern)?

    const auto numLanes = traces.LanesPerBatch();

    if (FramesAdded() == 0)
    {
        // This is the first trace batch.

        // Reset all the histograms.
        hist_.clear();
        hist_.reserve(traces.LanesPerBatch());
        isHistInitialized_ = false;
        histFrameCount_ = 0;

        // Reset baseline stat accumulators.
        InitStats(numLanes);
    }

    // Have we accumulated enough data for baseline statistics, which are
    // needed to initialize the histograms?
    const bool doInitHist = !isHistInitialized_
            && FramesAdded() + traces.NumFrames() >= NumFramesPreAccumStats();

    // For each lane/block in the batch ...
    const auto& statsView = stats.GetHostView();
    for (unsigned int lane = 0; lane < numLanes; ++lane)
    {
        // Accumulate baseliner stats.
        stats_[lane].Merge(Data::BaselinerStatAccumulator<Data::RawTraceElement>(statsView[lane]));

        if (doInitHist)
        {
            // Define histogram parameters and construct empty histogram.
            InitHistogram(lane);
            isHistInitialized_ = true;
        }

        if (isHistInitialized_) AddBlock(traces, lane);
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


void TraceHistogramAccumHost::AddBlock(const Data::TraceBatch<TraceElementType>& traces,
                                       unsigned int lane)
{
    assert(lane < hist_.size());

    auto& h = hist_[lane];

    // Get view to the trace data of lane i.
    const auto traceBlock = traces.GetBlockView(lane);

    // Iterate over lane-frames.
    for (auto lfi = traceBlock.CBegin(); lfi != traceBlock.CEnd(); ++lfi)
    {
        // TODO: Filter edge frames.
        // TODO: Would be nice to avoid copying to the temporary, x.
        // Note that there is a possible elemental type conversion here.
        const LaneArray<HistDataType> x {*lfi};
        h.AddDatum(x);
        ++histFrameCount_;
    }
}


void TraceHistogramAccumHost::InitStats(unsigned int numLanes)
{
    stats_.clear();
    stats_.resize(numLanes);
}


const TraceHistogramAccumHost::PoolHistType&
TraceHistogramAccumHost::HistogramImpl() const
{
    using std::copy;

    assert(hist_.size() == PoolSize());
    auto phv = poolHist_.data.GetHostView();

    for (unsigned int lane = 0; lane < PoolSize(); ++lane)
    {
        LaneHistType& phvl = phv[lane];
        const auto& histl = hist_[lane];

        const auto& lb = histl.LowerBound();
        assert(lb.Size() == phvl.lowBound.size());
        copy(lb.begin(), lb.end(), phvl.lowBound.begin());
        // TODO: Is this alternative std::copy any better?
        // LaneArrayRef<HistDataType>(phvl.lowBound.data()) = lb;

        const auto& bs = histl.BinSize();
        assert(bs.Size() == phvl.binSize.size());
        copy(bs.begin(), bs.end(), phvl.binSize.begin());

        const auto& ocLow = histl.LowOutlierCount();
        assert(ocLow.Size() == phvl.outlierCountLow.size());
        copy(ocLow.begin(), ocLow.end(), phvl.outlierCountLow.begin());

        const auto& ocHigh = histl.HighOutlierCount();
        assert(ocHigh.Size() == phvl.outlierCountHigh.size());
        copy(ocHigh.begin(), ocHigh.end(), phvl.outlierCountHigh.begin());

        assert(histl.NumBins() == phvl.numBins);
        for (unsigned int bin = 0; bin < phvl.numBins; ++bin)
        {
            const auto& bc = histl.BinCount(bin);
            assert(bc.Size() == phvl.binCount[bin].size());
            copy(bc.begin(), bc.end(), phvl.binCount[bin].begin());
        }
    }

    return poolHist_;
}


const TraceHistogramAccumulator::PoolTraceStatsType&
TraceHistogramAccumHost::TraceStatsImpl() const
{
    auto ptsv = poolTraceStats_.GetHostView();
    for (unsigned int lane = 0; lane < PoolSize(); ++lane)
    {
        ptsv[lane] = stats_[lane].GetState();
    }
    return poolTraceStats_;
}

}}}     // namespace PacBio::Mongo::Basecaller
