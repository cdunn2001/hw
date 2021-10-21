
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
//  Defines some members of class TraceHistogramAccumHost.

#include "TraceHistogramAccumHost.h"

#include <algorithm>

#include <tbb/task_arena.h>
#include <tbb/parallel_for.h>

#include <common/StatAccumulator.h>
#include <dataTypes/configs/BasecallerTraceHistogramConfig.h>

namespace PacBio {
namespace Mongo {
namespace Basecaller {

// static data
float TraceHistogramAccumHost::binSizeCoeff_;
unsigned int TraceHistogramAccumHost::baselineStatMinFrameCount_;
float TraceHistogramAccumHost::fallBackBaselineSigma_;

void TraceHistogramAccumHost::Configure(const Data::BasecallerTraceHistogramConfig& traceConfig)
{
    binSizeCoeff_ = traceConfig.BinSizeCoeff;
    PBLOG_INFO << "TraceHistogramAccumulator: BinSizeCoeff = "
               << binSizeCoeff_ << '.';

    baselineStatMinFrameCount_ = traceConfig.BaselineStatMinFrameCount;
    PBLOG_INFO << "TraceHistogramAccumulator: BaselineStatMinFrameCount = "
               << baselineStatMinFrameCount_ << '.';

    fallBackBaselineSigma_ = traceConfig.FallBackBaselineSigma;
    PBLOG_INFO << "TraceHistogramAccumulator: FallBackBaselineSigma = "
               << fallBackBaselineSigma_ << '.';
}


TraceHistogramAccumHost::TraceHistogramAccumHost(unsigned int poolId,
                                                 unsigned int poolSize)
    : TraceHistogramAccumulator(poolId, poolSize)
{ }

void TraceHistogramAccumHost::ResetImpl(const Cuda::Memory::UnifiedCudaArray<LaneHistBounds>& bounds)
{
    constexpr unsigned int numBins = LaneHistType::numBins;
    using Arr = LaneArray<TraceHistogramAccumHost::HistDataType, laneSize>;
    hist_.clear();
    auto view = bounds.GetHostView();
    for (size_t i = 0; i < view.Size(); ++i)
    {
        hist_.emplace_back(numBins, Arr(view[i].lowerBounds), Arr(view[i].upperBounds));
    }
}

void TraceHistogramAccumHost::ResetImpl(const Data::BaselinerMetrics& metrics)
{
    constexpr unsigned int numBins = LaneHistType::numBins;
    hist_.clear();
    auto view = metrics.baselinerStats.GetHostView();
    for (size_t lane = 0; lane < view.Size(); ++lane)
    {
        // Determine histogram parameters.
        const auto& laneBlStats = StatAccumulator<LaneArray<float>>(view[lane].baselineStats);

        const auto& blCount = laneBlStats.Count();
        const auto& sufficientData = blCount >= BaselineStatMinFrameCount();
        const auto& blMean = Blend(sufficientData,
                                   laneBlStats.Mean(),
                                   LaneArray<float>(0.0f));
        const auto& blSigma = Blend(sufficientData,
                                    sqrt(laneBlStats.Variance()),
                                    LaneArray<float>(FallBackBaselineSigma()));

        const auto binSize = BinSizeCoeff() * blSigma;
        const auto lower = blMean - 4.0f*blSigma;
        const auto upper = lower + float(numBins)*binSize;
        hist_.emplace_back(numBins, lower, upper);
    }
}

void TraceHistogramAccumHost::AddBatchImpl(const Data::TraceBatch<TraceElementType>& traces,
                                           const PoolDetModel& /* detModel */)
{
    const auto numLanes = traces.LanesPerBatch();

    // TODO: Pass detection model along and use for edge-frame scrubbing.
    
    tbb::task_arena().execute([&] {
        // For each lane/block in the batch ...
        tbb::parallel_for((size_t) {0}, numLanes, [&](size_t lane) {
            AddBlock(traces, lane);
        });
    });
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
        h.AddDatum(LaneArray<float>(lfi.Extract()));
    }
}

TraceHistogramAccumHost::PoolHistType
TraceHistogramAccumHost::HistogramImpl() const
{
    PoolHistType poolHist(PoolId(),
                          PoolSize(),
                          Cuda::Memory::SyncDirection::HostWriteDeviceRead);
    assert(hist_.size() == PoolSize());
    auto phv = poolHist.data.GetHostView();

    for (unsigned int lane = 0; lane < PoolSize(); ++lane)
    {
        LaneHistType& phvl = phv[lane];
        const auto& histl = hist_[lane];

        phvl.lowBound = histl.LowerBound();
        phvl.binSize = histl.BinSize();
        phvl.outlierCountLow = histl.LowOutlierCount();
        phvl.outlierCountHigh = histl.HighOutlierCount();

        assert(histl.NumBins() == phvl.numBins);
        for (unsigned int bin = 0; bin < phvl.numBins; ++bin)
        {
            phvl.binCount[bin] = histl.BinCount(bin);
        }
    }

    return poolHist;
}

}}}     // namespace PacBio::Mongo::Basecaller
