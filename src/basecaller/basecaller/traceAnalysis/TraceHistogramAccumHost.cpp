
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

#include <common/LaneArray.h>
#include <common/StatAccumulator.h>
#include <dataTypes/configs/AnalysisConfig.h>
#include <dataTypes/configs/BasecallerTraceHistogramConfig.h>

namespace PacBio {
namespace Mongo {
namespace Basecaller {

// static data
float TraceHistogramAccumHost::binSizeCoeff_;
unsigned int TraceHistogramAccumHost::baselineStatMinFrameCount_;
float TraceHistogramAccumHost::fallBackBaselineSigma_;
float TraceHistogramAccumHost::movieScaler_ = 1.0f;
float TraceHistogramAccumHost::binSizeLowBoundCoeff_ = 1.0f;

void TraceHistogramAccumHost::Configure(const Data::BasecallerTraceHistogramConfig& histConfig,
                                        const Data::AnalysisConfig& analysisConfig)
{
    movieScaler_ = analysisConfig.movieInfo.photoelectronSensitivity;
    binSizeLowBoundCoeff_ = histConfig.BinSizeLowBoundCoeff;

    binSizeCoeff_ = histConfig.BinSizeCoeff;
    PBLOG_INFO << "TraceHistogramAccumulator: BinSizeCoeff = "
               << binSizeCoeff_ << '.';

    baselineStatMinFrameCount_ = histConfig.BaselineStatMinFrameCount;
    PBLOG_INFO << "TraceHistogramAccumulator: BaselineStatMinFrameCount = "
               << baselineStatMinFrameCount_ << '.';

    fallBackBaselineSigma_ = histConfig.FallBackBaselineSigma;
    PBLOG_INFO << "TraceHistogramAccumulator: FallBackBaselineSigma = "
               << fallBackBaselineSigma_ << '.';
}


TraceHistogramAccumHost::TraceHistogramAccumHost(unsigned int poolId,
                                                 unsigned int poolSize)
    : TraceHistogramAccumulator(poolId, poolSize)
    , edgeClassifier_ (poolSize)
{ }

void TraceHistogramAccumHost::ResetImpl(const Cuda::Memory::UnifiedCudaArray<LaneHistBounds>& bounds)
{
    constexpr unsigned int numBins = LaneHistType::numBins;
    using Arr = LaneArray<TraceHistogramAccumHost::HistDataType, laneSize>;
    hist_.clear();
    auto view = bounds.GetHostView();
    assert(view.Size() == edgeClassifier_.size());
    for (size_t i = 0; i < view.Size(); ++i)
    {
        hist_.emplace_back(numBins, Arr(view[i].lowerBounds), Arr(view[i].upperBounds));
    }
}

void TraceHistogramAccumHost::ResetImpl(const Data::BaselinerMetrics& metrics)
{
    constexpr uint32_t numBins = LaneHistType::numBins;
    hist_.clear();
    auto view = metrics.baselinerStats.GetHostView();
    assert(view.Size() == edgeClassifier_.size());

    const float binSizeMin = binSizeLowBoundCoeff_ * std::max(1.0f, movieScaler_);

    for (size_t lane = 0; lane < view.Size(); ++lane)
    {
        // Determine histogram parameters
        const StatAccumulator<FloatVec>& laneBlStats = view[lane].baselineStats;

        const auto& sufficientData =
                    laneBlStats.Count() >= BaselineStatMinFrameCount();
        const FloatVec& blMean = Blend(sufficientData,
                                   laneBlStats.Mean(),
                                   FloatVec(0.0f));
        const FloatVec& blSigma = Blend(sufficientData,
                                    sqrt(laneBlStats.Variance()),
                                    FloatVec(FallBackBaselineSigma()));

        // TODO: Baseline estimation can be confused by trace data with very
        // sparse sampling of true baseline.  In such cases, estimates of the
        // mean and sigma of the baseline distribution are typically drastically
        // too high.

        const FloatVec binSize = max(binSizeCoeff_ * blSigma, binSizeMin);
        const auto loBound = blMean - 4.0f*blSigma;
        const auto upBound = loBound + float(numBins)*binSize;

        hist_.emplace_back(numBins, loBound, upBound);
    }
}

void TraceHistogramAccumHost::AddBatchImpl(const Data::TraceBatch<TraceElementType>& traces,
                                           const PoolDetModel& detModel)
{
    const auto numLanes = traces.LanesPerBatch();
    tbb::task_arena().execute([&] {
        // For each lane/block in the batch ...
        tbb::parallel_for((size_t) {0}, numLanes, [&](size_t lane) {
            AddBlock(traces, detModel, lane);
        });
    });
}

void TraceHistogramAccumHost::AddBlock(const Data::TraceBatch<TraceElementType>& traces,
                                       const PoolDetModel& pdm,
                                       unsigned int lane)
{
    assert(lane < hist_.size());

    auto& h = hist_[lane];

    // Get views to the trace data and detection model of lane i.
    const auto traceBlock = traces.GetBlockView(lane);
    const DetModelHost detModel {pdm, lane};
    const auto& bgMode = detModel.BaselineMode();

    // The threshold below which the sample is most likely full-frame baseline.
    // TODO: This value should be configurable, depend on the SNR of
    // the dimmest pulse component of the detection model, or both.
    static constexpr float threshSigma = 2.0f;
    const FrameArray threshold {roundCastInt(threshSigma * sqrt(bgMode.SignalCovar()) + bgMode.SignalMean())};

    // The edge-frame classifier for the specified lane.
    auto& efc = edgeClassifier_.at(lane);

    // Iterate over lane-frames.
    for (auto lfi = traceBlock.CBegin(); lfi != traceBlock.CEnd(); ++lfi)
    {
        FrameArray frame = lfi.Extract();
        const auto [isEdge, candidateFrame] = efc.IsEdgeFrame(threshold, frame);
        h.AddDatum(LaneArray<float>(candidateFrame), !isEdge);
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
