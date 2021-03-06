#ifndef mongo_dataTypes_TraceAnalysisMetrics_H_
#define mongo_dataTypes_TraceAnalysisMetrics_H_

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
//  Defines class TraceAnalysisMetrics, which are held by
//  BasecallingMetricAccumulators and populated by HFMetricsFilter from other
//  stages in the analysis pipeline

#include <math.h>
#include <numeric>

#include <common/cuda/memory/UnifiedCudaArray.h>
#include <common/LaneArray.h>

#include "BaselinerStatAccumulator.h"
#include "BasicTypes.h"

namespace PacBio {
namespace Mongo {
namespace Data {

class TraceAnalysisMetrics
{
public:
    template <typename T>
    using SingleMetric = LaneArray<T>;
    template <typename T>
    using AnalogMetric = std::array<LaneArray<T>, numAnalogs>;

public:

    TraceAnalysisMetrics()
        : startFrame_(0)
        , numFrames_(0)
        , pixelChecksum_(0)
        , pulseDetectionScore_(0)
        , baselineStatAccum_()
        , autocorrAccum_()
    {};

    TraceAnalysisMetrics(const TraceAnalysisMetrics&) = delete;
    TraceAnalysisMetrics& operator=(const TraceAnalysisMetrics&) = delete;
    ~TraceAnalysisMetrics() = default;

    // Resets most fields to 0, except startFrame which is set to the start of
    // the next block (assumed to be the previous startFrame + numFrames)
    void Reset()
    {
        baselineStatAccum_.Reset();
        autocorrAccum_.Reset();
        startFrame_ = startFrame_ + numFrames_;
        numFrames_ = 0;
        pulseDetectionScore_ = std::numeric_limits<float>::quiet_NaN();
        pulseDetectionScore_ = 0;
        // These aren't currently used:
        //confidenceScore_ = 0;
        //fullEstimationAttempted_ = 0;
        //modelUpdated_ = 0;
    }


public:
    // Start frame of trace metric block
    const SingleMetric<uint32_t>& StartFrame() const
    { return startFrame_; }

    SingleMetric<uint32_t>& StartFrame()
    { return startFrame_; }

    // First frame after end of trace metric block
    SingleMetric<uint32_t> StopFrame() const
    { return startFrame_ + numFrames_; }

    // Number of frames used to compute trace metrics.
    // These don't need to be stored for each ZMW in a lane...
    const SingleMetric<uint32_t>& NumFrames() const
    { return numFrames_; }

    SingleMetric<uint32_t>& NumFrames()
    { return numFrames_; }

    /// Autocorrelation of baseline-subtracted traces.
    SingleMetric<float> Autocorrelation() const
    { return autocorrAccum_.Autocorrelation(); }

    /// The mean of the DWS baseline as computed by the pulse detection filter.
    SingleMetric<float> FrameBaselineDWS() const
    { return baselineStatAccum_.Mean(); }

    /// The unbiased sample variance of the DWS baseline as computed by the pulse detection filter.
    SingleMetric<float> FrameBaselineVarianceDWS() const
    { return baselineStatAccum_.Variance(); }

    /// The sample standard deviation of the DWS baseline as computed by the pulse detection filter.
    SingleMetric<float> FrameBaselineSigmaDWS() const
    { return sqrt(FrameBaselineVarianceDWS()); }

    /// The number of baseline frames used by the pulse detection filter to
    /// compute DWS statistics, FrameBaselineDWS() and FrameBaselineVarianceDWS().
    const SingleMetric<uint16_t> NumFramesBaseline() const
    { return SingleMetric<uint16_t>(baselineStatAccum_.Count()); }

    /* TODO: to reactivate these, if needed, they can't be one-bit bool
     * LaneArrays
     *
    // Indicates whether complete analysis for estimation of the current
    // detection model was tried for this ZMW. This will be identical
    // for all ZMWs for a given lane and in the current implementation
    // all ZMWs for a given block.
    const SingleBoolMetric& FullEstimationAttempted() const
    { return fullEstimationAttempted_; }

    SingleBoolMetric& FullEstimationAttempted()
    { return fullEstimationAttempted_; }

    // Indicates whether the model was updated since the previous trace block
    // for this ZMW.
    const SingleBoolMetric& ModelUpdated() const
    { return modelUpdated_; }

    SingleBoolMetric& ModelUpdated()
    { return modelUpdated_; }

    /// The heuristic confidence score from the estimation algorithm for this ZMW.
    /// 0 indicates failure; 1, success with flying colors.
    const SingleFloatMetric& ConfidenceScore() const
    { return confidenceScore_; }

    SingleFloatMetric& ConfidenceScore()
    { return confidenceScore_; }
    */

    /// Returns pulse detection score.
    const SingleMetric<float>& PulseDetectionScore() const
    { return pulseDetectionScore_; }

    SingleMetric<float>& PulseDetectionScore()
    { return pulseDetectionScore_; }

    /// Returns pixel checksum
    const SingleMetric<int16_t>& PixelChecksum() const
    { return pixelChecksum_; }

    SingleMetric<int16_t>& PixelChecksum()
    { return pixelChecksum_; }

    /// Returns baseline stat accumulator
    const StatAccumulator<LaneArray<float>>& BaselinerStatAccum() const
    { return baselineStatAccum_; }

    StatAccumulator<LaneArray<float>>& BaselinerStatAccum()
    { return baselineStatAccum_; }

    /// Returns autocorrelation accumulator
    const AutocorrAccumulator<LaneArray<float>>& AutocorrAccum() const
    { return autocorrAccum_; }

     AutocorrAccumulator<LaneArray<float>>& AutocorrAccum()
    { return autocorrAccum_; }

private:
    SingleMetric<uint32_t> startFrame_;
    SingleMetric<uint32_t> numFrames_;
    SingleMetric<int16_t> pixelChecksum_;
    SingleMetric<float> pulseDetectionScore_;

    // We're not hanging on to the entire BaselineStatAccumulator because it
    // doesn't have a Reset or other niceties, and we don't need its extra
    // members
    StatAccumulator<LaneArray<float>> baselineStatAccum_;
    AutocorrAccumulator<LaneArray<float>> autocorrAccum_;

    // None of these are used yet, perhaps some will never be used:
    //SingleMetric<float> confidenceScore_;
    //SingleBoolMetric fullEstimationAttempted_;
    //SingleBoolMetric modelUpdated_;
};

}}}     // namespace PacBio::Mongo::Data

#endif // mongo_dataTypes_TraceAnalysisMetrics_H_
