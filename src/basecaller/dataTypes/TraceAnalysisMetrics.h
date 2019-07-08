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
//  Defines class TraceAnalysisMetrics, which are held by BasecallingMetrics
//  and populated by HFMetricsFilter from other stages in the analysis pipeline

#include <array>
#include <math.h>
#include <numeric>
#include <vector>

#include <pacbio/smrtdata/Basecall.h>

#include <common/cuda/utility/CudaArray.h>
#include <common/cuda/memory/UnifiedCudaArray.h>

namespace PacBio {
namespace Mongo {
namespace Data {

template <uint32_t LaneWidth>
class TraceAnalysisMetrics
{
public:
    static constexpr unsigned int numAnalogs = 4;
    using UInt = uint32_t;
    using Int = int16_t;
    using Flt = float;
    using SingleIntegerMetric = Cuda::Utility::CudaArray<Int, LaneWidth>;
    using SingleUIntegerMetric = Cuda::Utility::CudaArray<UInt, LaneWidth>;
    using SingleFloatMetric = Cuda::Utility::CudaArray<Flt, LaneWidth>;
    using SingleBoolMetric = Cuda::Utility::CudaArray<bool, LaneWidth>;

public:
    /*
    // TODO: CudaArray initializers might not work this way...
    TraceAnalysisMetrics(size_t startFrame = 0, size_t numFrames = 0)
        : startFrame_(startFrame)
        , numFrames_(numFrames)
        , autocorr_ (std::numeric_limits<Flt>::quiet_NaN())
        , pulseDetectionScore_(std::numeric_limits<Flt>::lowest())
    { }
    */

    TraceAnalysisMetrics() = default;
    TraceAnalysisMetrics(const TraceAnalysisMetrics&) = default;
    TraceAnalysisMetrics& operator=(const TraceAnalysisMetrics&) = default;
    ~TraceAnalysisMetrics() = default;

public:
    // Start frame of trace metric block
    SingleUIntegerMetric StartFrame() const
    { return startFrame_; }

    // First frame after end of trace metric block
    SingleUIntegerMetric StopFrame() const
    {
        SingleUIntegerMetric tbr;
        for (size_t i = 0; i < LaneWidth; ++i)
            tbr = startFrame_[i] + numFrames_[i];
        return tbr;
        // Can we re-enable this if "vector" operators are implemented for
        // CudaArrays?
        //return startFrame_ + numFrames_;
    }

    // Set start frame of trace metric block
    TraceAnalysisMetrics& StartFrame(uint32_t startFrame)
    {
        startFrame_ = startFrame;
        return *this;
    }

    // Number of frames used to compute trace metrics.
    SingleUIntegerMetric NumFrames() const
    { return numFrames_; }

    // Set the number of frames used to compute trace metrics.
    TraceAnalysisMetrics& NumFrames(uint16_t numFrames)
    {
        numFrames_ = numFrames;
        return *this;
    }

    /// Autocorrelation of baseline-subtracted traces.
    SingleFloatMetric Autocorrelation() const
    { return autocorr_; }

    /// Mutable reference to autocorrelation of baseline-subtraced traces.
    SingleFloatMetric& Autocorrelation()
    { return autocorr_; }

    /// Sets the autocorrelation.
    /// \returns *this.
    TraceAnalysisMetrics& Autocorrelation(const SingleFloatMetric& value)
    {
        autocorr_ = value;
        return *this;
    }

/*
    // Set detection modes for baseline and analog channels
    TraceAnalysisMetrics& DetectionModes(const std::array<DetectionMode<float,NCam>, 5>& detModes)
    {
        baseline_  = detModes.front();
        std::copy(detModes.begin()+1, detModes.end(), analog_.begin());
        return *this;
    }

    // Detection mode properties for baseline (background) channels
    const DetectionMode& Baseline() const
    { return baseline_; }

    // Detection mode properties for analog channels [A, C, G, T]
    const std::array<DetectionMode<Flt,NCam>,4>& Analog() const
    { return analog_; }
*/

    /// The mean of the DWS baseline as computed by the pulse detection filter.
    const SingleFloatMetric FrameBaselineDWS() const
    {
        return frameBaselineDWS_;
    }

    /// The mean of the DWS baseline as computed by the pulse detection filter.
    SingleFloatMetric FrameBaselineDWS()
    {
        return frameBaselineDWS_;
    }

    /// The unbiased sample variance of the DWS baseline as computed by the pulse detection filter.
    const SingleFloatMetric FrameBaselineVarianceDWS() const
    {
        return frameBaselineVarianceDWS_;
    }

    /// The unbiased sample variance of the DWS baseline as computed by the pulse detection filter.
    SingleFloatMetric FrameBaselineVarianceDWS()
    {
        return frameBaselineVarianceDWS_;
    }

    /// The sample standard deviation of the DWS baseline as computed by the pulse detection filter.
    SingleFloatMetric FrameBaselineSigmaDWS() const
    {

        SingleFloatMetric tbr;
        for (size_t i = 0; i < LaneWidth; ++i)
            tbr[i] = sqrtf(FrameBaselineVarianceDWS()[i]);
        return tbr;
        // Can we implement some of these for CudaArrays?
        //return sqrtf(FrameBaselineVarianceDWS());
    }

    /// The number of baseline frames used by the pulse detection filter to
    /// compute DWS statistics, FrameBaselineDWS() and FrameBaselineVarianceDWS().
    const uint16_t NumFramesBaseline() const
    {
        return numFramesBaseline_;
    }

    /// The number of baseline frames used by the pulse detection filter to
    /// compute DWS statistics, FrameBaselineDWS() and FrameBaselineVarianceDWS().
    uint16_t NumFramesBaseline()
    {
        return numFramesBaseline_;
    }

    // Indicates whether complete analysis for estimation of the current
    // detection model was tried for this ZMW. This will be identical
    // for all ZMWs for a given lane and in the current implementation
    // all ZMWs for a given block.
    SingleBoolMetric FullEstimationAttempted() const
    { return fullEstimationAttempted_; }

    TraceAnalysisMetrics& FullEstimationAttempted(
            const SingleBoolMetric fullEstimationAttempted)
    {
        fullEstimationAttempted_ = fullEstimationAttempted;
        return *this;
    }

    /// Indicates specifically whether estimation of dye angles (a.k.a. spectra)
    /// was successful for this unit cell. In the current implementation this
    /// will be identical for all ZMWs for a given lane.
    SingleBoolMetric AngleEstimationSuceeded() const
    { return angleEstimationSucceeded_; }

    TraceAnalysisMetrics& AngleEstimationSucceeded(
            const SingleBoolMetric angleEstimationSucceeded)
    {
        angleEstimationSucceeded_ = angleEstimationSucceeded;
        return *this;
    }

    // Indicates whether the model was updated since the previous trace block
    // for this ZMW.
    SingleBoolMetric ModelUpdated() const
    { return modelUpdated_; }

    TraceAnalysisMetrics& ModelUpdated(const SingleBoolMetric modelUpdated)
    {
        modelUpdated_ = modelUpdated;
        return *this;
    }

    /// The heuristic confidence score from the estimation algorithm for this ZMW.
    /// 0 indicates failure; 1, success with flying colors.
    SingleFloatMetric ConfidenceScore() const
    { return confidenceScore_; }

    TraceAnalysisMetrics& ConfidenceScore(const SingleFloatMetric confidenceScore)
    {
        confidenceScore_ = confidenceScore;
        return *this;
    }

    /// Returns pulse detection score.
    SingleFloatMetric PulseDetectionScore() const
    { return pulseDetectionScore_; }

    // Sets pulse detection score.
    TraceAnalysisMetrics& PulseDetectionScore(const SingleFloatMetric& value)
    {
        pulseDetectionScore_ = value;
        return *this;
    }

private:
    SingleUIntegerMetric startFrame_;
    SingleUIntegerMetric numFrames_;
public:
    SingleIntegerMetric pixelChecksum;
private:
    // TODO These are going to have to come from somewhere:
    //DetectionMode<Flt,NCam> baseline_;
    //std::array<DetectionMode<Flt,NCam>,numAnalogs> analog_;

    SingleFloatMetric frameBaselineVarianceDWS_;
    SingleFloatMetric frameBaselineDWS_;
    SingleUIntegerMetric numFramesBaseline_;


    SingleFloatMetric autocorr_;  // Autocorrelation coefficient at configured lag.
    SingleFloatMetric confidenceScore_;
    SingleFloatMetric pulseDetectionScore_;
    SingleBoolMetric fullEstimationAttempted_;
    SingleBoolMetric modelUpdated_;
    SingleBoolMetric angleEstimationSucceeded_;
};



}}}     // namespace PacBio::Mongo::Data

#endif // mongo_dataTypes_TraceAnalysisMetrics_H_
