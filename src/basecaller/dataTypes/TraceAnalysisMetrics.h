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

#include <array>
#include <math.h>
#include <numeric>
#include <vector>

#include <common/cuda/utility/CudaArray.h>
#include <common/cuda/memory/UnifiedCudaArray.h>

namespace PacBio {
namespace Mongo {
namespace Data {

// TODO: This could benefit from the StatAccumulator and Accumulator/State
// models demonstrated in BaselinerStatAccumulator
template <uint32_t LaneWidth>
class TraceAnalysisMetrics
{
public:
    static constexpr unsigned int numAnalogs = 4;
    using UnsignedInt = uint32_t;
    using Int = int16_t;
    using Flt = float;
    using SingleIntegerMetric = Cuda::Utility::CudaArray<Int, LaneWidth>;
    using SingleUnsignedIntegerMetric = Cuda::Utility::CudaArray<UnsignedInt, LaneWidth>;
    using SingleFloatMetric = Cuda::Utility::CudaArray<Flt, LaneWidth>;
    using SingleBoolMetric = Cuda::Utility::CudaArray<bool, LaneWidth>;

public:

    TraceAnalysisMetrics() = default;
    TraceAnalysisMetrics(const TraceAnalysisMetrics&) = default;
    TraceAnalysisMetrics& operator=(const TraceAnalysisMetrics&) = default;
    ~TraceAnalysisMetrics() = default;

    void Initialize(size_t startFrame = 0, size_t numFrames = 0)
    {
        for (size_t zmw = 0; zmw < LaneWidth; ++zmw)
        {
            startFrame_[zmw] = startFrame;
            numFrames_[zmw] = numFrames;
            autocorr_[zmw] = std::numeric_limits<Flt>::quiet_NaN();
            pulseDetectionScore_[zmw] = std::numeric_limits<Flt>::quiet_NaN();

            frameBaselineM0DWS_[zmw] = 0;
            frameBaselineM1DWS_[zmw] = 0;
            frameBaselineM2DWS_[zmw] = 0;
            autocorr_[zmw] = 0;
            confidenceScore_[zmw] = 0;
            pulseDetectionScore_[zmw] = 0;
            fullEstimationAttempted_[zmw] = 0;
            modelUpdated_[zmw] = 0;
        }
    }


public:
    // Start frame of trace metric block
    const SingleUnsignedIntegerMetric& StartFrame() const
    { return startFrame_; }

    SingleUnsignedIntegerMetric& StartFrame()
    { return startFrame_; }

    // First frame after end of trace metric block
    SingleUnsignedIntegerMetric StopFrame() const
    {
        SingleUnsignedIntegerMetric ret;
        for (size_t i = 0; i < LaneWidth; ++i)
            ret = startFrame_[i] + numFrames_[i];
        return ret;
    }

    // Number of frames used to compute trace metrics.
    // These don't need to be stored for each ZMW in a lane...
    const SingleUnsignedIntegerMetric& NumFrames() const
    { return numFrames_; }

    SingleUnsignedIntegerMetric& NumFrames()
    { return numFrames_; }

    /// Autocorrelation of baseline-subtracted traces.
    const SingleFloatMetric& Autocorrelation() const
    { return autocorr_; }

    SingleFloatMetric& Autocorrelation()
    { return autocorr_; }

    /// The mean of the DWS baseline as computed by the pulse detection filter.
    SingleFloatMetric FrameBaselineDWS() const
    {
        SingleFloatMetric ret;
        for (size_t zmw = 0; zmw < LaneWidth; ++zmw)
            ret[zmw] = frameBaselineM1DWS_[zmw] / frameBaselineM0DWS_[zmw];
        return ret;
    }

    /// The unbiased sample variance of the DWS baseline as computed by the pulse detection filter.
    SingleFloatMetric FrameBaselineVarianceDWS() const
    {
        SingleFloatMetric ret;
        for (size_t zmw = 0; zmw < LaneWidth; ++zmw)
        {
            //std::cout << frameBaselineM0DWS_[zmw] << " " << frameBaselineM1DWS_[zmw] << " " << frameBaselineM2DWS_[zmw] << std::endl;
            ret[zmw] = frameBaselineM1DWS_[zmw]
                       * frameBaselineM1DWS_[zmw]
                       / frameBaselineM0DWS_[zmw];
            ret[zmw] = (frameBaselineM2DWS_[zmw] - ret[zmw])
                       / (frameBaselineM0DWS_[zmw] - 1.0f);
            ret[zmw] = std::max(ret[zmw], 0.0f);
            ret[zmw] = frameBaselineM0DWS_[zmw] > 1.0f
                       ? ret[zmw] : std::numeric_limits<Flt>::quiet_NaN();
        }
        return ret;
    }

    const SingleFloatMetric& FrameBaselineM0DWS() const
    { return frameBaselineM0DWS_; }

    SingleFloatMetric& FrameBaselineM0DWS()
    { return frameBaselineM0DWS_; }

    const SingleFloatMetric& FrameBaselineM1DWS() const
    { return frameBaselineM1DWS_; }

    SingleFloatMetric& FrameBaselineM1DWS()
    { return frameBaselineM1DWS_; }

    const SingleFloatMetric& FrameBaselineM2DWS() const
    { return frameBaselineM2DWS_; }

    SingleFloatMetric& FrameBaselineM2DWS()
    { return frameBaselineM2DWS_; }

    /// The sample standard deviation of the DWS baseline as computed by the pulse detection filter.
    SingleFloatMetric FrameBaselineSigmaDWS() const
    {

        SingleFloatMetric ret;
        for (size_t i = 0; i < LaneWidth; ++i)
            ret[i] = sqrtf(FrameBaselineVarianceDWS()[i]);
        return ret;
    }

    /// The number of baseline frames used by the pulse detection filter to
    /// compute DWS statistics, FrameBaselineDWS() and FrameBaselineVarianceDWS().
    const SingleUnsignedIntegerMetric& NumFramesBaseline() const
    { return frameBaselineM0DWS_; }

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

    /// Returns pulse detection score.
    const SingleFloatMetric& PulseDetectionScore() const
    { return pulseDetectionScore_; }

    SingleFloatMetric& PulseDetectionScore()
    { return pulseDetectionScore_; }

private:
    SingleUnsignedIntegerMetric startFrame_;
    SingleUnsignedIntegerMetric numFrames_;

public:
    SingleIntegerMetric pixelChecksum;

private:
    SingleFloatMetric frameBaselineM0DWS_;
    SingleFloatMetric frameBaselineM1DWS_;
    SingleFloatMetric frameBaselineM2DWS_;

    // None of these are used yet, perhaps some will never be used:
    SingleFloatMetric autocorr_;
    SingleFloatMetric confidenceScore_;
    SingleFloatMetric pulseDetectionScore_;
    SingleBoolMetric fullEstimationAttempted_;
    SingleBoolMetric modelUpdated_;
};

}}}     // namespace PacBio::Mongo::Data

#endif // mongo_dataTypes_TraceAnalysisMetrics_H_
