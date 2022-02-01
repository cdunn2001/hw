// Copyright (c) 2015, Pacific Biosciences of California, Inc.
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
/// \brief  tbd

#ifndef PacBio_SmrtData_TraceAnalysisMetrics_h
#define PacBio_SmrtData_TraceAnalysisMetrics_h

#include <cstdint>
#include <array>

#include "DetectionModeScalar.h"

namespace PacBio {
namespace Primary {

template <size_t NCam>
struct pad_traits
{
    using type = int32_t;
};

template <>
struct pad_traits<1>
{
    using type = int16_t;
};

template <typename Flt, size_t NCam>
class TraceAnalysisMetrics
{
public:
    static constexpr size_t NFilter = NCam;
    static constexpr unsigned int numAnalogs = 4;
    static constexpr size_t NAngles = (NCam == 2) ? 2 : 0;

public:
    TraceAnalysisMetrics(size_t startFrame = 0, size_t numFrames = 0)
        : startFrame_(startFrame)
        , numFrames_(numFrames)
        , autocorr_ (std::numeric_limits<Flt>::quiet_NaN())
        , pulseDetectionScore_(std::numeric_limits<Flt>::lowest())
    { }

	TraceAnalysisMetrics(const TraceAnalysisMetrics&) = default;
	TraceAnalysisMetrics& operator=(const TraceAnalysisMetrics&) = default;
    ~TraceAnalysisMetrics() = default;

public:
    // Start frame of trace metric block
    uint32_t StartFrame() const
    { return startFrame_; }

	// First frame after end of trace metric block
	uint32_t StopFrame() const
	{ return startFrame_ + numFrames_; }

    // Set start frame of trace metric block
    TraceAnalysisMetrics<Flt,NCam>& StartFrame(uint32_t startFrame)
    {
        startFrame_ = startFrame;
        return *this;
    }

    // Number of frames used to compute trace metrics.
    uint16_t NumFrames() const
    { return numFrames_; }

    // Set the number of frames used to compute trace metrics.
    TraceAnalysisMetrics<Flt,NCam>& NumFrames(uint16_t numFrames)
    {
        numFrames_ = numFrames;
        return *this;
    }

    /// Autocorrelation of baseline-subtracted traces.
    Flt Autocorrelation() const
    { return autocorr_; }

    /// Mutable reference to autocorrelation of baseline-subtraced traces.
    Flt& Autocorrelation()
    { return autocorr_; }

    /// Sets the autocorrelation.
    /// \returns *this.
    TraceAnalysisMetrics& Autocorrelation(const Flt& value)
    {
        autocorr_ = value;
        return *this;
    }

	// Set detection modes for baseline and analog channels
	TraceAnalysisMetrics<Flt,NCam>& DetectionModes(const std::array<DetectionMode<float,NCam>, 5>& detModes)
    {
        baseline_  = detModes.front();
        std::copy(detModes.begin()+1, detModes.end(), analog_.begin());
        return *this;
    }

    // Detection mode properties for baseline (background) channels
    const DetectionMode<Flt,NCam>& Baseline() const
    { return baseline_; }

    // Detection mode properties for analog channels [A, C, G, T]
    const std::array<DetectionMode<Flt,NCam>,4>& Analog() const
    { return analog_; }

    /// The mean of the DWS baseline as computed by the pulse detection filter.
    const std::array<Flt,NCam>& FrameBaselineDWS() const
    {
        return frameBaselineDWS_;
    }

    /// The mean of the DWS baseline as computed by the pulse detection filter.
    std::array<Flt,NCam>& FrameBaselineDWS()
    {
        return frameBaselineDWS_;
    }

    /// The unbiased sample variance of the DWS baseline as computed by the pulse detection filter.
    const std::array<Flt,NCam>& FrameBaselineVarianceDWS() const
    {
        return frameBaselineVarianceDWS_;
    }

    /// The unbiased sample variance of the DWS baseline as computed by the pulse detection filter.
    std::array<Flt,NCam>& FrameBaselineVarianceDWS()
    {
        return frameBaselineVarianceDWS_;
    }

    /// The sample standard deviation of the DWS baseline as computed by the pulse detection filter.
    std::array<Flt,NCam> FrameBaselineSigmaDWS() const
    {
        auto sigmaDws = FrameBaselineVarianceDWS();
        for (auto& v : sigmaDws) v = sqrtf(v);
        return sigmaDws;
    }

    /// The number of baseline frames used by the pulse detection filter to
    /// compute DWS statistics, FrameBaselineDWS() and FrameBaselineVarianceDWS().
    const std::array<uint16_t,NCam>& NumFramesBaseline() const
    {
        return numFramesBaseline_;
    }

    /// The number of baseline frames used by the pulse detection filter to
    /// compute DWS statistics, FrameBaselineDWS() and FrameBaselineVarianceDWS().
    std::array<uint16_t,NCam>& NumFramesBaseline()
    {
        return numFramesBaseline_;
    }

    // Indicates whether complete analysis for estimation of the current
    // detection model was tried for this ZMW. This will be identical
    // for all ZMWs for a given lane and in the current implementation
    // all ZMWs for a given block.
    bool FullEstimationAttempted() const
    { return fullEstimationAttempted_; }

    TraceAnalysisMetrics<Flt,NCam>& FullEstimationAttempted(const bool fullEstimationAttempted)
    {
        fullEstimationAttempted_ = fullEstimationAttempted;
        return *this;
    }

    /// Indicates specifically whether estimation of dye angles (a.k.a. spectra)
    /// was successful for this unit cell. In the current implementation this
    /// will be identical for all ZMWs for a given lane.
    bool AngleEstimationSuceeded() const
    { return angleEstimationSucceeded_; }

    TraceAnalysisMetrics<Flt,NCam>& AngleEstimationSucceeded(const bool angleEstimationSucceeded)
    {
        angleEstimationSucceeded_ = angleEstimationSucceeded;
        return *this;
    }

    // Indicates whether the model was updated since the previous trace block for this ZMW.
    bool ModelUpdated() const
    { return modelUpdated_; }

    TraceAnalysisMetrics<Flt,NCam>& ModelUpdated(const bool modelUpdated)
    {
        modelUpdated_ = modelUpdated;
        return *this;
    }

    /// The heuristic confidence score from the estimation algorithm for this ZMW.
    /// 0 indicates failure; 1, success with flying colors.
    Flt ConfidenceScore() const
    { return confidenceScore_; }

    TraceAnalysisMetrics<Flt,NCam>& ConfidenceScore(const float confidenceScore)
    {
        confidenceScore_ = confidenceScore;
        return *this;
    }

    // Set the angle spectrum for all spectral channels
    TraceAnalysisMetrics<Flt,NCam>& AngleSpectrum(const std::array<Flt,NAngles>& spectrum)
    {
        angleSpectrum_ = spectrum;
        return *this;
    }

    /// Returns angle spectrum by spectral channels
    const std::array<Flt,NAngles>& AngleSpectrum() const
    { return angleSpectrum_; }

    /// Returns angle spectrum by spectral channels
    std::array<Flt,NAngles>& AngleSpectrum()
    { return angleSpectrum_; }

    /// Returns pulse detection score.
    Flt PulseDetectionScore() const
    { return pulseDetectionScore_; }

    // Sets pulse detection score.
    TraceAnalysisMetrics<Flt,NCam>& PulseDetectionScore(const Flt& value)
    {
        pulseDetectionScore_ = value;
        return *this;
    }

private:
    uint32_t startFrame_;
    uint16_t numFrames_;
public:
    int16_t pixelChecksum;
private:
    DetectionMode<Flt,NCam> baseline_;
    std::array<DetectionMode<Flt,NCam>,numAnalogs> analog_;

    std::array<Flt,NCam> frameBaselineVarianceDWS_;
    std::array<Flt,NCam> frameBaselineDWS_;
    std::array<uint16_t,NCam> numFramesBaseline_;

    // NOTE: The below padding is necessary since the size of std::array<T,0> is
    // different on x86_64 vs MIC.
    typename pad_traits<NCam>::type pad;

    std::array<Flt,NAngles> angleSpectrum_;

    Flt autocorr_;  // Autocorrelation coefficient at configured lag.
    Flt confidenceScore_;
    Flt pulseDetectionScore_;
    bool fullEstimationAttempted_;
    bool modelUpdated_;
    bool angleEstimationSucceeded_;
};

// NOTE: The size checks are to confirm the amount of storage required for
// these objects with type Flt=float and NCam=2,1. These should be updated
// whenever more metrics are added.
//
// IMPORTANT: These sizes must agree across all architectures as this data
// is copied raw based on the size of the objects.
//
static_assert(sizeof(TraceAnalysisMetrics<float,2>) == 176, "sizeof(TraceAnalysisMetrics<float,2>) is 176 bytes");
static_assert(sizeof(TraceAnalysisMetrics<float,1>) == 100, "sizeof(TraceAnalysisMetrics<float,2>) is 100 bytes");

template <typename Flt, size_t NCam>
std::ostream& operator<<(std::ostream& s, const TraceAnalysisMetrics<Flt,NCam>&)
{
    s << "(TraceAnalysisMetrics:tbd)";
    return s;
}

}}

#endif // PacBio_SmrtData_BasecallingMetrics_H_

