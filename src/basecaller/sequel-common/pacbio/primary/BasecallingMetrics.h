#ifndef PacBio_SmrtData_BasecallingMetrics_H_
#define PacBio_SmrtData_BasecallingMetrics_H_

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
/// \file	BasecallingMetrics.h
/// \brief	The class BasecallingMetrics is used to report per-ZMW metrics
///         that summarize the analysis of a block of unit-cell trace data.

#include <cstdint>
#include <array>
#include <deque>
#include <pacbio/primary/BlockActivityLabels.h>
#include <pacbio/primary/RTMetricsConfig.h>
#include <pacbio/smrtdata/Basecall.h>

#include "TraceAnalysisMetrics.h"
#include "PulseLookback.h"

namespace PacBio {
namespace Primary {

template <typename Flt, size_t NCam>
class BasecallingMetrics
{
public:
    static constexpr size_t NFilter = NCam;
    static constexpr unsigned int numAnalogs = 4;

public:     // Structors
    BasecallingMetrics(size_t startFrame = 0, size_t numFrames = 0)
      : traceMetrics_(startFrame, numFrames)
      , numPulseFrames_(0)
      , numPulses_(0)
      , numBaseFrames_(0)
      , numBases_(0)
      , numSandwiches_(0)
      , numHalfSandwiches_(0)
      , numPulseLabelStutters_(0)
      , activityLabel_(ActivityLabeler::HQRFPhysicalState::EMPTY)
      , pkmidSig_ {{ 0.0f, 0.0f, 0.0f, 0.0f }}
      , bpzvar_ {{ 0.0f, 0.0f, 0.0f, 0.0f }}
      , pkzvar_ {{ 0.0f, 0.0f, 0.0f, 0.0f }}
      , pkmidN_ {{ 0, 0, 0, 0 }}
      , pkMax_ {{ 0, 0, 0, 0 }}
      , numBasesByAnalog_ {{ 0, 0, 0, 0 }}
      , numPkmidBasesByAnalog_ {{ 0, 0, 0, 0 }}
      , numPulsesByAnalog_ {{ 0, 0, 0, 0 }}
    { }

    BasecallingMetrics(const BasecallingMetrics&) = default;
    ~BasecallingMetrics() = default;

public:     // Assignment
    BasecallingMetrics& operator=(const BasecallingMetrics&) = default;

public:     // Mutating

	/// Add metrics from base.  Need to know previous two pulses to compute
    /// some of the metrics.
	/// \returns Reference to \code *this.
    BasecallingMetrics& AddBasecall(const SmrtData::Basecall& base,
                                    const PulseLookback<2>& twoPulseLookback,
                                    uint32_t sandwichTolerance);

	/// Finalize variance calculation.
    void FinalizeVariance();

public:

    /// Mean pkmid signal per frame excluding edge frames by channel [A,C,G,T]
    std::array<Flt, numAnalogs> PkmidMean() const;

    /// Mean pulse signal to noise ratio by channel [A,C,G,T] using frame baseline sigma
    std::array<Flt, numAnalogs>  FrameSnr() const;

public:     // Property Accessors/Settors

    /// Trace analysis metrics
    const TraceAnalysisMetrics<Flt,NFilter>& TraceMetrics() const
    { return traceMetrics_; }

    TraceAnalysisMetrics<Flt,NFilter>& TraceMetrics()
    { return traceMetrics_; }

    /// Number of in-pulse labeled frames.
    uint16_t NumPulseFrames() const
    { return numPulseFrames_; }

    /// Number of detected pulses.
    uint16_t NumPulses() const
    { return numPulses_; }

    /// Number of pulse sandwiches.
    uint16_t NumSandwiches() const
    { return numSandwiches_; }

    /// Number of pulse half-sandwiches
    uint16_t NumHalfSandwiches() const
    { return numHalfSandwiches_; }

    /// Number of pulse-label stutters.
    uint16_t NumPulseLabelStutters() const
    { return numPulseLabelStutters_; }

    /// Number of in-pulse frames for pulses called as bases.
    uint16_t NumBaseFrames() const
    { return numBaseFrames_; }

    /// Number of called bases.
    uint16_t NumBases() const
    { return numBases_; }

    /// Total DWS signal from intra-pulse frames of pulses that
    /// are called as bases, units (photo e-), by analog channel
    /// [A, C, G, T].
    const std::array<Flt, numAnalogs>& PkmidSignal() const
    { return pkmidSig_; }

    /// Estimate of the inter-pulse frame DWS signal
    /// "sum of squares of differences from the mean",
    /// units (photo e-)^2, by analog channel [A, C, G, T].
    /// For computing inter-pulse variance; see for example
    /// https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    const std::array<Flt, numAnalogs>& Bpzvar() const
    { return bpzvar_; }

    /// Estimate of the intra-pulse frame DWS signal
    /// "sum of squares of differences from the mean",
    /// units (photo e-)^2, by analog channel [A, C, G, T].
    /// For computing intra-pulse variance; see for example
    /// https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    const std::array<Flt, numAnalogs>& Pkzvar() const
    { return pkzvar_; }

    /// Number of frames contributing to pkmid DWS quantities
    /// by analog channel, [A, C, G, T].
    const std::array<uint16_t, numAnalogs>& PkmidNumFrames() const
    { return pkmidN_; }

    /// Average base width in frames.
    Flt BaseWidth() const
    {
        return (numBases_ != 0) ? numBaseFrames_ / (float) numBases_ : 0;
    }

    /// Average pulse width in frames.
    Flt PulseWidth() const
    {
        return (numPulses_ != 0) ? numPulseFrames_ / (float) numPulses_ : 0;
    }

    /// Max signal over all pulses of each channel [A,C,G,T] in block.
    const std::array<Flt,numAnalogs> PkMax() const
    { return pkMax_; }

    /// Number of pmkid bases used to correct intra-pulse frame DWS
    /// signal variance.
    const std::array<uint16_t, numAnalogs>& NumPkmidBasesByAnalog() const
    { return numPkmidBasesByAnalog_; }

    /// Returns number of bases in each analog
    /// [A, C, G, T]
    const std::array<uint16_t, numAnalogs>& NumBasesByAnalog() const
    { return numBasesByAnalog_; }

    /// Number of pulses by analog channel, [A, C, G, T]
    const std::array<uint16_t, numAnalogs>& NumPulsesByAnalog() const
    { return numPulsesByAnalog_; }

    /// Estimated activity label
    ActivityLabeler::HQRFPhysicalState ActivityLabel() const
    { return activityLabel_; }

    /// Set the number of detected pulses.
    /// \returns Reference to \code *this.
    BasecallingMetrics& NumPulses(uint16_t val)
    {
        numPulses_ = val;
        return *this;
    }

    /// Set the number of called bases.
    /// \returns Reference to \code *this.
    BasecallingMetrics& NumBases(uint16_t val)
    {
        numBases_ = val;
        return *this;
    }

    /// Set the number of in-pulse frames for pulses called as bases.
    /// \returns Reference to \code *this.
    BasecallingMetrics& NumBaseFrames(uint16_t val)
    {
        numBaseFrames_ = val;
        return *this;
    }

    /// Set the number of in-pulse labeled frames.
    /// \returns Reference to \code *this.
    BasecallingMetrics& NumPulseFrames(uint16_t val)
    {
        numPulseFrames_ = val;
        return *this;
    }

    /// Set the number of sandwiches.
    /// \returns Reference to \code *this.
    BasecallingMetrics& NumSandwiches(uint16_t val)
    {
        numSandwiches_ = val;
        return *this;
    }

    /// Sets the number of half sandwiches.
    /// \returns Reference to \code *this.
    BasecallingMetrics& NumHalfSandwiches(uint16_t val)
    {
        numHalfSandwiches_ = val;
        return *this;
    }

    /// Sets number of pulse label stutters.
    /// \returns Reference to \code *this.
    BasecallingMetrics& NumPulseLabelStutters(uint16_t val)
    {
        numPulseLabelStutters_ = val;
        return *this;
    }

    /// Set the estimated activity label
    /// \returns Reference to \code *this.
    BasecallingMetrics& ActivityLabel(ActivityLabeler::HQRFPhysicalState val)
    {
        activityLabel_ = val;
        return *this;
    }

    /// Returns total DWS signal from intra-pulse frames of pulses
    /// by analog channel [A, C, G, T]
    std::array<Flt, numAnalogs>& PkmidSignal()
    { return pkmidSig_; }

    /// Returns inter-pulse frame DWS signal variance by
    /// analog channel [A, C, G, T]
    std::array<Flt, numAnalogs>& Bpzvar()
    { return bpzvar_; }

    /// Returns intra-pulse frame DWS signal variance by
    /// analog channel [A, C, G, T]
    std::array<Flt, numAnalogs>& Pkzvar()
    { return pkzvar_; }

    /// Returns number of frames contributing to pkmid DWS quantities
    /// by analog channel, [A, C, G, T].
    std::array<uint16_t, numAnalogs>& PkmidNumFrames()
    { return pkmidN_; }

    /// Returns max signal over all pulses of each channel [A,C,G,T].
    std::array<Flt, numAnalogs>& PkMax()
    { return pkMax_; };

    /// Returns number of bases used in pkmid calculations for
    /// correcting intra-pulse variance by analog channel
    /// [A, C, G, T]
    std::array<uint16_t, numAnalogs>& NumPkmidBasesByAnalog()
    { return numPkmidBasesByAnalog_; }

    /// Returns number of bases in each analog
    /// [A, C, G, T]
    std::array<uint16_t, numAnalogs>& NumBasesByAnalog()
    { return numBasesByAnalog_; }

    /// Returns number of pulses by analog channel [A, C, G, T]
    std::array<uint16_t, numAnalogs>& NumPulsesByAnalog()
    { return numPulsesByAnalog_; }



private:
    TraceAnalysisMetrics<Flt, NFilter>  traceMetrics_;
    uint16_t numPulseFrames_;
    uint16_t numPulses_;
    uint16_t numBaseFrames_;
    uint16_t numBases_;
    uint16_t numSandwiches_;
    uint16_t numHalfSandwiches_;
    uint16_t numPulseLabelStutters_;
    ActivityLabeler::HQRFPhysicalState activityLabel_;
    std::array<Flt, numAnalogs> pkmidSig_;
    std::array<Flt, numAnalogs> bpzvar_;
    std::array<Flt, numAnalogs> pkzvar_;
    std::array<uint16_t, numAnalogs> pkmidN_;
    std::array<Flt, numAnalogs> pkMax_;
    std::array<uint16_t, numAnalogs> numBasesByAnalog_;
    std::array<uint16_t, numAnalogs> numPkmidBasesByAnalog_;
    std::array<uint16_t, numAnalogs> numPulsesByAnalog_;
};

namespace BasecallingMetricsT {
    using Sequel = BasecallingMetrics<float, 2>;
    using Spider = BasecallingMetrics<float, 1>;
}

// NOTE: The size checks are to confirm the amount of storage required for
// these objects with type Flt=float and NCam=2,1. These should be updated
// whenever more metrics are added.
//
// IMPORTANT: These sizes must agree across all architectures as this data
// is copied raw based on the size of the objects.
//
static_assert(sizeof(BasecallingMetricsT::Sequel) == 288, "sizeof(BasecallingMetricsT::Sequel) is 288 bytes");
static_assert(sizeof(BasecallingMetricsT::Spider) == 212, "sizeof(BasecallingMetricsT::Spider) is 212 bytes");

BasecallingMetricsT::Sequel Convert1CamTo2Cam(const BasecallingMetricsT::Spider* buf);

}}


#endif // PacBio_SmrtData_BasecallingMetrics_H_
