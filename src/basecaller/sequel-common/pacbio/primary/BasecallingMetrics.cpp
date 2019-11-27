// Copyright (c) 2014, Pacific Biosciences of California, Inc.
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
// File Description:
/// \brief  tbd
//
// Programmer: John Nguyen

#include "BasecallingMetrics.h"

#include <pacbio/PBException.h>
#include <pacbio/logging/Logger.h>

namespace PacBio {
namespace Primary {

using SmrtData::Pulse;
using SmrtData::Basecall;
using SmrtData::NucleotideLabel;

namespace
{

/// Determines which sandwich "group" a pulse belongs to.  For the Sequel
/// pipeline analogs of the same color are grouped together.  For Spider each
/// analog is independent.  A Half sandwhich occurs when two "abutting" pulses
/// are of different groups, and a full sandwich occurs when three abutting
/// pulses switch from one group to another and then back.+
template<size_t NCam>
int SandwichGroup(const Pulse& pulse);

template<>
int SandwichGroup<2>(const Pulse& pulse)
{
    //
    // TODO: this should be implemented in terms of the basemap.
    //
    if (pulse.Label() == NucleotideLabel::A || pulse.Label() == NucleotideLabel::C)
        return 1;
    else if (pulse.Label() == NucleotideLabel::G || pulse.Label() == NucleotideLabel::T)
        return 2;
    else
        throw PBException("Invalid pulse label!");
}

template<>
int SandwichGroup<1>(const Pulse& pulse)
{
    int ret = static_cast<int>(pulse.Label());
    if (ret > 3)
        throw PBException("Invalid pulse label!");
    return ret;
}

} // ::anonymous namespace

template <typename Flt, size_t NCam>
BasecallingMetrics<Flt,NCam>& BasecallingMetrics<Flt,NCam>::AddBasecall(
        const Basecall& base,
        const PulseLookback<2>& twoPulseLookback,
        uint32_t sandwichTolerance)
{
#if 0 //!defined(NDEBUG)
    // Make this check/assert in DEBUG mode only
    if (base.GetPulse().Stop() > TraceMetrics().StopFrame())
    {
        PBLOG_ERROR << "Pulse stop frame: " << base.GetPulse().Stop()
                    << ", Metrics block end frame: " << TraceMetrics().StopFrame();
    }
    if (base.GetPulse().Stop() <= TraceMetrics().StartFrame())
    {
        PBLOG_ERROR << "Pulse stop frame: " << base.GetPulse().Stop()
                    << ", Metrics block start frame: " << TraceMetrics().StartFrame();
    }
    assert(TraceMetrics().StartFrame() < base.GetPulse().Stop());
    assert(base.GetPulse().Stop() <= TraceMetrics().StopFrame());
#endif

    // Account for the latency in Viterbi stitching. Currently hard-coded
    // but this parameter should be exposed from PdfViterbiFullFrame::MaxDetectionLatency.

    static const size_t MaxDetectionLatency = 64;

    if (base.GetPulse().Stop() > TraceMetrics().StopFrame() ||
            (TraceMetrics().StartFrame() >= MaxDetectionLatency &&
                base.GetPulse().Stop() <= (TraceMetrics().StartFrame() - MaxDetectionLatency)))
    {
        PBLOG_ERROR << "Pulse stop frame: " << base.GetPulse().Stop()
                    << ", Metrics block start frame: " << TraceMetrics().StartFrame()
                    << ", Metrics block end frame: " << TraceMetrics().StopFrame()
                    << ", Max detection latency: " << MaxDetectionLatency;
        return *this;
    }

    ++numPulses_;

    numPulseFrames_ += base.GetPulse().Width();
    uint8_t pulseLabel = static_cast<uint8_t>(base.GetPulse().Label());
    if (pulseLabel >= 4)
    {
        PBLOG_ERROR << "pulseLabel = " << (int)pulseLabel << " >= 4";
    } else
    {
        numPulsesByAnalog_[pulseLabel]++;
        pkMax_[pulseLabel] = std::max(pkMax_[pulseLabel], base.GetPulse().MaxSignal());
    }

    // Count half-sandwiches, pulse-label stutter events
    if (twoPulseLookback.Size() >= 1)
    {
        const Pulse& prevPulse = twoPulseLookback.GetPreviousPulse(0).GetPulse();
        const Pulse& curPulse = base.GetPulse();

        if (prevPulse.Label() == curPulse.Label())
            numPulseLabelStutters_++;

        // Half sandwich occurs when we are abutting the previous pulse, and we
        // are in different groups (see comments on SandwichGroup)
        bool abutted = (curPulse.Start() <= (sandwichTolerance + prevPulse.Stop()));
        auto prevGroup = SandwichGroup<NCam>(prevPulse);
        auto curGroup = SandwichGroup<NCam>(curPulse);
        bool differentGroups = (prevGroup != curGroup);
        if (abutted && differentGroups)
            numHalfSandwiches_++;

        // Count full sandwiches
        if (abutted && differentGroups && twoPulseLookback.Size() == 2)
        {
            const Pulse& prevPrevPulse = twoPulseLookback.GetPreviousPulse(1).GetPulse();

            bool allAbutted = (prevPulse.Start() <= (sandwichTolerance + prevPrevPulse.Stop())  &&
                               abutted);
            auto prevPrevGroup = SandwichGroup<NCam>(prevPrevPulse);
            bool groupSwitchAndBack = (prevPrevGroup != prevGroup &&
                                       prevPrevGroup == curGroup);
            if (allAbutted && groupSwitchAndBack)
                numSandwiches_++;
        }
    }



    if (!base.IsNoCall())
    {
        // Base-level metrics
        ++numBases_;
        numBaseFrames_ += base.GetPulse().Width();

        uint8_t baseLabel = static_cast<uint8_t>(base.Base()); // NOTE: safe to cast?
        numBasesByAnalog_[baseLabel]++;
        if (!isnan(base.GetPulse().MidSignal()))
        {
            numPkmidBasesByAnalog_[baseLabel]++;

            // Inter-pulse moments (in terms of frames)
            const uint16_t midWidth = static_cast<uint16_t>(base.GetPulse().Width() - 2);
            // count (M0)
            pkmidN_[baseLabel] += midWidth;
            // sum of signals (M1)
            pkmidSig_[baseLabel] += base.GetPulse().MidSignal()
                                    * midWidth;
            // sum of square of signals (M2)
            bpzvar_[baseLabel] += base.GetPulse().MidSignal()
                                  * base.GetPulse().MidSignal()
                                  * midWidth;

            // Intra-pulse M2
            pkzvar_[baseLabel] += base.GetPulse().SignalM2();
        }
    }

    return *this;
}

template <typename Flt,size_t NCam>
void BasecallingMetrics<Flt, NCam>::FinalizeVariance()
{
    for (size_t baseLabel = 0; baseLabel < numPkmidBasesByAnalog_.size(); baseLabel++)
    {
        if (numPkmidBasesByAnalog_[baseLabel] == 0)
        {
            // No bases called on channel.
            pkmidSig_[baseLabel] = std::numeric_limits<Flt>::quiet_NaN();
            bpzvar_[baseLabel] = std::numeric_limits<Flt>::quiet_NaN();
            pkzvar_[baseLabel] = std::numeric_limits<Flt>::quiet_NaN();
        } else if (numPkmidBasesByAnalog_[baseLabel] < 2 || pkmidN_[baseLabel] < 2)
        {
            bpzvar_[baseLabel] = std::numeric_limits<Flt>::quiet_NaN();
            pkzvar_[baseLabel] = std::numeric_limits<Flt>::quiet_NaN();
        }
        else
        {
            // Convert moments to interpulse variance
            bpzvar_[baseLabel] = (bpzvar_[baseLabel]
                                  - (pkmidSig_[baseLabel]
                                     * pkmidSig_[baseLabel]
                                     / pkmidN_[baseLabel]))
                                 / (pkmidN_[baseLabel]);
            // Bessel's correction with num bases, not frames
            bpzvar_[baseLabel] *= numPkmidBasesByAnalog_[baseLabel]
                                  / (numPkmidBasesByAnalog_[baseLabel] - 1);

            // Here we assume the base label order and dye color order:
            size_t dyeColorIndex = 0; // 1C
            if (NCam == 2 && baseLabel < 2) // 2C
                dyeColorIndex = 1;

            const auto baselineVariance =
                TraceMetrics().FrameBaselineVarianceDWS()[dyeColorIndex];


            bpzvar_[baseLabel] -= baselineVariance
                                  / (pkmidN_[baseLabel]
                                     / numPkmidBasesByAnalog_[baseLabel]);

            pkzvar_[baseLabel] = (pkzvar_[baseLabel]
                                  - (pkmidSig_[baseLabel]
                                     * pkmidSig_[baseLabel]
                                     / pkmidN_[baseLabel]))
                                  / (pkmidN_[baseLabel] - 1);

            // pkzvar up to this point contains total signal variance. We
            // subtract out interpulse variance and baseline variance to leave
            // intrapulse variance.
            pkzvar_[baseLabel] -= bpzvar_[baseLabel] + baselineVariance;

            const auto& modelIntraVars =
                TraceMetrics().Analog()[baseLabel].SignalCovar();


            // the model intrapulse variance still contains baseline variance,
            // remove before normalizing
            if (modelIntraVars[dyeColorIndex] > baselineVariance)
                pkzvar_[baseLabel] /= modelIntraVars[dyeColorIndex]
                                      - baselineVariance;
            else
                pkzvar_[baseLabel] = 0.0f;
            pkzvar_[baseLabel] = std::max(pkzvar_[baseLabel], 0.0f);

            const auto pkmid = pkmidSig_[baseLabel] / pkmidN_[baseLabel];
            if (pkmid > 0)
                bpzvar_[baseLabel] = std::max(bpzvar_[baseLabel] / (pkmid * pkmid),
                                              0.0f);
            else
                bpzvar_[baseLabel] = std::numeric_limits<Flt>::quiet_NaN();
        }
    }
}

template <typename Flt, size_t NCam>
std::array<Flt,BasecallingMetrics<Flt,NCam>::numAnalogs> BasecallingMetrics<Flt,NCam>::PkmidMean() const
{
    std::array<Flt,4> meanSignal =
    {{
         std::numeric_limits<Flt>::quiet_NaN(),
         std::numeric_limits<Flt>::quiet_NaN(),
         std::numeric_limits<Flt>::quiet_NaN(),
         std::numeric_limits<Flt>::quiet_NaN()
    }};

    for (size_t i = 0; i < meanSignal.size(); ++i)
    {
        const auto& pkmidTotal = PkmidSignal()[i];
        if (!isnan(pkmidTotal))
        {
            uint16_t numFrames = PkmidNumFrames()[i];
            if (numFrames > 0)
            {
                meanSignal[i] = pkmidTotal / numFrames;
            }
        }
    }

    return meanSignal;
}

template <typename Flt, size_t NCam>
std::array<Flt,BasecallingMetrics<Flt,NCam>::numAnalogs> BasecallingMetrics<Flt,NCam>::FrameSnr() const
{
    std::array<Flt,4> snr =
    {{
         std::numeric_limits<Flt>::quiet_NaN(),
         std::numeric_limits<Flt>::quiet_NaN(),
         std::numeric_limits<Flt>::quiet_NaN(),
         std::numeric_limits<Flt>::quiet_NaN()
    }};

    const auto& blS = TraceMetrics().FrameBaselineSigmaDWS();
    const auto& pkmidMean = PkmidMean();

    std::array<float,4> blSigma;
    if (NCam == 1)
        std::fill(blSigma.begin(), blSigma.end(), blS.front());
    else
        blSigma = {{ blS[1], blS[1], blS[0], blS[0] }};

    for (size_t i = 0; i < snr.size(); ++i)
    {
        auto bls = blSigma[i];
        if (bls > 0)
        {
            if (!isnan(pkmidMean[i]))
            {
                snr[i] = pkmidMean[i] / bls;
            }
        }
    }

    return snr;
};

// Explicit Instantiations
template class BasecallingMetrics<float,2>;
template class BasecallingMetrics<float,1>;

BasecallingMetricsT::Sequel Convert1CamTo2Cam(const BasecallingMetricsT::Spider* buf)
{
    // Convert metrics to 2-camera.
    BasecallingMetricsT::Sequel bm;

    bm.NumPulseFrames(buf->NumPulseFrames())
            .NumBaseFrames(buf->NumBaseFrames())
            .NumSandwiches(buf->NumSandwiches())
            .NumHalfSandwiches(buf->NumHalfSandwiches())
            .NumPulseLabelStutters(buf->NumPulseLabelStutters())
            .NumBases(buf->NumBases())
            .NumPulses(buf->NumPulses());
    bm.PkmidSignal() = buf->PkmidSignal();
    bm.Bpzvar() = buf->Bpzvar();
    bm.Pkzvar() = buf->Pkzvar();
    bm.PkmidNumFrames() = buf->PkmidNumFrames();
    bm.PkMax() = buf->PkMax();
    bm.NumPulsesByAnalog() = buf->NumPulsesByAnalog();
    bm.NumBasesByAnalog() = buf->NumBasesByAnalog();
    bm.NumPkmidBasesByAnalog() = buf->NumPkmidBasesByAnalog();

    auto& tm = bm.TraceMetrics();
    tm.StartFrame(buf->TraceMetrics().StartFrame())
            .NumFrames(buf->TraceMetrics().NumFrames())
            .ConfidenceScore(buf->TraceMetrics().ConfidenceScore())
            .PulseDetectionScore(buf->TraceMetrics().PulseDetectionScore())
            .FullEstimationAttempted(buf->TraceMetrics().FullEstimationAttempted())
            .ModelUpdated(buf->TraceMetrics().ModelUpdated())
            .AngleEstimationSucceeded(buf->TraceMetrics().AngleEstimationSuceeded())
            .Autocorrelation(buf->TraceMetrics().Autocorrelation());

    std::fill(tm.FrameBaselineVarianceDWS().begin(), tm.FrameBaselineVarianceDWS().end(),
              buf->TraceMetrics().FrameBaselineVarianceDWS().front());
    std::fill(tm.FrameBaselineDWS().begin(), tm.FrameBaselineDWS().end(),
              buf->TraceMetrics().FrameBaselineDWS().front());
    std::fill(tm.NumFramesBaseline().begin(), tm.NumFramesBaseline().end(),
              buf->TraceMetrics().NumFramesBaseline().front());
    std::fill(tm.AngleSpectrum().begin(), tm.AngleSpectrum().end(), 0);

    auto convertDetMode = [](const DetectionMode<float,1>& m)
    {
        auto sm = m.SignalMean().front();
        auto sc = m.SignalCovar().front();
        return DetectionMode<float,2>({{ sm, sm }}, {{ sc, sc, sc }}, m.Weight());
    };

    const auto& bl = buf->TraceMetrics().Baseline();
    const auto& am = buf->TraceMetrics().Analog();

    std::array<DetectionMode<float,2>, 5> detModes;
    detModes[0] = convertDetMode(bl);
    detModes[1] = convertDetMode(am[0]);
    detModes[2] = convertDetMode(am[1]);
    detModes[3] = convertDetMode(am[2]);
    detModes[4] = convertDetMode(am[3]);
    tm.DetectionModes(detModes);

    tm.pixelChecksum = buf->TraceMetrics().pixelChecksum;

    return bm;
}

}}
