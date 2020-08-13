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
//  Defines members of class BasecallingMetricsAccumulator

#include <array>
#include <vector>

#include "BasecallingMetricsAccumulator.h"
#include "BaselinerStatAccumulator.h"
#include <common/MongoConstants.h>
#include "TrainedCartParams.h"

using namespace PacBio::Cuda::Utility;

namespace PacBio {
namespace Mongo {
namespace Data {

namespace {

// TODO clean
template <typename T, size_t N, typename U>
void Fill(CudaArray<T, N>& arr, U val)
{
    std::fill(arr.begin(), arr.end(), val);
}

// Evaluates a polynomial with a list of coefficients by descending degree
// (e.g. y = ax^2 + bx + c)
template <size_t size>
float evaluatePolynomial(const std::array<float, size>& coeff, float x)
{
    static_assert(size > 0, "Unexpected empty array");
    float y = coeff[0];
    for (unsigned int i = 1; i < size; ++i)
        y = y * x + coeff[i];
    return y;
}

template <size_t size>
LaneArray<float> evaluatePolynomial(const std::array<float, size>& coeff, LaneArray<float> x)
{
    static_assert(size > 0, "Unexpected empty array");
    LaneArray<float> y(coeff[0]);
    for (unsigned int i = 1; i < size; ++i)
        y = y * x + LaneArray<float>(coeff[i]);
    return y;
}
}

void BasecallingMetricsAccumulator::LabelBlock(float frameRate)
{
    // Calculated in the accessor, so caching here:
    const auto& stdDev = traceMetrics_.FrameBaselineSigmaDWS();
    const auto& numBases = NumBases();
    const auto& numPulses = NumPulses();
    const auto& pulseWidth = PulseWidth();
    alignas(64) const auto pkmid = PkmidMean();
    LaneArray<float> zeros(0.0f);

    std::array<ArrayUnion<LaneArray<float>>, ActivityLabeler::NUM_FEATURES> features;
    for (auto& analog : features)
    {
        analog = zeros;
    }

    const auto& seconds = traceMetrics_.NumFrames() / frameRate;

    features[ActivityLabeler::PULSERATE] = numPulses / seconds;

    features[ActivityLabeler::SANDWICHRATE] = LaneArray<float>(numSandwiches_) / numPulses;
    features[ActivityLabeler::SANDWICHRATE] = Blend(
            isnan(features[ActivityLabeler::SANDWICHRATE]),
            zeros,
            features[ActivityLabeler::SANDWICHRATE]);

    auto hswr = LaneArray<float>(numHalfSandwiches_) / numPulses;
    hswr = Blend(isnan(hswr), zeros, hswr);
    auto hswrExp = evaluatePolynomial(ActivityLabeler::TrainedCart::hswCurve,
                                      features[ActivityLabeler::PULSERATE]);
    hswrExp = Blend(hswrExp > ActivityLabeler::TrainedCart::maxAcceptableHalfsandwichRate,
                    LaneArray<float>(ActivityLabeler::TrainedCart::maxAcceptableHalfsandwichRate),
                    hswrExp);

    features[ActivityLabeler::LOCALHSWRATENORM] = hswr - hswrExp;

    features[ActivityLabeler::VITERBISCORE] = traceMetrics_.PulseDetectionScore();
    features[ActivityLabeler::MEANPULSEWIDTH] = pulseWidth;
    features[ActivityLabeler::LABELSTUTTERRATE] = LaneArray<float>(numPulseLabelStutters_) / numPulses;
    features[ActivityLabeler::LABELSTUTTERRATE] = Blend(
            isnan(features[ActivityLabeler::LABELSTUTTERRATE]),
            zeros,
            features[ActivityLabeler::LABELSTUTTERRATE]);

    const auto& pkbases = numBasesByAnalog_;

    // make a copy of modelMean_
    AnalogMetric<float> relamps(modelMean_);

    LaneArray<float> maxamp(relamps[0]);
    for (size_t i = 1; i < numAnalogs; ++i)
    {
        maxamp = max(LaneArray<float>(relamps[i]), maxamp);
    }
    for (size_t i = 0; i < numAnalogs; ++i)
    {
        relamps[i] = LaneArray<float>(relamps[i]) / maxamp;
    }

    LaneArray<unsigned int> lowAmpIndex(0);
    LaneArray<float> minamp(relamps[0]);
    // TODO unify with above?
    for (size_t i = 1; i < numAnalogs; ++i)
    {
        // TODO rename
        LaneArray<float> tmp(relamps[i]);
        static_assert(sizeof(unsigned int) == 4u, "");
        const auto& ila = LaneArray<unsigned int>(i);
        lowAmpIndex = Blend(tmp < minamp, ila, lowAmpIndex);
        minamp = min(tmp, minamp);
    }

    for (size_t i = 0; i < numAnalogs; ++i)
    {
        features[ActivityLabeler::BLOCKLOWSNR] += LaneArray<float>(pkmid[i]) / stdDev
                                                * LaneArray<float>(pkbases[i]) / numBases
            // TODO we've converted relamps about a millio times now
                                                * minamp / LaneArray<float>(relamps[i]);
    }
    features[ActivityLabeler::BLOCKLOWSNR] = Blend(
            isnan(features[ActivityLabeler::BLOCKLOWSNR]),
            zeros,
            features[ActivityLabeler::BLOCKLOWSNR]);

    for (size_t i = 0; i < numAnalogs; ++i)
    {
        features[ActivityLabeler::MAXPKMAXNORM] = max(
            features[ActivityLabeler::MAXPKMAXNORM],
            (LaneArray<float>(pkMax_[i]) - LaneArray<float>(pkmid[i])) / stdDev);
    }
    features[ActivityLabeler::MAXPKMAXNORM] = Blend(
            isnan(features[ActivityLabeler::MAXPKMAXNORM]),
            zeros,
            features[ActivityLabeler::MAXPKMAXNORM]);

    features[ActivityLabeler::AUTOCORRELATION] = traceMetrics_.Autocorrelation();

    LaneArray<float> lowbp(0);
    LaneArray<float> lowpk(0);
    for (size_t i = 0; i < numAnalogs; ++i)
    {
        // Can we avoid this Blend?
        // TODO rename
        LaneArray<float> tmp(bpZvar_[i]);
        features[ActivityLabeler::BPZVARNORM] += Blend(isnan(tmp), zeros, tmp);
        lowbp = Blend((lowAmpIndex == i) & !isnan(tmp), tmp, lowbp);

        tmp = LaneArray<float>(pkZvar_[i]);
        features[ActivityLabeler::PKZVARNORM] += Blend(isnan(tmp), zeros, tmp);
        lowpk = Blend((lowAmpIndex == i) & !isnan(tmp), tmp, lowpk);
    }
    features[ActivityLabeler::BPZVARNORM] -=  lowbp;
    features[ActivityLabeler::BPZVARNORM] /= LaneArray<float>(3.0f);
    features[ActivityLabeler::PKZVARNORM] -=  lowpk;
    features[ActivityLabeler::PKZVARNORM] /= LaneArray<float>(3.0f);

    for (size_t i = 0; i < features.size(); ++i)
    {
        assert(none(isnan(features[i])));
    }

    // I don't think parallel array accessions are possible at the moment, so
    // this still loops overt the lane:
    for (size_t z = 0; z < laneSize; ++z)
    {
        size_t current = 0;
        while (ActivityLabeler::TrainedCart::feature[current] >= 0)
        {
            if (features[ActivityLabeler::TrainedCart::feature[current]][z]
                    <= ActivityLabeler::TrainedCart::threshold[current])
            {
                current = ActivityLabeler::TrainedCart::childrenLeft[current];
            }
            else
            {
                current = ActivityLabeler::TrainedCart::childrenRight[current];
            }
        }
        activityLabel_[z] = ActivityLabeler::TrainedCart::value[current];
    }
}

void BasecallingMetricsAccumulator::PopulateBasecallingMetrics(
        BasecallingMetrics& metrics)
{
    // Gather up the results of some non-trivial accessors, tweak those results
    // as necessary (which couldn't be tweaked in FinalizeMetrics() without
    // storing them until this Populate method is called)
    //
    const auto& numPulses = NumPulses();
    const auto& numBases = NumBases();
    auto autocorr = traceMetrics_.Autocorrelation();
    autocorr = Blend(isnan(autocorr), LaneArray<float>(0), autocorr);

    metrics.numPulseFrames = numPulseFrames_;
    metrics.numBaseFrames = numBaseFrames_;
    metrics.numSandwiches = numSandwiches_;
    metrics.numHalfSandwiches = numHalfSandwiches_;
    metrics.numPulseLabelStutters = numPulseLabelStutters_;
    metrics.numPulses = numPulses;
    metrics.numBases = numBases;
    metrics.activityLabel = activityLabel_;
    metrics.startFrame = traceMetrics_.StartFrame();
    metrics.numFrames = traceMetrics_.NumFrames();
    metrics.autocorrelation = autocorr;
    metrics.pulseDetectionScore = traceMetrics_.PulseDetectionScore();
    metrics.pixelChecksum = traceMetrics_.PixelChecksum();

    for (size_t a = 0; a < numAnalogs; ++a)
    {
        metrics.pkMidSignal[a] = pkMidSignal_[a];
        metrics.bpZvar[a] = bpZvar_[a];
        metrics.pkZvar[a] = pkZvar_[a];
        metrics.pkMax[a] = pkMax_[a];
        metrics.numPkMidFrames[a] = numPkMidFrames_[a];
        metrics.numPkMidBasesByAnalog[a] = numPkMidBasesByAnalog_[a];
        metrics.numBasesByAnalog[a] = numBasesByAnalog_[a];
        metrics.numPulsesByAnalog[a] = numPulsesByAnalog_[a];
    }
}

void BasecallingMetricsAccumulator::Reset()
{
    Fill(numPulseFrames_, 0);
    Fill(numBaseFrames_, 0);
    Fill(numSandwiches_, 0);
    Fill(numHalfSandwiches_, 0);
    Fill(numPulseLabelStutters_, 0);
    prevBasecallCache_.fill(Pulse().Start(0).Width(0).Label(Pulse::NucleotideLabel::NONE));
    prevprevBasecallCache_.fill(Pulse().Start(0).Width(0).Label(Pulse::NucleotideLabel::NONE));
    Fill(activityLabel_, static_cast<uint16_t>(HQRFPhysicalStates::EMPTY));
    for (size_t a = 0; a < numAnalogs; ++a)
    {
        Fill(pkMidSignal_[a], 0);
        Fill(bpZvar_[a], 0);
        Fill(pkZvar_[a], 0);
        Fill(pkMax_[a], 0);
        Fill(numPkMidFrames_[a], 0);
        Fill(numPkMidBasesByAnalog_[a], 0);
        Fill(numBasesByAnalog_[a], 0);
        Fill(numPulsesByAnalog_[a], 0);
    }
    traceMetrics_.Reset();
}

auto BasecallingMetricsAccumulator::NumBases() const -> LaneArray<uint16_t>
{
    LaneArray<uint16_t> ret(0);
    for (size_t a = 0; a < numAnalogs; ++a)
    {
        ret += LaneArray<uint16_t>(numBasesByAnalog_[a]);
    }
    return ret;
}

auto BasecallingMetricsAccumulator::NumPulses() const -> LaneArray<uint16_t>
{
    LaneArray<uint16_t> ret(0);
    for (size_t a = 0; a < numAnalogs; ++a)
    {
        ret += LaneArray<uint16_t>(numPulsesByAnalog_[a]);
    }
    return ret;
}

auto BasecallingMetricsAccumulator::PulseWidth() const -> LaneArray<float>
{
    auto ret = NumPulses() / LaneArray<float>(numPulseFrames_);
    return Blend(isnan(ret), LaneArray<float>(0), ret);
}


auto BasecallingMetricsAccumulator::PkmidMean() const -> AnalogMetric<float>
{
    AnalogMetric<float> ret;
    for (size_t ai = 0; ai < numAnalogs; ++ai)
    {
        ret[ai] = LaneArray<float>(pkMidSignal_[ai]) / LaneArray<float>(numPkMidFrames_[ai]);
    }
    return ret;
}

// This only keeps the most recent model, which should be fine
void BasecallingMetricsAccumulator::AddModels(
        const LaneModelParameters<Cuda::PBHalf, laneSize>& models)
{
    for (size_t ai = 0; ai < numAnalogs; ++ai)
    {
        for (size_t i = 0; i < laneSize; ++i)
        {
            modelVariance_[ai][i] = models.AnalogMode(ai).vars[i];
            modelMean_[ai][i] = models.AnalogMode(ai).means[i];
        }
    }
}

void BasecallingMetricsAccumulator::AddBatchMetrics(
        const BaselinerStatAccumState& baselinerStats,
        const Cuda::Utility::CudaArray<float, laneSize>& viterbiScore,
        const StatAccumState& pdBaselineStats)
{
    BaselinerStatAccumulator<BaselinedTraceElement> reconstitutedStats(baselinerStats);
    traceMetrics_.BaselinerStatAccum().Merge(reconstitutedStats.BaselineFramesStats());
    traceMetrics_.AutocorrAccum().Merge(reconstitutedStats.BaselineSubtractedStats());

    traceMetrics_.PulseDetectionScore() += LaneArray<float>(viterbiScore);

    // TODO: collect PulseAccumulator BaselineStats, replace
    // BaselinerStatAccum().Merge(... above
    (void)pdBaselineStats;
}

void BasecallingMetricsAccumulator::Count(
        const LaneVectorView<const Pulse>& pulses,
        uint32_t numFrames)
{
    traceMetrics_.NumFrames() += numFrames;
    for (size_t zi = 0; zi < laneSize; ++zi)
    {
        const Pulse* prevPulse = &prevBasecallCache_[zi];
        const Pulse* prevprevPulse = &prevprevBasecallCache_[zi];
        for (size_t bi = 0; bi < pulses.size(zi); ++bi)
        {
            const Pulse* pulse = &pulses(zi, bi);

            uint8_t pulseLabel = static_cast<uint8_t>(pulse->Label());
            numPulseFrames_[zi] += pulse->Width();
            numPulsesByAnalog_[pulseLabel][zi]++;
            pkMax_[pulseLabel][zi] = std::max(pkMax_[pulseLabel][zi],
                                              pulse->MaxSignal());

            // These caches are initialized to NONE:0-0
            // We'll not consider any previous pulses with label NONE anyway,
            // because a sandwich in such a case doesn't make much sense
            if (prevPulse->Label() != Pulse::NucleotideLabel::NONE)
            {
                if (prevPulse->Label() == pulse->Label())
                {
                    numPulseLabelStutters_[zi]++;
                }

                bool abutted = (pulse->Start() == prevPulse->Stop());

                if (abutted && prevPulse->Label() != pulse->Label())
                {
                    numHalfSandwiches_[zi]++;
                }

                if (prevprevPulse->Label() != Pulse::NucleotideLabel::NONE)
                {
                    bool prevAbutted = (prevPulse->Start() == prevprevPulse->Stop());
                    if (prevAbutted && abutted
                            && prevprevPulse->Label() == pulse->Label()
                            && prevprevPulse->Label() != prevPulse->Label())
                    {
                        numSandwiches_[zi]++;
                    }
                }
            }

            if (!pulse->IsReject())
            {
                numBaseFrames_[zi] += pulse->Width();
                numBasesByAnalog_[pulseLabel][zi]++;

                if (!isnan(pulse->MidSignal()))
                {
                    numPkMidBasesByAnalog_[pulseLabel][zi]++;

                    // Inter-pulse moments (in terms of frames)
                    const uint16_t midWidth = static_cast<uint16_t>(pulse->Width() - 2);
                    // count (M0)
                    numPkMidFrames_[pulseLabel][zi] += midWidth;
                    // sum of signals (M1)
                    pkMidSignal_[pulseLabel][zi] += pulse->MidSignal()
                                                    * midWidth;
                    // sum of square of signals (M2)
                    bpZvar_[pulseLabel][zi] += pulse->MidSignal()
                                               * pulse->MidSignal()
                                               * midWidth;

                    // Intra-pulse M2
                    pkZvar_[pulseLabel][zi] += pulse->SignalM2();
                }
            }
            prevprevPulse = prevPulse;
            prevPulse = pulse;
        }
        prevBasecallCache_[zi] = *prevPulse;
        prevprevBasecallCache_[zi] = *prevprevPulse;
    }
}

void BasecallingMetricsAccumulator::FinalizeMetrics(
        bool realtimeActivityLabels, float frameRate)
{
    LaneArray<float> nans(std::numeric_limits<float>::quiet_NaN());

    for (size_t pulseLabel = 0; pulseLabel < numAnalogs; pulseLabel++)
    {
        if (all(LaneArray<int16_t>(numPkMidBasesByAnalog_[pulseLabel]) == 0))
        {
            // No bases called on channel.
            Fill(pkMidSignal_[pulseLabel], std::numeric_limits<float>::quiet_NaN());
            Fill(bpZvar_[pulseLabel], std::numeric_limits<float>::quiet_NaN());
            Fill(pkZvar_[pulseLabel], std::numeric_limits<float>::quiet_NaN());
        }
        else if (all(LaneArray<float>(numPkMidBasesByAnalog_[pulseLabel]) < 2)
                 || all(LaneArray<float>(numPkMidFrames_[pulseLabel]) < 2))
        {
            Fill(bpZvar_[pulseLabel], std::numeric_limits<float>::quiet_NaN());
            Fill(pkZvar_[pulseLabel], std::numeric_limits<float>::quiet_NaN());
        }
        else
        {
            // Load up the values we ultimately intend to adjust
            LaneArray<float> bpZvar(bpZvar_[pulseLabel]);
            LaneArray<float> pkZvar(pkZvar_[pulseLabel]);
            LaneArray<float> pkMidSignal(pkMidSignal_[pulseLabel]);

            const LaneArray<float> nf(numPkMidFrames_[pulseLabel]);
            const LaneArray<float> nb(numPkMidBasesByAnalog_[pulseLabel]);
            // Convert moments to interpulse variance
            bpZvar = (bpZvar - (pkMidSignal * pkMidSignal / nf)) / nf;

            // Bessel's correction with num bases, not frames
            bpZvar = bpZvar * nb / (nb - 1.0f);

            const auto baselineVariance =
                traceMetrics_.FrameBaselineVarianceDWS();

            bpZvar = bpZvar - baselineVariance / (nf / nb);

            pkZvar = (pkZvar - (pkMidSignal * pkMidSignal / nf)) / (nf - 1.0f);

            // pkzvar up to this point contains total signal variance. We
            // subtract out interpulse variance and baseline variance to leave
            // intrapulse variance.
            pkZvar -= bpZvar + baselineVariance;

            const LaneArray<float> modelIntraVars(modelVariance_[pulseLabel]);
            // the model intrapulse variance still contains baseline variance,
            // remove before normalizing
            auto mask = (modelIntraVars < baselineVariance);
            pkZvar /= modelIntraVars - baselineVariance;
            mask |= (pkZvar < 0.0f);
            pkZvar = Blend(mask, LaneArray<float>(0.0f), pkZvar);


            const auto& pkMid = pkMidSignal / nf;
            mask = (pkMid < 0);
            bpZvar /= (pkMid * pkMid);
            mask |= (bpZvar < 0.0f);
            bpZvar = Blend(mask, nans, bpZvar);

            // No bases called on channel.
            const LaneArray<int16_t> numPkMidBasesByAnalog(numPkMidBasesByAnalog_[pulseLabel]);
            mask = (numPkMidBasesByAnalog == 0);
            pkMidSignal = Blend(mask, nans, pkMidSignal);
            // Not enough bases called on channel
            mask |= (numPkMidBasesByAnalog < 2);
            mask |= (LaneArray<int16_t>(numPkMidFrames_[pulseLabel]) < 2);
            bpZvar = Blend(mask, nans, bpZvar);
            pkZvar = Blend(mask, nans, pkZvar);

            // Store the results
            pkMidSignal_[pulseLabel] = pkMidSignal;
            bpZvar_[pulseLabel] = bpZvar;
            pkZvar_[pulseLabel] = pkZvar;
        }
    }

    auto& pdScore = traceMetrics_.PulseDetectionScore();
    pdScore = pdScore / traceMetrics_.NumFrames();
    pdScore = Blend(isnan(pdScore), LaneArray<float>(0), pdScore);

    if (realtimeActivityLabels)
    {
        LabelBlock(frameRate);
    }
}

}}}     // namespace PacBio::Mongo::Data
