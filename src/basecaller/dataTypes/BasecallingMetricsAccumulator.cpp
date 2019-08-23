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

namespace PacBio {
namespace Mongo {
namespace Data {

namespace {

// Evaluates a polynomial with a list of coefficients by descending degree
// (e.g. y = ax^2 + bx + c)
template <size_t size>
float evaluatePolynomial(const std::array<float, size>& coeff, float x)
{
    static_assert(size > 0);
    float y = coeff[0];
    for (unsigned int i = 1; i < size; ++i)
        y = y * x + coeff[i];
    return y;
}

template <size_t size>
LaneArray<float> evaluatePolynomial(const std::array<float, size>& coeff, LaneArray<float> x)
{
    static_assert(size > 0);
    LaneArray<float> y(coeff[0]);
    for (unsigned int i = 1; i < size; ++i)
        y = y * x + LaneArray<float>(coeff[i]);
    return y;
}
}

template <unsigned int LaneWidth>
void BasecallingMetricsAccumulator<LaneWidth>::LabelBlock(float frameRate)
{
    // Calculated in the accessor, so caching here:
    const auto& stdDev = traceMetrics_.FrameBaselineSigmaDWS();
    const auto& numBases = NumBases().AsFloat();
    const auto& numPulses = NumPulses().AsFloat();
    const auto& pulseWidth = PulseWidth();
    const auto& pkmid = PkmidMean();
    LaneArray<float> zeros(0);

    std::array<LaneArray<float>, ActivityLabeler::NUM_FEATURES> features;
    for (auto& analog : features)
    {
        analog = zeros;
    }

    const auto& seconds = traceMetrics_.NumFrames().AsFloat() / LaneArray<float>(frameRate);

    features[ActivityLabeler::PULSERATE] = numPulses / seconds;

    features[ActivityLabeler::SANDWICHRATE] = numSandwiches_.AsFloat() / numPulses;
    features[ActivityLabeler::SANDWICHRATE] = Blend(
            isnan(features[ActivityLabeler::SANDWICHRATE]),
            zeros,
            features[ActivityLabeler::SANDWICHRATE]);

    auto hswr = numHalfSandwiches_.AsFloat() / numPulses;
    hswr = Blend(isnan(hswr), zeros, hswr);
    auto hswrExp = evaluatePolynomial(ActivityLabeler::TrainedCart::hswCurve,
                                      features[ActivityLabeler::PULSERATE]);
    hswrExp = Blend(hswrExp > ActivityLabeler::TrainedCart::maxAcceptableHalfsandwichRate,
                    LaneArray<float>(ActivityLabeler::TrainedCart::maxAcceptableHalfsandwichRate),
                    hswrExp);

    features[ActivityLabeler::LOCALHSWRATENORM] = hswr - hswrExp;

    features[ActivityLabeler::VITERBISCORE] = traceMetrics_.PulseDetectionScore();
    features[ActivityLabeler::MEANPULSEWIDTH] = pulseWidth;
    features[ActivityLabeler::LABELSTUTTERRATE] = numPulseLabelStutters_.AsFloat() / numPulses;
    features[ActivityLabeler::LABELSTUTTERRATE] = Blend(
            isnan(features[ActivityLabeler::LABELSTUTTERRATE]),
            zeros,
            features[ActivityLabeler::LABELSTUTTERRATE]);

    const auto& pkbases = numBasesByAnalog_;

    // make a copy of modelMean_, probably won't work this way:
    AnalogFloatMetric relamps(modelMean_);

    LaneArray<float> maxamp = relamps[0];
    for (size_t i = 1; i < numAnalogs; ++i)
    {
        maxamp = max(relamps[i], maxamp);
    }
    for (size_t i = 0; i < numAnalogs; ++i)
    {
        relamps[i] /= maxamp;
    }

    LaneArray<unsigned int> lowAmpIndex(0);
    LaneArray<float> minamp(relamps[0]);
    for (size_t i = 1; i < numAnalogs; ++i)
    {
        static_assert(sizeof(unsigned int) == 4u);
        const auto& ila = LaneArray<unsigned int>(i);
        lowAmpIndex = Blend(relamps[i] < minamp, ila, lowAmpIndex);
        minamp = min(relamps[i], minamp);
    }

    for (size_t i = 0; i < numAnalogs; ++i)
    {
        features[ActivityLabeler::BLOCKLOWSNR] += pkmid[i] / stdDev
                                                * pkbases[i].AsFloat() / numBases
                                                * minamp / relamps[i];
    }
    features[ActivityLabeler::BLOCKLOWSNR] = Blend(
            isnan(features[ActivityLabeler::BLOCKLOWSNR]),
            zeros,
            features[ActivityLabeler::BLOCKLOWSNR]);

    for (size_t i = 0; i < numAnalogs; ++i)
    {
        features[ActivityLabeler::MAXPKMAXNORM] = max(
            features[ActivityLabeler::MAXPKMAXNORM],
            (pkMax_[i] - pkmid[i]) / stdDev);
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
        features[ActivityLabeler::BPZVARNORM] += Blend(isnan(bpZvar_[i]), zeros, bpZvar_[i]);
        lowbp = Blend(lowAmpIndex == LaneArray<unsigned int>(i) & !isnan(bpZvar_[i]), bpZvar_[i], lowbp);

        features[ActivityLabeler::PKZVARNORM] += Blend(isnan(pkZvar_[i]), zeros, pkZvar_[i]);
        lowpk = Blend(lowAmpIndex == LaneArray<unsigned int>(i) & !isnan(pkZvar_[i]), pkZvar_[i], lowpk);
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
        activityLabel_[z] = static_cast<HQRFPhysicalStates>(ActivityLabeler::TrainedCart::value[current]);
    }
}

template <unsigned int LaneWidth>
void BasecallingMetricsAccumulator<LaneWidth>::PopulateBasecallingMetrics(
        BasecallingMetricsT& metrics)
{
    // Gather up the results of some non-trivial accessors, tweak those results
    // as necessary (which couldn't be tweaked in FinalizeMetrics() without
    // storing them until this Populate method is called)
    //
    const auto& numPulses = NumPulses();
    const auto& numBases = NumBases();
    const auto& stopFrame = traceMetrics_.StopFrame();
    auto autocorr = traceMetrics_.Autocorrelation();
    autocorr = Blend(isnan(autocorr), LaneArray<float>(0), autocorr);

    for (size_t z = 0; z < LaneWidth; ++z)
    {
        metrics.numPulseFrames[z] = numPulseFrames_[z];
        metrics.numBaseFrames[z] = numBaseFrames_[z];
        metrics.numSandwiches[z] = numSandwiches_[z];
        metrics.numHalfSandwiches[z] = numHalfSandwiches_[z];
        metrics.numPulseLabelStutters[z] = numPulseLabelStutters_[z];
        metrics.numPulses[z] = numPulses[z];
        metrics.numBases[z] = numBases[z];
        metrics.activityLabel[z] = activityLabel_[z];
        metrics.startFrame[z] = traceMetrics_.StartFrame()[z];
        metrics.stopFrame[z] = stopFrame[z];
        metrics.numFrames[z] = traceMetrics_.NumFrames()[z];
        metrics.autocorrelation[z] = autocorr[z];
        metrics.pulseDetectionScore[z] = traceMetrics_.PulseDetectionScore()[z];
        metrics.pixelChecksum[z] = traceMetrics_.PixelChecksum()[z];
    }
    for (size_t a = 0; a < numAnalogs; ++a)
    {
        for (size_t z = 0; z < LaneWidth; ++z)
        {
            metrics.pkMidSignal[a][z] = pkMidSignal_[a][z];
            metrics.bpZvar[a][z] = bpZvar_[a][z];
            metrics.pkZvar[a][z] = pkZvar_[a][z];
            metrics.pkMax[a][z] = pkMax_[a][z];
            metrics.pkMidNumFrames[a][z] = pkMidNumFrames_[a][z];
            metrics.numPkMidBasesByAnalog[a][z] = numPkMidBasesByAnalog_[a][z];
            metrics.numBasesByAnalog[a][z] = numBasesByAnalog_[a][z];
            metrics.numPulsesByAnalog[a][z] = numPulsesByAnalog_[a][z];
        }
    }
}

template <unsigned int LaneWidth>
void BasecallingMetricsAccumulator<LaneWidth>::Reset()
{
    numPulseFrames_ = 0;
    numBaseFrames_ = 0;
    numSandwiches_ = 0;
    numHalfSandwiches_ = 0;
    numPulseLabelStutters_ = 0;
    prevBasecallCache_.fill(Pulse().Start(0).Width(0).Label(Pulse::NucleotideLabel::NONE));
    prevprevBasecallCache_.fill(Pulse().Start(0).Width(0).Label(Pulse::NucleotideLabel::NONE));
    for (size_t a = 0; a < numAnalogs; ++a)
    {
        pkMidSignal_[a] = 0;
        bpZvar_[a] = 0;
        pkZvar_[a] = 0;
        pkMax_[a] = 0;
        pkMidNumFrames_[a] = 0;
        numPkMidBasesByAnalog_[a] = 0;
        numBasesByAnalog_[a] = 0;
        numPulsesByAnalog_[a] = 0;
    }
    traceMetrics_.Reset();
}

template <unsigned int LaneWidth>
typename BasecallingMetricsAccumulator<LaneWidth>::SingleUnsignedIntegerMetric
BasecallingMetricsAccumulator<LaneWidth>::NumBases() const
{
    SingleUnsignedIntegerMetric ret(0);
    for (size_t a = 0; a < numAnalogs; ++a)
    {
        ret += numBasesByAnalog_[a];
    }
    return ret;
}

template <unsigned int LaneWidth>
typename BasecallingMetricsAccumulator<LaneWidth>::SingleUnsignedIntegerMetric
BasecallingMetricsAccumulator<LaneWidth>::NumPulses() const
{
    SingleUnsignedIntegerMetric ret(0);
    for (size_t a = 0; a < numAnalogs; ++a)
    {
        ret += numPulsesByAnalog_[a];
    }
    return ret;
}

template <unsigned int LaneWidth>
typename BasecallingMetricsAccumulator<LaneWidth>::SingleFloatMetric
BasecallingMetricsAccumulator<LaneWidth>::PulseWidth() const
{
    SingleFloatMetric ret(NumPulses() / numPulseFrames_);
    return Blend(isnan(ret), LaneArray<float>(0), ret);
}


template <unsigned int LaneWidth>
typename BasecallingMetricsAccumulator<LaneWidth>::AnalogFloatMetric
BasecallingMetricsAccumulator<LaneWidth>::PkmidMean() const
{
    AnalogFloatMetric ret;
    for (size_t ai = 0; ai < numAnalogs; ++ai)
    {
        ret[ai] = pkMidSignal_[ai] / pkMidNumFrames_[ai].AsFloat();
    }
    return ret;
}

template <unsigned int LaneWidth>
void BasecallingMetricsAccumulator<LaneWidth>::AddBaselinerStats(
        const InputBaselineStats& baselinerStats)
{
    BaselinerStatAccumulator<BaselinedTraceElement> reconstitutedStats(baselinerStats);
    traceMetrics_.BaselinerStatAccum().Merge(reconstitutedStats.BaselineFramesStats());
    traceMetrics_.AutocorrAccum().Merge(reconstitutedStats.BaselineSubtractedStats());
}

// This only keeps the most recent model, which should be fine
template <unsigned int LaneWidth>
void BasecallingMetricsAccumulator<LaneWidth>::AddModels(
        const InputModelsT& models)
{
    for (size_t ai = 0; ai < numAnalogs; ++ai)
    {
        for (size_t zi = 0; zi < LaneWidth; ++zi)
        {
            modelVariance_[ai][zi] = models.AnalogMode(ai).vars[zi];
            modelMean_[ai][zi] = models.AnalogMode(ai).means[zi];
        }
    }
}

template <unsigned int LaneWidth>
void BasecallingMetricsAccumulator<LaneWidth>::AddPulseDetectionMetrics(
        const PulseDetectionMetrics& pdMetrics)
{
    traceMetrics_.PulseDetectionScore() += LaneArray<float>(pdMetrics.viterbiScore);

    // TODO: collect PulseAccumulator BaselineStats, replace
    // 'BasecallingMetricsAccumulator::AddBaselinerStats'
}

template <unsigned int LaneWidth>
void BasecallingMetricsAccumulator<LaneWidth>::Count(
        const InputPulses& pulses,
        uint32_t numFrames)
{
    traceMetrics_.NumFrames() += numFrames;
    for (size_t zi = 0; zi < LaneWidth; ++zi)
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
                uint8_t pulseLabel = static_cast<uint8_t>(pulse->Label());
                numBaseFrames_[zi] += pulse->Width();

                if (!isnan(pulse->MidSignal()))
                {
                    numPkMidBasesByAnalog_[pulseLabel][zi]++;

                    // Inter-pulse moments (in terms of frames)
                    const uint16_t midWidth = static_cast<uint16_t>(pulse->Width() - 2);
                    // count (M0)
                    pkMidNumFrames_[pulseLabel][zi] += midWidth;
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

                numBasesByAnalog_[pulseLabel][zi]++;
            }
            prevprevPulse = prevPulse;
            prevPulse = pulse;
        }
        prevBasecallCache_[zi] = *prevPulse;
        prevprevBasecallCache_[zi] = *prevprevPulse;
    }
}

template <unsigned int LaneWidth>
void BasecallingMetricsAccumulator<LaneWidth>::FinalizeMetrics(
        bool realtimeActivityLabels, float frameRate)
{
    using Flt = typename BasecallingMetrics<LaneWidth>::Flt;
    LaneArray<float> nans(std::numeric_limits<Flt>::quiet_NaN());

    for (size_t pulseLabel = 0; pulseLabel < numAnalogs; pulseLabel++)
    {
        if (all(numPkMidBasesByAnalog_[pulseLabel] == 0))
        {
            // No bases called on channel.
            pkMidSignal_[pulseLabel] = std::numeric_limits<Flt>::quiet_NaN();
            bpZvar_[pulseLabel] = std::numeric_limits<Flt>::quiet_NaN();
            pkZvar_[pulseLabel] = std::numeric_limits<Flt>::quiet_NaN();
        }
        else if (all(numPkMidBasesByAnalog_[pulseLabel] < 2)
                || all(pkMidNumFrames_[pulseLabel] < 2))
        {
            bpZvar_[pulseLabel] = std::numeric_limits<Flt>::quiet_NaN();
            pkZvar_[pulseLabel] = std::numeric_limits<Flt>::quiet_NaN();
        }
        else
        {
            const auto& nf = pkMidNumFrames_[pulseLabel].AsFloat();
            const auto& nb = numPkMidBasesByAnalog_[pulseLabel].AsFloat();
            // Convert moments to interpulse variance
            bpZvar_[pulseLabel] = (bpZvar_[pulseLabel]
                                   - (pkMidSignal_[pulseLabel]
                                      * pkMidSignal_[pulseLabel]
                                      / nf))
                                  / (nf);
            // Bessel's correction with num bases, not frames
            bpZvar_[pulseLabel] *= nb / (nb - 1.0f);

            const auto baselineVariance =
                traceMetrics_.FrameBaselineVarianceDWS();

            bpZvar_[pulseLabel] -= baselineVariance / (nf / nb);

            pkZvar_[pulseLabel] = (pkZvar_[pulseLabel]
                                   - (pkMidSignal_[pulseLabel]
                                      * pkMidSignal_[pulseLabel]
                                      / nf))
                                  / (nf - 1.0f);

            // pkzvar up to this point contains total signal variance. We
            // subtract out interpulse variance and baseline variance to leave
            // intrapulse variance.
            pkZvar_[pulseLabel] -= bpZvar_[pulseLabel] + baselineVariance;

            const auto& modelIntraVars = modelVariance_[pulseLabel];
            // the model intrapulse variance still contains baseline variance,
            // remove before normalizing
            auto mask = (modelIntraVars < baselineVariance);
            pkZvar_[pulseLabel] /= modelIntraVars - baselineVariance;
            mask |= (pkZvar_[pulseLabel] < 0.0f);
            pkZvar_[pulseLabel] = Blend(
                    mask, LaneArray<float>(0.0f), pkZvar_[pulseLabel]);


            const auto& pkMid = pkMidSignal_[pulseLabel] / nf;
            mask = (pkMid < 0);
            bpZvar_[pulseLabel] /= (pkMid * pkMid);
            mask |= (bpZvar_[pulseLabel] < 0.0f);
            bpZvar_[pulseLabel] = Blend(mask, nans, bpZvar_[pulseLabel]);

            // No bases called on channel.
            mask = (numPkMidBasesByAnalog_[pulseLabel] == 0);
            pkMidSignal_[pulseLabel] = Blend(mask, nans, pkMidSignal_[pulseLabel]);
            // Not enough bases called on channel
            mask |= (numPkMidBasesByAnalog_[pulseLabel] < 2);
            mask |= (pkMidNumFrames_[pulseLabel] < 2);
            bpZvar_[pulseLabel] = Blend(mask, nans, bpZvar_[pulseLabel]);
            pkZvar_[pulseLabel] = Blend(mask, nans, pkZvar_[pulseLabel]);
        }
    }

    auto& pdScore = traceMetrics_.PulseDetectionScore();
    pdScore = pdScore / traceMetrics_.NumFrames().AsFloat();
    pdScore = Blend(isnan(pdScore), LaneArray<float>(0), pdScore);

    if (realtimeActivityLabels)
    {
        LabelBlock(frameRate);
    }
}

template class BasecallingMetricsAccumulator<laneSize>;

}}}     // namespace PacBio::Mongo::Data
