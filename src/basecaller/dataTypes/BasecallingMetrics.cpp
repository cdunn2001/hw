
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
//  Defines members of class BasecallingMetrics.

#include "BasecallingMetrics.h"
#include <common/MongoConstants.h>
#include <common/BlockActivityLabels.h>
#include <common/TrainedCartParams.h>

namespace PacBio {
namespace Mongo {
namespace Data {

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

template <unsigned int LaneWidth>
void BasecallingMetricsAccumulator<LaneWidth>::LabelBlock(float frameRate)
{
    // Calculated in the accessor, so caching here:
    const auto& stdDevAll = traceMetrics_.FrameBaselineSigmaDWS();
    const auto& numBases = NumBases();
    const auto& numPulses = NumPulses();
    const auto& pulseWidth = PulseWidth();
    const auto& pkmid = PkmidMean();
    for (size_t zmw = 0; zmw < laneSize; ++zmw)
    {
        // TODO: replace with non-dynamic CudaArray...
        std::vector<float> features;
        features.resize(ActivityLabeler::NUM_FEATURES, 0.0f);
        const float seconds = traceMetrics_.NumFrames()[zmw] / frameRate;

        features[ActivityLabeler::PULSERATE] = numPulses[zmw] / seconds;
        features[ActivityLabeler::SANDWICHRATE] = numPulses[zmw] > 0 ?
            static_cast<float>(numSandwiches_[zmw]) / numPulses[zmw] : 0.0f;

        const float hswr = (numPulses[zmw] > 0) ?
            static_cast<float>(numHalfSandwiches_[zmw]) / numPulses[zmw] : 0.0f;
        const float hswrExp = std::min(
                ActivityLabeler::TrainedCart::maxAcceptableHalfsandwichRate,
                evaluatePolynomial(ActivityLabeler::TrainedCart::hswCurve,
                                   features[ActivityLabeler::PULSERATE]));
        features[ActivityLabeler::LOCALHSWRATENORM] = hswr - hswrExp;

        features[ActivityLabeler::VITERBISCORE] = traceMetrics_.PulseDetectionScore()[zmw];
        features[ActivityLabeler::MEANPULSEWIDTH] = pulseWidth[zmw];
        features[ActivityLabeler::LABELSTUTTERRATE] = (numPulses[zmw] > 0) ?
            static_cast<float>(numPulseLabelStutters_[zmw]) / numPulses[zmw] : 0.0f;

        const float stdDev = stdDevAll[zmw];

        const auto& pkbases = numBasesByAnalog_;
        const float totBases = static_cast<float>(numBases[zmw]);

        std::vector<float> relamps;
        // these amps aren't relative, and therefore could be misleading. Scoping
        // maxamp to prevent misuse
        {
            std::vector<float> amps = {
                modelMean_[0][zmw],
                modelMean_[1][zmw],
                modelMean_[2][zmw],
                modelMean_[3][zmw]};

            const float maxamp = *std::max_element(amps.begin(), amps.end());
            std::transform(amps.begin(), amps.end(), std::back_inserter(relamps),
                           [maxamp](float amp) { return amp/maxamp; });
        }
        // This never changes, could be a member of an object
        const float minamp = *std::min_element(relamps.begin(), relamps.end());

        for (size_t i = 0; i < NumAnalogs; ++i)
        {
            if (!std::isnan(pkmid[i][zmw]) && pkbases[i][zmw] > 0 && stdDev > 0
                    && relamps[i] > 0 && totBases > 0)
            {
                features[ActivityLabeler::BLOCKLOWSNR] += pkmid[i][zmw] / stdDev * pkbases[i][zmw]
                                         / totBases * minamp / relamps[i];
            }
        }

        for (size_t aI = 0; aI < NumAnalogs && stdDev > 0; ++aI)
        {
            features[ActivityLabeler::MAXPKMAXNORM] = std::fmax(
                features[ActivityLabeler::MAXPKMAXNORM],
                (pkMax_[aI][zmw] - pkmid[aI][zmw]) / stdDev);
        }

        features[ActivityLabeler::AUTOCORRELATION] = traceMetrics_.Autocorrelation()[zmw];

        int lowAnalogIndex = std::distance(
            relamps.begin(),
            std::min_element(relamps.begin(), relamps.end()));

        for (const auto& val : bpZvar_)
        {
            features[ActivityLabeler::BPZVARNORM] += std::isnan(val[zmw]) ? 0.0f : val[zmw];
        }
        const auto& lowbp = bpZvar_[lowAnalogIndex][zmw];
        features[ActivityLabeler::BPZVARNORM] -=  std::isnan(lowbp) ? 0.0f : lowbp;
        features[ActivityLabeler::BPZVARNORM] /= 3.0f;

        for (const auto& val : pkZvar_)
        {
            features[ActivityLabeler::PKZVARNORM] += std::isnan(val[zmw]) ? 0.0f : val[zmw];
        }
        const auto& lowpk = pkZvar_[lowAnalogIndex][zmw];
        features[ActivityLabeler::PKZVARNORM] -=  std::isnan(lowpk) ? 0.0f : lowpk;
        features[ActivityLabeler::PKZVARNORM] /= 3.0f;

        for (size_t i = 0; i < features.size(); ++i)
        {
            assert(!std::isnan(features[i]));
        }

        size_t current = 0;
        while (ActivityLabeler::TrainedCart::feature[current] >= 0)
        {
            if (features[ActivityLabeler::TrainedCart::feature[current]]
                    <= ActivityLabeler::TrainedCart::threshold[current])
            {
                current = ActivityLabeler::TrainedCart::childrenLeft[current];
            }
            else
            {
                current = ActivityLabeler::TrainedCart::childrenRight[current];
            }
        }
        activityLabel_[zmw] = static_cast<ActivityLabeler::HQRFPhysicalState>(ActivityLabeler::TrainedCart::value[current]);
    }
}

template <unsigned int LaneWidth>
void BasecallingMetricsAccumulator<LaneWidth>::PopulateBasecallingMetrics(
        BasecallingMetricsT& metrics)
{
    const auto& numPulses = NumPulses();
    const auto& numBases = NumBases();
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
    }
    for (size_t a = 0; a < NumAnalogs; ++a)
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
void BasecallingMetricsAccumulator<LaneWidth>::Initialize()
{
    // These metrics serve as accumulators, zero-initialization of some sort is
    // necessary
    for (size_t z = 0; z < LaneWidth; ++z)
    {
        numPulseFrames_[z] = 0;
        numBaseFrames_[z] = 0;
        numSandwiches_[z] = 0;
        numHalfSandwiches_[z] = 0;
        numPulseLabelStutters_[z] = 0;
        prevBasecallCache_[z] = Pulse().Start(0).Width(0).Label(Pulse::NucleotideLabel::NONE);
        prevprevBasecallCache_[z] = Pulse().Start(0).Width(0).Label(Pulse::NucleotideLabel::NONE);
    }
    for (size_t a = 0; a < NumAnalogs; ++a)
    {
        for (size_t z = 0; z < LaneWidth; ++z)
        {
            pkMidSignal_[a][z] = 0;
            bpZvar_[a][z] = 0;
            pkZvar_[a][z] = 0;
            pkMax_[a][z] = 0;
            pkMidNumFrames_[a][z] = 0;
            numPkMidBasesByAnalog_[a][z] = 0;
            numBasesByAnalog_[a][z] = 0;
            numPulsesByAnalog_[a][z] = 0;
        }
    }
    traceMetrics_.Initialize();
}

template <unsigned int LaneWidth>
typename BasecallingMetricsAccumulator<LaneWidth>::SingleUnsignedIntegerMetric
BasecallingMetricsAccumulator<LaneWidth>::NumBases() const
{
    SingleUnsignedIntegerMetric ret;
    for (size_t z = 0; z < LaneWidth; ++z)
        ret[z] = 0;
    for (size_t a = 0; a < NumAnalogs; ++a)
    {
        for (size_t z = 0; z < LaneWidth; ++z)
        {
            ret[z] += numBasesByAnalog_[a][z];
        }
    }
    return ret;
}

template <unsigned int LaneWidth>
typename BasecallingMetricsAccumulator<LaneWidth>::SingleUnsignedIntegerMetric
BasecallingMetricsAccumulator<LaneWidth>::NumPulses() const
{
    SingleUnsignedIntegerMetric ret;
    for (size_t z = 0; z < LaneWidth; ++z)
        ret[z] = 0;
    for (size_t a = 0; a < NumAnalogs; ++a)
    {
        for (size_t z = 0; z < LaneWidth; ++z)
        {
            ret[z] += numPulsesByAnalog_[a][z];
        }
    }
    return ret;
}

template <unsigned int LaneWidth>
typename BasecallingMetricsAccumulator<LaneWidth>::SingleFloatMetric
BasecallingMetricsAccumulator<LaneWidth>::PulseWidth() const
{
    SingleFloatMetric ret;
    const auto& numPulses = NumPulses();
    for (size_t z = 0; z < LaneWidth; ++z)
        ret[z] = numPulses[z];
    for (size_t z = 0; z < LaneWidth; ++z)
    {
        if (ret[z] > 0)
            ret[z] /= numPulseFrames_[z];
    }
    return ret;
}


template <unsigned int LaneWidth>
typename BasecallingMetricsAccumulator<LaneWidth>::AnalogFloatMetric
BasecallingMetricsAccumulator<LaneWidth>::PkmidMean() const
{
    AnalogFloatMetric ret;
    for (size_t ai = 0; ai < NumAnalogs; ++ai)
    {
        for (size_t zi = 0; zi < LaneWidth; ++zi)
        {
            if (pkMidNumFrames_[ai][zi] > 0)
            {
                ret[ai][zi] = pkMidSignal_[ai][zi] / pkMidNumFrames_[ai][zi];
            }
            else
            {
                ret[ai][zi] = std::numeric_limits<Flt>::quiet_NaN();
            }
        }
    }
    return ret;
}

// TODO: this could be handled by a StatsAccumulator
template <unsigned int LaneWidth>
void BasecallingMetricsAccumulator<LaneWidth>::AddBaselineStats(
        const InputBaselineStats& baselineStats)
{
    /* TODO: segfaulting here, are baselineStats initialized?
    for (size_t zi = 0; zi < LaneWidth; ++zi)
    {
        traceMetrics_.FrameBaselineM0DWS()[zi] += baselineStats.m0_[zi];
        traceMetrics_.FrameBaselineM1DWS()[zi] += baselineStats.m1_[zi];
        traceMetrics_.FrameBaselineM2DWS()[zi] += baselineStats.m2_[zi];
    }
    */
}

// TODO: This only keeps the most recent model. Should this accumulate
// instead?
template <unsigned int LaneWidth>
void BasecallingMetricsAccumulator<LaneWidth>::AddModels(
        const InputModelsT& models)
{
    for (size_t ai = 0; ai < NumAnalogs; ++ai)
    {
        for (size_t zi = 0; zi < LaneWidth; ++zi)
        {
            modelVariance_[ai][zi] = models.AnalogMode(ai).vars[zi];
            modelMean_[ai][zi] = models.AnalogMode(ai).means[zi];
        }
    }
}

template <unsigned int LaneWidth>
void BasecallingMetricsAccumulator<LaneWidth>::Count(
        const InputPulses& pulses,
        uint32_t numFrames)
{
    for (size_t zi = 0; zi < LaneWidth; ++zi)
    {
        traceMetrics_.NumFrames()[zi] += numFrames;
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

    for (size_t pulseLabel = 0; pulseLabel < NumAnalogs; pulseLabel++)
    {
        for (size_t zi = 0; zi < LaneWidth; ++zi)
        {
            if (numPkMidBasesByAnalog_[pulseLabel][zi] == 0)
            {
                // No bases called on channel.
                pkMidSignal_[pulseLabel][zi] = std::numeric_limits<Flt>::quiet_NaN();
                bpZvar_[pulseLabel][zi] = std::numeric_limits<Flt>::quiet_NaN();
                pkZvar_[pulseLabel][zi] = std::numeric_limits<Flt>::quiet_NaN();
            } else if (numPkMidBasesByAnalog_[pulseLabel][zi] < 2
                    || pkMidNumFrames_[pulseLabel][zi] < 2)
            {
                bpZvar_[pulseLabel][zi] = std::numeric_limits<Flt>::quiet_NaN();
                pkZvar_[pulseLabel][zi] = std::numeric_limits<Flt>::quiet_NaN();
            }
            else
            {
                // Convert moments to interpulse variance
                bpZvar_[pulseLabel][zi] = (bpZvar_[pulseLabel][zi]
                                          - (pkMidSignal_[pulseLabel][zi]
                                             * pkMidSignal_[pulseLabel][zi]
                                             / pkMidNumFrames_[pulseLabel][zi]))
                                         / (pkMidNumFrames_[pulseLabel][zi]);
                // Bessel's correction with num bases, not frames
                bpZvar_[pulseLabel][zi] *= numPkMidBasesByAnalog_[pulseLabel][zi]
                                      / (numPkMidBasesByAnalog_[pulseLabel][zi] - 1);

                const auto baselineVariance =
                    traceMetrics_.FrameBaselineVarianceDWS()[zi];


                bpZvar_[pulseLabel][zi] -= baselineVariance
                                        / (pkMidNumFrames_[pulseLabel][zi]
                                           / numPkMidBasesByAnalog_[pulseLabel][zi]);

                pkZvar_[pulseLabel][zi] = (pkZvar_[pulseLabel][zi]
                                      - (pkMidSignal_[pulseLabel][zi]
                                         * pkMidSignal_[pulseLabel][zi]
                                         / pkMidNumFrames_[pulseLabel][zi]))
                                      / (pkMidNumFrames_[pulseLabel][zi] - 1);

                // pkzvar up to this point contains total signal variance. We
                // subtract out interpulse variance and baseline variance to leave
                // intrapulse variance.
                pkZvar_[pulseLabel][zi] -= bpZvar_[pulseLabel][zi] + baselineVariance;

                const auto& modelIntraVars = modelVariance_[pulseLabel][zi];
                // the model intrapulse variance still contains baseline variance,
                // remove before normalizing
                if (modelIntraVars > baselineVariance)
                    pkZvar_[pulseLabel][zi] /= modelIntraVars - baselineVariance;
                else
                    pkZvar_[pulseLabel][zi] = 0.0f;
                pkZvar_[pulseLabel][zi] = std::max(pkZvar_[pulseLabel][zi], 0.0f);


                const auto pkMid = pkMidSignal_[pulseLabel][zi]
                                 / pkMidNumFrames_[pulseLabel][zi];
                if (pkMid > 0)
                    bpZvar_[pulseLabel][zi] = std::max(
                        bpZvar_[pulseLabel][zi] / (pkMid * pkMid), 0.0f);
                else
                    bpZvar_[pulseLabel][zi] = std::numeric_limits<Flt>::quiet_NaN();
            }
        }
    }

/* TODO harvest these when available then re-enable (but merge with the loop
 * below)
    const auto& pdMetrics = pulseDetectMetrics_.Value();
    const auto& autocorr = autocorr_.Value().Autocorrelation();
    for (unsigned int z = 0; z < laneSize; ++z)
    {
        auto& tm = laneMetrics_[z].TraceMetrics();
        const auto& nf = laneMetrics_[z].TraceMetrics().NumFrames();
        // Normalize score by number of frames.
        tm.PulseDetectionScore((nf != 0) ? MakeUnion(pdMetrics.score)[z] / nf : 0);
        const auto acz = MakeUnion(autocorr)[z];
        tm.Autocorrelation(isnan(acz) ? 0 : acz);
    }
*/
    if (realtimeActivityLabels)
    {
        LabelBlock(frameRate);
    }
}


/*
template <unsigned int LaneWidth>
void SimpleAccumulationMethods<LaneWidth>::Count(
        BasecallingMetrics<LaneWidth>& bm,
        const typename BasecallingMetrics<LaneWidth>::InputPulses& pulses,
        uint32_t numFrames)
{
    (void)numFrames;
    for (size_t zi = 0; zi < LaneWidth; ++zi)
    {
        for (size_t bi = 0; bi < pulses.size(zi); ++bi)
        {
            const Pulse& pulse = pulses(zi, bi);
            uint8_t pulseLabel = static_cast<uint8_t>(pulse.Label());
            bm.numPulsesByAnalog_[pulseLabel][zi]++;
            if (!pulse.IsReject())
            {
                bm.numBasesByAnalog_[pulseLabel][zi]++;
            }
        }
    }
}
*/

template class BasecallingMetricsAccumulator<laneSize>;
template class BasecallingMetrics<laneSize>;

}}}     // namespace PacBio::Mongo::Data
