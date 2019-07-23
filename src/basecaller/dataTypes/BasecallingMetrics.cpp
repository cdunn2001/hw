
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

namespace PacBio {
namespace Mongo {
namespace Data {

template <unsigned int LaneWidth>
std::unique_ptr<AccumulationMethods<LaneWidth>> BasecallingMetrics<LaneWidth>::accumulator_;

template <unsigned int LaneWidth>
void BasecallingMetrics<LaneWidth>::Configure(std::unique_ptr<AccumulationMethods<LaneWidth>> accumulator)
{
    accumulator_ = std::move(accumulator);
}

template <unsigned int LaneWidth>
void BasecallingMetrics<LaneWidth>::Initialize()
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
    }
    for (size_t a = 0; a < BasecallingMetrics<LaneWidth>::NumAnalogs; ++a)
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
    //traceMetrics_.Initialize();
}

template <unsigned int LaneWidth>
typename BasecallingMetrics<LaneWidth>::AnalogFloatMetric BasecallingMetrics<LaneWidth>::PkmidMean() const
{
    BasecallingMetrics<LaneWidth>::AnalogFloatMetric ret;
    for (size_t ai = 0; ai < NumAnalogs; ++ai)
    {
        for (size_t zi = 0; zi < LaneWidth; ++zi)
        {
            if (PkMidNumFrames()[ai][zi] > 0)
            {
                ret[zi][zi] = PkMidSignal()[ai][zi] / PkMidNumFrames()[ai][zi];
            }
            else
            {
                ret[zi][zi] = std::numeric_limits<Flt>::quiet_NaN();
            }
        }
    }
    return ret;
}

template <unsigned int LaneWidth>
void BasecallingMetrics<LaneWidth>::AddBaselineStats(
        const BasecallingMetrics<LaneWidth>::InputBaselineStats& baselineStats)
{
    accumulator_->AddBaselineStats(*this, baselineStats);
}

// TODO: this could be handled by a StatsAccumulator
template <unsigned int LaneWidth>
void FullAccumulationMethods<LaneWidth>::AddBaselineStats(
        BasecallingMetrics<LaneWidth>& bm,
        const typename BasecallingMetrics<LaneWidth>::InputBaselineStats& baselineStats)
{
    for (size_t zi = 0; zi < LaneWidth; ++zi)
    {
        bm.traceMetrics_.FrameBaselineM0DWS()[zi] += baselineStats.m0_[zi];
        bm.traceMetrics_.FrameBaselineM1DWS()[zi] += baselineStats.m1_[zi];
        bm.traceMetrics_.FrameBaselineM2DWS()[zi] += baselineStats.m2_[zi];
    }
}

template <unsigned int LaneWidth>
void BasecallingMetrics<LaneWidth>::Count(
        const BasecallingMetrics<LaneWidth>::InputBasecalls& bases,
        uint32_t numFrames)
{
    accumulator_->Count(*this, bases, numFrames);
}

template <unsigned int LaneWidth>
void FullAccumulationMethods<LaneWidth>::Count(
        BasecallingMetrics<LaneWidth>& bm,
        const typename BasecallingMetrics<LaneWidth>::InputBasecalls& bases,
        uint32_t numFrames)
{
    using Basecall = typename BasecallingMetrics<LaneWidth>::Basecall;

    for (size_t zi = 0; zi < LaneWidth; ++zi)
    {
        bm.traceMetrics_.NumFrames()[zi] += numFrames;
        // TODO: fix this for the first block (currently initialized to A:0-0
        const Basecall* prevBasecall = &bm.prevBasecallCache_[zi];
        const Basecall* prevprevBasecall = &bm.prevprevBasecallCache_[zi];
        for (size_t bi = 0; bi < bases.size(zi); ++bi)
        {
            const Basecall* base = &bases(zi, bi);

            uint8_t pulseLabel = static_cast<uint8_t>(base->GetPulse().Label());
            bm.numPulseFrames_[zi] += base->GetPulse().Width();
            bm.numPulsesByAnalog_[pulseLabel][zi]++;
            bm.pkMax_[pulseLabel][zi] = std::max(bm.pkMax_[pulseLabel][zi],
                                              base->GetPulse().MaxSignal());

            // TODO: replace this size check...
            if (bm.prevBasecallCache_.size() > 0)
            {
                const auto& prevPulse = prevBasecall->GetPulse();
                const auto& curPulse = base->GetPulse();

                if (prevPulse.Label() == curPulse.Label())
                {
                    bm.numPulseLabelStutters_[zi]++;
                }

                bool abutted = (curPulse.Start() == prevPulse.Stop());

                if (abutted && prevPulse.Label() != curPulse.Label())
                {
                    bm.numHalfSandwiches_[zi]++;
                }

                // TODO: replace this size check...
                if (bm.prevprevBasecallCache_.size() > 0)
                {
                    const auto& prevprevPulse = prevprevBasecall->GetPulse();
                    bool prevAbutted = (prevPulse.Start() == prevprevPulse.Stop());
                    if (prevAbutted && abutted
                            && prevprevPulse.Label() == curPulse.Label()
                            && prevprevPulse.Label() != prevPulse.Label())
                    {
                        bm.numSandwiches_[zi]++;
                    }
                }
            }

            if (!base->IsNoCall())
            {
                uint8_t baseLabel = static_cast<uint8_t>(base->Base());
                bm.numBaseFrames_[zi] += base->GetPulse().Width();

                if (!isnan(base->GetPulse().MidSignal()))
                {
                    bm.numPkMidBasesByAnalog_[baseLabel][zi]++;

                    // TODO: These moments are already recorded in
                    // TraceMetrics, should pkzvar and bpzvar be in
                    // traceAnalysis metrics?

                    // Inter-pulse moments (in terms of frames)
                    const uint16_t midWidth = static_cast<uint16_t>(base->GetPulse().Width() - 2);
                    // count (M0)
                    bm.pkMidNumFrames_[baseLabel][zi] += midWidth;
                    // sum of signals (M1)
                    bm.pkMidSignal_[baseLabel][zi] += base->GetPulse().MidSignal()
                                                 * midWidth;
                    // sum of square of signals (M2)
                    bm.bpZvar_[baseLabel][zi] += base->GetPulse().MidSignal()
                                            * base->GetPulse().MidSignal()
                                            * midWidth;

                    // Intra-pulse M2
                    bm.pkZvar_[baseLabel][zi] += base->GetPulse().SignalM2();
                }

                bm.numBasesByAnalog_[baseLabel][zi]++;
            }
            prevprevBasecall = prevBasecall;
            prevBasecall = base;
        }
        bm.prevBasecallCache_[zi] = *prevBasecall;
        bm.prevprevBasecallCache_[zi] = *prevprevBasecall;
    }
}


template <unsigned int LaneWidth>
void BasecallingMetrics<LaneWidth>::FinalizeMetrics()
{
    accumulator_->FinalizeMetrics(*this);
}

template <unsigned int LaneWidth>
void FullAccumulationMethods<LaneWidth>::FinalizeMetrics(BasecallingMetrics<LaneWidth>& bm)
{
    using Flt = typename BasecallingMetrics<LaneWidth>::Flt;

    // Transcribe pulse detection scores to trace analysis metrics.
    // Also, transcribe autocorrelation.

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

    for (size_t baseLabel = 0; baseLabel < bm.NumAnalogs; baseLabel++)
    {
        for (size_t zi = 0; zi < LaneWidth; ++zi)
        {
            if (bm.numPkMidBasesByAnalog_[baseLabel][zi] == 0)
            {
                // No bases called on channel.
                bm.pkMidSignal_[baseLabel][zi] = std::numeric_limits<Flt>::quiet_NaN();
                bm.bpZvar_[baseLabel][zi] = std::numeric_limits<Flt>::quiet_NaN();
                bm.pkZvar_[baseLabel][zi] = std::numeric_limits<Flt>::quiet_NaN();
            } else if (bm.numPkMidBasesByAnalog_[baseLabel][zi] < 2
                    || bm.pkMidNumFrames_[baseLabel][zi] < 2)
            {
                bm.bpZvar_[baseLabel][zi] = std::numeric_limits<Flt>::quiet_NaN();
                bm.pkZvar_[baseLabel][zi] = std::numeric_limits<Flt>::quiet_NaN();
            }
            else
            {
                // Convert moments to interpulse variance
                bm.bpZvar_[baseLabel][zi] = (bm.bpZvar_[baseLabel][zi]
                                          - (bm.pkMidSignal_[baseLabel][zi]
                                             * bm.pkMidSignal_[baseLabel][zi]
                                             / bm.pkMidNumFrames_[baseLabel][zi]))
                                         / (bm.pkMidNumFrames_[baseLabel][zi]);
                // Bessel's correction with num bases, not frames
                bm.bpZvar_[baseLabel][zi] *= bm.numPkMidBasesByAnalog_[baseLabel][zi]
                                      / (bm.numPkMidBasesByAnalog_[baseLabel][zi] - 1);

                const auto baselineVariance =
                    bm.traceMetrics_.FrameBaselineVarianceDWS()[zi];


                bm.bpZvar_[baseLabel][zi] -= baselineVariance
                                        / (bm.pkMidNumFrames_[baseLabel][zi]
                                           / bm.numPkMidBasesByAnalog_[baseLabel][zi]);

                bm.pkZvar_[baseLabel][zi] = (bm.pkZvar_[baseLabel][zi]
                                      - (bm.pkMidSignal_[baseLabel][zi]
                                         * bm.pkMidSignal_[baseLabel][zi]
                                         / bm.pkMidNumFrames_[baseLabel][zi]))
                                      / (bm.pkMidNumFrames_[baseLabel][zi] - 1);

                // pkzvar up to this point contains total signal variance. We
                // subtract out interpulse variance and baseline variance to leave
                // intrapulse variance.
                bm.pkZvar_[baseLabel][zi] -= bm.bpZvar_[baseLabel][zi] + baselineVariance;


                /* TODO renable when the analog modes are available
                const auto& modelIntraVars =
                    bm.traceMetrics_.Analog()[baseLabel].SignalCovar()[zi];

                // the model intrapulse variance still contains baseline variance,
                // remove before normalizing
                if (modelIntraVars > baselineVariance)
                    bm.pkZvar_[baseLabel][zi] /= modelIntraVars - baselineVariance;
                else
                    bm.pkZvar_[baseLabel][zi] = 0.0f;
                bm.pkZvar_[baseLabel][zi] = std::max(bm.pkZvar_[baseLabel][zi], 0.0f);
                */


                const auto pkMid = bm.pkMidSignal_[baseLabel][zi]
                                 / bm.pkMidNumFrames_[baseLabel][zi];
                if (pkMid > 0)
                    bm.bpZvar_[baseLabel][zi] = std::max(
                        bm.bpZvar_[baseLabel][zi] / (pkMid * pkMid), 0.0f);
                else
                    bm.bpZvar_[baseLabel][zi] = std::numeric_limits<Flt>::quiet_NaN();
            }
        }
    }
}


template <unsigned int LaneWidth>
void SimpleAccumulationMethods<LaneWidth>::Count(
        BasecallingMetrics<LaneWidth>& bm,
        const typename BasecallingMetrics<LaneWidth>::InputBasecalls& bases,
        uint32_t numFrames)
{
    using Basecall = typename BasecallingMetrics<LaneWidth>::Basecall;

    (void)numFrames;
    for (size_t zi = 0; zi < LaneWidth; ++zi)
    {
        for (size_t bi = 0; bi < bases.size(zi); ++bi)
        {
            const Basecall& base = bases(zi, bi);
            uint8_t pulseLabel = static_cast<uint8_t>(base.GetPulse().Label());
            bm.numPulsesByAnalog_[pulseLabel][zi]++;

            if (!base.IsNoCall())
            {
                uint8_t baseLabel = static_cast<uint8_t>(base.Base());
                bm.numBasesByAnalog_[baseLabel][zi]++;
            }
        }
    }
}

template class BasecallingMetrics<laneSize>;
template class FullAccumulationMethods<laneSize>;
template class SimpleAccumulationMethods<laneSize>;

}}}     // namespace PacBio::Mongo::Data
