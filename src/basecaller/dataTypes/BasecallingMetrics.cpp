
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
void BasecallingMetrics<LaneWidth>::Initialize()
{
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
}

template <unsigned int LaneWidth>
void BasecallingMetrics<LaneWidth>::Count(
        const BasecallingMetrics<LaneWidth>::InputBasecalls& bases)
{
    for (size_t zi = 0; zi < LaneWidth; ++zi)
    {
        // TODO: fix this for the first block
        const Basecall* prevBasecall = &prevBasecallCache[zi];
        const Basecall* prevprevBasecall = &prevprevBasecallCache[zi];
        for (size_t bi = 0; bi < bases.size(zi); ++bi)
        {
            const Basecall* base = &bases(zi, bi);

            uint8_t pulseLabel = static_cast<uint8_t>(base->GetPulse().Label());
            numPulseFrames_[zi] += base->GetPulse().Width();
            numPulsesByAnalog_[pulseLabel][zi]++;
            pkMax_[pulseLabel][zi] = std::max(pkMax_[pulseLabel][zi],
                                              base->GetPulse().MaxSignal());

            if (prevBasecallCache.size() > 0)
            {
                const auto& prevPulse = prevBasecall->GetPulse();
                const auto& curPulse = base->GetPulse();

                if (prevPulse.Label() == curPulse.Label())
                {
                    ++numPulseLabelStutters_[zi];
                }

                bool abutted = (curPulse.Start() == prevPulse.Stop());

                if (abutted && prevPulse.Label() != curPulse.Label())
                {
                    ++numHalfSandwiches_[zi];
                }

                if (prevprevBasecallCache.size() > 0)
                {
                    const auto& prevprevPulse = prevprevBasecall->GetPulse();
                    bool prevAbutted = (prevPulse.Start() == prevprevPulse.Stop());
                    if (prevAbutted && abutted
                            && prevprevPulse.Label() == curPulse.Label()
                            && prevprevPulse.Label() != prevPulse.Label())
                    {
                        ++numSandwiches_[zi];
                    }
                }
            }

            if (!base->IsNoCall())
            {
                uint8_t baseLabel = static_cast<uint8_t>(base->Base());
                numBaseFrames_[zi] += base->GetPulse().Width();

                if (!isnan(base->GetPulse().MidSignal()))
                {
                    numPkMidBasesByAnalog_[baseLabel][zi]++;

                    // Inter-pulse moments (in terms of frames)
                    const uint16_t midWidth = static_cast<uint16_t>(base->GetPulse().Width() - 2);
                    // count (M0)
                    pkMidNumFrames_[baseLabel][zi] += midWidth;
                    // sum of signals (M1)
                    pkMidSignal_[baseLabel][zi] += base->GetPulse().MidSignal()
                                              * midWidth;
                    // sum of square of signals (M2)
                    bpZvar_[baseLabel][zi] += base->GetPulse().MidSignal()
                                            * base->GetPulse().MidSignal()
                                            * midWidth;

                    // Intra-pulse M2
                    pkZvar_[baseLabel][zi] += base->GetPulse().SignalM2();
                }

                numBasesByAnalog_[baseLabel][zi]++;
            }
            prevprevBasecall = prevBasecall;
            prevBasecall = base;
        }
        prevBasecallCache[zi] = *prevBasecall;
        prevprevBasecallCache[zi] = *prevprevBasecall;
    }
}


template <unsigned int LaneWidth>
void BasecallingMetrics<LaneWidth>::FinalizeMetrics()
{
    // Transcribe pulse detection scores to trace analysis metrics.
    // Also, transcribe autocorrelation.
/*
    assert(this->StartFrame() == pulseDetectMetrics_.Start());
    assert(this->StopFrame() == pulseDetectMetrics_.Stop());
    assert(this->StartFrame() == autocorr_.Start());
    assert(this->StopFrame() == autocorr_.Stop());
*/

/* TODO harvest these from somewhere (pdmetrics and autocorr are collected over chunks)
 * then re-enable (but merge with the loop below)
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

    // Transcribe pulse detector's DWS baseline statistics to trace
    // analysis metrics.
    const auto& bsc = pdMetrics.baselineStats;
    const auto& count = MakeUnion(bsc.Count());
    const auto& mean = MakeUnion(bsc.Mean());
    const auto& var = MakeUnion(bsc.Variance());
    for (unsigned int z = 0; z < laneSize; ++z)
    {
        auto& tm = laneMetrics_[z].TraceMetrics();
        tm.NumFramesBaseline() = round_cast<uint16_t>(count[z]);
        tm.FrameBaselineDWS() = mean[z];
        tm.FrameBaselineVarianceDWS() = var[z];
    }
*/

    for (size_t baseLabel = 0; baseLabel < NumAnalogs; baseLabel++)
    {
        for (size_t zi = 0; zi < LaneWidth; ++zi)
        {
            if (numPkMidBasesByAnalog_[baseLabel][zi] == 0)
            {
                // No bases called on channel.
                pkMidSignal_[baseLabel][zi] = std::numeric_limits<Flt>::quiet_NaN();
                bpZvar_[baseLabel][zi] = std::numeric_limits<Flt>::quiet_NaN();
                pkZvar_[baseLabel][zi] = std::numeric_limits<Flt>::quiet_NaN();
            } else if (numPkMidBasesByAnalog_[baseLabel][zi] < 2
                    || pkMidNumFrames_[baseLabel][zi] < 2)
            {
                bpZvar_[baseLabel][zi] = std::numeric_limits<Flt>::quiet_NaN();
                pkZvar_[baseLabel][zi] = std::numeric_limits<Flt>::quiet_NaN();
            }
            else
            {
                // Convert moments to interpulse variance
                bpZvar_[baseLabel][zi] = (bpZvar_[baseLabel][zi]
                                          - (pkMidSignal_[baseLabel][zi]
                                             * pkMidSignal_[baseLabel][zi]
                                             / pkMidNumFrames_[baseLabel][zi]))
                                         / (pkMidNumFrames_[baseLabel][zi]);
                // Bessel's correction with num bases, not frames
                bpZvar_[baseLabel][zi] *= numPkMidBasesByAnalog_[baseLabel][zi]
                                      / (numPkMidBasesByAnalog_[baseLabel][zi] - 1);

                const auto baselineVariance =
                    traceMetrics_.FrameBaselineVarianceDWS()[zi];


                bpZvar_[baseLabel][zi] -= baselineVariance
                                        / (pkMidNumFrames_[baseLabel][zi]
                                           / numPkMidBasesByAnalog_[baseLabel][zi]);

                pkZvar_[baseLabel][zi] = (pkZvar_[baseLabel][zi]
                                      - (pkMidSignal_[baseLabel][zi]
                                         * pkMidSignal_[baseLabel][zi]
                                         / pkMidNumFrames_[baseLabel][zi]))
                                      / (pkMidNumFrames_[baseLabel][zi] - 1);

                // pkzvar up to this point contains total signal variance. We
                // subtract out interpulse variance and baseline variance to leave
                // intrapulse variance.
                pkZvar_[baseLabel][zi] -= bpZvar_[baseLabel][zi] + baselineVariance;

                /*
                const auto& modelIntraVars =
                    traceMetrics_.Analog()[baseLabel].SignalCovar()[zi];


                // the model intrapulse variance still contains baseline variance,
                // remove before normalizing
                if (modelIntraVars > baselineVariance)
                    pkZvar_[baseLabel][zi] /= modelIntraVars - baselineVariance;
                else
                    pkZvar_[baseLabel][zi] = 0.0f;
                pkZvar_[baseLabel][zi] = std::max(pkZvar_[baseLabel][zi], 0.0f);
                */

                const auto pkMid = pkMidSignal_[baseLabel][zi]
                                 / pkMidNumFrames_[baseLabel][zi];
                if (pkMid > 0)
                    bpZvar_[baseLabel][zi] = std::max(
                        bpZvar_[baseLabel][zi] / (pkMid * pkMid), 0.0f);
                else
                    bpZvar_[baseLabel][zi] = std::numeric_limits<Flt>::quiet_NaN();
            }
        }
    }


/* TODO: Move over activity labeler, re-enable
    if (realtimeActivityLabels_)
    {
        for (unsigned int z = 0; z < laneSize; ++z)
        {
            const auto& activityLabel = ActivityLabeler::LabelBlock(
                laneMetrics_[z], frameRate_);
            laneMetrics_[z].ActivityLabel(activityLabel);
        }
    }
*/
}



MinimalBasecallingMetrics& MinimalBasecallingMetrics::Count(const MinimalBasecallingMetrics::Basecall& base)
{
    uint8_t pulseLabel = static_cast<uint8_t>(base.GetPulse().Label());
    numPulsesByAnalog_[pulseLabel]++;

    if (!base.IsNoCall())
    {
        uint8_t baseLabel = static_cast<uint8_t>(base.Base());
        numBasesByAnalog_[baseLabel]++;
    }

    return *this;
}

template class BasecallingMetrics<laneSize>;

}}}     // namespace PacBio::Mongo::Data
