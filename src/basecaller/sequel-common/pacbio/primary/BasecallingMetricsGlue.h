// Copyright (c) 2017, Pacific Biosciences of California, Inc.
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
/// \brief   Code used to bridge metrics from the basecaller to PPA, which use slightly different objects
///

#ifndef SEQUELACQUISITION_BASECALLINGMETRICSGLUE_H_H
#define SEQUELACQUISITION_BASECALLINGMETRICSGLUE_H_H

#include <boost/math/constants/constants.hpp>

#include <pacbio/primary/BasecallingMetrics.h>
#include <pacbio/primary/RTMetricsConfig.h>
#include <pacbio/primary/MetricBlock.h>

namespace PacBio {
namespace Primary {

template <typename Flt,size_t NCam>
inline void FillMetricBlock(const BasecallingMetrics<Flt,NCam>& inputMetric,
                            MetricBlock<Flt,NCam>& mb,
                            const RTMetricsConfig::BaselineMode::RawEnum bmode, size_t minFrames)
{
    mb.NumFrames(inputMetric.TraceMetrics().NumFrames());
    mb.NumPulses(inputMetric.NumPulses());
    mb.PixelChecksum(inputMetric.TraceMetrics().pixelChecksum);
    mb.ActivityLabel(inputMetric.ActivityLabel());

    bool dmeSuccess = inputMetric.TraceMetrics().FullEstimationAttempted() &&
                      inputMetric.TraceMetrics().ModelUpdated() &&
                      inputMetric.TraceMetrics().ConfidenceScore() > 0;

    bool accept = dmeSuccess;
    std::array<uint32_t,NCam> frames;
    frames.fill(1);

    if (bmode == RTMetricsConfig::BaselineMode::MODE1 || bmode == RTMetricsConfig::BaselineMode::MODE2)
    {
        const auto &nf = inputMetric.TraceMetrics().NumFramesBaseline();
        accept = std::all_of(nf.begin(), nf.end(), [minFrames](uint16_t f){ return f > minFrames; });
        std::transform(nf.begin(), nf.end(), frames.begin(),
                       [](uint16_t v) { return static_cast<uint32_t>(v); });
    }

    if (accept)
    {
        mb.NumBaselineFrames(frames);

        // Store number of pkmid frames.
        mb.NumPkmidFramesA(inputMetric.PkmidNumFrames()[0]);
        mb.NumPkmidFramesC(inputMetric.PkmidNumFrames()[1]);
        mb.NumPkmidFramesG(inputMetric.PkmidNumFrames()[2]);
        mb.NumPkmidFramesT(inputMetric.PkmidNumFrames()[3]);

        mb.Baselines(inputMetric.TraceMetrics().FrameBaselineDWS());
        mb.BaselineSds(inputMetric.TraceMetrics().FrameBaselineSigmaDWS());

        // Store the mean pkmid.
        mb.PkmidA(inputMetric.PkmidNumFrames()[0] > 0 ? inputMetric.PkmidSignal()[0] / inputMetric.PkmidNumFrames()[0] : 0);
        mb.PkmidC(inputMetric.PkmidNumFrames()[1] > 0 ? inputMetric.PkmidSignal()[1] / inputMetric.PkmidNumFrames()[1] : 0);
        mb.PkmidG(inputMetric.PkmidNumFrames()[2] > 0 ? inputMetric.PkmidSignal()[2] / inputMetric.PkmidNumFrames()[2] : 0);
        mb.PkmidT(inputMetric.PkmidNumFrames()[3] > 0 ? inputMetric.PkmidSignal()[3] / inputMetric.PkmidNumFrames()[3] : 0);

        // Store the bpzvar (inter-pulse variance)
        mb.BpzvarA(inputMetric.PkmidNumFrames()[0] > 0 ? inputMetric.Bpzvar()[0]: 0);
        mb.BpzvarC(inputMetric.PkmidNumFrames()[1] > 0 ? inputMetric.Bpzvar()[1]: 0);
        mb.BpzvarG(inputMetric.PkmidNumFrames()[2] > 0 ? inputMetric.Bpzvar()[2]: 0);
        mb.BpzvarT(inputMetric.PkmidNumFrames()[3] > 0 ? inputMetric.Bpzvar()[3]: 0);

        // Store the pkzvar (intra-pulse variance)
        mb.PkzvarA(inputMetric.PkmidNumFrames()[0] > 0 ? inputMetric.Pkzvar()[0]: 0);
        mb.PkzvarC(inputMetric.PkmidNumFrames()[1] > 0 ? inputMetric.Pkzvar()[1]: 0);
        mb.PkzvarG(inputMetric.PkmidNumFrames()[2] > 0 ? inputMetric.Pkzvar()[2]: 0);
        mb.PkzvarT(inputMetric.PkmidNumFrames()[3] > 0 ? inputMetric.Pkzvar()[3]: 0);
    }
    else
    {
        // NOTE: We keep zero below instead of NaN as the merging code in MetricBlock::Merge()
        // will effectively zero out these terms.

        std::array<uint32_t,NCam> zf;
        zf.fill(0);
        mb.NumBaselineFrames(zf);

        std::array<Flt,NCam> zv;
        zv.fill(0);
        mb.Baselines(zv);
        mb.BaselineSds(zv);

        mb.NumPkmidFramesA(0);
        mb.NumPkmidFramesC(0);
        mb.NumPkmidFramesG(0);
        mb.NumPkmidFramesT(0);

        mb.PkmidA(0);
        mb.PkmidC(0);
        mb.PkmidG(0);
        mb.PkmidT(0);
        mb.BpzvarA(0);
        mb.BpzvarC(0);
        mb.BpzvarG(0);
        mb.BpzvarT(0);
        mb.PkzvarA(0);
        mb.PkzvarC(0);
        mb.PkzvarG(0);
        mb.PkzvarT(0);
    }

    if (inputMetric.TraceMetrics().AngleEstimationSuceeded())
    {
        // Convert to degrees.
        std::array<Flt,MetricBlock<Flt,NCam>::NAngles> angleDegs;
        const auto& inputAngles = inputMetric.TraceMetrics().AngleSpectrum();
        std::transform(inputAngles.begin(), inputAngles.end(), angleDegs.begin(),
                       [](Flt a) { return a * (180.0f / boost::math::constants::pi<Flt>()); });
        mb.Angles(angleDegs);
    }
    else
    {
        std::array<Flt,MetricBlock<Flt,NCam>::NAngles> angles;
        angles.fill(std::numeric_limits<Flt>::quiet_NaN());
        mb.Angles(angles);
    }

    mb.NumBasesA(inputMetric.NumBasesByAnalog()[0]);
    mb.NumBasesC(inputMetric.NumBasesByAnalog()[1]);
    mb.NumBasesG(inputMetric.NumBasesByAnalog()[2]);
    mb.NumBasesT(inputMetric.NumBasesByAnalog()[3]);

    mb.NumPkmidBasesA(inputMetric.NumPkmidBasesByAnalog()[0]);
    mb.NumPkmidBasesC(inputMetric.NumPkmidBasesByAnalog()[1]);
    mb.NumPkmidBasesG(inputMetric.NumPkmidBasesByAnalog()[2]);
    mb.NumPkmidBasesT(inputMetric.NumPkmidBasesByAnalog()[3]);

    mb.PulseWidth(inputMetric.NumPulseFrames());
    mb.BaseWidth(inputMetric.NumBaseFrames());

    mb.NumSandwiches(inputMetric.NumSandwiches());
    mb.NumHalfSandwiches(inputMetric.NumHalfSandwiches());
    mb.NumPulseLabelStutters(inputMetric.NumPulseLabelStutters());

    mb.PkmaxA(inputMetric.PkMax()[0]);
    mb.PkmaxC(inputMetric.PkMax()[1]);
    mb.PkmaxG(inputMetric.PkMax()[2]);
    mb.PkmaxT(inputMetric.PkMax()[3]);

    mb.PulseDetectionScore(inputMetric.TraceMetrics().PulseDetectionScore());
    mb.TraceAutocorr(inputMetric.TraceMetrics().Autocorrelation());

    // TODO: dme status
    // mb.DmeStatus(inputMetric.TraceMetrics().DmeStatus());
}

}}

#endif //SEQUELACQUISITION_BASECALLINGMETRICSGLUE_H_H
