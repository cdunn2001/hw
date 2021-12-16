// Copyright (c) 2019-2020, Pacific Biosciences of California, Inc.
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

#ifndef mongo_dataTypes_configs_BasecallerDmeConfig_H_
#define mongo_dataTypes_configs_BasecallerDmeConfig_H_

#include <limits>

#include <pacbio/configuration/PBConfig.h>

#include <basecaller/traceAnalysis/ComputeDevices.h>

using std::numeric_limits;


namespace PacBio {
namespace Mongo {
namespace Data {

class FixedDmeConfig : public Configuration::PBConfig<FixedDmeConfig>
{
    PB_CONFIG(FixedDmeConfig);
    // Configuration parameters for a (temporary) fixed model DME, until we
    // can get a true model estimation filter in place.
    PB_CONFIG_PARAM(bool, useSimulatedBaselineParams, false);
    PB_CONFIG_PARAM(float, baselineMean, 200.0f);
    PB_CONFIG_PARAM(float, baselineVar, 33.0f);
};


class BasecallerDmeConfig : public Configuration::PBConfig<BasecallerDmeConfig>
{
public:
    PB_CONFIG(BasecallerDmeConfig);

    SMART_ENUM(MethodName, Fixed, EmHost, EmDevice);
    PB_CONFIG_PARAM(MethodName, Method, Configuration::DefaultFunc(
                        [](Basecaller::ComputeDevices device) -> MethodName
                        {
                            return device == Basecaller::ComputeDevices::Host ?
                                MethodName::EmHost:
                                MethodName::EmDevice;
                        },
                        {"analyzerHardware"}
    ));

    bool UsesGpu() const { return Method == MethodName::EmDevice; }

    // Parameters for the SpiderFixed model, when in use
    PB_CONFIG_OBJECT(FixedDmeConfig, SimModel);

    // Thresholds for mixing fractions of analog modes in detection model fit.
    //
    // AnalogMixFractionThreshold[0] < AnalogMixFractionThreshold[1].
    //
    // Associated confidence factor is defined using AnalogMixFractionThreshold[1]
    // AnalogMixFractionThreshold[1] must be <= 0.25. If <= 0, the threshold is
    // disabled (i.e., the associated confidence factor will always be 1).
    // AnalogMixFractionThreshold[0] defines the lower end of
    // a 0-1 ramp for the related confidence factor.
    // AnalogMixFractionThreshold[0] may be < 0. Since mixing fractions cannot be
    // negative, setting AnalogMixFractionThreshold[0] < 0 effectively sets a
    // positive lower bound for the confidence factor that is attained when
    // the mixing fraction is 0.
    PB_CONFIG_PARAM(std::vector<float>, AnalogMixFractionThreshold,
            std::vector<float>({ 0.02f / 3, 0.02f }));

    // If the confidence of the initial model for the core DME EM algorithm
    // is less than ScaleSnrConfTol, its SNR will be scaled toward a
    // fractile of the frame data that estimates the average pulse signal
    // level of the analogs.
    // Value must be positive.
    // Used only by DmeMonochrome.
    PB_CONFIG_PARAM(float, ScaleSnrConfTol, 1.0f);

    // Upper bound for expectation-maximization iterations.
    PB_CONFIG_PARAM(unsigned short, EmIterationLimit, 20);

    // A factor that is multiplied into the G-test statistic before
    // computing the p-value. Ideally, this would be 1.0.
    // If set <= 0, the associated confidence factor will always be 1.0.
    // If set < 0, the G-test computation is skipped entirely.
    PB_CONFIG_PARAM(float, GTestStatFactor, -1.0f);

    // If IterateToLimit is set, EM estimation algorithm will consistently
    // iterate until it reaches EmIterationLimit, regardless of meeting
    // the convergence criterion. This is primarily useful for
    // speed benchmarking.
    PB_CONFIG_PARAM(bool, IterateToLimit, false);

    // Parameters for the fuzzy threshold for minimum analog SNR in
    // DmeMonochrome confidence factor.
    // Largest SNR for which the confidence factor is 0.
    PB_CONFIG_PARAM(float, MinAnalogSnrThresh0, 2.0f);
    // Smallest SNR for which the confidence factor is 1.
    PB_CONFIG_PARAM(float, MinAnalogSnrThresh1, 4.0f);

    // A non-negative coefficient for the regularization term for pulse
    // amplitude scale estimation in DmeMonochrome. This is multiplied by
    // the confidence of the running-average model. Setting this parameter
    // to zero effectively disables the regularization.
    PB_CONFIG_PARAM(float, PulseAmpRegularization, 0.0f);

    // A coefficient to scale the threshold used in DmeMonochrome to
    // penalize the confidence if the SNR drops dramatically.
    // The primary motive for this confidence factor is to guard against
    // registration error in the fit when there are few data representing
    // incorporation of the brightest analog in the data.
    // If this parameter is set to 1.0, the SnrDrop confidence factor will
    // be zero if the signal level for the brightest analog is estimated to
    // be less than a threshold, which is defined to be logarithmically
    // one-third of the way from the second-brightest analog to the
    // brightest one, according to the running-average model, and possibly
    // reduced by low confidence.
    // Set this parameter to a negative value to effectively disable this
    // confidence factor (i.e., make it always evaluate to 1.0).
    // Cannot be larger than the pulse amplitude ratio of the brightest to
    // the second-brightest analog.
    PB_CONFIG_PARAM(float, SnrDropThresh, 1.0f);

    // If the confidence score for an estimate is less than
    // SuccessConfidenceThresh, it is set to zero.
    PB_CONFIG_PARAM(float, SuccessConfidenceThresh, 0.10f);

    // ----------------------------------------------------
    // Stuff below here was merely copied from Sequel and
    // is not _yet_ used in Mongo.

    // Model update is all or nothing (as opposed to mixing update)?
    PB_CONFIG_PARAM(bool, PureUpdate, false);

    // Maximum weight used for updating detection model.
    PB_CONFIG_PARAM(float, ModelUpdateWeightMax, 0.50f);

    // Threshold for Mahalanobis distance of analog means from background distribution.
    // Default corresponds to chi-square probability of 0.90;
    PB_CONFIG_PARAM(float, ThreshAnalogSNR, 2.149f);

    PB_CONFIG_PARAM(uint32_t, FirstAttemptsIterMultiple, 2);

    PB_CONFIG_PARAM(unsigned int, NumBinsMin, 500);

    PB_CONFIG_PARAM(float, ConvergeCoeff, 3.0e-5);

    // 1 will round up to a single full chunk.
    PB_CONFIG_PARAM(uint32_t, MinFramesForEstimate, 4000);

    // Frames required to estimate histogram bounds from
    // basline stats
    PB_CONFIG_PARAM(uint32_t, NumFramesPreAccumStats, 1000u);

    // Coefficient for the reduction of model confidence triggered by laser
    // power changes.
    // Set less than zero to preserve prior behavior, described by comment
    // for ConfidenceHalfLife[01].
    // If LaserPowerChangeReduceConfidence >= 0, the (nominal) confidence
    // half-life will be constant ConfidenceHalfLife1.
    // Model confidence will be reduced for each laser power change.
    // Used only by 1C4A (Spider).
    PB_CONFIG_PARAM(float, LaserPowerChangeReduceConfidence, -1.0f);

    // Half-life of confidence decay of dection model in units of frames.
    // Only used by 1C4A (Spider).
    // If LaserPowerChangeReduceConfidence < 0,
    // half-life is ConfidenceHalfLife0 until a fixed number of frames that
    // is defined to approximate the median of ALP duration.
    // Then half-life ramps to ConfidenceHalfLife1 at a frame count that
    // is defined to approximate the 97.5-th percentile of ALP duration.
    // Typically, ConfidenceHalfLife0 < ConfidenceHalfLife1.
    PB_CONFIG_PARAM(unsigned int, ConfidenceHalfLife0, 4096);
    PB_CONFIG_PARAM(unsigned int, ConfidenceHalfLife1, 12000);

    // Term to enhance the confidence half-life when the trace data exhibit
    // no sign of polymerization activity (e.g., low trace autocorrelation).
    // When set to a value x, the half-life will be enhanced by a factor of
    // up to (1 + x). Typically, x >= 0, but this is not strictly required.
    // Used only by 1C4A (Spider).
    PB_CONFIG_PARAM(float, ConfidenceHalfLifePauseEnhance, 0.0f);

    // Parameters to control the DmeMonochrome confidence factor that
    // applies a fuzzy threshold on the log of a Pearson's chi-square (PCS)
    // statistic. Both are offsets from a scale A set by the total of the
    // bin counts. If log(PCS) < A + GofLogChiSqrThresh1, the confidence
    // factor is 1. If log(PCS) > A + GofLogChiSqrThresh1 +
    // GofLogChiSqrThresh2, the confidence factor is 0.
    // Required: GofLogChiSqrThresh2 > 0.
    // To effectively disable this confidence factor, set
    // GofLogChiSqrThresh1 >= 100.
    // When enabled, suggest GofLogChiSqrThresh1 approx 1.0.
    PB_CONFIG_PARAM(float, GofLogChiSqrThresh1, 111.0f);
    PB_CONFIG_PARAM(float, GofLogChiSqrThresh2, 8.0f);
};

}}}     // namespace PacBio::Mongo::Data


// Define validation specialization.  Specializations must happen in the
// same namespace as the generic declaration.
namespace PacBio {
namespace Configuration {

using PacBio::Mongo::Data::BasecallerDmeConfig;

template <>
inline void ValidateConfig<BasecallerDmeConfig>(const BasecallerDmeConfig& dmeConfig, ValidationResults* results)
{
    const auto amft = dmeConfig.AnalogMixFractionThreshold;
    if (!(isfinite(amft[1]) && (amft[1] <= 0.25f)))  // 0.25 included
    {
        results->AddError("Incorrect upper bound: must be finite and <= 0.25");
    }

    if (!(isfinite(amft[0]) && (amft[0] <= amft[1])))
    {
        results->AddError("Incorrect lower bound: must be finite and <= upper one");
    }
}

}}

#endif //mongo_dataTypes_configs_BasecallerDmeConfig_H_
