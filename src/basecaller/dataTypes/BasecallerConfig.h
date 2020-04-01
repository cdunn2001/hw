#ifndef mongo_dataTypes_BasecallerConfig_H_
#define mongo_dataTypes_BasecallerConfig_H_

#include <array>

#include <pacbio/configuration/PBConfig.h>
#include <pacbio/utilities/SmartEnum.h>

#include "AnalogMode.h"
#include "PrimaryConfig.h"
#include "StaticDetModelConfig.h"

// TODO: After some mongo dust has settled, purge unused configuration properties.

namespace PacBio {
namespace Mongo {
namespace Data {

class BasecallerInitConfig : public Configuration::PBConfig<BasecallerInitConfig>
{
public:
    PB_CONFIG(BasecallerInitConfig);
    /// The number of host worker threads to use.  Most parallelism should be
    /// handled internal to the GPU, so this does not need to be large.
    /// A minimum of 3 will allow efficient overlap of upload/download/compute,
    /// but beyond that it shouldn't really be any higher than what is necessary
    /// for active host stages to keep up with the gpu

    // TODO add hooks so that we can switch between gpu and host centric defaults
    // without manually specifying a million parameters
    PB_CONFIG_PARAM(uint32_t, numWorkerThreads, 8);

    /// If true, the threads are bound to a particular set of cores for the
    /// Sequel Alpha machines when running on the host.
    PB_CONFIG_PARAM(bool, bindCores, false);
};


class BasecallerTraceHistogramConfig : public Configuration::PBConfig<BasecallerTraceHistogramConfig>
{
public:
    PB_CONFIG(BasecallerTraceHistogramConfig);

    SMART_ENUM(MethodName, Host, Gpu);
    PB_CONFIG_PARAM(MethodName, Method, MethodName::Host);
    PB_CONFIG_PARAM(unsigned int, NumFramesPreAccumStats, 1000u);

    // Bin size of data histogram is nominally defined as initial estimate
    // of baseline sigma multiplied by this coefficient.
    PB_CONFIG_PARAM(float, BinSizeCoeff, 0.25f);

    // Use fall-back baseline sigma when number of baseline frames is
    // less than this value.
    PB_CONFIG_PARAM(unsigned int, BaselineStatMinFrameCount, 50u);

    // Use this value as an estimate for baseline standard deviation when
    // we have insufficient data.
    PB_CONFIG_PARAM(float, FallBackBaselineSigma, 10.0f);
};

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

    SMART_ENUM(MethodName, Fixed, EmHost);
    PB_CONFIG_PARAM(MethodName, Method, MethodName::Fixed);

    // Parameters for the SpiderFixed model, when in use
    PB_CONFIG_OBJECT(FixedDmeConfig, SimModel);

    // Threshold for mixing fractions of analog modes in detection model fit.
    // Associated confidence factor is defined using this threshold.
    // Must be non-negative.
    PB_CONFIG_PARAM(float, AnalogMixFractionThreshold, 0.039f);

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

    // Number of frames to skip between estimation attempts.
    PB_CONFIG_PARAM(unsigned int, MinSkipFrames, 0);

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


class BasecallerFrameLabelerConfig : public Configuration::PBConfig<BasecallerFrameLabelerConfig>
{
public:
    PB_CONFIG(BasecallerFrameLabelerConfig);

    // TODO: When we are done testing subframe and it presumably becomes
    //       default, consider putting subframe specific options into a
    //       new subgroup

    SMART_ENUM(MethodName, NoOp, DeviceSubFrameGaussCaps)
    PB_CONFIG_PARAM(MethodName, Method, MethodName::DeviceSubFrameGaussCaps);

    PB_CONFIG_PARAM(float, UpperThreshold, 7.0f);
    PB_CONFIG_PARAM(float, LowerThreshold, 2.0f);
    PB_CONFIG_PARAM(float, Alpha, 1.0f);
    PB_CONFIG_PARAM(float, Beta, 1.0f);
    PB_CONFIG_PARAM(float, Gamma, 1.0f);
};

class BasecallerPulseAccumConfig : public Configuration::PBConfig<BasecallerPulseAccumConfig>
{
public:
    PB_CONFIG(BasecallerPulseAccumConfig);

    SMART_ENUM(MethodName, NoOp, HostSimulatedPulses, HostPulses, GpuPulses)
    PB_CONFIG_PARAM(MethodName, Method, MethodName::GpuPulses);

    // Increasing this number will directly increase memory usage, even if
    // we don't saturate the allowed number of calls, so be conservative
    PB_CONFIG_PARAM(uint32_t, maxCallsPerZmw, 12);
};

class BasecallerMetricsConfig : public Configuration::PBConfig<BasecallerMetricsConfig>
{
public:
    PB_CONFIG(BasecallerMetricsConfig);

    SMART_ENUM(MethodName, Host, NoOp, Gpu);
    PB_CONFIG_PARAM(MethodName, Method, MethodName::Gpu);

    PB_CONFIG_PARAM(uint32_t, sandwichTolerance, 0);
};


class BasecallerBaselinerConfig : public Configuration::PBConfig<BasecallerBaselinerConfig>
{
public:
    PB_CONFIG(BasecallerBaselinerConfig);

    SMART_ENUM(MethodName,
               MultiScaleLarge, MultiScaleMedium, MultiScaleSmall,
               TwoScaleLarge, TwoScaleMedium, TwoScaleSmall,
               DeviceMultiScale,
               NoOp);
    PB_CONFIG_PARAM(MethodName, Method, MethodName::DeviceMultiScale);
};


class BasecallerPulseToBaseConfig : public Configuration::PBConfig<BasecallerPulseToBaseConfig>
{
public:
    PB_CONFIG(BasecallerPulseToBaseConfig);

    SMART_ENUM(MethodName, Simple, Simulator, SigmaCut, ExShortPulse);
    PB_CONFIG_PARAM(MethodName, Method, MethodName::Simple);

    PB_CONFIG_PARAM(uint32_t, BasesPerZmwChunk, 50u); // used by p2bsimulator ... maybe moved up?
    PB_CONFIG_PARAM(float, SnrThresh, 100.0f);
    PB_CONFIG_PARAM(double, XspAmpThresh, 0.70);  // Valid range is [0, 1].
    PB_CONFIG_PARAM(float, XspWidthThresh, 3.5f); // Must be >= 0.
};


class SimulatedFaults : public Configuration::PBConfig<SimulatedFaults>
{
    PB_CONFIG(SimulatedFaults);

    PB_CONFIG_PARAM(int, negativeDyeSpectrumCounter,0);
};


class BasecallerAlgorithmConfig : public Configuration::PBConfig<BasecallerAlgorithmConfig>
{
public:
    PB_CONFIG(BasecallerAlgorithmConfig);

    PB_CONFIG_OBJECT(BasecallerBaselinerConfig, baselinerConfig);
    PB_CONFIG_OBJECT(BasecallerTraceHistogramConfig, traceHistogramConfig);
    PB_CONFIG_OBJECT(BasecallerDmeConfig, dmeConfig);
    PB_CONFIG_OBJECT(BasecallerFrameLabelerConfig, frameLabelerConfig);
    PB_CONFIG_OBJECT(BasecallerPulseAccumConfig, pulseAccumConfig);
    PB_CONFIG_OBJECT(BasecallerPulseToBaseConfig, PulseToBase);
    PB_CONFIG_OBJECT(BasecallerMetricsConfig, Metrics);
    PB_CONFIG_OBJECT(SimulatedFaults, simulatedFaults);

    PB_CONFIG_OBJECT(StaticDetModelConfig, staticDetModelConfig);
    PB_CONFIG_PARAM(bool, staticAnalysis, true);

public:
    std::string CombinedMethodName() const
    {
        return baselinerConfig.Method.toString() + "_"
             + dmeConfig.Method.toString() + "_"
             + frameLabelerConfig.Method.toString() + "_"
             + PulseToBase.Method.toString() + "_"
             + Metrics.Method.toString();
    }
};


class BasecallerConfig : public Configuration::PBConfig<BasecallerConfig>
{
    PB_CONFIG(BasecallerConfig);

    PB_CONFIG_OBJECT(BasecallerInitConfig, init);
    PB_CONFIG_OBJECT(BasecallerAlgorithmConfig, algorithm);
};

}}}     // namespace PacBio::Mongo::Data


#endif //mongo_dataTypes_BasecallerConfig_H_
