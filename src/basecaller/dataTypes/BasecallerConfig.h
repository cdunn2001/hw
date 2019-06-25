#ifndef mongo_dataTypes_BasecallerConfig_H_
#define mongo_dataTypes_BasecallerConfig_H_

#include <array>

#include <pacbio/process/ConfigurationBase.h>
#include <pacbio/utilities/SmartEnum.h>

#include "AnalogMode.h"
#include "PrimaryConfig.h"

// TODO: After some mongo dust has settled, purge unused configuration properties.

namespace PacBio {
namespace Mongo {
namespace Data {

    class BasecallerInitConfig : public PacBio::Process::ConfigurationObject
    {
    public:
        /// The number of host worker threads to use.  Most parallelism should be
        /// handled internal to the GPU, so this does not need to be large.
        /// A minimum of 3 will allow efficient overlap of upload/download/compute,
        /// but beyond that it shouldn't really be any higher than what is necessary
        /// for active host stages to keep up with the gpu

        // TODO add hooks so that we can switch between gpu and host centric defaults
        // without manually specifying a million parameters
        ADD_PARAMETER(uint32_t, numWorkerThreads, 8);

        /// If true, the threads are bound to a particular set of cores for the
        /// Sequel Alpha machines when running on the host.
        ADD_PARAMETER(bool, bindCores, false);

    public:
        BasecallerInitConfig()
        { }
    };


    class BasecallerTraceHistogramConfig : public PacBio::Process::ConfigurationObject
    {
    public:
        SMART_ENUM(MethodName, Host, Gpu);
        ADD_ENUM(MethodName, Method, MethodName::Host);
        // TODO
    };

    class SpiderFixedDmeConfig : public PacBio::Process::ConfigurationObject
    {
        // Configuration parameters for a (temporary) fixed model DME, until we
        // can get a true model estimation filter in place.
        ADD_PARAMETER(float, RefSNR, 60);
        ADD_PARAMETER(float, TAmp, 1.0f / 4.4f);
        ADD_PARAMETER(float, GAmp, 1.7f / 4.4f);
        ADD_PARAMETER(float, CAmp, 1.0f);
        ADD_PARAMETER(float, AAmp, 2.9f / 4.4f);
        ADD_PARAMETER(float, baselineMean, 200.0f);
        ADD_PARAMETER(float, baselineVar, 33.0f);
        ADD_PARAMETER(float, pulseCV, 0.1);
        ADD_PARAMETER(float, shotCoeff, 1.37);
    };


    class BasecallerDmeConfig : public PacBio::Process::ConfigurationObject
    {
    public:
        SMART_ENUM(MethodName, SpiderFixed, Monochrome);
        ADD_ENUM(MethodName, Method, MethodName::SpiderFixed);

        // Parameters for the SpiderFixed model, when in use
        ADD_OBJECT(SpiderFixedDmeConfig, SpiderSimModel);

        // Model update is all or nothing (as opposed to mixing update)?
        ADD_PARAMETER(bool, PureUpdate, false);

        // Upper limit for iteration of the EM algorithm used for bivariate
        // model estimation (phase 2).
        ADD_PARAMETER(uint32_t, IterationLimit, 20);

        // If IterateToLimit is set, EM estimation algorithm will consistently
        // iterate until it reaches the IterationLimit, regardless of meeting
        // the convergence criterion. This is primarily useful for
        // speed benchmarking.
        ADD_PARAMETER(bool, IterateToLimit, false);

        // Maximum weight used for updating detection model.
        ADD_PARAMETER(float, ModelUpdateWeightMax, 0.50f);

        // Threshold for Mahalanobis distance of analog means from background distribution.
        // Default corresponds to chi-square probability of 0.90;
        ADD_PARAMETER(float, ThreshAnalogSNR, 2.149f);

        ADD_PARAMETER(uint32_t, FirstAttemptsIterMultiple, 2);

        // Bin size of data histogram is nominally defined as initial estimated
        // of baseline sigma multiplied by this coefficient.
        // Used by both DmeTwoPhase and DmeMonochrome.
        ADD_PARAMETER(float, BinSizeCoeff, 0.20f);

        ADD_PARAMETER(unsigned int, NumBinsMin, 500);

        ADD_PARAMETER(float, ConvergeCoeff, 3.0e-5);

        // 1 will round up to a single full chunk.
        ADD_PARAMETER(uint32_t, MinFramesForEstimate, 4000);

        // Number of frames to skip between estimation attempts.
        ADD_PARAMETER(unsigned int, MinSkipFrames, 0);

        // If the confidence score for an estimate is less than
        // SuccessConfidenceThresh, it is set to zero.
        ADD_PARAMETER(float, SuccessConfidenceThresh, 0.10f);

        // Threshold for mixing fractions of analog modes in detection model fit.
        // Associated confidence factor is defined using this threshold.
        // Must be non-negative.
        ADD_PARAMETER(float, AnalogMixFractionThreshold, 0.039f);

        // Parameters for the fuzzy threshold for minimum analog SNR in
        // DmeMonochrome confidence factor.
        // Largest SNR for which the confidence factor is 0.
        ADD_PARAMETER(float, MinAnalogSnrThresh0, 2.0f);
        // Smallest SNR for which the confidence factor is 1.
        ADD_PARAMETER(float, MinAnalogSnrThresh1, 4.0f);

        // Coefficient for the reduction of model confidence triggered by laser
        // power changes.
        // Set less than zero to preserve prior behavior, described by comment
        // for ConfidenceHalfLife[01].
        // If LaserPowerChangeReduceConfidence >= 0, the (nominal) confidence
        // half-life will be constant ConfidenceHalfLife1.
        // Model confidence will be reduced for each laser power change.
        // Used only by 1C4A (Spider).
        ADD_PARAMETER(float, LaserPowerChangeReduceConfidence, -1.0f);

        // Half-life of confidence decay of dection model in units of frames.
        // Only used by 1C4A (Spider).
        // If LaserPowerChangeReduceConfidence < 0,
        // half-life is ConfidenceHalfLife0 until a fixed number of frames that
        // is defined to approximate the median of ALP duration.
        // Then half-life ramps to ConfidenceHalfLife1 at a frame count that
        // is defined to approximate the 97.5-th percentile of ALP duration.
        // Typically, ConfidenceHalfLife0 < ConfidenceHalfLife1.
        ADD_PARAMETER(unsigned int, ConfidenceHalfLife0, 4096);
        ADD_PARAMETER(unsigned int, ConfidenceHalfLife1, 12000);

        // Term to enhance the confidence half-life when the trace data exhibit
        // no sign of polymerization activity (e.g., low trace autocorrelation).
        // When set to a value x, the half-life will be enhanced by a factor of
        // up to (1 + x). Typically, x >= 0, but this is not strictly required.
        // Used only by 1C4A (Spider).
        ADD_PARAMETER(float, ConfidenceHalfLifePauseEnhance, 0.0f);

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
        ADD_PARAMETER(float, SnrDropThresh, 1.0f);

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
        ADD_PARAMETER(float, GofLogChiSqrThresh1, 111.0f);
        ADD_PARAMETER(float, GofLogChiSqrThresh2, 8.0f);

        // A factor that is multiplied into the G-test statistic before
        // computing the p-value. Ideally, this would be 1.0.
        // If set <= 0, the associated confidence factor will always be 1.0.
        // If set < 0, the G-test computation is skipped entirely.
        // Used only be DmeMonochrome.
        ADD_PARAMETER(float, GTestStatFactor, -1.0f);

        // A non-negative coefficient for the regularization term for pulse
        // amplitude scale estimation in DmeMonochrome. This is multiplied by
        // the confidence of the running-average model. Setting this parameter
        // to zero effectively disables the regularization.
        ADD_PARAMETER(float, PulseAmpRegularization, 0.0f);
    };


    class BasecallerFrameLabelerConfig : public PacBio::Process::ConfigurationObject
    {
    public:
        // TODO: When we are done testing subframe and it presumably becomes
        //       default, consider putting subframe specific options into a
        //       new subgroup

        SMART_ENUM(MethodName, DeviceSubFrameGaussCaps)
        ADD_ENUM(MethodName, Method, MethodName::DeviceSubFrameGaussCaps);

        ADD_PARAMETER(float, UpperThreshold, 7.0f);
        ADD_PARAMETER(float, LowerThreshold, 2.0f);
        ADD_PARAMETER(float, Alpha, 1.0f);
        ADD_PARAMETER(float, Beta, 1.0f);
        ADD_PARAMETER(float, Gamma, 1.0f);
    };


    class BasecallerMetricsConfig : public PacBio::Process::ConfigurationObject
    {
    public:
        SMART_ENUM(MethodName, HFMetrics, NoOp);
        ADD_ENUM(MethodName, Method, MethodName::HFMetrics);

        ADD_PARAMETER(uint32_t, sandwichTolerance, 0);
    };


    class BasecallerBaselinerConfig : public PacBio::Process::ConfigurationObject
    {
        CONF_OBJ_SUPPORT_COPY(BasecallerBaselinerConfig)
    public:
        SMART_ENUM(MethodName,
                   MultiScaleLarge, MultiScaleMedium, MultiScaleSmall,
                   TwoScaleLarge, TwoScaleMedium, TwoScaleSmall,
                   DeviceMultiScale,
                   NoOp);
        ADD_ENUM(MethodName, Method, MethodName::DeviceMultiScale);
        ADD_PARAMETER(uint16_t, AutocorrLagFrames, 4u);
    };


    class BasecallerPulseToBaseConfig : public PacBio::Process::ConfigurationObject
    {
        CONF_OBJ_SUPPORT_COPY(BasecallerPulseToBaseConfig)
    public:
        SMART_ENUM(MethodName, Simple, Simulator, SigmaCut, ExShortPulse);
        ADD_ENUM(MethodName, Method, MethodName::Simple);

        ADD_PARAMETER(uint32_t, BasesPerZmwChunk, 50u); // used by p2bsimulator ... maybe moved up?
        ADD_PARAMETER(float, SnrThresh, 100.0f);
        ADD_PARAMETER(double, XspAmpThresh, 0.70);  // Valid range is [0, 1].
        ADD_PARAMETER(float, XspWidthThresh, 3.5f); // Must be >= 0.
    };


    class SimulatedFaults : public PacBio::Process::ConfigurationObject
    {
        ADD_PARAMETER(int, negativeDyeSpectrumCounter,0);
    };


    class BasecallerAlgorithmConfig :  public PacBio::Process::ConfigurationObject
    {
        CONF_OBJ_SUPPORT_COPY(BasecallerAlgorithmConfig);
    public:
        ADD_OBJECT(BasecallerBaselinerConfig, baselinerConfig);
        ADD_OBJECT(BasecallerTraceHistogramConfig, traceHistogramConfig);
        ADD_OBJECT(BasecallerDmeConfig, dmeConfig);
        ADD_OBJECT(BasecallerFrameLabelerConfig, frameLabelerConfig);
        ADD_OBJECT(BasecallerPulseToBaseConfig, PulseToBase);
        ADD_OBJECT(BasecallerMetricsConfig, Metrics);
        ADD_OBJECT(SimulatedFaults, simulatedFaults);

        ADD_PARAMETER(bool, staticAnalysis, true);

    public:
        std::string CombinedMethodName() const
        {
            return baselinerConfig.Method().toString() + "_"
                 + dmeConfig.Method().toString() + "_"
                 + frameLabelerConfig.Method().toString() + "_"
                 + PulseToBase.Method().toString() + "_"
                 + Metrics.Method().toString();
        }

        void SetSpiderDefaults()
        {
        }
    };


    class BasecallerConfig : public PacBio::Process::ConfigurationObject
    {
        CONF_OBJ_SUPPORT_COPY(BasecallerConfig);
        ADD_OBJECT(BasecallerInitConfig, init);
        ADD_OBJECT(BasecallerAlgorithmConfig, algorithm);
    };

}}}     // namespace PacBio::Mongo::Data


#endif //mongo_dataTypes_BasecallerConfig_H_
