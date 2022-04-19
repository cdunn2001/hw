// Copyright (c) 2021, Pacific Biosciences of California, Inc.
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
//  Defines members of class DmeEmDevice.

#include "DmeEmDevice.h"

#include <algorithm>
#include <limits>

#include <common/IntInterval.h>

#include <common/cuda/memory/AllocationViews.cuh>
#include <common/cuda/PBCudaSimd.cuh>
#include <common/cuda/streams/LaunchManager.cuh>

#include <dataTypes/configs/BasecallerDmeConfig.h>
#include <dataTypes/configs/AnalysisConfig.h>
#include <basecaller/traceAnalysis/DmeDiagnostics.h>

///////////////////////////////////////////////////////////////////
// TODO There are a lot of commented out PBAssert statements below,
//      as they will not work on the GPU.  PTSD-267 will hopefully
//      result in a replacement that can be slotted in
///////////////////////////////////////////////////////////////////

using std::numeric_limits;
using namespace PacBio::Cuda;
using namespace PacBio::Cuda::Utility;
using namespace PacBio::Mongo::Data;

using FrameIntervalType = PacBio::Mongo::IntInterval<FrameIndexType>;


namespace PacBio {
namespace Mongo {
namespace Basecaller {

namespace {

struct AnalogMode
{
    char baseLabel;
    float relAmplitude;
    float excessNoiseCV;
    float interPulseDistance;   // seconds
    float pulseWidth;           // seconds
    float pw2SlowStepRatio;
    float ipd2SlowStepRatio;
};

__constant__ CoreDMEstimator::Configuration baseConfig;

// Wrapping all the static configurations into a single struct,
// as that will be easier to upload to the GPU.
struct StaticConfig
{
    CudaArray<AnalogMode, 4> analogs;
    float analogMixFracThresh0_;
    float analogMixFracThresh1_;
    float confidHalfLife0_;
    float confidHalfLife1_;
    float scaleSnrConfTol_;
    unsigned short emIterLimit_;
    float gTestFactor_;
    bool iterToLimit_;
    float pulseAmpRegCoeff_;
    float snrDropThresh_;
    float snrThresh0_;
    float snrThresh1_;
    float successConfThresh_;
    uint32_t updateMethod_;
    float refSnr_;       // Expected SNR for analog with relative amplitude of 1
};

__constant__ StaticConfig staticConfig;

/// Saturated linear activation function.
/// A fuzzy threshold that ramps from 0 at a to 1 at b.
/// \returns (x - a)/(b - a) clamped to [0, 1] range.
// TODO fix duplication?
__device__ float satlin(float a, float b, float x)
{
    const auto r = (x - a) / (b - a);
    return min(max(r, 0.f), 1.f);
}

using LaneDetModel = Data::LaneModelParameters<PBHalf2, laneSize/2>;

__device__ const AnalogMode& Analog(int i)
{
    return staticConfig.analogs[i];
}

static constexpr auto numBins = DmeEmDevice::LaneHist::numBins;

template <typename VF>
__device__ VF ModelSignalCovar(float excessNoiseCV2, VF sigMean, VF blVar)
{
    blVar += baseConfig.shotVarCoeff * sigMean;

    // Typically, excessNoiseCV2 << 1 and sigMean >> 1. Order multiplications
    // this way to avoid overflow, especially when VF is half-precision.
    blVar += excessNoiseCV2 * sigMean * sigMean;
    return blVar;
}

template <typename VF>
__device__ VF XsnCoeffCVSq(const VF& sigMean, const VF& sigCovar, const VF& blVar)
{
    VF r (sigCovar - blVar);
    r -= sigMean * baseConfig.shotVarCoeff;
    r /= sigMean * sigMean;
    return r;
}

__device__ void UpdateMode(const ZmwAnalogMode& from,
                             ZmwAnalogMode* to,
                             float fraction)
{
    const float a = fraction;
    const float b = 1 - fraction;
    to->mean = a * from.mean + b * to->mean;
    to->var  = a * from.var  + b * to->var;
}

__device__ void UpdateModel0(const ZmwDetectionModel& from,
                              ZmwDetectionModel *to,
                              float fraction)
{
    UpdateMode(from.baseline, &to->baseline, fraction);
    for (int i = 0; i < to->numAnalogs; ++i)
    {
        UpdateMode(from.analogs[i], &to->analogs[i], fraction);
    }
}

__device__ void UpdateModel1(const ZmwDetectionModel& from,
                              ZmwDetectionModel *to,
                              float fraction)
{

    const float a = 1.0f - fraction;
    const float b = fraction;

    auto &tbm = to->baseline;
    auto &obm = from.baseline;

    auto bw = a * tbm.weight + b * obm.weight;
    tbm.weight = bw;

    auto bm = a * tbm.mean + b * obm.mean;
    tbm.mean = bm;

    auto bv = powf(tbm.var, a) * powf(obm.var, b);
    tbm.var = bv;

    // Four analogs with remaining weight equally partitioned 
    auto aw = 0.25f * (1.0f - bw);

    for (int i = 0; i < to->numAnalogs; ++i)
    {
        auto& tdmi = to->analogs[i];
        auto& obmi = from.analogs[i];

        tdmi.weight = aw;

        auto am = powf(tdmi.mean, a) * powf(obmi.mean, b);
        tdmi.mean = am;

        auto cv = Analog(i).excessNoiseCV;
        auto av = ModelSignalCovar(cv*cv, am, bv);
        tdmi.var = av;
    }

}

__device__ void UpdateModel2(const ZmwDetectionModel& from,
                              ZmwDetectionModel *to,
                              float fraction)
{

    const float a = 1.0f - fraction;
    const float b = fraction;

    auto &tbm = to->baseline;
    auto &obm = from.baseline;

    const auto prevBlCovar = tbm.var;

    auto bw = a * tbm.weight + b * obm.weight;
    tbm.weight = bw;

    auto bm = a * tbm.mean + b * obm.mean;
    tbm.mean = bm;

    auto bv = powf(tbm.var, a) * powf(obm.var, b);
    tbm.var = bv;

    // Four analogs with remaining weight equally partitioned 
    auto aw = 0.25f * (1.0f - bw);

    for (int i = 0; i < to->numAnalogs; ++i)
    {
        auto& tdmi = to->analogs[i];
        auto& obmi = from.analogs[i];

        tdmi.weight = aw;

        const auto tXsnCVSq = XsnCoeffCVSq(tdmi.mean, tdmi.var, prevBlCovar);
        const auto oXsnCVSq = XsnCoeffCVSq(obmi.mean, obmi.var, obm.var);
        const auto newXsnCVSq = a * tXsnCVSq  + b * oXsnCVSq;

        auto am = powf(tdmi.mean, a) * powf(obmi.mean, b);
        tdmi.mean = am;

        auto av = ModelSignalCovar(newXsnCVSq, am, bv);
        tdmi.var = av;
    }

}

__device__ void UpdateModel(const ZmwDetectionModel& from,
                              ZmwDetectionModel *to)
{
    float toConfidence = 0;
    assert (from.confidence >= 0.0f);
    assert (toConfidence >= 0.0f);

    const auto confSum = from.confidence + toConfidence;
    const float fraction = confSum > 0.0f ? from.confidence / confSum: 0.f;

    assert (fraction >= 0.0f);
    assert (fraction <= 1.0f);
    //assert ((fraction > 0) | (confSum == Confidence())));
    switch (staticConfig.updateMethod_)
    {
        case 0: UpdateModel0(from, to, fraction); break;
        case 1: UpdateModel1(from, to, fraction); break;
        case 2: UpdateModel2(from, to, fraction); break;
        // hmm, can't throw an exception in device code, so defaulting to 0
        // and doing this explicitly just to prevent confusion
        default: UpdateModel0(from, to, fraction); break;
    }

    to->confidence=confSum;
}

}

DmeEmDevice::DmeEmDevice(uint32_t poolId, unsigned int poolSize)
    : CoreDMEstimator(poolId, poolSize)
{ }

// static
void DmeEmDevice::Configure(const Data::BasecallerDmeConfig &dmeConfig,
                            const Data::AnalysisConfig &analysisConfig)
{
    CoreDMEstimator::Configure(analysisConfig);
    Cuda::CudaCopyToSymbol(&baseConfig, &CoreDMEstimator::Config());

    // TODO: Validate values.
    // TODO: Log settings.
    StaticConfig config;
    auto& movieInfo = analysisConfig.movieInfo;
    for (size_t i = 0; i < movieInfo.analogs.size(); i++)
    {
        config.analogs[i].baseLabel = movieInfo.analogs[i].baseLabel;
        config.analogs[i].relAmplitude = movieInfo.analogs[i].relAmplitude;
        config.analogs[i].excessNoiseCV = movieInfo.analogs[i].excessNoiseCV;
        config.analogs[i].interPulseDistance = movieInfo.analogs[i].interPulseDistance;
        config.analogs[i].pulseWidth = movieInfo.analogs[i].pulseWidth;
        config.analogs[i].pw2SlowStepRatio = movieInfo.analogs[i].pw2SlowStepRatio;
        config.analogs[i].ipd2SlowStepRatio = movieInfo.analogs[i].ipd2SlowStepRatio;
    }

    config.analogMixFracThresh0_ = dmeConfig.AnalogMixFractionThreshold[0];
    config.analogMixFracThresh1_ = dmeConfig.AnalogMixFractionThreshold[1];
    config.confidHalfLife0_ = dmeConfig.ConfidenceHalfLife[0];
    config.confidHalfLife1_ = dmeConfig.ConfidenceHalfLife[1];
    config.scaleSnrConfTol_ = dmeConfig.ScaleSnrConfTol;

    config.emIterLimit_ = dmeConfig.EmIterationLimit;
    config.gTestFactor_ = dmeConfig.GTestStatFactor;
    config.iterToLimit_ = dmeConfig.IterateToLimit;
    config.pulseAmpRegCoeff_  = dmeConfig.PulseAmpRegularization;
    config.snrDropThresh_ = dmeConfig.SnrDropThresh;
    config.snrThresh0_ = dmeConfig.MinAnalogSnrThresh0;
    config.snrThresh1_ = dmeConfig.MinAnalogSnrThresh1;
    config.successConfThresh_ = dmeConfig.SuccessConfidenceThresh;
    config.updateMethod_   = dmeConfig.ModelUpdateMethod;
    config.refSnr_ = movieInfo.refSnr;

    Cuda::CudaCopyToSymbol(&staticConfig, &config);
}

__device__ int TotalCount(const DmeEmDevice::LaneHist& hist)
{
    int count = hist.outlierCountHigh[threadIdx.x] + hist.outlierCountLow[threadIdx.x];
    for (int i = 0; i < numBins; ++i)
    {
        count += hist.binCount[i][threadIdx.x];
    }
    return count;
}

__device__ float Fractile(const DmeEmDevice::LaneHist& hist, float frac)
{
    assert(frac >= 0.f);
    assert(frac <= 1.0f);

    static constexpr auto inf = std::numeric_limits<float>::infinity();

    int totalCount = hist.outlierCountHigh[threadIdx.x] + hist.outlierCountLow[threadIdx.x];
    for (int i = 0; i < numBins; ++i) totalCount += hist.binCount[i][threadIdx.x];

    float ret;
    const auto nf = frac * totalCount;
    // Find the critical bin.
    auto n = hist.outlierCountLow[threadIdx.x];
    if (n > 0 && n >= nf)
    {
        // The precise fractile is in the low-outlier bin.
        // Since this bin is unbounded below, ...
        ret = -inf;
        return ret;
    }

    int i = 0;
    while ((n == 0 || n < nf) && i < numBins)
    {
        n += hist.binCount[i++][threadIdx.x];
    }

    if (n < nf)
    {
        // The precise fractile is in the high-outlier bin.
        // Since this bin is unbounded above, ...
        ret = +inf;
        return ret;
    }

    // Otherwise, the precise fractile is in a normal bin.
    // Interpolate within the critical bin.
    assert(i > 0);
    assert(n >= nf);
    i -= 1;     // Back up to the bin that pushed us over the target.
    auto x0 = hist.lowBound[threadIdx.x] + i * hist.binSize[threadIdx.x];
    const auto ni = hist.binCount[i][threadIdx.x];
    auto m = n - ni;
    assert(m < nf || (m == 0 && nf == 0));
    ret = x0 + hist.binSize[threadIdx.x] * (nf - m) / (ni + 1);

    return ret;
}


// Compute a preliminary scaling factor based on a fractile statistic.
__device__ float PrelimScaleFactor(const ZmwDetectionModel& model,
                                   const DmeEmDevice::LaneHist& hist)
{
    using std::max;  using std::min;
    using std::sqrt;

    // Define a fractile that includes all of the background and half of the pulse frames.
    const auto& bgMode = model.baseline;
    const float& bgVar = bgMode.var;
    const float bgSigma = sqrt(bgVar);
    const float& bgMean = bgMode.mean;
    const float thresh = 2.5f * bgSigma + bgMean;
    const float binSize = hist.binSize[threadIdx.x];

    // Note: This replicates the host original, which never ever included upper
    //       outliers and always includes lower outliers?
    float bgCount = 0;
    float totalCount = hist.outlierCountLow[threadIdx.x];
    {
        int i = 0;
        float binX = hist.lowBound[threadIdx.x];
        float rem = thresh - binX;
        while (rem > binSize)
        {
            totalCount += hist.binCount[i][threadIdx.x];
            binX += hist.binSize[threadIdx.x];
            rem = thresh - binX;
            i++;
        }
        bgCount = totalCount + rem / binSize * hist.binCount[i][threadIdx.x];
        for (; i < numBins; ++i)
        {
            totalCount += hist.binCount[i][threadIdx.x];
        }
    }
    const float fractile = 0.5f * (bgCount + totalCount) / totalCount;

    // Define the scale factor as the ratio between this fractile and the
    // average of the pulse signal means.
    float avgSignalMean = 0.0f;
    assert(model.analogs.size() == numAnalogs);
    for (unsigned int a = 0; a < numAnalogs; ++a)
    {
        avgSignalMean += model.analogs[a].mean;
    }
    avgSignalMean /= static_cast<float>(numAnalogs);

    // Moderate scaling by the clamped model confidence
    const float w = satlin(0, staticConfig.scaleSnrConfTol_, model.confidence);
    auto scaleFactor = Fractile(hist, fractile) / avgSignalMean;
    scaleFactor = (1.0f - w) * scaleFactor + w;

    // Clamp the scale factor to a "reasonable" size
    return min(max(scaleFactor, 0.1f), 10.0f);
}

/// Updates *detModel by increasing the amplitude of all detection modes by
/// \a scale. Also updates all detection mode covariances according
/// to the standard noise model. Ratios of amplitudes among detection modes
/// and properties of the background mode are preserved.
void __device__ ScaleModelSnr(const float& scale,
                              ZmwDetectionModel* detModel)
{
    assert (scale > 0.0f);
    const auto baselineCovar = detModel->baseline.var;
    auto& detectionModes_ = detModel->analogs;
    assert (detectionModes_.size() == numAnalogs);
    for (unsigned int a = 0; a < numAnalogs; ++a)
    {
        auto& dmi = detectionModes_[a];
        dmi.mean *= scale;
        auto cv = Analog(a).excessNoiseCV;
        dmi.var = ModelSignalCovar(cv*cv, dmi.mean, baselineCovar);
    }
    // TODO: Should we update updated_?
}

// Constants
// TODO remove duplication (was from NumericUtil.h)
constexpr float pi_f = 3.1415926536f;

/// The cumulative probability of the standard normal distribution (mean=0,
/// variance=1).
// TODO this duplicated NumericalUtil.h
__device__ float normalStdCdf(float x)
{
    float s = 1.f / sqrt(2.f);
    x *= s;
    const float r = 0.5f * erfc(-x);
    assert((r >= 0.0f) && (r <= 1.0f) || isnan(x));
    return r;
}

/// The cumulative probability at \a x of the normal distribution with \mean
/// and standard deviation \a stdDev.
/// \tparam FP a floating-point numeric type (including m512f).
// TODO this duplicated NumericalUtil.h
__device__ float normalCdf(float x, float mean = 0.0f, float stdDev = 1.0f)
{
    assert(stdDev > 0);
    const float y = (x - mean) / stdDev;
    return normalStdCdf(y);
}

// Apply a G-test significance test to assess goodness of fit of the model
// to the trace histogram.
//
// Note: This is a mostly-complete port of the host version, but it is
//       currently non-functional because it relies on the chi squared
//       distribution which we don't have access to on the GPU.
__device__ GoodnessOfFitTest<float>
Gtest(const DmeEmDevice::LaneHist& histogram, const ZmwDetectionModel& model)
{
    assert(blockDim.x == 64);
    //const auto& bb = histogram.BinBoundaries();
    auto bb = [&](int idx){
        return histogram.lowBound[threadIdx.x] + idx*histogram.binSize[threadIdx.x];
    };
    const auto& bg = model.baseline;

    // Cache the standard deviations.
    const auto bgStdDev = sqrt(bg.var);
    CudaArray<float, numAnalogs> dmStdDev;
    assert(model.analogs.size() == numAnalogs);
    for (unsigned int j = 0; j < numAnalogs; ++j)
    {
        dmStdDev[j] = sqrt(model.analogs[j].var);
    }

    // Compute the bin probabilities according to the model.
    // TODO: Improve accuracy when normalCdf is close to 1.0.
    CudaArray<float, numBins+1> p;
    for (unsigned int i = 0; i < numBins+1; ++i)
    {
        p[i] = Basecaller::normalCdf(bb(i), bg.mean, bgStdDev) * bg.weight;
        for (unsigned int j = 0; j < numAnalogs; ++j)
        {
            const auto& dmj = model.analogs[j];
            p[i] += Basecaller::normalCdf(bb(i), dmj.mean, dmStdDev[j]) * dmj.weight;
        }
    }

    // TODO: By splitting out the first iteration, should be able to fuse the
    // loop in adjacent_difference with the loop above.
    //std::adjacent_difference(p.cbegin(), p.cend(), p.begin());
    float tmp = p[0];
    float diff;
    for (int i = 0; i < numBins; ++i)
    {
        diff = p[i+1] - tmp;
        tmp = p[i+1];
        p[i+1] = diff;
    }

    assert(p[0] >= 0.0f);
    assert(p[0] <= 1.0f);
    assert(p[numBins] >= 0.0f);
    assert(p[numBins] <= 1.0f);

    // Compute the G test statistic.
    const auto n = [&](){
        int n = 0;
        for (int i = 0; i < numBins; ++i)
        {
            n += histogram.binCount[i][threadIdx.x];
        }
        return n;
    }();
    //const auto n = FloatVec(histogram.InRangeCount());
    float g = 0.0f;
    for (int i = 0; i < numBins; ++i)
    {
        const auto obs = histogram.binCount[i][threadIdx.x];
        const auto mod = n * p[i+1];
        const auto t = obs * log(obs/mod);
        if (obs > 0.f) g += t;
    }
    g *= 2.0f;

    // Compute the p-value.
    assert(CoreDMEstimator::nModelParams + 1 < static_cast<unsigned int>(numBins));
    const auto dof = numBins - CoreDMEstimator::nModelParams - 1;
    // TODO disabled because I don't have access to a gpu chi2 distribution on the GPU.
    assert(false);
    const auto pval = 0.f;
    //const auto pval = chi2CdfComp(g * gTestFactor_, dof);

    return {g, static_cast<float>(dof), pval};
}

// Compute the confidence factors of a model estimate, given the
// diagnostics of the estimation, a reference model.
__device__ PacBio::Cuda::Utility::CudaArray<float, ConfFactor::NUM_CONF_FACTORS>
ComputeConfidence(const DmeDiagnostics<float>& dmeDx,
                  const ZmwDetectionModel& refModel,
                  const ZmwDetectionModel& modelEst)
{
    const auto mldx = dmeDx.mldx;
    CudaArray<float, ConfFactor::NUM_CONF_FACTORS> cf;
    for (auto& val : cf) val = 1.f;

    // Check EM convergence.
    cf[ConfFactor::CONVERGED] = mldx.converged;

    // Check for missing baseline component.
    const auto& bg = modelEst.baseline;
    // TODO: Make this configurable.
    // Threshold level for background fraction.
    static const float bgFracThresh0 = 0.05f;
    static const float bgFracThresh1 = 0.15f;
    float x = satlin(bgFracThresh0, bgFracThresh1, bg.weight);
    cf[ConfFactor::BL_FRACTION] = x;

    // Check magnitude of residual baseline mean.
    x = bg.mean * bg.mean / bg.var;
    // TODO: Make this configurable.
    static const float bgMeanTol = 1.0f;
    assert(bgMeanTol > 0.0f);
    x = exp(-x / (2*bgMeanTol*bgMeanTol));
    cf[ConfFactor::BL_CV] = x;

    // Check for large deviation of baseline variance from reference variance.
    const auto& refBgVar = refModel.baseline.var;
    x = log2(bg.var / refBgVar);
    // TODO: Make this configurable.
    const float bgVarTol = 1.5f / (0.5f + refModel.confidence);
    x = exp(-x*x / (2*bgVarTol*bgVarTol));
    cf[ConfFactor::BL_VAR_STABLE] = x;

    // Check for missing pulse components.
    // Require that the first (brightest) and last (dimmest) are not absent.
    x = 1.0f;
    const auto& detModes = modelEst.analogs;
    const float analogMixFracThresh1_ = staticConfig.analogMixFracThresh1_;
    const float analogMixFracThresh0_ = staticConfig.analogMixFracThresh0_;
    if (analogMixFracThresh1_ > 0.0f)
    {
        assert(detModes.size() >= 1);
        assert(analogMixFracThresh0_ < analogMixFracThresh1_);
        x *= satlin(analogMixFracThresh0_, analogMixFracThresh1_, detModes.front().weight);
        x *= satlin(analogMixFracThresh0_, analogMixFracThresh1_, detModes.back().weight);
    }
    cf[ConfFactor::ANALOG_REP] = x;

    // Check for low SNR.
    x = detModes.back().mean;  // Assumes last analog is dimmest.
    const auto bgSigma = sqrt(bg.var);
    x /= bgSigma;
    x = satlin(staticConfig.snrThresh0_, staticConfig.snrThresh1_, x);
    cf[ConfFactor::SNR_SUFFICIENT] = x;

    // Check for large decrease in SNR.
    // This factor is specifically designed to catch registration errors in the
    // fit when the brightest analog is absent. In such cases, the weight of
    // the dimmest fraction can be substantial (the fit presumably robs some
    // weight from the background component)
    if (staticConfig.snrDropThresh_ < 0.0f) cf[ConfFactor::SNR_DROP] = 1.0f;
    else
    {
        const auto snrEst = detModes[0].mean / bgSigma;
        const auto& refDetModes = refModel.analogs;
        assert(refDetModes.size() >= 2);
        const auto& refSignal0 = refDetModes[0].mean;
        //PBAssert(all(refSignal0 >= 0.0f), "Bad SignalMean.");
        const auto& refSignal1 = refDetModes[1].mean;
        //PBAssert(all(refSignal1 >= 0.0f), "Bad SignalMean.");
        const auto refBgSigma = sqrt(refModel.baseline.var);
        //PBAssert(all(refBgSigma >= 0.0f), "Bad baseline sigma.");
        const auto refSnr0 = refSignal0 / refBgSigma;
        auto refSnr1 = refSignal1 / refBgSigma;
        refSnr1 *= staticConfig.snrDropThresh_;
        refSnr1 *= min(refModel.confidence, 1.0f);
        //PBAssert(all(refSnr1 < refSnr0),
        //         "Bad threshold in SNR Drop confidence factor.");
        x = satlin(refSnr1, sqrt(refSnr0*refSnr1), snrEst);
        cf[ConfFactor::SNR_DROP] = x;
    }

    // The G-test as a goodness-of-fit score.
    cf[ConfFactor::G_TEST] = dmeDx.gTest.pValue;

    return cf;
}

// TODO this code replicates host stat accumulator for float and PBHalf2 below
__device__ float Mean(const StatAccumState& stats)
{
    float mean = stats.moment1[threadIdx.x] / stats.moment0[threadIdx.x];
    return mean + stats.offset[threadIdx.x];
};

/// The unbiased sample variance of the aggregated samples.
/// NaN if Count() < 2.
__device__ float Variance(const StatAccumState& stats)
{
    float var = stats.moment1[threadIdx.x] * stats.moment1[threadIdx.x];
    var /= stats.moment0[threadIdx.x];
    var = (stats.moment2[threadIdx.x] - var);
    var /= (stats.moment0[threadIdx.x] - 1.0f);
    var = max(var, 0.0f);

    const float nan = std::numeric_limits<float>::quiet_NaN();
    return stats.moment0[threadIdx.x] > 1.0f ? var : nan;
};

__device__ void EvolveModel(FiTypeDevice estFI,
                            const BaselinerStatAccumState& blStats,
                            ZmwDetectionModel* model,
                            FiTypeDevice modFI)
{
    // Current and next model
    auto mod0Cntr = modFI.lo + modFI.up / 2;
    auto mod1Cntr = estFI.lo + estFI.up / 2;

    const float thl = 0.5f * (mod0Cntr + mod1Cntr);

    // TODO: Make these configurable.       ALP: Adaptive Laser Power
    static const float t00 = 56160.0f;   // 50th percentile of ALP duration
    static const float t10 = 168480.0f;  // 97.5th %-ile of ALP duration

    // Compute the nominal confidence half-life.
    const float& hl0 = staticConfig.confidHalfLife0_;
    const float& hl1 = staticConfig.confidHalfLife1_;
    const float hl = hl0 + (hl1 - hl0) * satlin(t00, t10, thl);     // frames

    // EvolveConfidence
    const auto t01 = mod0Cntr;
    const auto t11 = mod1Cntr;
    auto confHalfLife = hl;
    const float tDiff = static_cast<float>(t11 - t01);
    const auto m = exp2(-abs(tDiff)/confHalfLife);
    model->confidence *= m;
}

__device__ void PrelimEstimate(const BaselinerStatAccumState& blStatAccState,
                               ZmwDetectionModel* model)
{
    assert(model != nullptr);

    const float nBlFrames   = blStatAccState.baselineStats.moment0[threadIdx.x];
    const float totalFrames = blStatAccState.fullAutocorrState.basicStats.moment0[threadIdx.x];
    const float blWeight    = max(nBlFrames / totalFrames, 0.01f);

    // Reject baseline statistics with insufficient data
    float nBaselineMin(3.0f);
    auto mask = nBlFrames >= nBaselineMin;
    ZmwAnalogMode& m0blm = model->baseline;
    const StatAccumState& blsa  = blStatAccState.baselineStats;

    auto blMean = mask ? Mean(blsa)     : m0blm.mean;
    auto blVar  = mask ? Variance(blsa) : m0blm.var;

    const auto blVarMax = baseConfig.BaselineVarianceMax();
    blVar = std::clamp(blVar, baseConfig.BaselineVarianceMin(), blVarMax);

    auto& detectionModes = model->analogs;

    assert(isfinite(blMean));
    assert(isfinite(blVar) && (blVar > 0.0f));
    assert (detectionModes.size() == numAnalogs);

    // Rescale
    auto scale = sqrt(blVar / m0blm.var);

    for (uint32_t i = 0; i < numAnalogs; ++i)
    {
        auto& mode = detectionModes[i];
        mode.mean = mode.mean * scale;
        auto cv = Analog(i).excessNoiseCV;
        mode.var  = ModelSignalCovar(cv*cv, mode.mean, blVar);
        mode.weight = 0.25f*(1.0f-blWeight);
    }

    model->baseline.mean = blMean;
    model->baseline.var = blVar;
    model->baseline.weight = blWeight;

    // Frame interval is not updated since it is not exported

    float conf = 0.1f * satlin(0.0f, 500.0f, nBlFrames - nBaselineMin);
    conf *= satlin(blVarMax, 0.5f * (baseConfig.BaselineVarianceNominal() + blVarMax), blVar);
    model->confidence = conf;
}

// Use the trace histogram and the input detection model to compute a new
// estimate for the detection model. Mix the new estimate with the input
// model, weighted by confidence scores. That result is returned in detModel.
__device__ void EstimateLaneDetModel(FiTypeDevice estFI,
                                     const DmeEmDevice::LaneHist& hist,
                                     const BaselinerStatAccumState& blStatAccState,
                                     LaneDetModel* detModel,
                                     FiTypeDevice modFI)
{
    assert(detModel != nullptr);

    ZmwDetectionModel model0;
    if (threadIdx.x%2 == 0)
        model0.Assign<0>(*detModel, threadIdx.x/2);
    else
        model0.Assign<1>(*detModel, threadIdx.x/2);

    FiTypeDevice wmFI;
    EvolveModel(estFI, blStatAccState, &model0, modFI);
    modFI = estFI;

    // Update model based on estimate of baseline variance
    // with confidence-weighted method
    ZmwDetectionModel workModel = model0;
    wmFI = modFI;

    PrelimEstimate(blStatAccState, &workModel);

    UpdateModel(workModel, &model0);
    modFI = wmFI;

    // Make a working copy of the detection model.
    workModel = model0;
    wmFI = modFI;

    // The term "mode" refers to a component of the mixture model.
    auto& bgMode = workModel.baseline;
    auto& pulseModes = workModel.analogs;

    const auto& numFrames = TotalCount(hist);

    // Scale the model based on fractile of the data.
    const auto scaleFactor = PrelimScaleFactor(workModel, hist);
    ScaleModelSnr(scaleFactor, &workModel);

    const auto binSize = hist.binSize[threadIdx.x];
    const auto hBinSize = 0.5f * binSize;

    // Define working variables for model parameters.
    const auto nModes = numAnalogs + 1;
    using ModeArray = CudaArray<float, nModes>;
    ModeArray rho;     // Mixture fraction for each mode.
    ModeArray mu;      // Mean of each mode.
    ModeArray var;     // Variance of each mode.

    const float rhoMin = 0.001f;
    const float blVarMin = baseConfig.BaselineVarianceMin();
    const float blVarMax = baseConfig.BaselineVarianceMax();

    // Variance associated with data binning.
    const auto varQuant = binSize * binSize / 12.f;

    rho[0] = max(bgMode.weight, rhoMin);
    assert(rho[0] <= 1.0f);
    mu[0] = bgMode.mean;
    var[0] = bgMode.var + varQuant;
    assert(var[0] >= blVarMin);
    for (unsigned int a = 0; a < numAnalogs; ++a)
    {
        const auto k = a + 1;
        //PBAssert(k < nModes, "k < nModes");
        const auto& pma = pulseModes[a];
        rho[k] = pma.weight;
        assert(rho[k] <= 1.0f);
        mu[k] = pma.mean;
        var[k] = pma.var + varQuant;
        assert(var[k] >= blVarMin);
    }

    // Enforce sanity bound on baseline variance.
    var[0] = min(var[0], blVarMax);

    // Enforce normalization of mixture fractions.
    {
        float rhoSum = 0.f;
        for (int i = 0; i < numAnalogs+1; ++i)
        {
            rhoSum += rho[i];
        }
        auto rhoSumInv = 1.f / rhoSum;
        for (int i = 0; i < numAnalogs+1; ++i)
        {
            rho[i] *= rhoSumInv;
        }
    }

    // Initialize estimates to initial model.
    ModeArray rhoEst = rho;
    ModeArray muEst = mu;
    ModeArray varEst = var;

    // The relative pulse amplitudes.
    CudaArray<float, numAnalogs> rpa;
    for (unsigned int a = 0; a < numAnalogs; ++a)
    {
        rpa[a] = Analog(a).relAmplitude;
        // TODO data error handling
        //if (rpa[a] <= 0.0f)
        //{
        //    throw PBException("Bad relative amplitude in analog properties."
        //                      " Relative amplitudes must be greater than zero.");
        //}
    }

    // Initialize the model parameter for pulse amplitude scale, which is
    // needed for the prior when calculating the posterior.
    float s = [&]()
    {
        float tmp1 = 0.f;
        float tmp2 = 0.f;
        for (int i = 0; i < numAnalogs; ++i)
        {
            tmp1 += mu[i+1]*rpa[i];
            tmp2 += rpa[i]*rpa[i];
        }
        return tmp1 / tmp2;
    }();
    //FloatVec s = (mu.tail(numAnalogs) * rpa.cast<FloatVec>()).sum() / rpa.square().sum();

    // Prior hyperparameters for scale parameter.
    // Undo the preliminary scaling based on percentile to get back to the
    // means of initModel.
    const float sExpect = s / scaleFactor;

    // sExpectWeight is the inverse of the variance of the normal prior for s.
    const float sExpectWeight = model0.confidence * staticConfig.pulseAmpRegCoeff_;

    // Log likelihood--really a posterior since we've added a prior for s.
    float logLike {numeric_limits<float>::lowest()};
    float logLikePrev {logLike};

    // Iteration limit for EM.
    const unsigned int iterLimit = staticConfig.emIterLimit_;

    // Define lower bound for bin probability.
    const float binProbMin
            = max(numFrames * binSize, 1.0f)
            * numeric_limits<float>::min();

    // Initialize intra-lane failure codes.
    int32_t zStatus = DmeEmDevice::OK;
    if (numFrames < CoreDMEstimator::nFramesMin) zStatus |= DmeEmDevice::INSUF_DATA;

    DmeDiagnostics<float> dmeDx {};
    dmeDx.fullEstimation = true;

    // TODO: Need to track frame interval.
//    dmeDx.startFrame = dtbs.front()->StartFrame();
//    dmeDx.stopFrame = dtbs.back()->StopFrame();

    MaxLikelihoodDiagnostics<float>& mldx = dmeDx.mldx;
    mldx.degOfFreedom = numFrames - CoreDMEstimator::nModelParams;

    // Expectation-maximization of grouped data.
    // See G. J. McLachlan and P. N. Jones,
    // Biometrics, Vol. 44, No. 2 (June, 1988), pp. 571-578.
    // http://www.jstor.org/stable/2531869

    const float n_j_sum = [&](){
        float sum = 0;
        for (int i = 0; i < numBins; ++i)
        {
            sum += hist.binCount[i][threadIdx.x];
        }
        return sum;
    }();

    unsigned int it = 0;
    for (; it < iterLimit; ++it)
    {
        // E-step

        // Inverses of variances.
        ModeArray hVarinv = var;
        for (auto& val : hVarinv) val = 0.5f / val;

        const float log_2pi = log(2.0f * pi_f);
        ModeArray prefactors;
        ModeArray c_i;
        ModeArray mom1;
        for (int i = 0; i < rho.size(); ++i)
        {
            prefactors[i] = log(rho[i]) - 0.5f * log(var[i]) - 0.5f * log_2pi;
            c_i[i] = 0.f;
            mom1[i] = 0.f;
        }

        float probSum(0.0f);
        float weightedProbSum(0.0f);
        float mom2 = 0.0f;   // Only needed for background mode.
        // correction terms to account for the fact that we'll use the
        // current mu in the loop below, while the math really wants
        // the updated mu not computed until later.
        float correct1 = 0.0f;
        float correct2 = 0.0f;

        struct CornerVals
        {
            float tau[nModes];
            float cProb;
        };
        auto cornerComp = [&](int b, CornerVals& cv, float x)
        {
            // First compute the log of the component probabilities.
            for (int i = 0; i < nModes; ++i)
            {
                const auto& y = x - mu[i];
                cv.tau[i] = prefactors[i] - y*y*hVarinv[i];
            }

            // Next compute the log likelihood of each bin boundary, which is
            // the log of the sum of exp(tau_i_x) over modes (i).
            // Use log-sum-exp trick to avoid uniform underflow. (PTVF2007, Equation 16.1.9)
            float tauMax = cv.tau[0];
            for (int i = 1; i < nModes; ++i)
            {
                tauMax = max(tauMax, cv.tau[i]);
            }
            float llb = [&](){
                float ret = 0.f;
                for (int i = 0; i < nModes; ++i)
                {
                    ret += __expf(cv.tau[i] - tauMax);
                }
                return __logf(ret) + tauMax;
            }();

            // Convert to likelihood ratio.
            for (int i = 0; i < nModes; ++i)
            {
                cv.tau[i] = __expf(cv.tau[i] - llb);
            }

            // Record the total probability density at the bin boundaries.
            cv.cProb = __expf(llb);
        };

        auto centerComp = [&](int b, const CornerVals& c1, const CornerVals& c2, float x0, float x1)
        {
            // Constrain bin probability to a minimum positive value.
            // Avoids 0 * log(0) in computation of log likelihood.
            auto binProb = max(hBinSize * (c1.cProb + c2.cProb), binProbMin);
            float binCount = hist.binCount[b][threadIdx.x];
            probSum += binProb;
            weightedProbSum += binCount * __logf(binProb);

            const auto factor = binCount / binProb;

            for (unsigned int m = 0; m < nModes; ++m)
            {
                // Relative weight of each mode. CSMM2002, Equation 5.
                // Use trapezoidal approximation of expectation of tau over each bin.
                auto tmp1 = c1.cProb * c1.tau[m];
                auto tmp2 = c2.cProb * c2.tau[m];

                c_i[m] += (tmp1 + tmp2) * factor;
                mom1[m] += (tmp1 * x0 + tmp2 * x1) * factor;
            }

            x0 -= mu[0];
            x1 -= mu[0];

            float co2 = c1.cProb * c1.tau[0];
            float co1 = co2 * x0;
            float m = x0 * co1;

            float tmp = c2.cProb * c2.tau[0];
            co2 += tmp;
            tmp *= x1;
            co1 += tmp;
            m += x1 * tmp;

            tmp = factor;
            mom2 += m * tmp;
            correct2 += co2 * tmp;
            correct1 += co1 * tmp;
        };

        // TODO: Check for bins where binProb == 0 but n_j > 0.
        CornerVals c1;
        CornerVals c2;
        float x0 = hist.lowBound[threadIdx.x];
        float x1 = x0 + binSize;
        cornerComp(0, c1, x0);
        for (unsigned int b = 0; b < numBins; ++b)
        {
            cornerComp(b+1, c2, x1);
            centerComp(b, c1, c2, x0, x1);
            c1 = c2;
            x0 = x1;
            x1 += binSize;
        }
        // TODO: Note the cancellation of the (0.5f * binSize) factor. Possible minor optimization opportunity.
        // TODO: When used, tau is always multiplied by boundaryProb. So why bother dividing that factor out when first computing tau?

        for (int i = 0; i < nModes; ++i)
        {
            c_i[i] *= 0.5f * binSize;
            mom1[i] *= 0.5f * binSize;
        }

        // Update log posterior.
        logLikePrev = logLike;
        logLike = weightedProbSum - n_j_sum * log(probSum);
        auto sdiff = s-sExpect;
        logLike -= 0.5f * sdiff * sdiff * sExpectWeight;

        // Check for convergence.
        const float deltaLogLike {logLike - logLikePrev};
        const float convTol = 1e-4f * abs(logLike);  // TODO: Make this configurable.
        // TODO if this loop body ever gets any code that does anything, may need to
        // use warp voting to share information for the whole lane?
        if (deltaLogLike < -convTol)
        {
            // TODO: We need lane tracking.
            // Ideally, this should not happen.
            // TODO: Should we abort estimation for these ZMWs?
            // TODO: PBLOGGER_DEBUG is broken.
//            PBLOGGER_DEBUG(logger_)
//                    << "Likelihood decreased during EM: Pool " << poolId_
//                    << ", iteration " << it << '.';
        }

        // TODO: Seems like we ought to eliminate or relax the lower bound
        // condition on deltaLogLike here.
        bool conv = (deltaLogLike >= 0) && (deltaLogLike < convTol)
                  && (logLike >= mldx.logLike);
        mldx.Converged(conv, it, logLike, deltaLogLike);

        // Update result for converged ZMWs.
        if (conv) for (unsigned int k = 0; k < nModes; ++k)
        {
            rhoEst[k] = rho[k];
            muEst[k]  = mu[k];
            varEst[k] = var[k];
        }

        // Note: This __all_sync is a bit weird.  It was put in place to match the host
        //       impl where the SIMD approach required all ZMW in a lane to converge if
        //       we're going to break early.  However here:
        //       * This sync is only for a warp, which really is only 32 ZMW not 64, so
        //         we're already inconsistent
        //       * With the CUDA SIMT model we could have threads break
        //         and exit individually when they are done with no
        //       Ideally we'd match perfectly and do a whole lane-level synchronization,
        //       but that's nontrivial to write since it involves coordination between
        //       two warps
        if (!staticConfig.iterToLimit_ && __all_sync(0xFFFFFFFF, mldx.converged || (zStatus != static_cast<int>(DmeEmDevice::OK))))
        {
            // TODO: Maybe count how many times we achieve this?
            break;
        }

        // M-step

        // Mixing fractions.
        // Constrain each component to have some minimal mixing fraction.
        float rhoSum = 0.0f;
        for (int i = 0; i < nModes; ++i)
        {
            c_i[i] = max(c_i[i], 0.1f);
            rho[i] = c_i[i] / n_j_sum;
            rhoSum += rho[i];
        }
        for (auto& x : rho) x /= rhoSum;

        auto oldMu = mu[0];
        // Background mean.
        mu[0] = mom1[0] / c_i[0];
        auto muDiff = oldMu - mu[0];

        // Amplitude scale parameter.
        float numer = 0.0f;
        float denom = 0.0f;
        for (unsigned int a = 0; a < numAnalogs; ++a)
        {
            const auto i = a + 1;
            auto varinv = 2.0f * hVarinv[i];
            numer += rpa[a] * varinv * mom1[i];
            denom += rpa[a] * rpa[a] * varinv * c_i[i];
        }

        // Add regularization/prior terms.
        numer += sExpect * sExpectWeight;
        denom += sExpectWeight;

        s = numer / denom;

        // The minimum bound for the pulse-amplitude scale parameter.
        static const float minSnr = 0.5f;
        const float minPulseMean = max(1.0f, baseConfig.signalScaler);
        auto rpaMin = rpa[0]; for (int i = 1; i < rpa.size(); ++i) rpaMin = min(rpaMin, rpa[i]);
        const auto sMin = max(minSnr * sqrt(var[0]), minPulseMean) / rpaMin;

        // Constrain s > sMin.
        const auto veryLowSignal = (s < sMin);
        if (veryLowSignal)
        {
            s = sMin;
            zStatus |= DmeEmDevice::VLOW_SIGNAL;
        }

        // Pulse mode means.
        for (int i = 0; i < numAnalogs; ++i) mu[i+1] = s * static_cast<float>(rpa[i]);

        // Background variance.
        // Need to apply correction terms since it was computed above with the old
        // mean instead of the newest update.
        // So far we've only run on "friendly" data, in which case it appears
        // that these correction terms make no practical difference.  Their
        // utility needs to be evaluated on more realistic data, as we will
        // run measurably faster if we can skip the computation of one or even
        // both of these terms.
        mom2 += muDiff * correct1 + muDiff * muDiff * correct2;
        mom2 *= hBinSize;
        var[0] = mom2 / c_i[0] + varQuant;
        var[0] = std::clamp(var[0], blVarMin, blVarMax);

        // Each pulse mode variance is computed as a function of the background
        // variance and the pulse mode mean.
        // Note that we've ignored these dependencies when updating the means
        // and the background variance above.
        for (unsigned int i = 0; i < numAnalogs; ++i)
        {
            auto cv = Analog(i).excessNoiseCV;
            var[i+1] = ModelSignalCovar(cv*cv, mu[i+1], var[0]);
        }
    }

    if (!mldx.converged) zStatus |= DmeEmDevice::NO_CONVERGE;

    using std::isfinite;

    // Package estimation results.
    //PBAssert(all(isfinite(rhoEst[0])), "all(isfinite(rhoEst[0])");
    bgMode.weight = rhoEst[0];
    //PBAssert(all(isfinite(muEst[0])), "all(isfinite(muEst[0]))");
    bgMode.mean = muEst[0];
    //PBAssert(all(isfinite(varEst[0])), "all(isfinite(varEst[0]))");
    bgMode.var = varEst[0];
    for (unsigned int a = 0; a < numAnalogs; ++a)
    {
        auto& pda = pulseModes[a];
        const auto i = a + 1;
        //PBAssert(all(isfinite(rhoEst[i])), "all(isfinite(rhoEst[i])");
        pda.weight = rhoEst[i];
        //PBAssert(all(isfinite(muEst[i])), "all(isfinite(muEst[i]))");
        pda.mean = muEst[i];
        //PBAssert(all(isfinite(varEst[i])), "all(isfinite(varEst[i]))");
        pda.var = varEst[i];
    }

    // Note: this is disabled until we have a chi squared cfd on the gpu.
    assert (staticConfig.gTestFactor_ < 0.);
    if (staticConfig.gTestFactor_ >= 0.0f) dmeDx.gTest = Gtest(hist, workModel);
    else assert(dmeDx.gTest.pValue == 1.0f);

    // Compute confidence score.
    dmeDx.confidFactors = ComputeConfidence(dmeDx, model0, workModel);
    {
        using std::min;  using std::max;
        float conf = 1.0f;
        for (const auto& cf : dmeDx.confidFactors) conf *= cf;
        conf = min(max(0.0f, conf), 1.0f);
        if (conf < staticConfig.successConfThresh_) conf = 0;
        workModel.confidence = conf;
    }
    // TODO: Push results to DmeDumpCollector.
    //    if (this->dmeDumpCollector_)
    //    {
    //        // TODO: Add lane and zmw event codes.
    //        this->dmeDumpCollector_->CollectRawEstimate1D(*dtbs.front(), hist, dmeDx, *workModel);
    //    }

    // Update the current model, model0.
    UpdateModel(workModel, &model0);
    modFI = wmFI;

    // Transcribe back to *detModel.
    model0.Export(threadIdx.x/2, threadIdx.x%2, detModel);
}

__global__ void EstimateKernel(FiTypeDevice estFI,
                               Cuda::Memory::DeviceView<const DmeEmDevice::LaneHist> hists,
                               Cuda::Memory::DeviceView<const BaselinerStatAccumState> blStatsState,
                               Cuda::Memory::DeviceView<LaneDetModel> models,
                               FiTypeDevice currFI)
{
    EstimateLaneDetModel(estFI, hists[blockIdx.x], blStatsState[blockIdx.x], &models[blockIdx.x], currFI);
}


void DmeEmDevice::EstimateImpl(const PoolHist &hist, 
                               const Data::BaselinerMetrics& metrics,
                               PoolDetModel *detModelPool) const
{
    // We assume that frame intervals of the trace histogram and the baseliner
    // statistics are equal.
    const auto& hfi = hist.frameInterval;
    assert(hfi == metrics.frameInterval);

    const auto& pfi = detModelPool->frameInterval;

    FiTypeDevice hstFI { hfi.Lower(), hfi.Upper() };
    FiTypeDevice modFI { pfi.Lower(), pfi.Upper() };

    Cuda::PBLauncher(EstimateKernel, hist.data.Size(), laneSize)
                    (hstFI, hist.data, metrics.baselinerStats, detModelPool->data, modFI);
    Cuda::CudaSynchronizeDefaultStream();

    detModelPool->frameInterval = hfi;
}

__global__ void InitModel(Cuda::Memory::DeviceView<const BaselinerStatAccumState> stats,
                          Cuda::Memory::DeviceView<LaneDetModel> models)
{
    auto& blStats = stats[blockIdx.x];
    auto& model = models[blockIdx.x];

    const auto& blsa = blStats.baselineStats;
    const auto& basicStats = blStats.fullAutocorrState.basicStats;

    // TODO this code replicates host stat accumulator
    auto half2Mean = [](const auto& stats) -> PBHalf2
    {
        return { stats.moment1[threadIdx.x*2]/stats.moment0[threadIdx.x*2],
                 stats.moment1[threadIdx.x*2+1]/stats.moment0[threadIdx.x*2+1]
        };
    };

    /// The unbiased sample variance of the aggregated samples.
    /// NaN if Count() < 2.
    auto half2Variance = [](const auto& stats) -> PBHalf2
    {
        const PBHalf2 nan = std::numeric_limits<float>::quiet_NaN();
        const PBHalf2 mom0 { stats.moment0[2*threadIdx.x], stats.moment0[2*threadIdx.x+1]};
        const PBFloat2 mom1 { stats.moment1[2*threadIdx.x], stats.moment1[2*threadIdx.x+1]};
        const PBFloat2 mom2 { stats.moment2[2*threadIdx.x], stats.moment2[2*threadIdx.x+1]};

        PBFloat2 tmp = mom1 * mom1 / mom0;
        tmp = (mom2 - tmp) / (mom0 - 1.0f);
        // We're having problems with the baseline variance overflowing a half precision
        // storage.  Something is already terribly wrong if our variance is over
        // 65k, but we'll put a limiter here because having a literal infinity run
        // around is causing problems elsewhere
        PBHalf2 var = clamp(PBHalf2{tmp.X(), tmp.Y()}, 0.0f, 65000.0f);
        return Blend(mom0 > 1.0f, var, nan);
    };

    PBHalf2 blMean =  half2Mean(blsa);
    PBHalf2 blVar  = half2Variance(blsa);
    const PBHalf2 blWeight = {
        blsa.moment0[2*threadIdx.x] / basicStats.moment0[2*threadIdx.x],
        blsa.moment0[2*threadIdx.x+1] / basicStats.moment0[2*threadIdx.x+1]
    };

    // Set the confidence to a small nominal value.
    model.Confidence()[threadIdx.x] = 0.1f;

    // If baseline frame count is insufficient, blMean and blVar can be NaN.
    // In this case, just adopt some semi-arbitrary fallback value.
    blMean = Blend(isnan(blMean), baseConfig.fallbackBaselineMean, blMean);
    blVar = Blend(isnan(blVar), baseConfig.fallbackBaselineVariance, blVar);

    // Constrain variance to reasonable range.
    const float blVarMin = baseConfig.BaselineVarianceMin();
    const float blVarMax = baseConfig.BaselineVarianceMax();
    blVar = clamp(blVar, blVarMin, blVarMax);


    const auto refSignal = staticConfig.refSnr_ * sqrt(blVar);
    const auto& aWeight = 0.25f * (1.0f - blWeight);
    model.BaselineMode().means[threadIdx.x] = blMean;
    model.BaselineMode().vars[threadIdx.x] = blVar;
    model.BaselineMode().weights[threadIdx.x] = blWeight;
    for (int a = 0; a < numAnalogs; ++a)
    {
        const auto aMean = max(blMean + staticConfig.analogs[a].relAmplitude * refSignal, 0);
        auto& aMode = model.AnalogMode(a);
        aMode.means[threadIdx.x] = aMean;

        // This noise model assumes that the trace data have been converted to
        // photoelectron units.
        auto cv = Analog(a).excessNoiseCV;
        aMode.vars[threadIdx.x] = min(65000.0f, ModelSignalCovar(cv*cv, aMean, blVar));

        aMode.weights[threadIdx.x] = aWeight;
    }
}

DmeEmDevice::PoolDetModel
DmeEmDevice::InitDetectionModels(const PoolBaselineStats& blStats) const
{
    PoolDetModel pdm (PoolSize(), Cuda::Memory::SyncDirection::HostReadDeviceWrite, SOURCE_MARKER());

    Cuda::PBLauncher(InitModel, PoolSize(), laneSize/2)(blStats.baselinerStats, pdm.data);
    Cuda::CudaSynchronizeDefaultStream();

    pdm.frameInterval = blStats.frameInterval;
    return pdm;
}

}}}     // namespace PacBio::Mongo::Basecaller
