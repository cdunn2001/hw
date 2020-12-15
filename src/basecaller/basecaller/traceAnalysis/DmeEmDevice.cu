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
//  Defines members of class DmeEmHost.

#include "DmeEmDevice.h"

#include <limits>

#include <common/cuda/memory/AllocationViews.cuh>
#include <common/cuda/streams/LaunchManager.cuh>

#include <dataTypes/configs/BasecallerDmeConfig.h>
#include <dataTypes/configs/MovieConfig.h>

using std::numeric_limits;
using namespace PacBio::Cuda::Utility;
using namespace PacBio::Mongo::Data;


struct StaticConfig{
    CudaArray<AnalogMode, 4> analogs;
    float analogMixFracThresh_;
    unsigned short emIterLimit_;
    float gTestFactor_;
    bool iterToLimit_;
    float pulseAmpRegCoeff_;
    float snrDropThresh_;
    float snrThresh0_;
    float snrThresh1_;
    float successConfThresh_;
    float fixedBaselineParams_;
    float fixedBaselineMean_;
    float fixedBaselineVar_;
    float refSnr_;
};

__constant__ StaticConfig staticConfig;

__device__ const AnalogMode& Analog(int i)
{
    return staticConfig.analogs[i];
}

namespace {

//template <typename T>
//using DynamicArray1 = Eigen::Array<T, 1, Eigen::Dynamic>;
//
//template <typename T>
//using DynamicArray2 = Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>;

template <typename T, size_t len>
using Array1D = CudaArray<T, len>;

template <typename T, size_t len1, size_t len2>
using Array2D = CudaArray<CudaArray<T, len2>, len1>;

}   // anonymous


namespace PacBio {
namespace Mongo {
namespace Basecaller {

template <int low>
__device__ void UpdateTo(const ZmwAnalogMode& from, LaneAnalogMode<Cuda::PBHalf2, 32>& to, int idx, float fraction)
{
    const float a = fraction;
    const float b = 1 - fraction;
    to.means[idx].Set<low>(a * from.mean + b * to.means[idx].Get<low>());
    to.vars[idx].Set<low>(a * from.var + b * to.vars[idx].Get<low>());
}

template <int low>
__device__ void UpdateTo(const ZmwDetectionModel& from, LaneModelParameters<Cuda::PBHalf2, 32>& to, int idx, float fraction)
{
    // BENTODO no updated boolean
    UpdateTo<low>(from.baseline, to.BaselineMode(), idx, fraction);
    for (int i = 0; i < to.numAnalogs; ++i)
    {
        UpdateTo<low>(from.analogs[i], to.AnalogMode(i), idx, fraction);
    }
    // BENTODO no frame interval to update
}

template <int low>
__device__ void UpdateTo(const ZmwDetectionModel& from, LaneModelParameters<Cuda::PBHalf2, 32>& to, int idx)
{
    // BENTODO no confidence stored in LaneModelParamters
    float toConfidence = 0;
    assert (from.confidence >= 0.0f);
    assert (toConfidence >= 0.0f);

    const auto confSum = from.confidence + toConfidence;
    const float fraction = confSum > 0.0f ? from.confidence / confSum: 0.f;

    assert (fraction >= 0.0f);
    assert (fraction <= 1.0f);
    //assert ((fraction > 0) | (confSum == Confidence())));

    UpdateTo<low>(from, to, idx, fraction);
    // BENTODO no confidence stored in LaneModelParamters
    //Confidence(confSum);
}

// Static constants
constexpr unsigned short DmeEmDevice::nModelParams;
constexpr unsigned int DmeEmDevice::nFramesMin;

// BENTODO clean duplication?
// Duplication used to live in DetectionModelEstimator, but now lives in DmeEmHost
__device__ float ModelSignalCovar(
        const Data::AnalogMode& analog,
        float signalMean,
        float baselineVar)
{
    baselineVar += signalMean;
    auto tmp = analog.excessNoiseCV * signalMean;
    baselineVar+= tmp*tmp;
    return baselineVar;
}


DmeEmDevice::DmeEmDevice(uint32_t poolId, unsigned int poolSize)
    : CoreDMEstimator(poolId, poolSize)
{ }

// static
void DmeEmDevice::Configure(const Data::BasecallerDmeConfig &dmeConfig,
                          const Data::MovieConfig &movConfig)
{
    CoreDMEstimator::Configure(dmeConfig, movConfig);

    // TODO: Validate values.
    // TODO: Log settings.
    StaticConfig config;
    for (size_t i = 0; i < movConfig.analogs.size(); i++)
    {
        config.analogs[i] = movConfig.analogs[i];
    }
    config.analogMixFracThresh_ = dmeConfig.AnalogMixFractionThreshold;
    config.emIterLimit_ = dmeConfig.EmIterationLimit;
    config.gTestFactor_ = dmeConfig.GTestStatFactor;
    config.iterToLimit_ = dmeConfig.IterateToLimit;
    config.pulseAmpRegCoeff_ = dmeConfig.PulseAmpRegularization;
    config.snrDropThresh_ = dmeConfig.SnrDropThresh;
    config.snrThresh0_ = dmeConfig.MinAnalogSnrThresh0;
    config.snrThresh1_ = dmeConfig.MinAnalogSnrThresh1;
    config.successConfThresh_ = dmeConfig.SuccessConfidenceThresh;
    config.refSnr_ = movConfig.refSnr;
    // BENTODO this duplicates DetectionModelFilter class?
    if (dmeConfig.Method == Data::BasecallerDmeConfig::MethodName::Fixed &&
        dmeConfig.SimModel.useSimulatedBaselineParams == true)
    {
        config.fixedBaselineParams_ = true;
        config.fixedBaselineMean_ = dmeConfig.SimModel.baselineMean;
        config.fixedBaselineVar_ = dmeConfig.SimModel.baselineVar;
    } else
    {
        config.fixedBaselineParams_ = false;
        config.fixedBaselineMean_ = 0;
        config.fixedBaselineVar_ = 0;
    }
    Cuda::CudaCopyToSymbol(staticConfig, &config);
}

__global__ void EstimateKernel(Cuda::Memory::DeviceView<const DmeEmDevice::LaneHist> hists,
                               Cuda::Memory::DeviceView<DmeEmDevice::LaneDetModel> models)
{

    DmeEmDevice::EstimateLaneDetModel(hists[blockIdx.x], &models[blockIdx.x]);
}


void DmeEmDevice::EstimateImpl(const PoolHist &hist, PoolDetModel *detModel) const
{
    Cuda::PBLauncher(EstimateKernel, hist.data.Size(), laneSize)(hist.data, *detModel);
    Cuda::CudaSynchronizeDefaultStream();
}

__device__ int TotalCount(const DmeEmDevice::LaneHist& hist)
{
    int count = hist.outlierCountHigh[threadIdx.x] + hist.outlierCountLow[threadIdx.x];
    for (int i = 0; i < DmeEmDevice::LaneHist::numBins; ++i)
    {
        count += hist.binCount[i][threadIdx.x];
    }
    return count;
}

// Constants
// BENTODO remove duplication (was from NumericUtil.h)
constexpr float pi_f = 3.1415926536f;

__device__ void DmeEmDevice::EstimateLaneDetModel(const LaneHist& hist,
                                                  LaneDetModel* detModel)
{
    // TODO: Evolve model. Use trace autocorrelation to adjust confidence
    // half-life.

    const auto& numFrames = TotalCount(hist);

    // Make a working copy of the detection model.
    ZmwDetectionModel workModel;
    if (threadIdx.x % 2 == 0)
        workModel.Assign<0>(*detModel, threadIdx.x/2);
    else
        workModel.Assign<1>(*detModel, threadIdx.x/2);

    // Keep a copy of the initial model.
    const auto initModel = workModel;

    // The term "mode" refers to a component of the mixture model.
    auto& bgMode = workModel.baseline;
    auto& pulseModes = workModel.analogs;

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

    // Variance associated with data binning.
    // BENTODO restore to const?
    auto varQuant = hist.binSize[threadIdx.x];
    varQuant = varQuant*varQuant / 12.f;
    //const auto& varQuant = pow2(hist.BinSize()) / 12.0f;

    rho[0] = bgMode.weight;
    mu[0] = bgMode.mean;
    var[0] = bgMode.var + varQuant;
    for (unsigned int a = 0; a < numAnalogs; ++a)
    {
        const auto k = a + 1;
        // BENTODO error handling, all PBAsserts commented out
        //PBAssert(k < nModes, "k < nModes");
        const auto& pma = pulseModes[a];
        rho[k] = pma.weight;
        mu[k] = pma.mean;
        var[k] = pma.var + varQuant;
    }

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
        rpa[a] = ::Analog(a).relAmplitude;
        // BENTODO data error handling
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
    const float sExpectWeight = initModel.confidence * staticConfig.pulseAmpRegCoeff_;

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
    int32_t zStatus = OK;
    SetBits(numFrames < nFramesMin, INSUF_DATA, &zStatus);

    DmeDiagnostics<float> dmeDx {};
    dmeDx.fullEstimation = true;

    // TODO: Need to track frame interval.
//    dmeDx.startFrame = dtbs.front()->StartFrame();
//    dmeDx.stopFrame = dtbs.back()->StopFrame();

    MaxLikelihoodDiagnostics<float>& mldx = dmeDx.mldx;
    mldx.degOfFreedom = numFrames - nModelParams;

    // See I. V. Cadez, P. Smyth, G. J. McLachlan, and C. E. McLaren,
    // Machine Learning 47:7 (2002). [CSMM2002]

    // BENTODO was boost numeric_cast
    static constexpr auto nBins = static_cast<unsigned short>(LaneHist::numBins);

    const float n_j_sum = [&](){
        float sum = 0;
        for (int i = 0; i < nBins; ++i)
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
        for (unsigned int b = 0; b < nBins; ++b)
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

        // BENTODO do we need to keep warps in sync here?
        if (!staticConfig.iterToLimit_ && __all_sync(0xFFFFFFFF, mldx.converged || (zStatus != static_cast<int>(OK))))
        {
            // TODO: Maybe count how many times we achieve this?
            break;
        }

        // M-step

        // Mixing fractions.
        for (int i = 0; i < nModes; ++i)
        {
            rho[i] = c_i[i] / n_j_sum;
        }

        auto oldMu = mu[0];
        // Background mean.
        mu[0] = mom1[0] / c_i[0];
        // BENTODO check for sign error.  I think this makes the correction term positive later
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
        static const float minSnr = 0.1f;
        auto rpaMin = rpa[0];
        for (int i = 1; i < rpa.size(); ++i) rpaMin = min(rpaMin, rpa[i]);
        const auto sMin = minSnr * sqrt(var[0]) / rpaMin;

        // Constrain s > sMin.
        const auto veryLowSignal = (s < sMin);
        if (veryLowSignal) s = sMin;
        SetBits(veryLowSignal, VLOW_SIGNAL, &zStatus);

        // Pulse mode means.
        for (int i = 0; i < numAnalogs; ++i) mu[i+1] = s * static_cast<float>(rpa[i]);

        // Background variance.
        // Need to apply correction terms since it was computed above with the old
        // mean instead of the newest update.
        mom2 += muDiff * correct1 + muDiff * muDiff * correct2;
        mom2 *= hBinSize;
        var[0] = mom2 / c_i[0] + varQuant;

        // Each pulse mode variance is computed as a function of the background
        // variance and the pulse mode mean.
        // Note that we've ignored these dependencies when updating the means
        // and the background variance above.
        for (unsigned int a = 0; a < numAnalogs; ++a)
        {
            const auto i = a + 1;
            var[i] = Basecaller::ModelSignalCovar(::Analog(a), mu[i], var[0]);
        }
    }

    SetBits(!mldx.converged, NO_CONVERGE, &zStatus);

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

    // BENTODO this is disabled until I get a chi squared cfd.
    assert (staticConfig.gTestFactor_ < 0.);
    if (staticConfig.gTestFactor_ >= 0.0f) dmeDx.gTest = Gtest(hist, workModel);
    else assert(dmeDx.gTest.pValue == 1.0f);

    // Compute confidence score.
    dmeDx.confidFactors = ComputeConfidence(dmeDx, initModel, workModel);
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

    // Blend the estimate into the output model.
    if (threadIdx.x % 2 == 0) UpdateTo<0>(workModel, *detModel, threadIdx.x/2);
    else UpdateTo<1>(workModel, *detModel, threadIdx.x/2);
}


// static

__device__ float Fractile(const DmeEmDevice::LaneHist& hist, float frac)
{
    assert(frac >= 0.f);
    assert(frac <= 1.0f);

    static constexpr auto inf = std::numeric_limits<float>::infinity();
    static constexpr auto numBins = DmeEmDevice::LaneHist::numBins;

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

__device__ float DmeEmDevice::PrelimScaleFactor(const ZmwDetectionModel& model,
                                                const LaneHist& hist)
{
    using std::max;  using std::min;
    using std::sqrt;

    // Define a fractile that includes all of the background and half of the pulse frames.
    const auto& bgMode = model.baseline;
    const float& bgVar = bgMode.var;
    const float bgSigma = sqrt(bgVar);
    const float& bgMean = bgMode.mean;
    const float thresh = 2.5f * bgSigma + bgMean;
    float bgCount = 0;
    float totalCount = hist.outlierCountLow[threadIdx.x];
    // BENTODO verify this
    int i = 0;
    float binX = hist.lowBound[threadIdx.x];
    // BENTODO warp convergence issues?
    while (binX < thresh)
    {
        totalCount += hist.binCount[i][threadIdx.x];
        binX += hist.binSize[threadIdx.x];
        i++;
    }
    bgCount = totalCount;
    // BENTODO original never ever included upper outliers and always includes
    //         lower outliers?
    for (; i < LaneHist::numBins; ++i)
    {
        totalCount += hist.binCount[i][threadIdx.x];
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

    auto scaleFactor = Fractile(hist, fractile) / avgSignalMean;

    // Moderate scaling by the clamped confidence.
    const auto cc = min(model.confidence, 1.0f);
    scaleFactor = (1.0f - cc) * scaleFactor + cc;

    // Make sure that scaleFactor > 0.
    scaleFactor = max(scaleFactor, 0.1f);

    return scaleFactor;
}

/// The cumulative probability of the standard normal distribution (mean=0,
/// variance=1).
// BENTODO this duplicated NumericalUtil.h
__device__ float normalStdCdf(float x)
{
    float s = 1.f / sqrt(2.f);
    x *= s;
    // BENTODO this used a boost numeric cast.  Was that caution or need?
    const float r = 0.5f * erfc(-x);
    // BENTODO better error handling?
    assert((r >= 0.0f) && (r <= 1.0f) || isnan(x));
    return r;
}

/// The cumulative probability at \a x of the normal distribution with \mean
/// and standard deviation \a stdDev.
/// \tparam FP a floating-point numeric type (including m512f).
// BENTODO this duplicated NumericalUtil.h
__device__ float normalCdf(float x, float mean = 0.0f, float stdDev = 1.0f)
{
    assert(stdDev > 0);
    const float y = (x - mean) / stdDev;
    return normalStdCdf(y);
}

// static
__device__ GoodnessOfFitTest<float>
DmeEmDevice::Gtest(const LaneHist& histogram, const ZmwDetectionModel& model)
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
    CudaArray<float, LaneHist::numBins+1> p;
    for (unsigned int i = 0; i < LaneHist::numBins+1; ++i)
    {
        p[i] = Basecaller::normalCdf(bb(i), bg.mean, bgStdDev) * bg.weight;
        for (unsigned int j = 0; j < numAnalogs; ++j)
        {
            const auto& dmj = model.analogs[j];
            p[i] += normalCdf(bb(i), dmj.mean, dmStdDev[j]) * dmj.weight;
        }
    }

    // TODO: By splitting out the first iteration, should be able to fuse the
    // loop in adjacent_difference with the loop above.
    //std::adjacent_difference(p.cbegin(), p.cend(), p.begin());
    float tmp = p[0];
    float diff;
    for (int i = 0; i < LaneHist::numBins; ++i)
    {
        diff = p[i+1] - tmp;
        tmp = p[i+1];
        p[i+1] = diff;
    }

    assert(p[0] >= 0.0f);
    assert(p[0] <= 1.0f);
    assert(p[LaneHist::numBins] >= 0.0f);
    assert(p[LaneHist::numBins] <= 1.0f);

    // Compute the G test statistic.
    const auto n = [&](){
        int n = 0;
        for (int i = 0; i < LaneHist::numBins; ++i)
        {
            n += histogram.binCount[i][threadIdx.x];
        }
        return n;
    }();
    //const auto n = FloatVec(histogram.InRangeCount());
    float g = 0.0f;
    for (int i = 0; i < LaneHist::numBins; ++i)
    {
        const auto obs = histogram.binCount[i][threadIdx.x];
        const auto mod = n * p[i+1];
        const auto t = obs * log(obs/mod);
        if (obs > 0.f) g += t;
    }
    g *= 2.0f;

    // Compute the p-value.
    assert(nModelParams + 1 < static_cast<unsigned int>(LaneHist::numBins));
    const auto dof = LaneHist::numBins - nModelParams - 1;
    // BENTODO I don't have access to a gpu chi2 distribution on the GPU.
    assert(false);
    const auto pval = 0.f;
    //const auto pval = chi2CdfComp(g * gTestFactor_, dof);

    return {g, static_cast<float>(dof), pval};
}

/// Saturated linear activation function.
/// A fuzzy threshold that ramps from 0 at a to 1 at b.
/// \returns (x - a)/(b - a) clamped to [0, 1] range.
// BENTODO fix duplication?
__device__ float satlin(float a, float b, float x)
{
    const auto r = (x - a) / (b - a);
    return min(max(r, 0.f), 1.f);
}

// static
__device__ PacBio::Cuda::Utility::CudaArray<float, DmeEmDevice::NUM_CONF_FACTORS>
DmeEmDevice::ComputeConfidence(const DmeDiagnostics<float>& dmeDx,
                               const ZmwDetectionModel& refModel,
                               const ZmwDetectionModel& modelEst)
{
    const auto mldx = dmeDx.mldx;
    CudaArray<float, DmeEmDevice::NUM_CONF_FACTORS> cf;
    for (auto val : cf) val = 1.f;

    // Check EM convergence.
    cf[CONVERGED] = mldx.converged;

    // Check for missing baseline component.
    const auto& bg = modelEst.baseline;
    // TODO: Make this configurable.
    // Threshold level for background fraction.
    static const float bgFracThresh0 = 0.05f;
    static const float bgFracThresh1 = 0.15f;
    float x = satlin(bgFracThresh0, bgFracThresh1, bg.weight);
    cf[BL_FRACTION] = x;

    // Check magnitude of residual baseline mean.
    x = bg.mean * bg.mean / bg.var;
    // TODO: Make this configurable.
    static const float bgMeanTol = 1.0f;
    assert(bgMeanTol > 0.0f);
    x = exp(-x / (2*bgMeanTol*bgMeanTol));
    cf[BL_CV] = x;

    // Check for large deviation of baseline variance from reference variance.
    const auto& refBgVar = refModel.baseline.var;
    x = log2(bg.var / refBgVar);
    // TODO: Make this configurable.
    static const float bgVarTol = 1.0f;
    x = exp(-x*x / (2*bgVarTol*bgVarTol));
    cf[BL_VAR_STABLE] = x;

    // Check for missing pulse components.
    // Require that the first (brightest) and last (dimmest) are not absent.
    // TODO: Make this configurable. Should this threshold be defined in terms
    // of data count instead of fraction?
    const float dmFracThresh1 = staticConfig.analogMixFracThresh_;
    const float dmFracThresh0 = dmFracThresh1 / 3.0f;
    const auto& detModes = modelEst.analogs;
    assert(detModes.size() == 4);
    // BENTODO hard code index uses to be front/back
    x = satlin(dmFracThresh0, dmFracThresh1, detModes[0].weight);
    x *= satlin(dmFracThresh0, dmFracThresh1, detModes[3].weight);
    cf[ANALOG_REP] = x;

    // Check for low SNR.
    // BENTODO hard coded index
    x = detModes[3].mean;  // Assumes last analog is dimmest.
    const auto bgSigma = sqrt(bg.var);
    x /= bgSigma;
    x = satlin(staticConfig.snrThresh0_, staticConfig.snrThresh1_, x);
    cf[SNR_SUFFICIENT] = x;

    // Check for large decrease in SNR.
    // This factor is specifically designed to catch registration errors in the
    // fit when the brightest analog is absent. In such cases, the weight of
    // the dimmest fraction can be substantial (the fit presumably robs some
    // weight from the background component)
    if (staticConfig.snrDropThresh_ < 0.0f) cf[SNR_DROP] = 1.0f;
    else
    {
        const auto snrEst = detModes[0].mean / bgSigma;
        const auto& refDetModes = refModel.analogs;
        assert(refDetModes.size() >= 2);
        const auto& refSignal0 = refDetModes[0].mean;
        // BENTODO need something to replace PBAssert
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
        cf[SNR_DROP] = x;
    }
    __syncwarp();

    // The G-test as a goodness-of-fit score.
    cf[G_TEST] = dmeDx.gTest.pValue;

    return cf;
}

// static
void __device__ DmeEmDevice::ScaleModelSnr(const float& scale,
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
        dmi.var = Basecaller::ModelSignalCovar(::Analog(a),
                                               dmi.mean,
                                               baselineCovar);
    }
    // TODO: Should we update updated_?
}

__global__ void InitModel(Cuda::Memory::DeviceView<const Data::BaselinerStatAccumState> stats,
                          Cuda::Memory::DeviceView<DmeEmDevice::LaneDetModel> models)
{
    auto& blStats = stats[blockIdx.x];
    auto& model = models[blockIdx.x];
    using ElementType = typename Data::BaselinerStatAccumState::StatElement;

    auto& blsa = blStats.baselineStats;

    // BENTODO this replicates host stat accumulator
    auto Mean = [](const auto& stats)
    {
        return stats.moment1[threadIdx.x]/stats.moment0[threadIdx.x];
    };

    /// The unbiased sample variance of the aggregated samples.
    /// NaN if Count() < 2.
    auto Variance = [](const auto& stats)
    {
        static const float nan = std::numeric_limits<float>::quiet_NaN();
        auto var = stats.moment1[threadIdx.x] * stats.moment1[threadIdx.x] / stats.moment0[threadIdx.x];
        var = (stats.moment2[threadIdx.x] - var) / (stats.moment0[threadIdx.x] - 1.0f);
        var = max(var, 0.0f);
        return stats.moment0[threadIdx.x] > 1.0f ? var : nan;
    };

    const auto& blMean = staticConfig.fixedBaselineParams_ ? staticConfig.fixedBaselineMean_ : Mean(blsa);
    const auto& blVar = staticConfig.fixedBaselineParams_ ? staticConfig.fixedBaselineVar_ : Variance(blsa);
    const auto& blWeight = blsa.moment0[threadIdx.x] / blStats.fullAutocorrState.basicStats.moment0[threadIdx.x];

    const auto refSignal = staticConfig.refSnr_ * sqrt(blVar);
    const auto& aWeight = 0.25f * (1.0f - blWeight);
    // BENTODO rethink all this pattern, it could be done better
    if (threadIdx.x % 2 == 0)
    {
        model.BaselineMode().means[threadIdx.x/2].Set<0>(blMean);
        model.BaselineMode().vars[threadIdx.x/2].Set<0>(blVar);
        model.BaselineMode().weights[threadIdx.x/2].Set<0>(blWeight);
        for (int a = 0; a < numAnalogs; ++a)
        {
            const auto aMean = blMean + staticConfig.analogs[a].relAmplitude * refSignal;
            auto& aMode = model.AnalogMode(a);
            aMode.means[threadIdx.x/2].Set<0>(aMean);

            // This noise model assumes that the trace data have been converted to
            // photoelectron units.
            aMode.vars[threadIdx.x/2].Set<0>(ModelSignalCovar(Analog(a), aMean, blVar));

            aMode.weights[threadIdx.x/2].Set<0>(aWeight);
        }
    } else {
        model.BaselineMode().means[threadIdx.x/2].Set<1>(blMean);
        model.BaselineMode().vars[threadIdx.x/2].Set<1>(blVar);
        model.BaselineMode().weights[threadIdx.x/2].Set<1>(blWeight);
        for (int a = 0; a < numAnalogs; ++a)
        {
            const auto aMean = blMean + staticConfig.analogs[a].relAmplitude * refSignal;
            auto& aMode = model.AnalogMode(a);
            aMode.means[threadIdx.x/2].Set<1>(aMean);

            // This noise model assumes that the trace data have been converted to
            // photoelectron units.
            aMode.vars[threadIdx.x/2].Set<1>(ModelSignalCovar(Analog(a), aMean, blVar));

            aMode.weights[threadIdx.x/2].Set<1>(aWeight);
        }
    }
}

DmeEmDevice::PoolDetModel
DmeEmDevice::InitDetectionModels(const PoolBaselineStats& blStats) const
{
    PoolDetModel pdm (PoolSize(), Cuda::Memory::SyncDirection::HostReadDeviceWrite, SOURCE_MARKER());

    Cuda::PBLauncher(InitModel, PoolSize(), laneSize)(blStats, pdm);

    return pdm;
}

}}}     // namespace PacBio::Mongo::Basecaller
