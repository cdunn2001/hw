
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

#include "DmeEmHost.h"

#include <common/StatAccumulator.h>
#include <dataTypes/configs/MovieConfig.h>

#include <limits>
#include <boost/numeric/conversion/cast.hpp>
#include <Eigen/Dense>

#include <tbb/task_arena.h>
#include <tbb/parallel_for.h>

#include <common/LaneArray.h>
#include <common/NumericUtil.h>
#include <dataTypes/configs/BasecallerDmeConfig.h>
#include <dataTypes/UHistogramSimd.h>

using std::numeric_limits;
using boost::numeric_cast;

namespace {

template <typename T>
using DynamicArray1 = Eigen::Array<T, 1, Eigen::Dynamic>;

template <typename T>
using DynamicArray2 = Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>;

}   // anonymous


namespace PacBio {
namespace Mongo {
namespace Basecaller {

// Static configuration parameters
Cuda::Utility::CudaArray<Data::AnalogMode, numAnalogs>
DmeEmHost::analogs_;
float DmeEmHost::refSnr_;
float DmeEmHost::movieScaler_ = 1.0f;
bool DmeEmHost::fixedModel_ = false;
bool DmeEmHost::fixedBaselineParams_ = false;
float DmeEmHost::fixedBaselineMean_ = 0;
float DmeEmHost::fixedBaselineVar_ = 0;

float DmeEmHost::analogMixFracThresh_ = 0.0f;
unsigned short DmeEmHost::emIterLimit_ = 0;
float DmeEmHost::gTestFactor_ = 1.0f;
bool DmeEmHost::iterToLimit_ = false;
float DmeEmHost::pulseAmpRegCoeff_ = 0.0f;
float DmeEmHost::snrDropThresh_ = 1.0f;
float DmeEmHost::snrThresh0_ = 0.0f;
float DmeEmHost::snrThresh1_ = 0.0f;
float DmeEmHost::successConfThresh_ = 0.0f;


DmeEmHost::DmeEmHost(uint32_t poolId, unsigned int poolSize)
    : CoreDMEstimator(poolId, poolSize)
{ }

// static
void DmeEmHost::Configure(const Data::BasecallerDmeConfig &dmeConfig,
                          const Data::MovieConfig &movConfig)
{
    refSnr_ = movConfig.refSnr;
    movieScaler_ = movConfig.photoelectronSensitivity;
    for (size_t i = 0; i < movConfig.analogs.size(); i++)
    {
        analogs_[i] = movConfig.analogs[i];
    }

    fixedModel_ = (dmeConfig.Method == Data::BasecallerDmeConfig::MethodName::Fixed);

    if (fixedModel_ && dmeConfig.SimModel.useSimulatedBaselineParams)
    {
        fixedBaselineParams_ = true;
        fixedBaselineMean_ = dmeConfig.SimModel.baselineMean;
        fixedBaselineVar_ = dmeConfig.SimModel.baselineVar;
    }

    // TODO: Validate values.
    // TODO: Log settings.
    analogMixFracThresh_ = dmeConfig.AnalogMixFractionThreshold;
    emIterLimit_ = dmeConfig.EmIterationLimit;
    gTestFactor_ = dmeConfig.GTestStatFactor;
    iterToLimit_ = dmeConfig.IterateToLimit;
    pulseAmpRegCoeff_ = dmeConfig.PulseAmpRegularization;
    snrDropThresh_ = dmeConfig.SnrDropThresh;
    snrThresh0_ = dmeConfig.MinAnalogSnrThresh0;
    snrThresh1_ = dmeConfig.MinAnalogSnrThresh1;
    successConfThresh_ = dmeConfig.SuccessConfidenceThresh;
}


void DmeEmHost::EstimateImpl(const PoolHist &hist, PoolDetModel *detModelPool) const
{
    if (fixedModel_) return;

    const auto& hView = hist.data.GetHostView();
    auto dmView = detModelPool->GetHostView();

    LaneDetModel& model0 = dmView[0]; // using LaneDetModel = Data::LaneDetectionModel<DetModelElementType>;
    LaneDetModelHost ldModel0(model0);
    
#ifndef NDEBUG
    for (LaneDetModel &dmLane : dmView)
    {
        LaneDetModelHost laneDetModel(dmLane);  // using LaneDetModelHost = Data::DetectionModelHost<FloatVec>;

        assert(laneDetModel.FrameInterval() == ldModel0.FrameInterval());
    }
#endif // NDEBUG

    const LaneDetModelHost ldModel1 = PrelimEstimate(detModelPool);
    
    tbb::task_arena().execute([&] {
        tbb::parallel_for((unsigned int) {0}, PoolSize(), [&](unsigned int lane) {
            auto& dmLane = dmView[lane];

            // Convert to host-friendly data types (e.g., LaneHist-->UHistogramSimd).
            const UHistType uhist(hView[lane]);
            LaneDetModelHost laneDetModel(dmLane);

            // Estimate parameters for this lane.
            EstimateModel(uhist, &laneDetModel);

            // Transcribe results back into *detModel.
            laneDetModel.ExportTo(&dmLane);
        });
    });
}

void DmeEmHost::EstimateModel(const UHistType& hist, LaneDetModelHost* detModel) const
{
    // TODO: Evolve model. Use trace autocorrelation to adjust confidence
    // half-life.

    EstimateFiniteMixture(hist, detModel);
}

DmeEmHost::LaneDetModelHost DmeEmHost::PrelimEstimate(PoolDetModel *detModelPool) const
{
    using std::max;
    using std::isfinite;

    assert(detModelPool->Size());

    // Aggregate the block-specific baseline statistics.
    FloatVec nFrames = 0.0f;
    FloatVec nBaseline = 0.0f;
    FloatVec resid = 0.0f;
    FloatVec var = 0.0f;
    auto dmView = detModelPool->GetHostView();
    for (LaneDetModel &ldm : dmView)
    {
        LaneDetModelHost laneDetModel(ldm);
        auto& blm = laneDetModel.BaselineMode();

        // const auto n = dtb->Size();   // How to get this one? 
        // const auto n = laneDetModel.FrameInterval(); // Seems like FrameInterval
        // const auto nf = static_cast<float>(n);

        // auto w2 = ldm.BaselineMode().weights;  // weights ??

        float nf = 1000;

        auto weights = blm.Weight();
        auto nbl = nf * weights;
        const auto mask = (nbl >= 2.0f);
        nbl = Blend(mask, nbl, 0.0f);
        nBaseline += nbl;
        nFrames   += Blend(mask, nf, FloatVec(0));
        var       += Blend(mask, (nbl-1) * blm.SignalCovar(), 0.0f);
        // resid   dtb->BaselineResidual ?????
    }
    assert(all(nFrames >= nBaseline));
    resid /= nBaseline;
    var /= nBaseline - 1;

    LaneDetModel& ldm0 = dmView[0]; // using LaneDetModel = Data::LaneDetectionModel<DetModelElementType>;
    LaneDetModelHost m0(ldm0);

    // Reject statistics with insufficient data.
    constexpr float nBaselineMin = 2.0f;
    assert(nBaselineMin > 1.0f);
    const BoolVec mask = nBaseline >= nBaselineMin;
    const auto& m0blm = m0.BaselineMode();

    var = Blend(mask, var, m0blm.SignalCovar());
    resid = Blend(mask, resid, m0blm.SignalMean());
    assert(all(isfinite(var)) && all(var > 0.0f));
    assert(all(isfinite(resid)));

    const FloatVec blWeight = max(nBaseline / nFrames, 0.01f);
    // m0.Rescale(blVar, blResid, blWeight, frameInterval); // What's that???

    FloatVec conf = 0.1f * satlin<FloatVec>(0.0f, 500.0f, nBaseline - nBaselineMin);
    m0.Confidence(conf);

    return m0;
}

void DmeEmHost::EstimateFiniteMixture(const UHistType& hist, LaneDetModelHost* detModel) const
{
    const auto& numFrames = hist.TotalCount();

    // Make a working copy of the detection model.
    LaneDetModelHost workModel = *detModel;

    // Keep a copy of the initial model.
    const auto initModel = workModel;

    // The term "mode" refers to a component of the mixture model.
    auto& bgMode = workModel.BaselineMode();
    auto& pulseModes = workModel.DetectionModes();

    // Scale the model based on fractile of the data.
    const auto scaleFactor = PrelimScaleFactor(workModel, hist);
    ScaleModelSnr(scaleFactor, &workModel);

    const auto& binSize = hist.BinSize();

    // Define working variables for model parameters.
    const auto nModes = numAnalogs + 1;
    using ModeArray = Eigen::Array<FloatVec, nModes, 1>;
    ModeArray rho;     // Mixture fraction for each mode.
    ModeArray mu;      // Mean of each mode.
    ModeArray var;     // Variance of each mode.

    rho[0] = bgMode.Weight();
    mu[0] = bgMode.SignalMean();
    var[0] = bgMode.SignalCovar();
    for (unsigned int a = 0; a < numAnalogs; ++a)
    {
        const auto k = a + 1;
        PBAssert(k < nModes, "k < nModes");
        const auto& pma = pulseModes[a];
        rho[k] = pma.Weight();
        mu[k] = pma.SignalMean();
        var[k] = pma.SignalCovar();
    }

    // Variance associated with data binning.
    const auto& varQuant = pow2(hist.BinSize()) / 12.0f;
    var += varQuant;

    // Enforce normalization of mixture fractions.
    rho = rho / rho.sum();

    // Initialize estimates to initial model.
    ModeArray rhoEst = rho;
    ModeArray muEst = mu;
    ModeArray varEst = var;

    // The relative pulse amplitudes.
    Eigen::Array<float, numAnalogs, 1> rpa;
    for (unsigned int a = 0; a < numAnalogs; ++a)
    {
        rpa[a] = Analog(a).relAmplitude;
        if (rpa[a] <= 0.0f)
        {
            throw PBException("Bad relative amplitude in analog properties."
                              " Relative amplitudes must be greater than zero.");
        }
    }

    // Initialize the model parameter for pulse amplitude scale, which is
    // needed for the prior when calculating the posterior.
    FloatVec s = (mu.tail(numAnalogs) * rpa.cast<FloatVec>()).sum() / rpa.square().sum();

    // Prior hyperparameters for scale parameter.
    // Undo the preliminary scaling based on percentile to get back to the
    // means of initModel.
    const FloatVec sExpect = s / scaleFactor;

    // sExpectWeight is the inverse of the variance of the normal prior for s.
    const FloatVec sExpectWeight = initModel.Confidence() * pulseAmpRegCoeff_;

    // Log likelihood--really a posterior since we've added a prior for s.
    FloatVec logLike {numeric_limits<float>::lowest()};
    FloatVec logLikePrev {logLike};

    // Iteration limit for EM.
    const unsigned int iterLimit = emIterLimit_;

    // Define lower bound for bin probability.
    const FloatVec binProbMin
            = max(FloatVec(numFrames) * binSize, 1.0f)
            * numeric_limits<float>::min();

    // Initialize intra-lane failure codes.
    IntVec zStatus = OK;
    SetBits(numFrames < CoreDMEstimator::nFramesMin, INSUF_DATA, &zStatus);

    DmeDiagnostics<FloatVec> dmeDx {};
    dmeDx.fullEstimation = true;

    // TODO: Need to track frame interval.
//    dmeDx.startFrame = dtbs.front()->StartFrame();
//    dmeDx.stopFrame = dtbs.back()->StopFrame();

    MaxLikelihoodDiagnostics<FloatVec>& mldx = dmeDx.mldx;
    mldx.degOfFreedom = LaneArray<int>(numFrames - CoreDMEstimator::nModelParams);

    // See I. V. Cadez, P. Smyth, G. J. McLachlan, and C. E. McLaren,
    // Machine Learning 47:7 (2002). [CSMM2002]

    const auto nBins = numeric_cast<unsigned short>(hist.NumBins());

    // Total probability density at the bin boundaries.
    DynamicArray1<FloatVec> boundaryProb (nBins+1);

    // Probability of each bin.
    DynamicArray1<FloatVec> binProb (nBins);

    // "Responsibility" (tau) of mode i at bin boundary x.
    // CSMM2002, Equation 4.
    DynamicArray2<FloatVec> tau (nModes, nBins+1);

    // Bin counts. CSMM2002, Equation 3.
    DynamicArray1<FloatVec> n_j (nBins);

    FloatVec n_j_sum(0.0f);
    for (unsigned int j = 0; j < nBins; ++j)
    {
        n_j[j] = FloatVec(hist.BinCount(j));
        n_j_sum += n_j[j];
    }

    unsigned int it = 0;
    for (; it < iterLimit; ++it)
    {
        // E-step

        // Inverses of variances.
        ModeArray varinv = var.array().inverse();

        static const float log_2pi = std::log(2.0f * pi_f);
        ModeArray prefactors = rho.log() - 0.5f*var.log() - 0.5f*log_2pi;

        FloatVec probSum(0.0f);
        FloatVec weightedProbSum(0.0f);
        ModeArray c_i;
        c_i.fill(0.0f);
        // Means
        ModeArray mom1;
        mom1.fill(0.0f);

        auto cornerComp = [&](int b)
        {
            // First compute the log of the component probabilities.
            const auto& x = hist.BinStart(b);
            for (unsigned int i = 0; i < nModes; ++i)
            {
                const auto& y = x - mu[i];
                tau(i, b) = prefactors[i] - 0.5f*y*y*varinv[i];
            }

            // Next compute the log likelihood of each bin boundary, which is
            // the log of the sum of exp(tau_i_x) over modes (i).
            // Use log-sum-exp trick to avoid uniform underflow. (PTVF2007, Equation 16.1.9)
            const FloatVec tauMax = tau.col(b).maxCoeff();
            FloatVec llb = tauMax + log((tau.col(b) - tauMax).exp().sum());

            // Convert to likelihood ratio.
            tau.col(b) = exp(tau.col(b) - llb);

            // Record the total probability density at the bin boundaries.
            boundaryProb(b) = exp(llb);
        };

        auto centerComp = [&](int b)
        {
            // Constrain bin probability to a minimum positive value.
            // Avoids 0 * log(0) in computation of log likelihood.
            binProb(b) = max(0.5f * binSize * (boundaryProb(b) + boundaryProb(b+1)), binProbMin);
            probSum += binProb(b);
            weightedProbSum += n_j[b] * log(binProb(b));

            auto factor = n_j[b] / binProb[b];

            const auto& x0 = hist.BinStart(b);
            const auto& x1 = hist.BinStart(b+1);
            for (unsigned int m = 0; m < nModes; ++m)
            {
                // Relative weight of each mode. CSMM2002, Equation 5.
                // Use trapezoidal approximation of expectation of tau over each bin.
                auto tmp1 = boundaryProb(b) * tau(m, b);
                auto tmp2 = boundaryProb(b+1) * tau(m, b+1);

                c_i[m] += (tmp1 + tmp2) * factor;
                mom1[m] += (tmp1 * x0 + tmp2 * x1) * factor;
            }
        };

        // TODO: Check for bins where binProb == 0 but n_j > 0.
        cornerComp(0);
        for (unsigned int b = 0; b < nBins; ++b)
        {
            cornerComp(b+1);
            centerComp(b);
        }
        // TODO: Note the cancellation of the (0.5f * binSize) factor. Possible minor optimization opportunity.
        // TODO: When used, tau is always multiplied by boundaryProb. So why bother dividing that factor out when first computing tau?

        c_i *= 0.5f * binSize;
        mom1 *= 0.5f * binSize;

        // Update log posterior.
        logLikePrev = logLike;
        logLike = weightedProbSum - n_j_sum * log(probSum);
        logLike -= 0.5f * pow2(s - sExpect) * sExpectWeight;

        // Check for convergence.
        const FloatVec deltaLogLike {logLike - logLikePrev};
        const FloatVec convTol = 1e-4f * abs(logLike);  // TODO: Make this configurable.
        if (any(deltaLogLike < -convTol))
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
        BoolVec conv = (deltaLogLike >= 0) & (deltaLogLike < convTol);
        conv &= (logLike >= mldx.logLike);
        mldx.Converged(conv, it, logLike, deltaLogLike);

        // Update result for converged ZMWs.
        if (any(conv)) for (unsigned int k = 0; k < nModes; ++k)
        {
            rhoEst[k] = Blend(conv, rho[k], rhoEst[k]);
            muEst[k] = Blend(conv, mu[k], muEst[k]);
            varEst[k] = Blend(conv, var[k], varEst[k]);
        }

        if (!iterToLimit_ && all(mldx.converged | (zStatus != static_cast<int>(OK))))
        {
            // TODO: Maybe count how many times we achieve this?
            break;
        }

        // M-step

        // Mixing fractions.
        rho = c_i / n_j_sum;

        // Background mean.
        mu[0] = mom1[0] / c_i[0];

        // Amplitude scale parameter.
        FloatVec numer = 0.0f;
        FloatVec denom = 0.0f;
        for (unsigned int a = 0; a < numAnalogs; ++a)
        {
            const auto i = a + 1;
            numer += rpa[a] * varinv[i] * mom1[i];
            denom += pow2(rpa[a]) * varinv[i] * c_i[i];
        }

        // Add regularization/prior terms.
        numer += sExpect * sExpectWeight;
        denom += sExpectWeight;

        s = numer / denom;

        // The minimum bound for the pulse-amplitude scale parameter.
        using std::min;  using std::max;

        static const float minSnr = 0.5f;
        const float minPulseMean = max(1.0f, movieScaler_);
        const auto sMin = max(minSnr * sqrt(var[0]), minPulseMean) / rpa.minCoeff();

        // Constrain s > sMin.
        const auto veryLowSignal = (s < sMin);
        s = Blend(veryLowSignal, sMin, s);
        SetBits(veryLowSignal, VLOW_SIGNAL, &zStatus);

        // Pulse mode means.
        mu.tail(numAnalogs) = s * rpa.cast<FloatVec>();

        // Background variance.
        // Could potentially be combined with the single pass loop above, but
        // would require a correction term to be added (since this loop requires
        // the updated rho), and combining this loop with the above didn't show
        // any major speed gains.
        FloatVec mom2 = 0.0f;   // Only needed for background mode.
        for (unsigned int j = 0; j < nBins; ++j)
        {
            const auto k = j + 1;
            const auto& x0 = hist.BinStart(j) - mu[0];
            const auto& x1 = hist.BinStart(k) - mu[0];
            FloatVec tmp = boundaryProb[j] * x0 * x0 * tau(0,j);
            tmp += boundaryProb[k] * x1 * x1 * tau(0,k);
            tmp *= n_j[j] / binProb[j];
            mom2 += tmp;
        }
        mom2 *= 0.5f * binSize;
        var[0] = mom2 / c_i[0] + varQuant;

        // Each pulse mode variance is computed as a function of the background
        // variance and the pulse mode mean.
        // Note that we've ignored these dependencies when updating the means
        // and the background variance above.
        for (unsigned int a = 0; a < numAnalogs; ++a)
        {
            const auto i = a + 1;
            var[i] = ModelSignalCovar(Analog(a), mu[i], var[0]);
        }
    }

    SetBits(!mldx.converged, NO_CONVERGE, &zStatus);

    using std::isfinite;

    // Package estimation results.
    PBAssert(all(isfinite(rhoEst[0])), "all(isfinite(rhoEst[0])");
    bgMode.Weight(rhoEst[0]);
    PBAssert(all(isfinite(muEst[0])), "all(isfinite(muEst[0]))");
    bgMode.SignalMean(muEst[0]);
    PBAssert(all(isfinite(varEst[0])), "all(isfinite(varEst[0]))");
    bgMode.SignalCovar(varEst[0]);
    for (unsigned int a = 0; a < numAnalogs; ++a)
    {
        auto& pda = pulseModes[a];
        const auto i = a + 1;
        PBAssert(all(isfinite(rhoEst[i])), "all(isfinite(rhoEst[i])");
        pda.Weight(rhoEst[i]);
        PBAssert(all(isfinite(muEst[i])), "all(isfinite(muEst[i]))");
        pda.SignalMean(muEst[i]);
        PBAssert(all(isfinite(varEst[i])), "all(isfinite(varEst[i]))");
        pda.SignalCovar(varEst[i]);
    }

    if (gTestFactor_ >= 0.0f) dmeDx.gTest = Gtest(hist, workModel);
    else assert(all(dmeDx.gTest.pValue == 1.0f));

    // Compute confidence score.
    dmeDx.confidFactors = ComputeConfidence(dmeDx, initModel, workModel);
    {
        using std::min;  using std::max;
        FloatVec conf = 1.0f;
        for (const auto& cf : dmeDx.confidFactors) conf *= cf;
        conf = min(max(0.0f, conf), 1.0f);
        conf = Blend(conf >= successConfThresh_, conf, FloatVec(0.0f));
        workModel.Confidence(conf);
    }

    // TODO: Push results to DmeDumpCollector.
    //    if (this->dmeDumpCollector_)
    //    {
    //        // TODO: Add lane and zmw event codes.
    //        this->dmeDumpCollector_->CollectRawEstimate1D(*dtbs.front(), hist, dmeDx, *workModel);
    //    }

    // Blend the estimate into the output model.
    detModel->Update(workModel);
}


// static
DmeEmHost::FloatVec
DmeEmHost::PrelimScaleFactor(const LaneDetModelHost& model,
                             const UHistType& hist)
{
    using std::max;  using std::min;
    using std::sqrt;

    // Define a fractile that includes all of the background and half of the pulse frames.
    const auto& bgMode = model.BaselineMode();
    const FloatVec& bgVar = bgMode.SignalCovar();
    const FloatVec bgSigma = sqrt(bgVar);
    const FloatVec& bgMean = bgMode.SignalMean();
    const FloatVec thresh = 2.5f * bgSigma + bgMean;
    const FloatVec bgCount = hist.CumulativeCount(thresh);
    const FloatVec totalCount =  FloatVec(hist.LowOutlierCount() + hist.InRangeCount());
    const FloatVec fractile = 0.5f * (bgCount + totalCount) / totalCount;

    // Define the scale factor as the ratio between this fractile and the
    // average of the pulse signal means.
    FloatVec avgSignalMean = 0.0f;
    assert(model.DetectionModes().size() == numAnalogs);
    for (unsigned int a = 0; a < numAnalogs; ++a)
    {
        avgSignalMean += model.DetectionModes()[a].SignalMean();
    }
    avgSignalMean /= static_cast<float>(numAnalogs);

    auto scaleFactor = hist.Fractile(fractile) / avgSignalMean;

    // Moderate scaling by the clamped confidence.
    const auto cc = min(model.Confidence(), 1.0f);
    scaleFactor = (1.0f - cc) * scaleFactor + cc;

    // Make sure that scaleFactor > 0.
    scaleFactor = max(scaleFactor, 0.1f);

    return scaleFactor;
}


// static
GoodnessOfFitTest<typename DmeEmHost::FloatVec>
DmeEmHost::Gtest(const UHistType& histogram, const LaneDetModelHost& model)
{
    const auto& bb = histogram.BinBoundaries();
    const auto& bg = model.BaselineMode();

    // Cache the standard deviations.
    const auto bgStdDev = sqrt(bg.SignalCovar());
    AlignedVector<FloatVec> dmStdDev (numAnalogs);
    assert(model.DetectionModes().size() == numAnalogs);
    for (unsigned int j = 0; j < numAnalogs; ++j)
    {
        dmStdDev[j] = sqrt(model.DetectionModes()[j].SignalCovar());
    }

    // Compute the bin probabilities according to the model.
    // TODO: Improve accuracy when normalCdf is close to 1.0.
    AlignedVector<FloatVec> p (bb.size());
    for (unsigned int i = 0; i < bb.size(); ++i)
    {
        p[i] = normalCdf(bb[i], bg.SignalMean(), bgStdDev) * bg.Weight();
        for (unsigned int j = 0; j < numAnalogs; ++j)
        {
            const auto& dmj = model.DetectionModes()[j];
            p[i] += normalCdf(bb[i], dmj.SignalMean(), dmStdDev[j]) * dmj.Weight();
        }
    }

    // TODO: By splitting out the first iteration, should be able to fuse the
    // loop in adjacent_difference with the loop above.
    std::adjacent_difference(p.cbegin(), p.cend(), p.begin());

    assert(all(p.front() >= 0.0f));
    assert(all(p.front() <= 1.0f));
    assert(all(p.back() >= 0.0f));
    assert(all(p.back() <= 1.0f));

    // Compute the G test statistic.
    const auto n = FloatVec(histogram.InRangeCount());
    FloatVec g = 0.0f;
    for (int i = 0; i < histogram.NumBins(); ++i)
    {
        const auto obs = FloatVec(histogram.BinCount(i));
        const auto mod = n * p[i+1];
        const auto t = obs * log(obs/mod);
        g += Blend(obs > 0.0f, t, FloatVec(0.0f));
    }
    g *= 2.0f;

    // Compute the p-value.
    assert(CoreDMEstimator::nModelParams + 1 < static_cast<unsigned int>(histogram.NumBins()));
    const auto dof = histogram.NumBins() - CoreDMEstimator::nModelParams - 1;
    const auto pval = chi2CdfComp(g * gTestFactor_, dof);

    return {g, static_cast<float>(dof), pval};
}

// static
Cuda::Utility::CudaArray<DmeEmHost::FloatVec, ConfFactor::NUM_CONF_FACTORS>
DmeEmHost::ComputeConfidence(const DmeDiagnostics<FloatVec>& dmeDx,
                             const LaneDetModelHost& refModel,
                             const LaneDetModelHost& modelEst)
{
    const auto mldx = dmeDx.mldx;
    Cuda::Utility::CudaArray<FloatVec, ConfFactor::NUM_CONF_FACTORS> cf;

    // Check EM convergence.
    cf[ConfFactor::CONVERGED] = Blend(mldx.converged, FloatVec(1), FloatVec(0));

    // Check for missing baseline component.
    const auto& bg = modelEst.BaselineMode();
    // TODO: Make this configurable.
    // Threshold level for background fraction.
    static const float bgFracThresh0 = 0.05f;
    static const float bgFracThresh1 = 0.15f;
    FloatVec x = satlin<FloatVec>(bgFracThresh0, bgFracThresh1, bg.Weight());
    cf[ConfFactor::BL_FRACTION] = x;

    // Check magnitude of residual baseline mean.
    x = pow2(bg.SignalMean()) / bg.SignalCovar();
    // TODO: Make this configurable.
    static const float bgMeanTol = 1.0f;
    assert(bgMeanTol > 0.0f);
    x = exp(-x / (2*pow2(bgMeanTol)));
    cf[ConfFactor::BL_CV] = x;

    // Check for large deviation of baseline variance from reference variance.
    const auto& refBgVar = refModel.BaselineMode().SignalCovar();
    x = log2(bg.SignalCovar() / refBgVar);
    // TODO: Make this configurable.
    static const float bgVarTol = 1.0f;
    x = exp(-x*x / (2*pow2(bgVarTol)));
    cf[ConfFactor::BL_VAR_STABLE] = x;

    // Check for missing pulse components.
    // Require that the first (brightest) and last (dimmest) are not absent.
    // TODO: Make this configurable. Should this threshold be defined in terms
    // of data count instead of fraction?
    const float dmFracThresh1 = analogMixFracThresh_;
    const float dmFracThresh0 = dmFracThresh1 / 3.0f;
    const auto& detModes = modelEst.DetectionModes();
    assert(detModes.size() > 0);
    x = satlin<FloatVec>(dmFracThresh0, dmFracThresh1, detModes.front().Weight());
    x *= satlin<FloatVec>(dmFracThresh0, dmFracThresh1, detModes.back().Weight());
    cf[ConfFactor::ANALOG_REP] = x;

    // Check for low SNR.
    x = detModes.back().SignalMean();  // Assumes last analog is dimmest.
    const auto bgSigma = sqrt(bg.SignalCovar());
    x /= bgSigma;
    x = satlin<FloatVec>(snrThresh0_, snrThresh1_, x);
    cf[ConfFactor::SNR_SUFFICIENT] = x;

    // Check for large decrease in SNR.
    // This factor is specifically designed to catch registration errors in the
    // fit when the brightest analog is absent. In such cases, the weight of
    // the dimmest fraction can be substantial (the fit presumably robs some
    // weight from the background component)
    if (snrDropThresh_ < 0.0f) cf[ConfFactor::SNR_DROP] = 1.0f;
    else
    {
        const auto snrEst = detModes.front().SignalMean() / bgSigma;
        const auto& refDetModes = refModel.DetectionModes();
        assert(refDetModes.size() >= 2);
        const auto& refSignal0 = refDetModes[0].SignalMean();
        PBAssert(all(refSignal0 >= 0.0f), "Bad SignalMean.");
        const auto& refSignal1 = refDetModes[1].SignalMean();
        PBAssert(all(refSignal1 >= 0.0f), "Bad SignalMean.");
        const auto refBgSigma = sqrt(refModel.BaselineMode().SignalCovar());
        PBAssert(all(refBgSigma >= 0.0f), "Bad baseline sigma.");
        const auto refSnr0 = refSignal0 / refBgSigma;
        auto refSnr1 = refSignal1 / refBgSigma;
        refSnr1 *= snrDropThresh_;
        refSnr1 *= min(refModel.Confidence(), 1.0f);
        PBAssert(all(refSnr1 < refSnr0),
                 "Bad threshold in SNR Drop confidence factor.");
        x = satlin(refSnr1, sqrt(refSnr0*refSnr1), snrEst);
        cf[ConfFactor::SNR_DROP] = x;
    }

    // The G-test as a goodness-of-fit score.
    cf[ConfFactor::G_TEST] = dmeDx.gTest.pValue;

    return cf;
}

// static
void DmeEmHost::ScaleModelSnr(const FloatVec& scale,
                              LaneDetModelHost* detModel)
{
    assert (all(scale > 0.0f));
    const auto baselineCovar = detModel->BaselineMode().SignalCovar();
    auto& detectionModes_ = detModel->DetectionModes();
    assert (detectionModes_.size() == numAnalogs);
    for (unsigned int a = 0; a < numAnalogs; ++a)
    {
        auto& dmi = detectionModes_[a];
        dmi.SignalMean(scale * dmi.SignalMean());
        dmi.SignalCovar(ModelSignalCovar(Analog(a),
                                         dmi.SignalMean(),
                                         baselineCovar));
    }
    // TODO: Should we update updated_?
}

DmeEmHost::PoolDetModel
DmeEmHost::InitDetectionModels(const PoolBaselineStats& blStats) const
{
    PoolDetModel pdm (PoolSize(), Cuda::Memory::SyncDirection::HostWriteDeviceRead, SOURCE_MARKER());

    auto pdmHost = pdm.GetHostView();
    const auto& blStatsHost = blStats.GetHostView();
    for (unsigned int lane = 0; lane < PoolSize(); ++lane)
    {
        InitLaneDetModel(blStatsHost[lane], pdmHost[lane]);
    }

    return pdm;
}

void DmeEmHost::InitLaneDetModel(const Data::BaselinerStatAccumState& blStats,
                                 LaneDetModel& ldm) const
{
    using ElementType = typename Data::BaselinerStatAccumState::StatElement;
    using LaneArr = LaneArray<ElementType>;

    StatAccumulator<LaneArr> blsa (blStats.baselineStats);

    const auto& blMean = fixedBaselineParams_ ? fixedBaselineMean_ : blsa.Mean();
    const auto& blVar = fixedBaselineParams_ ? fixedBaselineVar_ : blsa.Variance();
    const auto& blWeight = LaneArr(blStats.NumBaselineFrames()) / LaneArr(blStats.fullAutocorrState.basicStats.moment0);

    ldm.BaselineMode().means = blMean;
    ldm.BaselineMode().vars = blVar;
    ldm.BaselineMode().weights = blWeight;
    assert(numAnalogs <= analogs_.size());
    const auto refSignal = refSnr_ * sqrt(blVar);
    const auto& aWeight = 0.25f * (1.0f - blWeight);
    for (unsigned int a = 0; a < numAnalogs; ++a)
    {
        const auto aMean = blMean + analogs_[a].relAmplitude * refSignal;
        auto& aMode = ldm.AnalogMode(a);
        aMode.means = aMean;

        // This noise model assumes that the trace data have been converted to
        // photoelectron units.
        aMode.vars = ModelSignalCovar(Analog(a), aMean, blVar);

        aMode.weights = aWeight;
    }
}

// static
LaneArray<float> DmeEmHost::ModelSignalCovar(
        const Data::AnalogMode& analog,
        const LaneArray<float>& signalMean,
        const LaneArray<float>& baselineVar)
{
    LaneArray<float> r {baselineVar};
    r += signalMean * CoreDMEstimator::shotVarCoeff;
    r += pow2(signalMean * analog.excessNoiseCV);
    return r;
}


}}}     // namespace PacBio::Mongo::Basecaller
