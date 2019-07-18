
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

#include <limits>
#include <boost/numeric/conversion/cast.hpp>
#include <Eigen/Dense>

#include <common/LaneArray.h>
#include <common/NumericUtil.h>
#include <dataTypes/BasecallerConfig.h>
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

// Static constants
constexpr unsigned short DmeEmHost::nModelParams;
constexpr unsigned int DmeEmHost::nFramesMin;

// Static configuration parameters
unsigned short DmeEmHost::emIterLimit_ = 0;
float DmeEmHost::gTestFactor_ = 1.0f;
bool DmeEmHost::iterToLimit_ = false;
float DmeEmHost::pulseAmpRegCoeff_ = 0.0f;

DmeEmHost::DmeEmHost(uint32_t poolId, unsigned int poolSize)
    : DetectionModelEstimator(poolId, poolSize)
{ }

// static
void DmeEmHost::Configure(const Data::BasecallerDmeConfig &dmeConfig,
                          const Data::MovieConfig &movConfig)
{
    // TODO: Validate values.
    // TODO: Log settings.
    emIterLimit_ = dmeConfig.EmIterationLimit;
    gTestFactor_ = dmeConfig.GTestStatFactor;
    iterToLimit_ = dmeConfig.IterateToLimit;
    pulseAmpRegCoeff_ = dmeConfig.PulseAmpRegularization;
}

void DmeEmHost::EstimateImpl(const PoolHist &hist, PoolDetModel *detModel) const
{
    const auto& hView = hist.data.GetHostView();
    auto dmView = detModel->GetHostView();
    for (unsigned int lane = 0; lane < PoolSize(); ++lane)
    {
        // Convert to host-friendly data types (e.g., LaneHist-->UHistogramSimd).
        const UHistType uhist (hView[lane]);
        LaneDetModelHost laneDetModel (dmView[lane]);

        EstimateLaneDetModel(uhist, &laneDetModel);
    }
}


void DmeEmHost::EstimateLaneDetModel(const UHistType& hist, LaneDetModelHost* detModel) const
{
    // TODO

    //    // Evolve model0. Use trace autocorrelation to adjust confidence half-life.
    //    EvolveModel(dtbs, &model0);

    const auto& numFrames = hist.TotalCount();

    // Make a working copy of the detection model.
    LaneDetModelHost workModel = *detModel;

    // Keep a copy of the initial model.
    const auto initModel = workModel;

    // The term "mode" refers to a component of the mixture model.
    auto& bgMode = workModel.BaselineMode();
    auto& pulseModes = workModel.DetectionModes();
    const LaneArray<float>& bgVar = bgMode.SignalCovar();
    const auto bgSigma = sqrt(bgVar);

    // Scale the model based on fractile of the data.
    const auto scaleFactor = PrelimScaleFactor(workModel, hist);
    workModel.ScaleSnr(scaleFactor);

    const auto& binSize = hist.BinSize();

    // Define working variables for model parameters.
    const auto nModes = numAnalogs + 1;
    using ModeArray = Eigen::Array<FloatVec, nModes, 1>;
    ModeArray rho;     // Mixture fraction for each mode.
    ModeArray mu;      // Mean of each mode.
    ModeArray var;     // Variance of each mode.

    rho[0] = bgMode.Weight();
    mu[0] = bgMode.SignalMean()[0];
    var[0] = bgMode.SignalCovar()[0];
    for (unsigned int a = 0; a < numAnalogs; ++a)
    {
        const auto k = a + 1;
        PBAssert(k < nModes, "k < nModes");
        const auto& pma = pulseModes[a];
        rho[k] = pma.Weight();
        mu[k] = pma.SignalMean()[0];
        var[k] = pma.SignalCovar()[0];
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
    SetBits(numFrames < numeric_cast<int>(nFramesMin),
            INSUF_DATA, &zStatus);

    DmeDiagnostics<FloatVec> dmeDx {};
    dmeDx.fullEstimation = true;

    // TODO: Need to track frame interval.
//    dmeDx.startFrame = dtbs.front()->StartFrame();
//    dmeDx.stopFrame = dtbs.back()->StopFrame();

    MaxLikelihoodDiagnostics<FloatVec>& mldx = dmeDx.mldx;
    mldx.degOfFreedom = numFrames - nModelParams;

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

        static const float log_2pi = log(2.0f * pi_f);
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

        // TODO: We can probably eliminate the Pearson's chi-square.
        // Compute Pearson's chi-squared statistic.
        const FloatVec pcs = n_j_sum * ((n_j / n_j_sum - binProb).square() / binProb).sum();

        // TODO: Seems like we ought to eliminate or relax the lower bound
        // condition on deltaLogLike here.
        BoolVec conv = (deltaLogLike >= 0) & (deltaLogLike < convTol);
        conv &= (logLike >= mldx.logLike);
        mldx.Converged(conv, it, logLike, deltaLogLike, pcs);

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
        static const float minSnr = 0.1f;
        const auto sMin = minSnr * sqrt(var[0]) / rpa.minCoeff();

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

    //    // Compute confidence score.
    //    dmeDx.confidFactors = ComputeConfidence(dmeDx, initModel, *workModel);
    //    {
    //        using std::min;  using std::max;
    //        FloatVec conf = 1.0f;
    //        for (const auto& cf : dmeDx.confidFactors) conf *= cf;
    //        conf = min(max(0.0f, conf), 1.0f);
    //        conf = Blend(conf >= successConfThresh_, conf, 0.0f);
    //        workModel->Confidence(conf);
    //        mldx.confidenceScore = conf;
    //    }

    //    // Push results to DmeDumpCollector.
    //    if (this->dmeDumpCollector_)
    //    {
    //        // TODO: Add lane and zmw event codes.
    //        this->dmeDumpCollector_->CollectRawEstimate1D(*dtbs.front(), hist, dmeDx, *workModel);
    //    }


    //    // Blend the estimate into the first of the models in the trace
    //    // block sequence.
    //    model0.Update(workModel);
    //    dtbs.front()->ModelEstimationDx(dmeDx);

}


// static
DmeEmHost::FloatVec
DmeEmHost::PrelimScaleFactor(const LaneDetModelHost& model,
                             const UHistType& hist)
{
    using std::max;  using std::min;
    using std::sqrt;

    using ClarFloat = ConstLaneArrayRef<float>;

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
    assert(nModelParams + 1 < static_cast<unsigned int>(histogram.NumBins()));
    const auto dof = histogram.NumBins() - nModelParams - 1;
    const auto pval = chi2CdfComp(g * gTestFactor_, dof);

    return {g, static_cast<float>(dof), pval};
}

}}}     // namespace PacBio::Mongo::Basecaller
