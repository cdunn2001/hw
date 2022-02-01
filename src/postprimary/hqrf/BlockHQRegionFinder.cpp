// Copyright (c) 2018, Pacific Biosciences of California, Inc.
//
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted (subject to the limitations in the
// disclaimer below) provided that the following conditions are met:
//
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//
//  * Redistributions in binary form must reproduce the above
//    copyright notice, this list of conditions and the following
//    disclaimer in the documentation and/or other materials provided
//    with the distribution.
//
//  * Neither the name of Pacific Biosciences nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE
// GRANTED BY THIS LICENSE. THIS SOFTWARE IS PROVIDED BY PACIFIC
// BIOSCIENCES AND ITS CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
// OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL PACIFIC BIOSCIENCES OR ITS
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
// USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
// OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
// SUCH DAMAGE.

#include <numeric>

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>

#include "BlockHQRegionFinder.h"
#include "SequelCrfParams.h"
#include "SpiderCrfParams.h"
#include "ZOffsetCrfParams.h"
#include <pacbio/primary/HQRFParamsMCart.h>
#include <pacbio/primary/HQRFParamsNCart.h>
#include "ClassificationTree.h"

using fmat_t = boost::numeric::ublas::matrix<float>;
using u8mat_t = boost::numeric::ublas::matrix<uint8_t>;
using boost::numeric::ublas::row;

///
/// Utils
///
namespace {

template<typename T>
inline size_t argmax(T begin, T end)
{
    return std::max_element(begin, end) - begin;
}

template<typename mat_t>
void PrintMatrix(const mat_t& m)
{
    std::cout.precision(3);
    for (size_t i = 0; i < m.size1(); i++) {
        for (size_t j = 0; j < m.size2(); j++) {
            std::cout << m(i, j) << "\t";
        }
        std::cout << std::endl;
    }
}

} // anon namespace.

namespace PacBio {
namespace Primary {
namespace Postprimary {


template <HQRFMethod Method>
BlockHQRegionFinder<Method>::BlockHQRegionFinder(
        float frameRate,
        float snrThresh,
        bool ignoreBazAL)
    : frameRate_(frameRate)
    , snrThresh_(snrThresh)
    , ignoreBazAL_(ignoreBazAL)
{}

template <HQRFMethod Method>
struct crf_traits;
template <> struct crf_traits<HQRFMethod::SEQUEL_CRF_HMM> {
    using Model = SequelCrfModel;
};
template <> struct crf_traits<HQRFMethod::SPIDER_CRF_HMM> {
    using Model = SpiderCrfModel;
};
template <> struct crf_traits<HQRFMethod::ZOFFSET_CRF_HMM> {
    using Model = ZOffsetCrfModel;
};

// This is a somewhat generic viterbi algorithm implementation for CRFs
// specified in various training files. The number of states and features can
// vary from model to model, this takes whatever is in the training and
// produces a sequence of model states. The Model provides a function for
// transforming those into Activity labels, which is the alphabet shared
// by all block labeling methods and ultimately the product of this method.
template <HQRFMethod Method>
std::vector<ActivityLabeler::Activity> BlockHQRegionFinder<Method>::LabelActivities(const BlockLevelMetrics& metrics) const
{
    using Model = typename crf_traits<Method>::Model;

    constexpr size_t numTransStates = static_cast<size_t>(Model::State::CARDINALITY);
    const auto swCurve = std::vector<float>(Model::hswCurve.begin(), Model::hswCurve.end());

    const auto& numPulses         = metrics.NumPulsesAll().data();
    const auto& totalPulseWidth   = metrics.PulseWidth().data();
    const auto& numHalfSandwiches = metrics.NumHalfSandwiches().data();
    const auto& numSandwiches     = metrics.NumSandwiches().data();
    const auto& numLabelStutters  = metrics.NumPulseLabelStutters().data();
    const auto& numFrames         = metrics.NumFrames().data();

    const auto& viterbiScores     = metrics.PulseDetectionScore().data();
    const auto& autocorrelations  = metrics.TraceAutocorr().data();
    const auto& blockSNRs         = metrics.BlockMinSNR().data();
    const auto& blockLowSNRs      = metrics.BlockLowSNR().data();
    const auto& maxPkMaxNorms     = metrics.MaxPkMaxNorms().data();

    const auto& pkzvarNorms       = metrics.PkzVarNorms().data();
    const auto& bpzvarNorms       = metrics.BpzVarNorms().data();

    size_t numBlocks = numPulses.size();

    // Test if the continuous feature value is within the discrete feature
    // function's bin
    auto inBin = [](float value, float start, float end) -> bool
    {
        return (value >= start && value < end);
    };

    // Fill the feature function values for each block
    fmat_t featureEmissions(numBlocks, Model::numFeatures);
    for (size_t t = 0; t < numBlocks; t++)
    {
        float mfSeconds = float(numFrames[t]) / frameRate_;
        float pulserate = float(numPulses[t]) / mfSeconds;
        float meanPulseWidth = (numPulses[t] > 0) ?
            totalPulseWidth[t] / static_cast<float>(numPulses[t]) : 0.0f;
        float hswr = (numPulses[t] > 0) ?
            numHalfSandwiches[t] / static_cast<float>(numPulses[t]) : 0.0f;
        float swr = (numPulses[t] > 0) ?
            numSandwiches[t] / static_cast<float>(numPulses[t]) : 0.0f;
        float labelStutterRate = (numPulses[t] > 0) ?
            numLabelStutters[t] / static_cast<float>(numPulses[t]) : 0.0f;
        float blockSNR = blockSNRs[t];
        float blockLowSNR = blockLowSNRs[t];
        float viterbiScore = viterbiScores[t];
        float maxPkMaxNorm = maxPkMaxNorms[t];
        float autocorrelation = autocorrelations[t];
        float hswrExp = std::min(0.12f, evaluatePolynomial(swCurve, pulserate));
        float hswrNorm = hswr - hswrExp;
        for (size_t fi = 0; fi < Model::features.size(); ++fi)
        {
            CrfFeature name;
            float start, end;
            std::tie(name, start, end) = Model::features[fi];
            switch(name)
            {
                case CrfFeature::HALFSANDWICHRATE:
                    featureEmissions(t, fi) = inBin(hswr, start, end) ? 1.0f : 0.0f;
                    break;
                case CrfFeature::PULSERATE:
                    featureEmissions(t, fi) = inBin(pulserate, start, end) ? 1.0f : 0.0f;
                    break;
                case CrfFeature::MEANPULSEWIDTH:
                    featureEmissions(t, fi) = inBin(meanPulseWidth, start, end) ? 1.0f : 0.0f;
                    break;
                case CrfFeature::SANDWICHRATE:
                    featureEmissions(t, fi) = inBin(swr, start, end) ? 1.0f : 0.0f;
                    break;
                case CrfFeature::LABELSTUTTERRATE:
                    featureEmissions(t, fi) = inBin(labelStutterRate, start, end) ? 1.0f : 0.0f;
                    break;
                case CrfFeature::MINSNR:
                    featureEmissions(t, fi) = inBin(blockSNR, start, end) ? 1.0f : 0.0f;
                    break;
                case CrfFeature::BLOCKLOWSNR:
                    featureEmissions(t, fi) = inBin(blockLowSNR, start, end) ? 1.0f : 0.0f;
                    break;
                case CrfFeature::VITERBISCORE:
                    featureEmissions(t, fi) = inBin(viterbiScore, start, end) ? 1.0f : 0.0f;
                    break;
                case CrfFeature::MAXPKMAXNORM:
                    featureEmissions(t, fi) = inBin(maxPkMaxNorm, start, end) ? 1.0f : 0.0f;
                    break;
                case CrfFeature::PKZVARNORM:
                    featureEmissions(t, fi) = inBin(pkzvarNorms[t], start, end) ? 1.0f : 0.0f;
                    break;
                case CrfFeature::BPZVARNORM:
                    featureEmissions(t, fi) = inBin(bpzvarNorms[t], start, end) ? 1.0f : 0.0f;
                    break;
                case CrfFeature::AUTOCORRELATION:
                    featureEmissions(t, fi) = inBin(autocorrelation, start, end) ? 1.0f : 0.0f;
                    break;
                case CrfFeature::LOCALHSWRATENORM:
                    featureEmissions(t, fi) = inBin(hswrNorm, start, end) ? 1.0f : 0.0f;
                    break;
                default:
                    throw UnknownFeatureException();
            }
        }
    }

    // Score matrix
    fmat_t scoreMatrix(numBlocks + 1, numTransStates);

    u8mat_t tracebackMatrix(numBlocks + 1, numTransStates);

    // Allow start from any state
    for (size_t t = 0; t < numTransStates; t++)
        scoreMatrix(0, t) = 0;

    std::vector<float> P(numTransStates);
    // Fill the score matrix one block at a time
    for (size_t t = 1; t <= numBlocks; ++t)
    {
        // For each possible current state
        for (size_t q = 0; q < numTransStates; ++q)
        {
            // For each possible previous state
            for (size_t qp = 0; qp < numTransStates; qp++)
            {
                P[qp] = scoreMatrix(t-1, qp) + Model::transitionScores[qp][q];
            }
            // Choose the highest probability from all options for previous
            // state
            size_t qMax = argmax(P.begin(), P.end());
            tracebackMatrix(t, q) = static_cast<uint8_t>(qMax);

            float emiss = 0.0;
            // probability of emission given this state
            for (uint32_t i = 0; i < Model::numFeatures; ++i)
            {
                // the featureEmissions matrix is aligned with the features,
                // not the dynamic programming matrices (which have an extra
                // starting state row)
                emiss += Model::featureWeights[i][q] * featureEmissions(t - 1, i);
            }
            scoreMatrix(t, q) = P[qMax] + emiss;
        }
    }
    //PrintMatrix(scoreMatrix);

    // Trace best path
    std::vector<typename Model::State> traceback(numBlocks);
    auto lastRow = row(scoreMatrix, numBlocks);
    size_t q = argmax(std::begin(lastRow), std::end(lastRow));

    assert(q < numTransStates);
    traceback[numBlocks-1] = static_cast<typename Model::State>(q);

    for (int t = numBlocks-1; t > 0; t--)
    {
        q = tracebackMatrix(t, q);
        assert(q < numTransStates);
        traceback[t-1] = static_cast<typename Model::State>(q);
    }
    return Model::State2Activity(traceback, blockSNRs, snrThresh_);
}

std::vector<ActivityLabeler::Activity> GetBazAL(const BlockLevelMetrics& metrics)
{
    const auto& rtSeq = metrics.ActivityLabels().data();
    std::vector<ActivityLabeler::Activity> ret;
    if (rtSeq.size() > 0)
    {
        std::transform(rtSeq.begin(),
                       rtSeq.end(),
                       std::back_inserter(ret),
                       [](const auto& val){ return static_cast<ActivityLabeler::Activity>(val); });
    }
    return ret;
}

template <>
std::vector<ActivityLabeler::Activity> BlockHQRegionFinder<HQRFMethod::TRAINED_CART_HMM>::LabelActivities(const BlockLevelMetrics& metrics) const
{
    const auto& rtSeq = GetBazAL(metrics);
    if (rtSeq.size() == 0 || ignoreBazAL_)
        return ActivityLabeler::LabelActivities<ActivityLabeler::HQRFParamsMCart>(metrics, frameRate_);
    return rtSeq;
}

template <>
std::vector<ActivityLabeler::Activity> BlockHQRegionFinder<HQRFMethod::TRAINED_CART_CART>::LabelActivities(const BlockLevelMetrics& metrics) const
{
    const auto& rtSeq = GetBazAL(metrics);
    if (rtSeq.size() == 0 || ignoreBazAL_)
        return ActivityLabeler::LabelActivities<ActivityLabeler::HQRFParamsMCart>(metrics, frameRate_);
    return rtSeq;
}

template <>
std::vector<ActivityLabeler::Activity> BlockHQRegionFinder<HQRFMethod::ZOFFSET_CART_HMM>::LabelActivities(const BlockLevelMetrics& metrics) const
{
    const auto& rtSeq = GetBazAL(metrics);
    if (rtSeq.size() == 0 || ignoreBazAL_)
        return ActivityLabeler::LabelActivities<ActivityLabeler::HQRFParamsNCart>(metrics, frameRate_);
    return rtSeq;
}

template <>
std::vector<ActivityLabeler::Activity> BlockHQRegionFinder<HQRFMethod::BAZ_HMM>::LabelActivities(const BlockLevelMetrics& metrics) const
{
    const auto& rtSeq = GetBazAL(metrics);
    if (rtSeq.size() == 0 || ignoreBazAL_)
        throw PBException("HQRF was configured to use Activity Labels from baz "
                          "file, but no labels were found. Please check the "
                          "contents of the baz file, PA/PPA versions, or choose "
                          "a different HQRF method");
    return rtSeq;
}

template <HQRFMethod Method>
std::pair<int, int> BlockHQRegionFinder<Method>::LabelRegions(
        const std::vector<ActivityLabeler::Activity>& activitySeq,
        const float transitionProb[NUM_STATES][NUM_STATES],
        const float emissionProb[NUM_STATES][ActivityLabeler::LABEL_CARDINALITY],
        std::vector<HQRFState>* stateSequence,
        float* loglikelihood) const
{
    size_t T = activitySeq.size();
    // Log-space.  TODO: do this only once.
    float logTransitionProb[NUM_STATES][NUM_STATES];
    for (size_t i = 0; i < NUM_STATES; i++)
        for (size_t j = 0; j < NUM_STATES; j++)
            logTransitionProb[i][j] = logf(transitionProb[i][j]);

    float logEmissionProb[NUM_STATES][ActivityLabeler::LABEL_CARDINALITY];
    for (size_t i = 0; i < NUM_STATES; i++)
        for (size_t j = 0; j < ActivityLabeler::LABEL_CARDINALITY; j++)
            logEmissionProb[i][j] = logf(emissionProb[i][j]);


    // Basic viterbi decoding, in the log domain.
    // We can do something fancier later
    fmat_t dpM(T+1, NUM_STATES);

    // Allow start from any state, except POST_ACTIVE, (which has the
    // same emission pmf as PRE)
    dpM(0, PRE)         = logf(0.33);
    dpM(0, HQ)          = logf(0.33);
    dpM(0, POST_QUIET)  = logf(0.33);
    dpM(0, POST_ACTIVE) = logf(0.0);

    for (size_t t = 1; t <= T; t++) {
        ActivityLabeler::Activity e = activitySeq[t-1];
        for (size_t q = 0; q < NUM_STATES; q++) {
            std::vector<float> P(NUM_STATES);
            for (size_t qp = 0; qp < NUM_STATES; qp++) {
                P[qp] = dpM(t-1, qp) + logTransitionProb[qp][q] + logEmissionProb[q][e];
            }
            size_t qMax = argmax(P.begin(), P.end());
            dpM(t, q) = P[qMax];
        }
    }
    // PrintMatrix(dpM);

    // Traceback
    std::vector<HQRFState> traceback(T);
    auto lastRow = row(dpM, T);
    size_t q = argmax(std::begin(lastRow), std::end(lastRow));
    if (loglikelihood != nullptr) { *loglikelihood = dpM(T, q); }
    traceback[T-1] = (HQRFState) q;

    for (int t = T-1; t > 0; t--)
    {
        ActivityLabeler::Activity e = activitySeq[t-1];
        std::vector<float> P(NUM_STATES);
        for (size_t qp = 0; qp < NUM_STATES; qp++) {
            P[qp] = dpM(t, qp) + logTransitionProb[qp][q] + logEmissionProb[q][e];
        }
        q = argmax(P.begin(), P.end());
        traceback[t-1] = (HQRFState) q;
    }

    // Identify HQ extent... null HQ region should be coded as [0,0);
    // maximal extent is [0, T)
    size_t hqEnd, hqBegin;
    for (hqEnd = T; hqEnd > 0 && traceback[hqEnd-1] != HQ; hqEnd--);
    for (hqBegin = hqEnd; hqBegin > 0 && traceback[hqBegin-1] == HQ; hqBegin--);

    // N.B. (dalexander) If the inferred HQR only includes one window,
    // this is likely an artifact of the restricted transitions in the
    // Markov model (can't go 0->2, have to go 0->1->2) which match
    // physical expectation.  I don't want to add such a transition at
    // this time because I don't understand the physical process where
    // we can go 0->2.  For now, we just replace single-window HQRs
    // with empty HQRs.
    if (hqEnd == hqBegin + 1)
    {
        hqEnd = hqBegin = 0;
    }

    if (stateSequence != nullptr)
    {
        // could do a swap here instead...
        stateSequence->clear();
        stateSequence->resize(traceback.size());
        std::copy(traceback.begin(), traceback.end(), (*stateSequence).begin());
    }

    return std::make_pair(hqBegin, hqEnd);
}

template <HQRFMethod Method>
std::pair<int, int> BlockHQRegionFinder<Method>::FindBlockHQRegion(
        const std::vector<ActivityLabeler::Activity>& activitySeq,
        std::vector<HQRFState>* stateSequence,
        float* loglikelihood) const
{
    size_t T = activitySeq.size();
    float a = 1.0f / T;

    // We scale the transition probs according to the length, so we
    // keep the average expected state dwell times proportional to the
    // length.  We could revisit this in the future, making the
    // expected dwell times be in terms of expected polymerase
    // lifetimes.
    // TODO: gcc seems to be OK with these VLAs, but icc might not be.
    float transitionProb[NUM_STATES][NUM_STATES] =      \
        //{ { 1-a,   a,   0,   0 },   // Pre ->
        { { 1-a, a/2, a/2,   0 },   // Pre ->
          {   0, 1-a, a/2, a/2 },   // HQ  ->
          {   0,   0, 1-a,   a },   // PostQuiet ->
          {   0,   0,   a, 1-a } }; // PostActive ->

    float emissionProb[NUM_STATES][ActivityLabeler::LABEL_CARDINALITY] = \
        { { 0.007f, 0.029f, 0.964f },    // Emit | Pre
          { 0.053f, 0.937f, 0.010f },    // Emit | HQ
          { 0.993f, 0.006f, 0.002f },    // Emit | PostQuiet
          { 0.013f, 0.102f, 0.884f } };  // Emit | PostActive
    //          A0      A1      A2

    return LabelRegions(activitySeq, transitionProb, emissionProb, stateSequence, loglikelihood);
}

template <>
std::pair<int, int> BlockHQRegionFinder<HQRFMethod::TRAINED_CART_HMM>::FindBlockHQRegion(
        const std::vector<ActivityLabeler::Activity>& activitySeq,
        std::vector<HQRFState>* stateSequence,
        float* loglikelihood) const
{
    size_t T = activitySeq.size();
    float a = 1.0f / T;

    // We scale the transition probs according to the length, so we
    // keep the average expected state dwell times proportional to the
    // length.  We could revisit this in the future, making the
    // expected dwell times be in terms of expected polymerase
    // lifetimes.
    // TODO: gcc seems to be OK with these VLAs, but icc might not be.
    float transitionProb[NUM_STATES][NUM_STATES] =      \
        //{ { 1-a,   a,   0,   0 },   // Pre ->
        { { 1-a, a/2, a/2,   0 },   // Pre ->
          {   0, 1-a, a/2, a/2 },   // HQ  ->
          {   0,   0,   1,   0 },   // PostQuiet ->
          {   0,   0,   a, 1-a } }; // PostActive ->

    float emissionProb[NUM_STATES][ActivityLabeler::LABEL_CARDINALITY] = \
        { { 0.007f, 0.054f, 0.939f },    // Emit | Pre
          { 0.067f, 0.867f, 0.066f },    // Emit | HQ
          { 0.965f, 0.014f, 0.021f },    // Emit | PostQuiet
          { 0.026f, 0.186f, 0.788f } };  // Emit | PostActive
    //          A0      A1      A2

    return BlockHQRegionFinder<HQRFMethod::TRAINED_CART_HMM>::LabelRegions(activitySeq, transitionProb, emissionProb, stateSequence, loglikelihood);
}

template <>
std::pair<int, int> BlockHQRegionFinder<HQRFMethod::TRAINED_CART_CART>::FindBlockHQRegion(
        const std::vector<ActivityLabeler::Activity>& activitySeq,
        std::vector<HQRFState>* stateSequence,
        float* loglikelihood) const
{
    (void)stateSequence;
    (void)loglikelihood;
    size_t modeledLatency = 10;
    return ActivityLabeler::LabelRegions(activitySeq, modeledLatency);
}


template <HQRFMethod Method>
std::pair<size_t, size_t> BlockHQRegionFinder<Method>::FindHQRegion(
        const BlockLevelMetrics& metrics,
        const EventData& zmw) const
{
    if (zmw.Truncated()) return {0, 0};

    const auto& activitySeq = LabelActivities(metrics);

    size_t begin, end;
    std::tie(begin, end) = FindBlockHQRegion(activitySeq);

    assert(begin <= end && end <= activitySeq.size());
    // NB: this takes some care to get the end-exclusive convention right.

    const auto& numPulses = zmw.Internal() ? metrics.NumPulsesAll().data() : metrics.NumBasesAll().data();
    assert(numPulses.size() == activitySeq.size());
    size_t pulseBegin = std::accumulate(numPulses.begin(), numPulses.begin()+begin, 0);
    size_t pulseEnd = std::accumulate(numPulses.begin()+begin, numPulses.begin() + end, pulseBegin);

    // If the interval is empty, normalize to 0-0
    if (pulseBegin == pulseEnd)
        return {0,0};
    else
        return {pulseBegin, pulseEnd};
}

template class BlockHQRegionFinder<HQRFMethod::SEQUEL_CRF_HMM>;
template class BlockHQRegionFinder<HQRFMethod::SPIDER_CRF_HMM>;
template class BlockHQRegionFinder<HQRFMethod::ZOFFSET_CRF_HMM>;
template class BlockHQRegionFinder<HQRFMethod::TRAINED_CART_HMM>;
template class BlockHQRegionFinder<HQRFMethod::TRAINED_CART_CART>;
template class BlockHQRegionFinder<HQRFMethod::BAZ_HMM>;
template class BlockHQRegionFinder<HQRFMethod::ZOFFSET_CART_HMM>;

}}} // ::PacBio::Primary::Postprimary

