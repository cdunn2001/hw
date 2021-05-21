#ifndef Sequel_PostPrimary_ClassificationTree_H_
#define Sequel_PostPrimary_ClassificationTree_H_
// Copyright (c) 2019, Pacific Biosciences of California, Inc.
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

#include <vector>
#include <bazio/BlockActivityLabels.h>
#include <pacbio/primary/HQRFMethod.h>

namespace PacBio {
namespace Primary {
namespace ActivityLabeler {

template <typename Model>
std::vector<Activity> LabelActivities(const BlockLevelMetrics& metrics,
                                      float frameRate)
{
    const auto swCurve = std::vector<float>(Model::hswCurve.begin(), Model::hswCurve.end());
    const auto& numPulses         = metrics.NumPulsesAll().data();
    const auto& totalPulseWidth   = metrics.PulseWidth().data();
    const auto& numHalfSandwiches = metrics.NumHalfSandwiches().data();
    const auto& numSandwiches     = metrics.NumSandwiches().data();
    const auto& numLabelStutters  = metrics.NumPulseLabelStutters().data();
    const auto& numFrames         = metrics.NumFrames().data();
    const auto& viterbiScores     = metrics.PulseDetectionScore().data();
    const auto& autocorrelations  = metrics.TraceAutocorr().data();
    const auto& blockLowSNRs      = metrics.BlockLowSNR().data();
    const auto& maxPkMaxNorms     = metrics.MaxPkMaxNorms().data();
    const auto& pkzvarNorms       = metrics.PkzVarNorms().data();
    const auto& bpzvarNorms       = metrics.BpzVarNorms().data();

    const size_t numBlocks = numFrames.size();

    std::vector<Activity> stateSeq(numBlocks, Activity::A0);
    for (size_t t = 0; t < numBlocks; t++)
    {
        const float mfSeconds = float(numFrames[t]) / frameRate;
        const float pulserate = float(numPulses[t]) / mfSeconds;
        const float hswr = (numPulses[t] > 0) ?
            numHalfSandwiches[t] / static_cast<float>(numPulses[t]) : 0.0f;
        const float hswrExp = std::min(Model::maxAcceptableHalfsandwichRate,
                                       Postprimary::evaluatePolynomial(
                                           swCurve, pulserate));

        std::vector<float> features;
        features.resize(Model::featureOrder.size(), 0.0f);
        for (size_t fidx = 0; fidx < Model::featureOrder.size(); ++fidx)
        {
            switch (Model::featureOrder[fidx])
            {
            case CART_FEATURES::PULSERATE:
                features[fidx] = pulserate;
                break;
            case CART_FEATURES::SANDWICHRATE:
                features[fidx] = (numPulses[t] > 0) ?
                    numSandwiches[t] / static_cast<float>(numPulses[t]) : 0.0f;
                break;
            case CART_FEATURES::LOCALHSWRATENORM:
                features[fidx] = hswr - hswrExp;
                break;
            case CART_FEATURES::VITERBISCORE:
                features[fidx] = viterbiScores[t];
                break;
            case CART_FEATURES::MEANPULSEWIDTH:
                features[fidx] = (numPulses[t] > 0) ?
                    totalPulseWidth[t] / static_cast<float>(numPulses[t]) : 0.0f;
                break;
            case CART_FEATURES::LABELSTUTTERRATE:
                features[fidx] = (numPulses[t] > 0) ?
                    numLabelStutters[t] / static_cast<float>(numPulses[t]) : 0.0f;
                break;
            case CART_FEATURES::BLOCKLOWSNR:
                features[fidx] = blockLowSNRs[t];
                break;
            case CART_FEATURES::MAXPKMAXNORM:
                features[fidx] = maxPkMaxNorms[t];
                break;
            case CART_FEATURES::AUTOCORRELATION:
                features[fidx] = autocorrelations[t];
                break;
            case CART_FEATURES::BPZVARNORM:
                features[fidx] = bpzvarNorms[t];
                break;
            case CART_FEATURES::PKZVARNORM:
                features[fidx] = pkzvarNorms[t];
                break;
            default:
                throw PBException("ActivityLabeler feature specified by model but not supported by software!");
                break;
            }
            assert(!std::isnan(features[fidx]));
        }

        // Thresholds should not be for standardized or normalized features. If
        // they are, then we should implement standardization or normalization
        size_t current = 0;
        while (Model::feature[current] >= 0)
        {
            if (features[Model::feature[current]] <= Model::threshold[current])
            {
                current = Model::childrenLeft[current];
            }
            else
            {
                current = Model::childrenRight[current];
            }
        }
        stateSeq[t] = static_cast<Activity>(Model::value[current]);
    }
    return stateSeq;
}

std::pair<int, int> LabelRegions(
        const std::vector<Activity>& activityLabels,
        size_t latency);

}}} // ::PacBio::Primary::ActivityLabeler
#endif // Sequel_PostPrimary_ClassificationTree_H_
