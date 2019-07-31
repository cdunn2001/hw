#ifndef Mongo_BaseCaller_TraceAnalysis_ActivityLabeler_H_
#define Mongo_BaseCaller_TraceAnalysis_ActivityLabeler_H_
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
// File Description:
//  The per-block classification tree for sequencing activity. These block
//  labels are then used to define a high quality region as part of the high
//  quality region finder (HQRF)

#include <common/BlockActivityLabels.h>
#include <common/TrainedCartParams.h>

namespace PacBio {
namespace Mongo {
namespace ActivityLabeler {

namespace {

// Evaluates a polynomial with a list of coefficients by descending degree
// (e.g. y = ax^2 + bx + c)
template <size_t size>
float evaluatePolynomial(const float (&coeff)[size], float x)
{
    static_assert(size > 0);
    float y = coeff[0];
    for (unsigned int i = 1; i < size; ++i)
        y = y * x + coeff[i];
    return y;
}

}

// perhaps this would be better if it accepted const bm and returned the
// labels...
template <typename BasecallingMetricsType>
void LabelBlock(const BasecallingMetricsType& bm, float frameRate, unsigned int laneSize)
{
    const auto& stdDevAll = bm.TraceMetrics().FrameBaselineSigmaDWS();
    for (size_t zmw = 0; zmw < laneSize; ++zmw)
    {
        std::vector<float> features;
        features.resize(NUM_FEATURES, 0.0f);
        const float seconds = bm.TraceMetrics().NumFrames()[zmw] / frameRate;

        features[PULSERATE] = bm.NumPulses()[zmw] / seconds;
        features[SANDWICHRATE] = bm.NumPulses()[zmw] > 0 ?
            static_cast<float>(bm.NumSandwiches()[zmw]) / bm.NumPulses()[zmw] : 0.0f;

        const float hswr = (bm.NumPulses()[zmw] > 0) ?
            static_cast<float>(bm.NumHalfSandwiches()[zmw]) / bm.NumPulses()[zmw] : 0.0f;
        const float hswrExp = std::min(
                TrainedCart::maxAcceptableHalfsandwichRate,
                evaluatePolynomial(TrainedCart::hswCurve,
                                   features[PULSERATE]));
        features[LOCALHSWRATENORM] = hswr - hswrExp;

        features[VITERBISCORE] = bm.TraceMetrics().PulseDetectionScore()[zmw];
        features[MEANPULSEWIDTH] = bm.PulseWidth()[zmw];
        features[LABELSTUTTERRATE] = (bm.NumPulses()[zmw] > 0) ?
            static_cast<float>(bm.NumPulseLabelStutters()[zmw]) / bm.NumPulses()[zmw] : 0.0f;

        const float stdDev = stdDevAll[zmw];
        const auto& pkmid = bm.PkmidMean();
        /* TODO re-enable when DetectionModel Analogs are available
        const auto& pkbases = bm.NumBasesByAnalog();
        const float totBases = static_cast<float>(bm.NumBases()[zmw]);

        std::vector<float> relamps;
        // these amps aren't relative, and therefore could be misleading. Scoping
        // maxamp to prevent misuse
        {
            std::vector<float> amps = {
                bm.TraceMetrics().Analog()[0][zmw].Amplitude(),
                bm.TraceMetrics().Analog()[1][zmw].Amplitude(),
                bm.TraceMetrics().Analog()[2][zmw].Amplitude(),
                bm.TraceMetrics().Analog()[3][zmw].Amplitude()};

            const float maxamp = *std::max_element(amps.begin(), amps.end());
            std::transform(amps.begin(), amps.end(), std::back_inserter(relamps),
                           [maxamp](float amp) { return amp/maxamp; });
        }
        // This never changes, could be a member of an object
        const float minamp = *std::min_element(relamps.begin(), relamps.end());

        for (size_t i = 0; i < BasecallingMetricsType::NumAnalogs; ++i)
        {
            if (!std::isnan(pkmid[i][zmw]) && pkbases[i][zmw] > 0 && stdDev > 0
                    && relamps[i][zmw] > 0 && totBases > 0)
            {
                features[BLOCKLOWSNR] += pkmid[i][zmw] / stdDev * pkbases[i][zmw]
                                         / totBases * minamp / relamps[i];
            }
        }
        */

        const auto& pkmax = bm.PkMax();
        for (size_t aI = 0; aI < BasecallingMetricsType::NumAnalogs && stdDev > 0; ++aI)
        {
            features[MAXPKMAXNORM] = std::fmax(
                features[MAXPKMAXNORM],
                (pkmax[aI][zmw] - pkmid[aI][zmw]) / stdDev);
        }

        features[AUTOCORRELATION] = bm.TraceMetrics().Autocorrelation()[zmw];

        /* TODO re-enable when DetectionModel Analogs are available
        int lowAnalogIndex = std::distance(
            relamps.begin(),
            std::min_element(relamps.begin(), relamps.end()));

        for (const auto& val : bm.Bpzvar())
        {
            features[BPZVARNORM] += std::isnan(val) ? 0.0f : val;
        }
        const auto& lowbp = bm.Bpzvar()[lowAnalogIndex];
        features[BPZVARNORM] -=  std::isnan(lowbp) ? 0.0f : lowbp;
        features[BPZVARNORM] /= 3.0f;

        for (const auto& val : bm.Pkzvar())
        {
            features[PKZVARNORM] += std::isnan(val) ? 0.0f : val;
        }
        const auto& lowpk = bm.Pkzvar()[lowAnalogIndex];
        features[PKZVARNORM] -=  std::isnan(lowpk) ? 0.0f : lowpk;
        features[PKZVARNORM] /= 3.0f;
        */

        for (size_t i = 0; i < features.size(); ++i)
        {
            assert(!std::isnan(features[i]));
        }

        size_t current = 0;
        while (TrainedCart::feature[current] >= 0)
        {
            if (features[TrainedCart::feature[current]]
                    <= TrainedCart::threshold[current])
            {
                current = TrainedCart::childrenLeft[current];
            }
            else
            {
                current = TrainedCart::childrenRight[current];
            }
        }
        bm.ActivityLabel()[zmw] = static_cast<HQRFPhysicalState>(TrainedCart::value[current]);
    }
}

}}} // ::PacBio::Mongo::ActivityLabeler

#endif // Mongo_BaseCaller_TraceAnalysis_ActivityLabeler_H_
