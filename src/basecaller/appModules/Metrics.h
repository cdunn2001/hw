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

#ifndef PACBIO_APPLICATION_METRICS_H
#define PACBIO_APPLICATION_METRICS_H

#include <half.hpp>

#include <dataTypes/BasecallingMetrics.h>
#include <bazio/writing/MetricBlock.h>

namespace PacBio::Application {

/// Metric representation used by the prelimHQ and Baz writer.
/// We have two flavors:
///
/// Production metrics: Default, used for production workflows.
/// Internal metrics: Used for generated data for prelimHQ and HQRF training.
///
/// \tparam UInt    Unsigned integer type
/// \tparam Flt     Float type
template <typename UInt, typename Flt>
struct ProductionMetrics : BazIO::MetricBlock<ProductionMetrics<UInt,Flt>>
{
    using UIntType = UInt;
    using FltType = Flt;
    using UIntArray = std::array<UInt,4>;
    using FltArray = std::array<Flt,4>;

    ProductionMetrics() = default;

    ProductionMetrics(const Mongo::Data::BasecallingMetrics& bm, size_t zmwIndex)
    : numFrames { bm.numFrames[zmwIndex] }
    , pulseWidth { bm.numPulseFrames[zmwIndex] }
    , baseWidth { bm.numBaseFrames[zmwIndex] }
    , numPulses { bm.numPulses[zmwIndex] }
    , numBases { bm.numBases[zmwIndex] }
    , baselineVar { bm.frameBaselineVarianceDWS[zmwIndex] }
    , baselineMean { bm.frameBaselineDWS[zmwIndex] }
    , numBaselineFrames { bm.numFramesBaseline[zmwIndex] }
    , activityLabel { static_cast<uint8_t>(bm.activityLabel[zmwIndex]) }
    {
        for (size_t a = 0; a < pkmid.size(); a++)
        {
            pkmid[a] = bm.pkMidSignal[a][zmwIndex];
            numPkmidFrames[a] = bm.numPkMidFrames[a][zmwIndex];
            if (numPkmidFrames[a] > 0) pkmid[a] /= numPkmidFrames[a];
        }
    }
  
    template <typename U,typename F>
    void Aggregate(const ProductionMetrics<U,F>& bmb)
    {
         // Weighted metrics
        UIntType nbf = (numBaselineFrames + bmb.numBaselineFrames);
        baselineMean = (numBaselineFrames * baselineMean) +
                       (bmb.numBaselineFrames * bmb.baselineMean);
        if (nbf > 0)
        {
            baselineMean /= nbf;

            // Compute pooled variance.
            auto computePooledVal = [](const auto& nbf, const auto& var, auto& f)
            {
                size_t n = (nbf > 1) ? nbf - 1 : nbf;
                f += (n * var);
                return (nbf > 0);
            };

            FltType bv = FltType(0);
            UIntType nk = nbf;
            if (computePooledVal(numBaselineFrames, baselineVar, bv)) nk--;
            if (computePooledVal(bmb.numBaselineFrames, bmb.baselineVar, bv)) nk--;
            baselineVar = bv / nk;

            numBaselineFrames = nbf;
        }

        for (size_t a = 0; a < pkmid.size(); a++)
        {
            pkmid[a] = (numPkmidFrames[a] * pkmid[a]) + (bmb.numPkmidFrames[a] * bmb.pkmid[a]);
            numPkmidFrames[a] = numPkmidFrames[a] + bmb.numPkmidFrames[a];
            if (numPkmidFrames[a] > 0) pkmid[a] /= numPkmidFrames[a];
        }

        numFrames = numFrames + bmb.numFrames;
        pulseWidth = pulseWidth + bmb.pulseWidth;
        baseWidth = baseWidth + bmb.baseWidth;
        numPulses = numPulses + bmb.numPulses;
        numBases = numBases + bmb.numBases;

        activityLabel = bmb.activityLabel;
    }

    template <typename U, typename F>
    void Set(const ProductionMetrics<U,F>& bmb)
    {
        numFrames = bmb.numFrames;
        pulseWidth = bmb.pulseWidth;
        baseWidth = bmb.baseWidth;
        numPulses = bmb.numPulses;
        numBases = bmb.numBases;
        pkmid = { bmb.pkmid[0], bmb.pkmid[1], bmb.pkmid[2], bmb.pkmid[3] };
        numPkmidFrames = { bmb.numPkmidFrames[0], bmb.numPkmidFrames[1], bmb.numPkmidFrames[2], bmb.numPkmidFrames[3] };
        baselineVar = bmb.baselineVar;
        baselineMean = bmb.baselineMean;
        numBaselineFrames = bmb.numBaselineFrames;
        activityLabel = bmb.activityLabel;
    }

    void Reset()
    {
        numFrames = 0;
        pulseWidth = 0;
        baseWidth = 0;
        numPulses = 0;
        numBases = 0;
        pkmid = {FltType(0), FltType(0), FltType(0), FltType(0)};
        numPkmidFrames = {0, 0, 0, 0};
        baselineVar = 0;
        baselineMean = 0;
        numBaselineFrames = 0;
        activityLabel = static_cast<uint8_t>(Mongo::Data::HQRFPhysicalStates::NUM_PHYS_STATES);
    }

    void Convert(Primary::SpiderMetricBlock& sm) const
    {
        sm.NumFrames(numFrames);
        sm.PulseWidth(pulseWidth);
        sm.BaseWidth(baseWidth);
        sm.NumPulses(numPulses);

        sm.PkmidA(pkmid[0]);
        sm.PkmidC(pkmid[1]);
        sm.PkmidG(pkmid[2]);
        sm.PkmidT(pkmid[3]);
        sm.NumPkmidFramesA(numPkmidFrames[0]);
        sm.NumPkmidFramesC(numPkmidFrames[1]);
        sm.NumPkmidFramesG(numPkmidFrames[2]);
        sm.NumPkmidFramesT(numPkmidFrames[3]);

        sm.Baselines({ baselineMean });
        sm.BaselineSds({ sqrt(baselineVar) });
        sm.NumBaselineFrames({ numBaselineFrames });
        sm.ActivityLabel(activityLabel);
    }

    uint8_t ActivityLabel() const { return activityLabel; }

    bool HasData() const { return numFrames != 0; }

    UIntType    numFrames = 0;
    UIntType    pulseWidth = 0;
    UIntType    baseWidth = 0;
    UIntType    numPulses = 0;
    UIntType    numBases = 0;
    FltArray    pkmid = {FltType(0), FltType(0), FltType(0), FltType(0)};
    UIntArray   numPkmidFrames = {0, 0, 0, 0};
    FltType     baselineVar = FltType(0);
    FltType     baselineMean = FltType(0);
    UIntType    numBaselineFrames = 0;
    uint8_t     activityLabel = static_cast<uint8_t>(Mongo::Data::HQRFPhysicalStates::NUM_PHYS_STATES);
};

template <typename UInt, typename Flt>
struct InternalMetrics : BazIO::MetricBlock<InternalMetrics<UInt,Flt>>
{
    using UIntType = UInt;
    using FltType = Flt;
    using UIntArray = std::array<UInt,4>;
    using FltArray = std::array<Flt,4>;

    InternalMetrics() = default;

    InternalMetrics(const Mongo::Data::BasecallingMetrics& bm, size_t zmwIndex)
    : productionMetrics { bm, zmwIndex }
    , numHalfSandwiches { bm.numHalfSandwiches[zmwIndex] }
    , numSandwiches { bm.numSandwiches[zmwIndex] }
    , numPulseLabelStutters { bm.numPulseLabelStutters[zmwIndex] }
    , pulseDetectionScore { bm.pulseDetectionScore[zmwIndex] }
    , traceAutoCorrelation { bm.autocorrelation[zmwIndex] }
    {
        for (size_t a = 0; a < pkmax.size(); a++)
        {
            pkmax[a] = bm.pkMax[a][zmwIndex];
            numPkmidBases[a] = bm.numPkMidBasesByAnalog[a][zmwIndex];
            bpzvar[a] = bm.bpZvar[a][zmwIndex];
            pkzvar[a] = bm.pkZvar[a][zmwIndex];
        }
    }

    template <typename U,typename F>
    void Aggregate(const InternalMetrics<U,F>& bmb)
    {
        numHalfSandwiches = numHalfSandwiches + bmb.numHalfSandwiches;
        numSandwiches = numSandwiches + bmb.numSandwiches;
        numPulseLabelStutters = numPulseLabelStutters + bmb.numPulseLabelStutters;

        FltArray pkmid = {FltType(0), FltType(0), FltType(0), FltType(0)};
        UIntArray numPkmidFrames = {0, 0, 0, 0};
        UIntArray numPkmidBasesTmp = {0, 0, 0, 0};
        FltArray pkzvarTmp = {FltType(0), FltType(0), FltType(0), FltType(0)};
        FltArray bpzvarTmp = {FltType(0), FltType(0), FltType(0), FltType(0)};
        for (size_t a = 0; a < pkmid.size(); a++)
        {
            pkmid[a] = (productionMetrics.numPkmidFrames[a] * productionMetrics.pkmid[a]) + (bmb.productionMetrics.numPkmidFrames[a] * bmb.productionMetrics.pkmid[a]);
            numPkmidFrames[a] = productionMetrics.numPkmidFrames[a] + bmb.productionMetrics.numPkmidFrames[a];
            if (numPkmidFrames[a] > 0) pkmid[a] /= numPkmidFrames[a];

            if (productionMetrics.numPkmidFrames[a] > 1) pkzvarTmp[a] += productionMetrics.numPkmidFrames[a] * pkzvar[a];
            if (bmb.productionMetrics.numPkmidFrames[a] > 1) pkzvarTmp[a] += bmb.productionMetrics.numPkmidFrames[a] * bmb.pkzvar[a];

            numPkmidBasesTmp[a] = numPkmidBases[a] + bmb.numPkmidBases[a];
            if (numPkmidBases[a] > 1) bpzvarTmp[a] += bpzvar[a] * productionMetrics.pkmid[a] * productionMetrics.pkmid[a] * numPkmidBases[a];
            if (bmb.numPkmidBases[a] > 1) bpzvarTmp[a] += bmb.bpzvar[a] * bmb.productionMetrics.pkmid[a] * bmb.productionMetrics.pkmid[a] * bmb.numPkmidBases[a];
        }

        for (size_t a = 0; a < pkmid.size(); a++)
        {
            pkzvar[a] = pkzvarTmp[a];
            if (numPkmidFrames[a] > 1) pkzvar[a] /= numPkmidFrames[a];

            bpzvar[a] = bpzvarTmp[a];
            numPkmidBases[a] = numPkmidBasesTmp[a];
            if (numPkmidFrames[a] > 0 && numPkmidBases[a] > 1) bpzvar[a] /= (numPkmidBases[a] * pkmid[a] * pkmid[a]);

            pkmax[a] = std::max(pkmax[a], bmb.pkmax[a]);
        }

        UIntType numFrames = productionMetrics.numFrames + bmb.productionMetrics.numFrames;
        pulseDetectionScore = (productionMetrics.numFrames * pulseDetectionScore) +
                              (bmb.productionMetrics.numFrames * bmb.pulseDetectionScore);
        traceAutoCorrelation = (productionMetrics.numFrames * traceAutoCorrelation) +
                               (bmb.productionMetrics.numFrames * bmb.traceAutoCorrelation);
        if (numFrames > 0)
        {
            pulseDetectionScore /= numFrames;
            traceAutoCorrelation /= numFrames;
        }

        productionMetrics.Aggregate(bmb.productionMetrics);
    }

    template <typename U,typename F>
    void Set(const InternalMetrics<U,F>& bmb)
    {
        productionMetrics.Set(bmb.productionMetrics);

        numHalfSandwiches = bmb.numHalfSandwiches;
        numSandwiches = bmb.numSandwiches;
        numPulseLabelStutters = bmb.numPulseLabelStutters;
        pulseDetectionScore = bmb.pulseDetectionScore;
        traceAutoCorrelation = bmb.traceAutoCorrelation;
        pkmax = { bmb.pkmax[0], bmb.pkmax[1], bmb.pkmax[2], bmb.pkmax[3] };
        numPkmidBases = { bmb.numPkmidBases[0], bmb.numPkmidBases[1], bmb.numPkmidBases[2], bmb.numPkmidBases[3] };
        bpzvar = { bmb.bpzvar[0], bmb.bpzvar[1], bmb.bpzvar[2], bmb.bpzvar[3] };
        pkzvar = { bmb.pkzvar[0], bmb.pkzvar[1], bmb.pkzvar[2], bmb.pkzvar[3] };
    }

    void Reset()
    {
        productionMetrics.Reset();

        numHalfSandwiches = 0;
        numSandwiches = 0;
        numPulseLabelStutters = 0;
        pulseDetectionScore = 0;
        traceAutoCorrelation = 0;
        pkmax = {FltType(0), FltType(0), FltType(0), FltType(0)};
        numPkmidBases = {0, 0, 0, 0};
        bpzvar = {FltType(0), FltType(0), FltType(0), FltType(0)};
        pkzvar = {FltType(0), FltType(0), FltType(0), FltType(0)};
    }

    void Convert(Primary::SpiderMetricBlock& sm) const
    {
        productionMetrics.Convert(sm);

        sm.NumHalfSandwiches(numHalfSandwiches);
        sm.NumSandwiches(numSandwiches);
        sm.NumPulseLabelStutters(numPulseLabelStutters);
        sm.PulseDetectionScore(pulseDetectionScore);
        sm.TraceAutocorr(traceAutoCorrelation);

        sm.PkmaxA(pkmax[0]);
        sm.PkmaxC(pkmax[1]);
        sm.PkmaxG(pkmax[2]);
        sm.PkmaxT(pkmax[3]);

        sm.NumPkmidBasesA(numPkmidBases[0]);
        sm.NumPkmidBasesC(numPkmidBases[1]);
        sm.NumPkmidBasesG(numPkmidBases[2]);
        sm.NumPkmidBasesT(numPkmidBases[3]);

        sm.BpzvarA(bpzvar[0]);
        sm.BpzvarC(bpzvar[1]);
        sm.BpzvarG(bpzvar[2]);
        sm.BpzvarT(bpzvar[3]);
    }

    uint8_t ActivityLabel() const { return productionMetrics.ActivityLabel(); }

    bool HasData() const { return productionMetrics.HasData(); }

    ProductionMetrics<UInt,Flt> productionMetrics;

    UIntType    numHalfSandwiches = 0;
    UIntType    numSandwiches = 0;
    UIntType    numPulseLabelStutters = 0;
    FltType     pulseDetectionScore = FltType(0);
    FltType     traceAutoCorrelation = FltType(0);
    FltArray    pkmax = {FltType(0), FltType(0), FltType(0), FltType(0)};
    UIntArray   numPkmidBases = {0, 0, 0, 0};
    FltArray    bpzvar = {FltType(0), FltType(0), FltType(0), FltType(0)};
    FltArray    pkzvar = {FltType(0), FltType(0), FltType(0), FltType(0)};
};

struct ProductionMetricsGroup
{
    using MetricT = ProductionMetrics<uint16_t,half_float::half>;
    using MetricAggregatedT = ProductionMetrics<uint32_t,half_float::half>;
};

struct InternalMetricsGroup
{
    using MetricT = InternalMetrics<uint16_t,half_float::half>;
    using MetricAggregatedT = InternalMetrics<uint16_t,half_float::half>;
};


} // namespace PacBio::Application


#endif // PACBIO_APPLICATION_METRICS_H