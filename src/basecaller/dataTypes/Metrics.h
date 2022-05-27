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

#ifndef PACBIO_MONGO_DATA_METRICS_H
#define PACBIO_MONGO_DATA_METRICS_H

#include <dataTypes/BasecallingMetrics.h>
#include <bazio/writing/MetricBlock.h>

namespace PacBio::Mongo::Data {

/// Metric representation used by the prelimHQ and Baz writer.
/// We have two flavors:
///
/// Production metrics: Default, used for production workflows.
/// Complete metrics:   The entire set of metrics, contains the production metrics, used
///                     for generated data for prelimHQ and HQRF training.
///
/// \tparam UInt    Unsigned integer type
/// \tparam Flt     Float type
template <typename UInt, typename Flt>
struct ProductionMetrics : BazIO::MetricBlock<ProductionMetrics<UInt,Flt>>
{
    using UIntType = UInt;
    using FltType = Flt;
    using UIntArray = std::array<UInt, 4>;
    using FltArray = std::array<Flt, 4>;

    ProductionMetrics() = default;

    ProductionMetrics(const Mongo::Data::BasecallingMetrics& bm, size_t zmwIndex)
        : numFrames_ {bm.numFrames[zmwIndex]}
        , pulseWidth_ {bm.numPulseFrames[zmwIndex]}
        , baseWidth_ {bm.numBaseFrames[zmwIndex]}
        , numPulses_ {bm.numPulses[zmwIndex]}
        , numBases_ {bm.numBases[zmwIndex]}
        , baselineVar_ {bm.frameBaselineVarianceDWS[zmwIndex]}
        , baselineMean_ {bm.frameBaselineDWS[zmwIndex]}
        , numBaselineFrames_ {bm.numFramesBaseline[zmwIndex]}
        , activityLabel_ {static_cast<uint8_t>(bm.activityLabel[zmwIndex])}
    {
        for (size_t a = 0; a < pkmid_.size(); a++)
        {
            float pkmid = bm.pkMidSignal[a][zmwIndex];
            numPkmidFrames_[a] = bm.numPkMidFrames[a][zmwIndex];
            if (numPkmidFrames_[a] > 0) pkmid /= numPkmidFrames_[a];
            pkmid_[a] = pkmid;
            numBasesByAnalog_[a] = bm.numBasesByAnalog[a][zmwIndex];
        }
    }

    template<typename U, typename F>
    void Aggregate(const ProductionMetrics<U, F>& bmb)
    {
        static_assert(sizeof(UIntType) >= 4, "Unsigned integer type for metric aggregation must be at least 4 bytes!");
        static_assert(sizeof(FltType) >= 4, "Float type for metric aggregation must be at least 4 bytes!");

        // Weighted metrics
        UIntType nbf = (numBaselineFrames_ + bmb.NumBaselineFrames());
        baselineMean_ = (numBaselineFrames_ * baselineMean_) +
                        (bmb.NumBaselineFrames() * static_cast<FltType>(bmb.BaselineMean()));
        if (nbf > 0)
        {
            baselineMean_ /= nbf;

            // Compute pooled variance.
            auto computePooledVal = [](const UIntType& nbf, const FltType& var, FltType& f) {
                size_t n = (nbf > 1) ? nbf - 1 : nbf;
                f += (n * var);
                return (nbf > 0);
            };

            FltType bv = FltType(0);
            UIntType nk = nbf;
            if (computePooledVal(numBaselineFrames_, baselineVar_, bv)) nk--;
            if (computePooledVal(bmb.NumBaselineFrames(), static_cast<FltType>(bmb.BaselineVar()), bv)) nk--;
            baselineVar_ = bv / nk;
            numBaselineFrames_ = nbf;
        }

        for (size_t a = 0; a < pkmid_.size(); a++)
        {
            pkmid_[a] = (numPkmidFrames_[a] * pkmid_[a]) +
                        (bmb.NumPkmidFrames()[a] * static_cast<FltType>(bmb.Pkmid()[a]));
            numPkmidFrames_[a] = numPkmidFrames_[a] + bmb.NumPkmidFrames()[a];
            if (numPkmidFrames_[a] > 0) pkmid_[a] /= numPkmidFrames_[a];

            numBasesByAnalog_[a] = numBasesByAnalog_[a] + bmb.NumBasesByAnalog()[a];
        }

        numFrames_ = numFrames_ + bmb.NumFrames();
        pulseWidth_ = pulseWidth_ + bmb.PulseWidth();
        baseWidth_ = baseWidth_ + bmb.BaseWidth();
        numPulses_ = numPulses_ + bmb.NumPulses();
        numBases_ = numBases_ + bmb.NumBases();


        activityLabel_ = bmb.ActivityLabel();
    }

    void Convert(Primary::SpiderMetricBlock& sm) const
    {
        sm.NumFrames(numFrames_);
        sm.PulseWidth(pulseWidth_);
        sm.BaseWidth(baseWidth_);
        sm.NumPulses(numPulses_);

        sm.PkmidA(pkmid_[0]);
        sm.PkmidC(pkmid_[1]);
        sm.PkmidG(pkmid_[2]);
        sm.PkmidT(pkmid_[3]);
        sm.NumPkmidFramesA(numPkmidFrames_[0]);
        sm.NumPkmidFramesC(numPkmidFrames_[1]);
        sm.NumPkmidFramesG(numPkmidFrames_[2]);
        sm.NumPkmidFramesT(numPkmidFrames_[3]);
        sm.NumBasesA(numBasesByAnalog_[0]);
        sm.NumBasesC(numBasesByAnalog_[1]);
        sm.NumBasesG(numBasesByAnalog_[2]);
        sm.NumBasesT(numBasesByAnalog_[3]);

        sm.Baselines({baselineMean_});
        sm.BaselineSds({sqrt(baselineVar_)});
        sm.NumBaselineFrames({numBaselineFrames_});
        sm.ActivityLabel(activityLabel_);
    }

    uint8_t ActivityLabel() const
    { return activityLabel_; }

    bool HasData() const
    { return numFrames_ != 0; }

    UIntType NumFrames() const
    { return numFrames_; }

    UIntType PulseWidth() const
    { return pulseWidth_; }

    UIntType BaseWidth() const
    { return baseWidth_; }

    UIntType NumPulses() const
    { return numPulses_; }

    UIntType NumBases() const
    { return numBases_; }

    const FltArray& Pkmid() const
    { return pkmid_; }

    const UIntArray& NumPkmidFrames() const
    { return numPkmidFrames_; }

    const UIntArray& NumBasesByAnalog() const
    { return numBasesByAnalog_; }

    FltType BaselineVar() const
    { return baselineVar_; }

    FltType BaselineMean() const
    { return baselineMean_; }

    UIntType NumBaselineFrames() const
    { return numBaselineFrames_; }

private:
    UIntType    numFrames_ = 0;
    UIntType    pulseWidth_ = 0;
    UIntType    baseWidth_ = 0;
    UIntType    numPulses_ = 0;
    UIntType    numBases_ = 0;
    FltArray    pkmid_ = {FltType(0), FltType(0), FltType(0), FltType(0)};
    UIntArray   numPkmidFrames_ = {0, 0, 0, 0};
    UIntArray   numBasesByAnalog_ = {0, 0, 0, 0};
    FltType     baselineVar_ = FltType(0);
    FltType     baselineMean_ = FltType(0);
    UIntType    numBaselineFrames_ = 0;
    uint8_t     activityLabel_ = static_cast<uint8_t>(Mongo::Data::HQRFPhysicalStates::NUM_PHYS_STATES);
};

template <typename UInt, typename Flt>
struct CompleteMetrics : BazIO::MetricBlock<CompleteMetrics<UInt,Flt>>
{
    using UIntType = UInt;
    using FltType = Flt;
    using ProductionMetricsT = ProductionMetrics<UInt,Flt>;
    using UIntArray = std::array<UInt,4>;
    using FltArray = std::array<Flt,4>;

    CompleteMetrics() = default;

    CompleteMetrics(const Mongo::Data::BasecallingMetrics& bm, size_t zmwIndex)
        : productionMetrics_ { bm, zmwIndex }
        , numHalfSandwiches_ { bm.numHalfSandwiches[zmwIndex] }
        , numSandwiches_ { bm.numSandwiches[zmwIndex] }
        , numPulseLabelStutters_ { bm.numPulseLabelStutters[zmwIndex] }
        , pulseDetectionScore_ { bm.pulseDetectionScore[zmwIndex] }
        , traceAutoCorrelation_ { bm.autocorrelation[zmwIndex] }
    {
        for (size_t a = 0; a < pkmax_.size(); a++)
        {
            pkmax_[a] = bm.pkMax[a][zmwIndex];
            numPkmidBases_[a] = bm.numPkMidBasesByAnalog[a][zmwIndex];
            bpzvar_[a] = bm.bpZvar[a][zmwIndex];
            pkzvar_[a] = bm.pkZvar[a][zmwIndex];
        }
    }

    template <typename U,typename F>
    void Aggregate(const CompleteMetrics<U,F>& bmb)
    {
        numHalfSandwiches_ = numHalfSandwiches_ + bmb.NumHalfSandwiches();
        numSandwiches_ = numSandwiches_ + bmb.NumSandwiches();
        numPulseLabelStutters_ = numPulseLabelStutters_ + bmb.NumPulseLabelStutters();

        FltArray pkmid = {FltType(0), FltType(0), FltType(0), FltType(0)};
        UIntArray numPkmidFrames = {0, 0, 0, 0};
        UIntArray numPkmidBasesTmp = {0, 0, 0, 0};
        FltArray pkzvarTmp = {FltType(0), FltType(0), FltType(0), FltType(0)};
        FltArray bpzvarTmp = {FltType(0), FltType(0), FltType(0), FltType(0)};
        for (size_t a = 0; a < pkzvar_.size(); a++)
        {
            pkmid[a] = (productionMetrics_.NumPkmidFrames()[a] * productionMetrics_.Pkmid()[a]) +
                       (bmb.ProdMetrics().NumPkmidFrames()[a] * static_cast<FltType>(bmb.ProdMetrics().Pkmid()[a]));
            numPkmidFrames[a] = productionMetrics_.NumPkmidFrames()[a] + bmb.ProdMetrics().NumPkmidFrames()[a];
            if (numPkmidFrames[a] > 0) pkmid[a] /= numPkmidFrames[a];

            if (productionMetrics_.NumPkmidFrames()[a] > 1)
                pkzvarTmp[a] += productionMetrics_.NumPkmidFrames()[a] * pkzvar_[a];
            if (bmb.ProdMetrics().NumPkmidFrames()[a] > 1)
                pkzvarTmp[a] += bmb.ProdMetrics().NumPkmidFrames()[a] * static_cast<FltType>(bmb.Pkzvar()[a]);

            numPkmidBasesTmp[a] = numPkmidBases_[a] + bmb.NumPkmidBases()[a];
            if (numPkmidBases_[a] > 1)
                bpzvarTmp[a] += bpzvar_[a] * productionMetrics_.Pkmid()[a] * productionMetrics_.Pkmid()[a] * numPkmidBases_[a];
            if (bmb.NumPkmidBases()[a] > 1)
                bpzvarTmp[a] += static_cast<FltType>(bmb.Bpzvar()[a]) *
                                static_cast<FltType>(bmb.ProdMetrics().Pkmid()[a]) *
                                static_cast<FltType>(bmb.ProdMetrics().Pkmid()[a]) * bmb.NumPkmidBases()[a];
        }

        for (size_t a = 0; a < pkzvar_.size(); a++)
        {
            pkzvar_[a] = pkzvarTmp[a];
            if (numPkmidFrames[a] > 1) pkzvar_[a] /= numPkmidFrames[a];

            bpzvar_[a] = bpzvarTmp[a];
            numPkmidBases_[a] = numPkmidBasesTmp[a];
            if (numPkmidFrames[a] > 0 && numPkmidBases_[a] > 1) bpzvar_[a] /= (numPkmidBases_[a] * pkmid[a] * pkmid[a]);

            pkmax_[a] = std::max(pkmax_[a], static_cast<FltType>(bmb.Pkmax()[a]));
        }

        UIntType numFrames = productionMetrics_.NumFrames() + bmb.ProdMetrics().NumFrames();
        pulseDetectionScore_ = (productionMetrics_.NumFrames() * pulseDetectionScore_) +
                              (bmb.ProdMetrics().NumFrames() * static_cast<FltType>(bmb.PulseDetectionScore()));
        traceAutoCorrelation_ = (productionMetrics_.NumFrames() * traceAutoCorrelation_) +
                               (bmb.ProdMetrics().NumFrames() * static_cast<FltType>(bmb.TraceAutoCorrelation()));
        if (numFrames > 0)
        {
            pulseDetectionScore_ /= numFrames;
            traceAutoCorrelation_ /= numFrames;
        }

        productionMetrics_.Aggregate(bmb.ProdMetrics());
    }

    void Convert(Primary::SpiderMetricBlock& sm) const
    {
        productionMetrics_.Convert(sm);

        sm.NumHalfSandwiches(numHalfSandwiches_);
        sm.NumSandwiches(numSandwiches_);
        sm.NumPulseLabelStutters(numPulseLabelStutters_);
        sm.PulseDetectionScore(pulseDetectionScore_);
        sm.TraceAutocorr(traceAutoCorrelation_);

        sm.PkmaxA(pkmax_[0]);
        sm.PkmaxC(pkmax_[1]);
        sm.PkmaxG(pkmax_[2]);
        sm.PkmaxT(pkmax_[3]);

        sm.NumPkmidBasesA(numPkmidBases_[0]);
        sm.NumPkmidBasesC(numPkmidBases_[1]);
        sm.NumPkmidBasesG(numPkmidBases_[2]);
        sm.NumPkmidBasesT(numPkmidBases_[3]);

        sm.BpzvarA(bpzvar_[0]);
        sm.BpzvarC(bpzvar_[1]);
        sm.BpzvarG(bpzvar_[2]);
        sm.BpzvarT(bpzvar_[3]);

        sm.PkzvarA(pkzvar_[0]);
        sm.PkzvarC(pkzvar_[1]);
        sm.PkzvarG(pkzvar_[2]);
        sm.PkzvarT(pkzvar_[3]);
    }

    uint8_t ActivityLabel() const { return productionMetrics_.ActivityLabel(); }

    bool HasData() const { return productionMetrics_.HasData(); }

    const ProductionMetricsT& ProdMetrics() const
    { return productionMetrics_; }

    UIntType NumHalfSandwiches() const
    { return numHalfSandwiches_; }

    UIntType NumSandwiches() const
    { return numSandwiches_; }

    UIntType NumPulseLabelStutters() const
    { return numPulseLabelStutters_; }

    FltType PulseDetectionScore() const
    { return pulseDetectionScore_; }

    FltType TraceAutoCorrelation() const
    { return traceAutoCorrelation_; }

    const FltArray& Pkmax() const
    { return pkmax_; }

    const UIntArray& NumPkmidBases() const
    { return numPkmidBases_; }

    const FltArray& Bpzvar() const
    { return bpzvar_; }

    const FltArray& Pkzvar() const
    { return pkzvar_; }

private:

    ProductionMetricsT productionMetrics_;

    UIntType    numHalfSandwiches_ = 0;
    UIntType    numSandwiches_ = 0;
    UIntType    numPulseLabelStutters_ = 0;
    FltType     pulseDetectionScore_ = FltType(0);
    FltType     traceAutoCorrelation_ = FltType(0);
    FltArray    pkmax_ = {FltType(0), FltType(0), FltType(0), FltType(0)};
    UIntArray   numPkmidBases_ = {0, 0, 0, 0};
    FltArray    bpzvar_ = {FltType(0), FltType(0), FltType(0), FltType(0)};
    FltArray    pkzvar_ = {FltType(0), FltType(0), FltType(0), FltType(0)};
};

struct ProductionMetricsGroup
{
    using MetricT = ProductionMetrics<uint16_t,half_float::half>;
    using MetricAggregatedT = ProductionMetrics<uint32_t,float>;
};

struct CompleteMetricsGroup
{
    using MetricT = CompleteMetrics<uint16_t,half_float::half>;
    using MetricAggregatedT = CompleteMetrics<uint32_t,float>;
};


} // namespace PacBio::Mongo::Data


#endif // PACBIO_MONGO_DATA_METRICS_H