
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
//  Defines members of class DetectionModelHost.

#include "DetectionModelHost.h"

#include <common/LaneArray.h>
#include <common/cuda/PBCudaSimd.h>
#include <common/simd/SimdConvTraits.h>

#include "DmeEmHost.h"

namespace PacBio::Mongo::Data {

// Static configuration parameters
template <typename VF>
uint32_t DetectionModelHost<VF>::updateMethod_ = 0;

template <typename VF>
template <typename VF2>
DetectionModelHost<VF>::DetectionModelHost(const LaneDetectionModel<VF2>& ldm,
                                           const FrameIntervalType& fi)
    : baselineMode_ (ldm.BaselineMode())
{
    static_assert(numAnalogs == LaneDetectionModel<VF2>::numAnalogs, "Mismatch in number of analogs.");
    detectionModes_.reserve(numAnalogs);
    for (unsigned int a = 0; a < numAnalogs; ++a)
    {
        detectionModes_.emplace_back(ldm.AnalogMode(a));
    }

    confid_ = ldm.Confidence();
    frameInterval_ = fi;
}

template <typename VF>
void DetectionModelHost<VF>::Configure(const Data::BasecallerDmeConfig &dmeConfig)
{
    updateMethod_ = dmeConfig.ModelUpdateMethod;
}

template <typename VF>
void DetectionModelHost<VF>::Update0(const DetectionModelHost& other, VF fraction)
{
    const auto a = 1.0f - fraction;
    const auto& b = fraction;

    auto& tbm = (*this).BaselineMode();
    auto& obm = other.BaselineMode();

    // Baseline mode
    VF bm = a * tbm.SignalMean() + b * obm.SignalMean();
    tbm.SignalMean(bm);

    VF bv = a * tbm.SignalCovar() + b * obm.SignalCovar();
    tbm.SignalCovar(bv);

    for (unsigned int i = 0; i < detectionModes_.size(); ++i)
    {
        auto& tdmi = (*this).DetectionModes()[i];
        auto& odmi = other.DetectionModes()[i];

        VF am = a * tdmi.SignalMean() + b * odmi.SignalMean();
        tdmi.SignalMean(am);

        VF av = a * tdmi.SignalCovar() + b * odmi.SignalCovar();
        tdmi.SignalCovar(av);
    }
}

template <typename VF>
void DetectionModelHost<VF>::Update1(const DetectionModelHost& other, VF fraction)
{
    const auto a = 1.0f - fraction;
    const auto& b = fraction;

    auto& tbm = (*this).BaselineMode();
    auto& obm = other.BaselineMode();

    // Baseline mode
    VF bw = a * tbm.Weight() + b * obm.Weight();
    tbm.Weight(bw);

    VF bm = a * tbm.SignalMean() + b * obm.SignalMean();
    tbm.SignalMean(bm);

    VF bv = pow(tbm.SignalCovar(), a) * pow(obm.SignalCovar(), b);
    tbm.SignalCovar(bv);

    // Equally partition remaining weight among four analogs.
    const VF aw = 0.25f * (1.0f - bw);

    for (size_t i = 0; i < detectionModes_.size(); ++i)
    {
        auto& tdmi = (*this).DetectionModes()[i];
        auto& odmi = other.DetectionModes()[i];

        tdmi.Weight(aw);

        auto am = pow(tdmi.SignalMean(), a) * pow(odmi.SignalMean(), b);
        tdmi.SignalMean(am);

        const auto cv = Basecaller::DmeEmHost::Analog(i).excessNoiseCV;
        const VF av = ModelSignalCovar(VF(cv*cv), am, bv);
        tdmi.SignalCovar(av);
    }
}

template <typename VF>
void DetectionModelHost<VF>::Update2(const DetectionModelHost& other, VF fraction)
{
    const auto a = 1.0f - fraction;
    const auto& b = fraction;

    auto& tbm = (*this).BaselineMode();
    auto& obm = other.BaselineMode();

    const auto prevBlCovar = tbm.SignalCovar();

    // Baseline mode
    VF bw = a * tbm.Weight() + b * obm.Weight();
    tbm.Weight(bw);

    VF bm = a * tbm.SignalMean() + b * obm.SignalMean();
    tbm.SignalMean(bm);

    VF bv = pow(tbm.SignalCovar(), a) * pow(obm.SignalCovar(), b);
    tbm.SignalCovar(bv);

    // Equally partition remaining weight among four analogs.
    const VF aw = 0.25f * (1.0f - bw);

    for (size_t i = 0; i < detectionModes_.size(); ++i)
    {
        auto& tdmi = (*this).DetectionModes()[i];
        auto& odmi = other.DetectionModes()[i];

        tdmi.Weight(aw);

       // For simplicity, using weighted sum of xsnCV^2, rather than of xsnCV
        const auto tXsnCVSq = XsnCoeffCVSq(tdmi.SignalMean(), tdmi.SignalCovar(), prevBlCovar);
        const auto oXsnCVSq = XsnCoeffCVSq(odmi.SignalMean(), odmi.SignalCovar(), obm.SignalCovar());
        const auto newXsnCVSq = a * tXsnCVSq + b * oXsnCVSq;

        auto am = pow(tdmi.SignalMean(), a) * pow(odmi.SignalMean(), b);
        tdmi.SignalMean(am);

        const VF av = ModelSignalCovar(newXsnCVSq, am, bv);
        tdmi.SignalCovar(av);
    }

}

template <typename VF>
DetectionModelHost<VF>&
DetectionModelHost<VF>::Update(const DetectionModelHost& other, VF fraction)
{
    assert (all((fraction >= 0.0f) & (fraction <= 1.0f)));

    switch (updateMethod_)
    {
    case 0: Update0(other, fraction); break;
    case 1: Update1(other, fraction); break;
    case 2: Update2(other, fraction); break;
    default: throw PBException("DetectionModel: Bad update method id.");
    }

    SetNonemptyFrameInterval(other.FrameInterval());

    // Do not update confidence.

    return *this;
}


template <typename VF>
DetectionModelHost<VF>&
DetectionModelHost<VF>::Update(const DetectionModelHost& other)
{
    // It might be useful to replace NaNs in either confidence with 0.
    assert (this->FrameInterval() == other.FrameInterval());
    assert (all(this->Confidence() >= 0.0f));
    assert (all(other.Confidence() >= 0.0f));

    const auto confSum = this->Confidence() + other.Confidence();
    const VF fraction = Blend(confSum > 0.0f,
                              other.Confidence() / confSum, FloatVec(0.0f));

    assert (all(fraction >= 0.0f));
    assert (all(fraction <= 1.0f));
    assert (all((fraction > 0) | (confSum == Confidence())));

    Update(other, fraction);
    Confidence(confSum);
    return *this;
}


template <typename VF>
template <typename VF2>
void DetectionModelHost<VF>::ExportTo(LaneDetectionModel<VF2>* ldm) const
{
    assert(ldm);
    baselineMode_.ExportTo(&ldm->BaselineMode());
    for (unsigned int a = 0; a < numAnalogs; ++a)
    {
        detectionModes_[a].ExportTo(&ldm->AnalogMode(a));
    }
    ldm->Confidence() = confid_;

    // Note that LaneDetectionModel does not include frame interval, which is
    // tracked at the pool level.
}

// static
template <typename VF>
VF DetectionModelHost<VF>::ModelSignalCovar(
        const VF& excessNoiseCV2,
        const VF& sigMean,
        const VF& blVar)
{
    VF r {blVar};
    r += sigMean * Basecaller::CoreDMEstimator::shotVarCoeff;
    r += pow2(sigMean) * excessNoiseCV2;
    return r;
}

template <typename VF>
VF DetectionModelHost<VF>::XsnCoeffCVSq(
        const VF& sigMean,
        const VF& sigCovar,
        const VF& blVar)
{
    VF r {sigCovar - blVar};
    r -= sigMean * Basecaller::CoreDMEstimator::shotVarCoeff;
    r /= pow2(sigMean);
    return r;
}

template <typename VF>
template <typename FloatT>
SignalModeHost<VF>::SignalModeHost(const LaneAnalogMode<FloatT, laneSize>& lam)
    : weight_ (lam.weights)
    , mean_ (lam.means)
    , var_ (lam.vars)
{ }

template <typename VF>
template <typename VF2>
void SignalModeHost<VF>::ExportTo(LaneAnalogMode<VF2, laneSize>* lam) const
{
    assert(lam);
    lam->means =  mean_;
    lam->vars = var_;
    lam->weights = weight_;
}


// Explicit instantiation
template class DetectionModelHost<LaneArray<float>>;
template DetectionModelHost<LaneArray<float>>::DetectionModelHost(const LaneDetectionModel<Cuda::PBHalf>& ldm,
                                                                  const FrameIntervalType& fi);
template void DetectionModelHost<LaneArray<float>>::ExportTo(LaneDetectionModel<Cuda::PBHalf>* ldm) const;
template SignalModeHost<LaneArray<float>>::SignalModeHost(const LaneAnalogMode<Cuda::PBHalf, laneSize>& lam);

}   // namespace PacBio::Mongo::Data
