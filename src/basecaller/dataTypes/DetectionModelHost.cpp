
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

using PacBio::Simd::MakeUnion;

namespace PacBio {
namespace Mongo {
namespace Data {

template <typename VF>
template <typename VF2>
DetectionModelHost<VF>::DetectionModelHost(const LaneDetectionModel<VF2>& ldm)
    : baselineMode_ (ldm.BaselineMode(), ldm.BaselineWeight())
    , updated_ (false)  // TODO: Is this right?
{
    static_assert(numAnalogs == ldm.numAnalogs, "Mismatch in number of analogs.");
    auto analogWeight = 0.25f * (1.0f - FloatVec(ldm.BaselineWeight()));
    detectionModes_.reserve(numAnalogs);
    for (unsigned int a = 0; a < numAnalogs; ++a)
    {
        detectionModes_.emplace_back(ldm.AnalogMode(a), analogWeight);
    }

    // TODO: What about confid_ and frameInterval_?
}


template <typename VF>
DetectionModelHost<VF>&
DetectionModelHost<VF>::Update(const DetectionModelHost& other, VF fraction)
{
    assert (all((fraction >= 0.0f) & (fraction <= 1.0f)));

    const auto mask = (fraction > 0.0f);
    updated_ |= mask;
    baselineMode_.Update(other.baselineMode_, fraction);
    for (unsigned int i = 0; i < detectionModes_.size(); ++i)
    {
        detectionModes_[i].Update(other.detectionModes_[i], fraction);
    }

    FrameInterval(other.FrameInterval());

    // Do not update confidence.

    return *this;
}


template <typename VF>
DetectionModelHost<VF>&
DetectionModelHost<VF>::Update(const DetectionModelHost& other)
{
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
    // TODO: What about confid_, updated_, and frameInterval_?
}

template <typename VF>
template <typename FloatT>
SignalModeHost<VF>::SignalModeHost(const LaneAnalogMode<FloatT, laneSize>& lam, const Cuda::Utility::CudaArray<FloatT, laneSize>& weight)
    : weight_ (weight)
    , mean_ (lam.means)
    , var_ (lam.vars)
{ }

template <typename VF>
template <typename FloatT>
SignalModeHost<VF>::SignalModeHost(const LaneAnalogMode<FloatT, laneSize>& lam, const FloatVec& weight)
    : weight_ (weight)
    , mean_ (lam.means)
    , var_ (lam.vars)
{ }

template <typename VF>
template <typename VF2>
void SignalModeHost<VF>::ExportTo(LaneAnalogMode<VF2, laneSize>* lam) const
{
    assert(lam);
    lam->means = mean_;
    lam->vars = var_;
}


// Explicit instantiation
template class DetectionModelHost<LaneArray<float>>;
template DetectionModelHost<LaneArray<float>>::DetectionModelHost(const LaneDetectionModel<Cuda::PBHalf>& ldm);
template void DetectionModelHost<LaneArray<float>>::ExportTo(LaneDetectionModel<Cuda::PBHalf>* ldm) const;
template SignalModeHost<LaneArray<float>>::SignalModeHost(const LaneAnalogMode<Cuda::PBHalf, laneSize>& lam,
                                                          const Cuda::Utility::CudaArray<Cuda::PBHalf, laneSize>& weight);

}}}     // namespace PacBio::Mongo::Data
