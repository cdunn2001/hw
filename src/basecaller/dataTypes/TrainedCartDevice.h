#ifndef mongo_dataTypes_TrainedCartDevice_H_
#define mongo_dataTypes_TrainedCartDevice_H_
// Copyright (c) 2018-2019, Pacific Biosciences of California, Inc.
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
// File Description: A POD struct that is populated on the host, then copied to
// device __constant)) memory

#include "TrainedCartParams.h"

namespace PacBio {
namespace Mongo {
namespace Data {

struct alignas(64) TrainedCartDevice
{
    TrainedCartDevice() = default;

    // Named constructor for the host that actually does the populating:
    static TrainedCartDevice PopulatedModel()
    {
        TrainedCartDevice ret;
        ret.maxAcceptableHalfsandwichRate = ActivityLabeler::TrainedCart::maxAcceptableHalfsandwichRate;
        ret.hswCurve = ActivityLabeler::TrainedCart::hswCurve;
        ret.childrenLeft = ActivityLabeler::TrainedCart::childrenLeft;
        ret.childrenRight = ActivityLabeler::TrainedCart::childrenRight;
        ret.feature = ActivityLabeler::TrainedCart::feature;
        ret.threshold = ActivityLabeler::TrainedCart::threshold;
        ret.value = ActivityLabeler::TrainedCart::value;
        return ret;
    };

    float maxAcceptableHalfsandwichRate;
    Cuda::Utility::CudaArray<float, ActivityLabeler::TrainedCart::hswCurve.size()> hswCurve;
    Cuda::Utility::CudaArray<int16_t, ActivityLabeler::TrainedCart::childrenLeft.size()> childrenLeft;
    Cuda::Utility::CudaArray<int16_t, ActivityLabeler::TrainedCart::childrenRight.size()> childrenRight;
    Cuda::Utility::CudaArray<int8_t, ActivityLabeler::TrainedCart::feature.size()> feature;
    Cuda::Utility::CudaArray<float, ActivityLabeler::TrainedCart::threshold.size()> threshold;
    Cuda::Utility::CudaArray<int8_t, ActivityLabeler::TrainedCart::value.size()> value;
};

}}} // PacBio::Mongo::Data
#endif // mongo_dataTypes_TrainedCartDevice_H_
