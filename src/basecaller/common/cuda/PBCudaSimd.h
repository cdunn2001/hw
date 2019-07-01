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

#ifndef PACBIO_CUDA_SIMD_H
#define PACBIO_CUDA_SIMD_H

#include <cuda_fp16.h>
#include <common/cuda/CudaFunctionDecorators.h>

#include <vector_types.h>

namespace PacBio {
namespace Cuda {

// Aliasing for now, mostly so there is an easy hook if we want to either
// have a proper wrapper type like PBHalf2, or need to stub in third
// part half precision float for the host.
using PBHalf = half;

class PBHalf2
{
public:
    PBHalf2() = default;

    CUDA_ENABLED PBHalf2(float f) : data_{__float2half2_rn(f)} {}
    CUDA_ENABLED PBHalf2(float f1, float f2) : data_{__floats2half2_rn(f1, f2)} {}
    CUDA_ENABLED PBHalf2(short2 f) : PBHalf2(static_cast<float>(f.x), static_cast<float>(f.y)) {}
    CUDA_ENABLED PBHalf2(half f)  : data_{f,f} {}
    CUDA_ENABLED PBHalf2(half2 f) : data_{f} {}

    CUDA_ENABLED PBHalf2& operator=(PBHalf2 o) { data_ = o.data_; return *this;}
    CUDA_ENABLED void SetX(float f) {data_.x = __float2half(f); }
    CUDA_ENABLED void SetY(float f) {data_.y = __float2half(f); }
    CUDA_ENABLED float X() const { return __half2float(data_.x); }
    CUDA_ENABLED float Y() const { return __half2float(data_.y); }

    half2 CUDA_ENABLED data() const { return data_; }
private:
    half2 data_;
};

}}

#endif // PACBIO_CUDA_SIMD_H
