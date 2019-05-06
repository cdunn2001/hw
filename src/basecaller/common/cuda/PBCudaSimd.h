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

class PBHalf2
{
public:
    PBHalf2() = default;

    explicit CUDA_ENABLED PBHalf2(float f) : data_{__float2half2_rn(f)} {}
    explicit CUDA_ENABLED PBHalf2(float f1, float f2) : data_{__floats2half2_rn(f1, f2)} {}
    explicit CUDA_ENABLED PBHalf2(short2 f) : PBHalf2(static_cast<float>(f.x), static_cast<float>(f.y)) {}
    explicit CUDA_ENABLED PBHalf2(half f)  : data_{f,f} {}
    explicit CUDA_ENABLED PBHalf2(half2 f) : data_{f} {}

    CUDA_ENABLED PBHalf2& operator=(PBHalf2 o) { data_ = o.data_; return *this;}

    half2 CUDA_ENABLED data() const { return data_; }
private:
    half2 data_;
};

}}

#endif // PACBIO_CUDA_SIMD_H
