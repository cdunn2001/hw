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

#ifndef PACBIO_CUDA_SIMD_CUH
#define PACBIO_CUDA_SIMD_CUH

#include <common/cuda/PBCudaSimd.h>

namespace PacBio {
namespace Cuda {

inline __device__ PBHalf2 Blend(PBHalf2 cond, PBHalf2 l, PBHalf2 r)
{
    half zero = __float2half(0.0f);
    half low =  (__low2half(cond.data())  == zero) ? __low2half(r.data())  : __low2half(l.data());
    half high = (__high2half(cond.data()) == zero) ? __high2half(r.data()) : __high2half(l.data());
    return PBHalf2(__halves2half2(low, high));
}

inline __device__ PBHalf2 operator ||(PBHalf2 first, PBHalf2 second)
{
    half zero = __float2half(0.0f);
    half low  = (__low2half(first.data())  != zero) || (__low2half(second.data())  != zero);
    half high = (__high2half(first.data()) != zero) || (__high2half(second.data()) != zero);
    return PBHalf2(__halves2half2(low, high));
}

inline __device__ PBHalf2 operator + (PBHalf2 l, PBHalf2 r) { return PBHalf2(l.data() + r.data()); }
inline __device__ PBHalf2 operator - (PBHalf2 l, PBHalf2 r) { return PBHalf2(l.data() - r.data()); }
inline __device__ PBHalf2 operator * (PBHalf2 l, PBHalf2 r) { return PBHalf2(l.data() * r.data()); }
inline __device__ PBHalf2 operator / (PBHalf2 l, PBHalf2 r) { return PBHalf2(l.data() / r.data()); }

inline __device__ PBHalf2& operator +=(PBHalf2& l, const PBHalf2 r) { l = l + r; return l;}
inline __device__ PBHalf2& operator -=(PBHalf2& l, const PBHalf2 r) { l = l - r; return l;}
inline __device__ PBHalf2& operator *=(PBHalf2& l, const PBHalf2 r) { l = l * r; return l;}
inline __device__ PBHalf2& operator /=(PBHalf2& l, const PBHalf2 r) { l = l / r; return l;}

inline __device__ PBHalf2 operator < (PBHalf2 l, PBHalf2 r) { return PBHalf2(__hltu2(l.data(), r.data())); }
inline __device__ PBHalf2 operator <=(PBHalf2 l, PBHalf2 r) { return PBHalf2(__hleu2(l.data(), r.data())); }
inline __device__ PBHalf2 operator > (PBHalf2 l, PBHalf2 r) { return PBHalf2(__hgtu2(l.data(), r.data())); }
inline __device__ PBHalf2 operator >=(PBHalf2 l, PBHalf2 r) { return PBHalf2(__hgeu2(l.data(), r.data())); }
inline __device__ PBHalf2 operator ==(PBHalf2 l, PBHalf2 r) { return PBHalf2(l.data() == r.data()); }

inline __device__ PBHalf2 pow2(PBHalf2 h) { return PBHalf2(h.data() * h.data()); }
inline __device__ PBHalf2 log(PBHalf2 h) { return PBHalf2(h2log(h.data())); }
inline __device__ PBHalf2 sqrt(PBHalf2 h) { return PBHalf2(h2sqrt(h.data())); }

inline __device__ PBHalf2 min(PBHalf2 l, PBHalf2 r)
{
    auto cond = l < r;
    return Blend(cond, l, r);
}
inline __device__ PBHalf2 max(PBHalf2 l, PBHalf2 r)
{
    auto cond = l > r;
    return Blend(cond, l, r);
}

inline __device__ short2 Blend(PBHalf2 cond, short2 l, short2 r)
{
    half zero = __float2half(0.0f);
    short2 ret;
    ret.x =  (__low2half(cond.data())  == zero) ? r.x  : l.x;
    ret.y =  (__low2half(cond.data())  == zero) ? r.y  : l.y;
    return ret;
}

}}

#endif //PACBIO_CUDA_SIMD_CUH
