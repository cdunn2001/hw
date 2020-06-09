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

inline __device__ PBHalf2 Blend(PBBool2 cond, PBHalf2 l, PBHalf2 r)
{
    half zero = __float2half(0.0f);
    half low =  (__low2half(cond.data())  == zero) ? __low2half(r.data())  : __low2half(l.data());
    half high = (__high2half(cond.data()) == zero) ? __high2half(r.data()) : __high2half(l.data());
    return PBHalf2(__halves2half2(low, high));
}

inline __device__ PBFloat2 Blend(PBShort2 cond, PBFloat2 l, PBFloat2 r)
{
    float low  = cond.X() ? l.X() : r.X();
    float high = cond.Y() ? l.Y() : r.Y();
    return PBFloat2(low, high);
}

inline __device__ PBHalf2 Blend(PBShort2 cond, PBHalf2 l, PBHalf2 r)
{
    half zero = __float2half(0.0f);
    half low  = cond.X() ? __low2half(l.data())  : __low2half(r.data());
    half high = cond.Y() ? __high2half(l.data()) : __high2half(r.data());
    return PBHalf2(__halves2half2(low, high));
}

inline __device__ PBShort2 Blend(PBShort2 cond, PBShort2 l, PBShort2 r)
{
    return PBShort2::FromRaw( (cond.data() & l.data()) | ((~cond.data()) & r.data()));
}

inline __device__ PBShort2 Blend(PBBool2 cond, PBShort2 l, PBShort2 r)
{
    half zero = __float2half(0.0f);
    uint32_t mask = 0;
    if (cond.X()) mask |= 0x0000FFFF;
    if (cond.Y()) mask |= 0xFFFF0000;
    return Blend(PBShort2::FromRaw(mask), l, r);
}

inline __device__ PBBool2 operator ||(PBBool2 first, PBBool2 second)
{
    half zero = __float2half(0.0f);
    bool low  = (__low2half(first.data())  != zero) || (__low2half(second.data())  != zero);
    bool high = (__high2half(first.data()) != zero) || (__high2half(second.data()) != zero);
    return PBBool2(low, high);
}

inline __device__ PBBool2 operator &&(PBBool2 first, PBBool2 second)
{
    half zero = __float2half(0.0f);
    bool low  = (__low2half(first.data())  != zero) && (__low2half(second.data())  != zero);
    bool high = (__high2half(first.data()) != zero) && (__high2half(second.data()) != zero);
    return PBBool2(low, high);
}

// Simd logical "or" and bitwise "or" are the same thing on boolean PBShort2 values,
// as cuda represents true as 0xFFFF an false as 0x0;
inline __device__ PBShort2 operator ||(PBShort2 first, PBShort2 second)
{
    return PBShort2::FromRaw(first.data() | second.data());
}

inline __device__ PBShort2 operator &&(PBShort2 first, PBShort2 second)
{
    return PBShort2::FromRaw(first.data() & second.data());
}


// TODO we need to come to a concensus about things like && vs &.  These overloads are to unify
// with host side code that does it this way
inline __device__ PBBool2 operator |(PBBool2 first, PBBool2 second)
{ return first || second; }
inline __device__ PBBool2 operator &(PBBool2 first, PBBool2 second)
{ return first && second; }

inline __device__ PBShort2 operator |(PBShort2 first, PBShort2 second)
{ return first || second; }
inline __device__ PBShort2 operator &(PBShort2 first, PBShort2 second)
{ return first && second; }



inline __device__ PBBool2 operator!(PBBool2 b)
{
    half zero = __float2half(0.0f);
    bool low  = (__low2half(b.data()) == zero);
    bool high = (__high2half(b.data()) == zero);
    return PBBool2(low, high);
}
inline __device__ PBShort2 operator!(PBShort2 b)
{
    return PBShort2::FromRaw(~b.data());
}

inline __device__ PBHalf2 operator + (PBHalf2 l, PBHalf2 r) { return PBHalf2(l.data() + r.data()); }
inline __device__ PBHalf2 operator - (PBHalf2 l, PBHalf2 r) { return PBHalf2(l.data() - r.data()); }
inline __device__ PBHalf2 operator * (PBHalf2 l, PBHalf2 r) { return PBHalf2(l.data() * r.data()); }
inline __device__ PBHalf2 operator / (PBHalf2 l, PBHalf2 r) { return PBHalf2(l.data() / r.data()); }

inline __device__ PBHalf2& operator +=(PBHalf2& l, const PBHalf2 r) { l = l + r; return l;}
inline __device__ PBHalf2& operator -=(PBHalf2& l, const PBHalf2 r) { l = l - r; return l;}
inline __device__ PBHalf2& operator *=(PBHalf2& l, const PBHalf2 r) { l = l * r; return l;}
inline __device__ PBHalf2& operator /=(PBHalf2& l, const PBHalf2 r) { l = l / r; return l;}

inline __device__ PBBool2 operator < (PBHalf2 l, PBHalf2 r) { return PBBool2(__hltu2(l.data(), r.data())); }
inline __device__ PBBool2 operator <=(PBHalf2 l, PBHalf2 r) { return PBBool2(__hleu2(l.data(), r.data())); }
inline __device__ PBBool2 operator > (PBHalf2 l, PBHalf2 r) { return PBBool2(__hgtu2(l.data(), r.data())); }
inline __device__ PBBool2 operator >=(PBHalf2 l, PBHalf2 r) { return PBBool2(__hgeu2(l.data(), r.data())); }
inline __device__ PBBool2 operator ==(PBHalf2 l, PBHalf2 r) { return PBBool2(l.data() == r.data()); }

inline __device__ PBHalf2 pow2(PBHalf2 h) { return PBHalf2(h.data() * h.data()); }
inline __device__ PBHalf2 log(PBHalf2 h) { return PBHalf2(h2log(h.data())); }
inline __device__ PBHalf2 sqrt(PBHalf2 h) { return PBHalf2(h2sqrt(h.data())); }
inline __device__ PBHalf2 exp(PBHalf2 h) { return PBHalf2(h2exp(h.data())); }

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

inline __device__ PBShort2 ToShort(PBHalf2 h)
{
    return PBShort2(__half2short_rn(h.data().x),
                    __half2short_rn(h.data().y));
}

// Cuda integral intrinsics do not supply multiplication and division.  If they really
// become necessary then operators can be added to emulate it, but that will be costly.
// If that need arises we should consider maybe having separate PBShort2 and SimdPBShort2
//
// Be aware that CUDA integral intrinsic arithmetic saturates instead of over/underflow
inline __device__ PBShort2 operator + (PBShort2 l, PBShort2 r) { return PBShort2::FromRaw(__vaddss2(l.data(), r.data())); }
inline __device__ PBShort2 operator - (PBShort2 l, PBShort2 r) { return PBShort2::FromRaw(__vsubss2(l.data(), r.data())); }

inline __device__ PBShort2& operator +=(PBShort2& l, const PBShort2 r) { l = l + r; return l;}
inline __device__ PBShort2& operator -=(PBShort2& l, const PBShort2 r) { l = l - r; return l;}

// For each 16 bit slot, 0xFFFF represents true and 0x0000 represents false;
inline __device__ PBShort2 operator <  (PBShort2 l, PBShort2 r) { return PBShort2::FromRaw(__vcmplts2(l.data(), r.data())); }
inline __device__ PBShort2 operator <= (PBShort2 l, PBShort2 r) { return PBShort2::FromRaw(__vcmples2(l.data(), r.data())); }
inline __device__ PBShort2 operator >  (PBShort2 l, PBShort2 r) { return PBShort2::FromRaw(__vcmpgts2(l.data(), r.data())); }
inline __device__ PBShort2 operator >= (PBShort2 l, PBShort2 r) { return PBShort2::FromRaw(__vcmpges2(l.data(), r.data())); }
inline __device__ PBShort2 operator == (PBShort2 l, PBShort2 r) { return PBShort2::FromRaw(__vcmpeq2(l.data(), r.data())); }
inline __device__ PBShort2 operator != (PBShort2 l, PBShort2 r) { return PBShort2::FromRaw(__vcmpne2(l.data(), r.data())); }

inline __device__ PBShort2 min(PBShort2 l, PBShort2 r)
{
    auto cond = l < r;
    return Blend(cond, l, r);
}
inline __device__ PBShort2 max(PBShort2 l, PBShort2 r)
{
    auto cond = l > r;
    return Blend(cond, l, r);
}

}}

#endif //PACBIO_CUDA_SIMD_CUH
