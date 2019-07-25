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

// TODO we need to come to a concensus about things like && vs &.  These overloads are to unify
// with host side code that does it this way
inline __device__ PBBool2 operator |(PBBool2 first, PBBool2 second)
{ return first || second; }
inline __device__ PBBool2 operator &(PBBool2 first, PBBool2 second)
{ return first && second; }


inline __device__ PBBool2 operator!(PBBool2 b)
{
    half zero = __float2half(0.0f);
    bool low  = (__low2half(b.data()) == zero);
    bool high = (__high2half(b.data()) == zero);
    return PBBool2(low, high);
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

// Note: Do not enable these functions unless necessary.

// Rational: Right now PBShort2 is based on a short2 implementation.  A downside of this
//           is that working with short2 requires twice as many 16 bit load/store/operations
//           than if things were bit packed into a single 32 bit type.  The gpu has
//           a word access size of 32 which makes this less than ideal, as we'll
//           still end up fetching 32 bits even if we really only wanted 16.  Memory
//           loads can be a significan bottleneck, and having 2x loads are potentially
//           problematic, even if we can expect alternating loads to be cached.
//
//           Cuda does provide some SIMD intrinsics, but they seem better suited towards
//           conditonals than arithmetic, as addition/subtraction seem limited to saturation
//           rather than rollover, and mul/div are simply lacking.  Our current usages do actually
//           function perfectly well without arithmetic, but I've not yet had a chance to confirm
//           the 16 bit loads are signficantly impacting performance and I don't want to limit the
//           API of this class until I know it serves a necessary purpose.
//
//           Having these functions here but disabled is just a way to walk the middle road.  We'll
//           keep the short2 implementation until I can confirm there is benefit from a SIMD
//           implementation, and until that time the use of the arithmetic operations is discouraged
//           so that a switch to SIMD later will be less painful.  If you find yourself having real
//           need to enable these functions then do so, but that also increases our need to evaluate
//           a SIMD implementation sooner than later.

inline __device__ PBShort2 operator + (PBShort2 l, PBShort2 r) { return PBShort2(l.X() + r.X(), l.Y() + r.Y()); }
inline __device__ PBShort2 operator - (PBShort2 l, PBShort2 r) { return PBShort2(l.X() - r.X(), l.Y() - r.Y()); }
//inline __device__ PBShort2 operator * (PBShort2 l, PBShort2 r) { return PBShort2(l.X() * r.X(), l.Y() * r.Y()); }
//inline __device__ PBShort2 operator / (PBShort2 l, PBShort2 r) { return PBShort2(l.X() / r.X(), l.Y() / r.Y()); }

inline __device__ PBShort2& operator +=(PBShort2& l, const PBShort2 r) { l = l + r; return l;}
inline __device__ PBShort2& operator -=(PBShort2& l, const PBShort2 r) { l = l - r; return l;}
//inline __device__ PBShort2& operator *=(PBShort2& l, const PBShort2 r) { l = l * r; return l;}
//inline __device__ PBShort2& operator /=(PBShort2& l, const PBShort2 r) { l = l / r; return l;}

inline __device__ PBBool2 operator <  (PBShort2 l, PBShort2 r) { return PBBool2(l.X() <  r.X(), l.Y() <  r.Y()); }
inline __device__ PBBool2 operator <= (PBShort2 l, PBShort2 r) { return PBBool2(l.X() <= r.X(), l.Y() <= r.Y()); }
inline __device__ PBBool2 operator >  (PBShort2 l, PBShort2 r) { return PBBool2(l.X() >  r.X(), l.Y() >  r.Y()); }
inline __device__ PBBool2 operator >= (PBShort2 l, PBShort2 r) { return PBBool2(l.X() >= r.X(), l.Y() >= r.Y()); }
inline __device__ PBBool2 operator == (PBShort2 l, PBShort2 r) { return PBBool2(l.X() == r.X(), l.Y() == r.Y()); }
inline __device__ PBBool2 operator != (PBShort2 l, PBShort2 r) { return PBBool2(l.X() != r.X(), l.Y() != r.Y()); }

inline __device__ PBShort2 Blend(PBBool2 cond, PBShort2 l, PBShort2 r)
{
    half zero = __float2half(0.0f);
    return PBShort2((__low2half(cond.data())  == zero) ? r.X()  : l.X(),
                    (__high2half(cond.data())  == zero) ? r.Y()  : l.Y());
}

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
