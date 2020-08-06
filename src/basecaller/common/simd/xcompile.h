#ifndef mongo_common_simd_xcompile_H_
#define mongo_common_simd_xcompile_H_

// Copyright (c) 2015, Pacific Biosciences of California, Inc.
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
/// \file   xcompile.h
/// \brief  Bits for compiler cross-compatibility of SIMD-implemented types.

#include <cmath>
#include <cstring>
#include <iostream>
#include <immintrin.h>

#define CLASS_ALIGNAS(bytes) class alignas(bytes)

#if !defined(__INTEL_COMPILER)

#if defined(PB_MIC_COPROCESSOR)
#error Builds for Intel MIC must use the Intel compiler.
#endif

#if defined(__SSE2__)

// Back-fill some key Intel SVML-defined SSE intrinsics
// http://gruntthepeon.free.fr/ssemath/
#include "sse_mathfun.h"

inline __m128 _mm_sin_ps(__m128 x) { return sin_ps(x); }
inline __m128 _mm_cos_ps(__m128 x) { return cos_ps(x); }
inline __m128 _mm_atan2_ps(__m128 y, __m128 x)
{ 
    __m128 r;
    float* py = reinterpret_cast<float*>(&y);
    float* px = reinterpret_cast<float*>(&x);
    float* pr = reinterpret_cast<float*>(&r);

    for (size_t i = 0; i < 4; ++i) pr[i] = std::atan2(py[i], px[i]);
    return r;
}

inline __m128i _mm_div_epu16(__m128i a, __m128i b)
{
    uint16_t pa[8];
    uint16_t pb[8];
    uint16_t pr[8];
    memcpy(pa, &a, sizeof(a));
    memcpy(pb, &b, sizeof(b));

    for (size_t i = 0; i < 8; ++i) pr[i] = pa[i] / pb[i];
    __m128i r;
    memcpy(&r, pr, sizeof(r));
    return r;
}

inline __m128i _mm_div_epi16(__m128i a, __m128i b)
{
    //__m128i r;
    //short* pa = reinterpret_cast<short*>(&a);
    //short* pb = reinterpret_cast<short*>(&b);
    //short* pr = reinterpret_cast<short*>(&r);
    // TODO Should probably port this fix to the other
    // functions, but this is the one crashing on me
    // right now.
    short pa[8];
    short pb[8];
    short pr[8];
    memcpy(pa, &a, sizeof(a));
    memcpy(pb, &b, sizeof(b));

    for (size_t i = 0; i < 8; ++i) pr[i] = pa[i] / pb[i];
    __m128i r;
    memcpy(&r, pr, sizeof(r));
    return r;
}

inline __m128i _mm_div_epu32(__m128i a, __m128i b)
{
    uint32_t pa[4];
    uint32_t pb[4];
    uint32_t pr[4];
    memcpy(pa, &a, sizeof(a));
    memcpy(pb, &b, sizeof(b));

    for (size_t i = 0; i < 4; ++i) pr[i] = pa[i] / pb[i];
    __m128i r;
    memcpy(&r, pr, sizeof(r));
    return r;
}

inline __m128i _mm_div_epi32(__m128i a, __m128i b)
{
    int pa[4];
    int pb[4];
    int pr[4];
    memcpy(pa, &a, sizeof(a));
    memcpy(pb, &b, sizeof(b));

    for (size_t i = 0; i < 4; ++i) pr[i] = pa[i] / pb[i];
    __m128i r;
    memcpy(&r, pr, sizeof(r));
    return r;
}

inline __m128 _mm_erf_ps(__m128 a)
{
    __m128 r;
    float* pa = reinterpret_cast<float*>(&a);
    float* pr = reinterpret_cast<float*>(&r);

    for (size_t i = 0; i < 4; ++i) pr[i] = std::erf(pa[i]);
    return r;
}

inline __m128 _mm_erfc_ps(__m128 a)
{
    __m128 r;
    float* pa = reinterpret_cast<float*>(&a);
    float* pr = reinterpret_cast<float*>(&r);

    for (size_t i = 0; i < 4; ++i) pr[i] = std::erfc(pa[i]);
    return r;
}

inline __m128 _mm_exp_ps(__m128 a)
{ return exp_ps(a); }

inline __m128 _mm_log_ps(__m128 a)
{ return log_ps(a); }

#endif  // defined(__SSE2__)

#endif  // !defined(__INTEL_COMPILER)


inline __m128 _mm_isfinite_ps(__m128 a)
{
    __m128 r;
    float* pa = reinterpret_cast<float*>(&a);
    float* pr = reinterpret_cast<float*>(&r);

    for (size_t i = 0; i < 4; ++i) pr[i] = std::isfinite(pa[i]) ? -1.0f : 0;
    return r;
}

inline __m128 _mm_isnormal_ps(__m128 a)
{
    __m128 r;
    float* pa = reinterpret_cast<float*>(&a);
    float* pr = reinterpret_cast<float*>(&r);

    for (size_t i = 0; i < 4; ++i) pr[i] = std::isnormal(pa[i]) ? -1.0f : 0;
    return r;
}


inline __m128 _mm_exp2_ps(__m128 a)
{
    __m128 r;
    float* pa = reinterpret_cast<float*>(&a);
    float* pr = reinterpret_cast<float*>(&r);

    for (size_t i = 0; i < 4; ++i) pr[i] = std::exp2(pa[i]);
    return r;
}

inline __m128 _mm_log2_ps(__m128 a)
{
    __m128 r;
    float* pa = reinterpret_cast<float*>(&a);
    float* pr = reinterpret_cast<float*>(&r);

    for (size_t i = 0; i < 4; ++i) pr[i] = std::log2(pa[i]);
    return r;
}

#endif  // mongo_common_simd_xcompile_H_
