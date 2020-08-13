#ifndef mongo_common_simd_m512f_AVX512_H_
#define mongo_common_simd_m512f_AVX512_H_

// Copyright (c) 2015,2020 Pacific Biosciences of California, Inc.
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
/// \file   m512f_AVX512.h
/// \brief  SIMD 512-bit float vector (16 packed 32-bit float values) for AVX-512.

#if !defined(PB_CORE_AVX512)
#error This type requires AVX-512 intrinsics.
#endif

#include <cassert>
#include <cmath>
#include <immintrin.h>
#include <ostream>

#include "xcompile.h"

#include "m512b_AVX512.h"
#include "m512i_AVX512.h"
#include "m512ui_AVX512.h"

#define CMP_MASK _mm512_cmp_ps_mask

namespace PacBio {
namespace Simd {

// SIMD 512-bit float vector (16 packed 32-bit float values).
CLASS_ALIGNAS(64) m512f
{
public:     // Types
    typedef m512f type;

public:     // Static constants
    /// The number of floats represented by one instance.
    static constexpr size_t size() { return sizeof(m512f) / sizeof(float); }

private:    // Implementation
    using ImplType = __m512;

    ImplType v;

public:     // Structors
    // Purposefully do not initialize v.
    m512f() = default;

    // Replicate scalar x across v.
    m512f(float x) : v(_mm512_set1_ps(x)) {}

    m512f(int32_t x) : m512f(static_cast<float>(x)) {}
    m512f(uint32_t x) : m512f(static_cast<float>(x)) {}

    // Load x from pointer px. px must be aligned to 16 bytes.
    m512f(const float *px) : v(_mm512_load_ps(px)) {}

    // Copy constructor
    m512f(const m512f& x) = default;

    // Construct from native vector type
    m512f(ImplType v_) : v(v_) {}

    // Construct from 16 floats
    m512f(float f0, float f1, float f2, float f3, float f4, float f5, float f6,
          float f7, float f8, float f9, float f10, float f11, float f12,
          float f13, float f14, float f15)
        : v(_mm512_setr_ps(f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11,
                           f12, f13, f14, f15)) {}

    m512f(const m512i& i)
        : v(_mm512_cvt_roundepi32_ps(i.data(), _MM_FROUND_NO_EXC))
    {}

    explicit operator m512i() const
    {
        constexpr auto mode = _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;
        return m512i(_mm512_cvt_roundps_epi32(v, mode));
    }

    m512f(const m512ui& i)
        : v(_mm512_cvt_roundepu32_ps(i.data(), _MM_FROUND_NO_EXC))
    {}

    explicit operator m512ui() const
    {
        constexpr auto mode = _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;
        return m512ui(_mm512_cvt_roundps_epu32(v, mode));
    }


public:     // Assignment
    m512f& operator=(const m512f& x) = default;

    // Assignment from scalar value
    m512f& operator = (float fv) { v = _mm512_set1_ps(fv); return *this; }

public:    // Compound assignment operators
    m512f& operator += (const m512f& x) { v = _mm512_add_ps(this->v, x.v); return *this; }
    m512f& operator -= (const m512f& x) { v = _mm512_sub_ps(this->v, x.v); return *this; }
    m512f& operator *= (const m512f& x) { v = _mm512_mul_ps(this->v, x.v); return *this; }
    m512f& operator /= (const m512f& x) { v = _mm512_div_ps(this->v, x.v); return *this; }

    m512f operator - () const { return m512f(_mm512_sub_ps(_mm512_setzero_ps(), v)); }

public:     // Scalar access

    // Return a scalar value
    float operator[](unsigned int i) const
    {
        assert(static_cast<size_t>(i) < this->size());
        const auto mask = static_cast<uint16_t>(1 << i);
        return _mm512_mask_reduce_add_ps(mask, v);
    }

    const ImplType& data() const
    { return v; }

public:     // Non-member (friend) functions
    friend m512f operator +  (const m512f& l, const m512f& r) { return m512f(_mm512_add_ps(l.v, r.v)); }
    friend m512f operator -  (const m512f& l, const m512f& r) { return m512f(_mm512_sub_ps(l.v, r.v)); }
    friend m512f operator *  (const m512f& l, const m512f& r) { return m512f(_mm512_mul_ps(l.v, r.v)); }
    friend m512f operator /  (const m512f& l, const m512f& r) { return m512f(_mm512_div_ps(l.v, r.v)); }

    friend m512b operator == (const m512f& l, const m512f& r) { return m512b(CMP_MASK(l.v, r.v, _CMP_EQ_OQ)); }
    friend m512b operator != (const m512f& l, const m512f& r) { return m512b(CMP_MASK(l.v, r.v, _CMP_NEQ_OQ)); }
    friend m512b operator >  (const m512f& l, const m512f& r) { return m512b(CMP_MASK(l.v, r.v, _CMP_GT_OS)); }
    friend m512b operator >= (const m512f& l, const m512f& r) { return m512b(CMP_MASK(l.v, r.v, _CMP_GE_OS)); }
    friend m512b operator <  (const m512f& l, const m512f& r) { return m512b(CMP_MASK(l.v, r.v, _CMP_LT_OS)); }
    friend m512b operator <= (const m512f& l, const m512f& r) { return m512b(CMP_MASK(l.v, r.v, _CMP_LE_OS)); }

    friend float reduceMax(const m512f& a) { return _mm512_reduce_max_ps(a.v); }
    friend float reduceMin(const m512f& a) { return _mm512_reduce_min_ps(a.v); }

    friend m512f min      (const m512f& l, const m512f& r) { return m512f(_mm512_min_ps(l.v, r.v)); }
    friend m512f max      (const m512f& l, const m512f& r) { return m512f(_mm512_max_ps(l.v, r.v)); }
    friend m512f erfDiff  (const m512f& l, const m512f& r) { return m512f(erf(l) - erf(r)); }
    friend m512f atan2    (const m512f& l, const m512f& r) { return m512f(_mm512_atan2_ps(l.v, r.v)); }

    friend m512b isnan    (const m512f& x) { return m512b(_mm512_knot(_mm512_cmpord_ps_mask(x.v, x.v))); }
    friend m512b isnotnan (const m512f& x) { return m512b(_mm512_cmpord_ps_mask(x.v, x.v)); }
    // 0x01 checks for quiet nan
    // 0x08 checks for positive infinity
    // 0x10 checks for negative infinity
    // 0x80 checks for signaling nan
    // Note, isfinite is implemented like std::isfinite, meaning it also returns
    // false for NaNs.  To match previous implementations, this also means
    // that isinf will return true for nans...
    friend m512b isfinite (const m512f& x) { return m512b(_mm512_knot(_mm512_fpclass_ps_mask(x.v, 0x01 | 0x08 | 0x10 | 0x80))); }
    friend m512b isinf    (const m512f& x) { return m512b(_mm512_fpclass_ps_mask(x.v, 0x08 | 0x10 | 0x01 | 0x80)); }
    // 0x20 checks for denormal numbers
    // 0x02 checks for positive 0
    // 0x04 checks for negative 0
    friend m512b isnormal (const m512f& x) { return m512b(_mm512_knot(_mm512_fpclass_ps_mask(x.v, 0x01 | 0x08 | 0x20 | 0x80 | 0x20 | 0x02 | 0x04))); }

    friend m512f sqrt     (const m512f& x) { return m512f(_mm512_sqrt_ps(x.v)); }
    friend m512f cos      (const m512f& x) { return m512f(_mm512_cos_ps(x.v)); }
    friend m512f sin      (const m512f& x) { return m512f(_mm512_sin_ps(x.v)); }
    friend m512f log      (const m512f& x) { return m512f(_mm512_log_ps(x.v)); }
    friend m512f log2     (const m512f& x) { return log(x) / log(m512f(2.0f)); }
    friend m512f exp      (const m512f& x) { return m512f(_mm512_exp_ps(x.v)); }
    friend m512f exp2     (const m512f& x) { return m512f(_mm512_exp2_ps(x.v));}
    friend m512f erf      (const m512f& x) { return m512f(_mm512_erf_ps(x.v)); }

    /// Complementary error function for AVX-512.
    friend m512f erfc(const m512f& x)
    { return m512f(_mm512_erfc_ps(x.v)); }

    friend m512f square   (const m512f& x) { return x * x; }

public: // Masked
    friend m512f Blend(const m512b& mask, const m512f& success, const m512f& failure)
    { return m512f(_mm512_mask_blend_ps(mask.data(), failure.v, success.v)); }

    friend m512f inc (const m512f& a, const m512b& mask)
    { return m512f(_mm512_mask_add_ps(a.v, mask.data(), a.v, m512f(1).v)); }

public: // stream
    friend std::ostream& operator << (std::ostream& stream, const m512f& vec)
    {
        stream << vec[0] << "\t" << vec[1] << "\t" << vec[2] << "\t" << vec[3]
               << "\t" << vec[4] << "\t" << vec[5] << "\t" << vec[6] << "\t"
               << vec[7] << "\t" << vec[8] << "\t" << vec[9] << "\t" << vec[10]
               << "\t" << vec[11] << "\t" << vec[12] << "\t" << vec[13] << "\t"
               << vec[14] << "\t" << vec[15];

        return stream;
    }
};

inline m512i floorCastInt(const m512f& x)
{
    return m512i(_mm512_cvt_roundps_epi32(
            x.data(),
            (_MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC)));
}

inline m512ui floorCastUInt(const m512f& x)
{
    return m512ui(_mm512_cvt_roundps_epu32(
            x.data(),
            (_MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC)));
}

}}      // namespace PacBio::Simd

#endif  // mongo_common_simd_m512f_AVX512_H_
