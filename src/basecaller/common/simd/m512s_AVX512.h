#ifndef mongo_common_simd_m512s_AVX512_H_
#define mongo_common_simd_m512s_AVX512_H_

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
/// \file   m512s_MIC.h
/// \brief  SIMD 512-bit short vector (32 packed 16-bit short int values) using AVX-512.

#if !defined(PB_CORE_AVX512)
#error This type requires AVX-512 intrinsics.
#endif

#include <cassert>
#include <immintrin.h>
#include <ostream>
#include <pacbio/PBException.h>

#include "m512f_AVX512.h"
#include "m512i_AVX512.h"
#include "m512b_AVX512.h"
#include "xcompile.h"

namespace PacBio {
namespace Simd {

// SIMD 512-bit short vector (32 packed 16-bit short int values).
CLASS_ALIGNAS(64) m512s
{
public:     // Types
    typedef m512s type;

public:     // Static constants
    /// The number of short ints represented by one instance.
    static constexpr size_t size() { return sizeof(m512s) / sizeof(short); }

    //private:    // Implementation
 public: // TODO access
    using ImplType = __m512i;
    ImplType v;

public:     // Structors
    // Purposefully do not initialize v.
    m512s() = default;

    // Replicate scalar x across v
    m512s(short x)  : v(_mm512_set1_epi16(x)) {}

    explicit m512s(int x) : m512s(static_cast<short>(x)) {}

    // Load x from pointer px. px must be aligned to 16 bytes.
    m512s(const short *px) : v(_mm512_load_si512(reinterpret_cast<const __m512i*>(px))) {}

    // Copy constructor
    m512s(const m512s& x) = default;

    // Construct from native vector type
    m512s(ImplType v_) : v(v_) {}

    m512s(const m512i& low, const m512i& high)
    {
        __m512i l = _mm512_castsi256_si512(_mm512_cvtepi32_epi16(low.data()));
        __m256i h = _mm512_cvtepi32_epi16(high.data());
        v = _mm512_inserti64x4(l, h, 1);
    }

    m512s(const m512ui& low, const m512ui& high)
        : m512s(m512i(low), m512i(high))
    {}


    m512s(const m512f& low, const m512f& high)
        : m512s(m512i(low), m512i(high))
    {}

    operator std::pair<m512i, m512i>() const
    {
        return {LowInts(*this), HighInts(*this)};
    }

    explicit operator std::pair<m512ui, m512ui>() const
    {
        return {LowUInts(*this), HighUInts(*this)};
    }

    operator std::pair<m512f, m512f>() const
    {
        return {LowFloats(*this), HighFloats(*this)};
    }

public:     // Assignment
    m512s& operator=(const m512s& x) = default;

    // Assignment from scalar value
    m512s& operator=(short x)
    {
        v = _mm512_set1_epi16(x);
        return *this;
    }

    m512s& operator+=(const m512s& x)
    {
        v = _mm512_add_epi16(v, x.v);
        return *this;
    }
    m512s& operator-=(const m512s& x)
    {
        v = _mm512_sub_epi16(v, x.v);
        return *this;
    }
    m512s& operator*=(const m512s& x)
    {
        v = _mm512_mullo_epi16(v, x.v);
        return *this;
    }
    m512s& operator/=(const m512s& x)
    {
        v = _mm512_div_epi16(v, x.v);
        return *this;
    }

    m512s operator - () const { return m512s(_mm512_sub_epi16(_mm512_setzero_si512(), v)); }

    friend m512s operator+(const m512s& l, const m512s& r)
    {
        return m512s(_mm512_add_epi16(l.v, r.v));
    }
    friend m512s operator-(const m512s& l, const m512s& r)
    {
        return m512s(_mm512_sub_epi16(l.v, r.v));
    }
    friend m512s operator*(const m512s& l, const m512s& r)
    {
        return m512s(_mm512_mullo_epi16(l.v, r.v));
    }
    friend m512s operator/(const m512s& l, const m512s& r)
    {
        return m512s(_mm512_div_epi16(l.v, r.v));
    }

    // Turns an __mask32 mask returned from the intrinsic into a pair
    // of __mmask16 types as our m512b type requires
    static std::pair<m512b,m512b> CompHelper(__mmask32 mask)
    {
        auto low = static_cast<__mmask16>(mask & 0xFFFF);
        auto high = static_cast<__mmask16>((mask & 0xFFFF0000) >> 16);
        return {m512b{low}, m512b{high}};
    }

    friend std::pair<m512b,m512b> operator!=(const m512s& a, const m512s& b)
    {
        return CompHelper(_mm512_cmpneq_epi16_mask(a.v, b.v));
    }
    friend std::pair<m512b,m512b> operator==(const m512s& a, const m512s& b)
    {
        return CompHelper(_mm512_cmpeq_epi16_mask(a.v, b.v));
    }
    friend std::pair<m512b,m512b> operator>(const m512s& a, const m512s& b)
    {
        return CompHelper(_mm512_cmpgt_epi16_mask(a.v, b.v));
    }
    friend std::pair<m512b,m512b> operator<(const m512s& a, const m512s& b)
    {
        return CompHelper(_mm512_cmplt_epi16_mask(a.v, b.v));
    }
    friend std::pair<m512b,m512b> operator<=(const m512s& a, const m512s& b)
    {
        return CompHelper(_mm512_cmple_epi16_mask(a.v, b.v));
    }
    friend std::pair<m512b,m512b> operator>=(const m512s& a, const m512s& b)
    {
        return CompHelper(_mm512_cmpge_epi16_mask(a.v, b.v));
    }

public:     // Conversion methods
    // Converts index 0-15 into an m512f
    friend m512f LowFloats(const m512s& in)
    {
        auto tmp = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(in.v, 0));
        return m512f(_mm512_cvt_roundepi32_ps(tmp, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    }

    // Converts index 0-15 into an m512i
    friend m512i LowInts(const m512s& in)
    {
        return m512i(_mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(in.v, 0)));
    }

    // Converts index 0-15 into an m512ui
    friend m512ui LowUInts(const m512s& in)
    {
        return m512ui(LowInts(in));
    }

    // Converts index 16-31 into an m512f
    friend m512f HighFloats(const m512s& in)
    {
        auto tmp = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(in.v, 1));
        return m512f(_mm512_cvt_roundepi32_ps(tmp, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    }

    // Converts index 16-31 into an m512i
    friend m512i HighInts(const m512s& in)
    {
        return m512i(_mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(in.v, 1)));
    }

    // Converts index 16-31 into an m512ui
    friend m512ui HighUInts(const m512s& in)
    {
        return m512ui(HighInts(in));
    }

public:     // Non-member (friend) functions

    friend m512s min(const m512s& a, const m512s&b)
    {
        return m512s(_mm512_min_epi16(a.v, b.v));
    }

    friend m512s max(const m512s& a, const m512s&b)
    {
        return m512s(_mm512_max_epi16(a.v, b.v));
    }

    friend m512s Blend(const m512b& bLow, const m512b& bHigh, const m512s& l, const m512s& r)
    {
        __mmask32 mask = (bHigh.data() << 16) | bLow.data();
        return m512s(_mm512_mask_blend_epi16(mask, r.v, l.v));
    }
};

}}      // namespace PacBio::Simd

#endif  // mongo_common_simd_m512s_AVX512_H_
