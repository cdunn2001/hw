#ifndef mongo_common_simd_m512s_SSE_H_
#define mongo_common_simd_m512s_SSE_H_

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
/// \file   m512s_SSE.h
/// \brief  SIMD 512-bit short vector (32 packed 16-bit short int values) using SSE.

#if !defined(__SSE2__)
#error This type requires SSE2 intrinsics.
#endif

#include <cassert>
#include <cstring>
#include <immintrin.h>
#include <ostream>

#include "m512f_SSE.h"
#include "m512i_SSE.h"
#include "xcompile.h"

namespace PacBio {
namespace Simd {

// SIMD 512-bit short vector (32 packed 16-bit short int values).
CLASS_ALIGNAS(16) m512s
{
public:     // Types
    typedef m512s type;

public:     // Static constants
    /// The number of floats represented by one instance.
    static constexpr size_t size()
    {
        return sizeof(m512s) / sizeof(short);
    }

private:    // Implementation
    using ImplType = __m128i;
    static const size_t implOffsetElems = sizeof(ImplType) / sizeof(short);

    union
    {
        ImplType simd[4];
        short raw[32];
    } data;

public:     // Structors
    // Purposefully do not initialize v.
    m512s() = default;

    // Replicate scalar x across v.
    m512s(short x)
        : data{{_mm_set1_epi16(x)
              , _mm_set1_epi16(x)
              , _mm_set1_epi16(x)
              , _mm_set1_epi16(x)}}
    {}

    // Load x from pointer px. px must be aligned to 16 bytes.
    m512s(const short *px)
        : data{{_mm_load_si128(reinterpret_cast<const ImplType*>(px))
              , _mm_load_si128(reinterpret_cast<const ImplType*>(px +   implOffsetElems))
              , _mm_load_si128(reinterpret_cast<const ImplType*>(px + 2*implOffsetElems))
              , _mm_load_si128(reinterpret_cast<const ImplType*>(px + 3*implOffsetElems))}}
    {}

    // Copy constructor
    m512s(const m512s& x) = default;

    // Construct from native vector type
    m512s(ImplType v1, ImplType v2, ImplType v3, ImplType v4)
        : data{{v1, v2, v3, v4}}
    {}

    explicit m512s(const m512i& low, const m512i& high)
    {
        // Unless I'm missing something, no good SSE
        // intrinsics for packing 16 bit values.
        // Emulating this here since that won't be the
        // case for avx512
        for (size_t i = 0; i < 16; ++i)
        {
            data.raw[i] = low[i];
            data.raw[i+16] = high[i];
        }

    }

    explicit m512s(const m512f& low, const m512f& high)
        : m512s(m512i(low), m512i(high))
    {}

    operator std::pair<m512i, m512i>() const
    {
        return {LowInts(*this), HighInts(*this)};
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
        data.simd[0] = _mm_set1_epi16(x);
        data.simd[1] = _mm_set1_epi16(x);
        data.simd[2] = _mm_set1_epi16(x);
        data.simd[3] = _mm_set1_epi16(x);
        return *this;
    }

    m512s& operator+=(const m512s& x)
    {
        data.simd[0] = _mm_add_epi16(this->data.simd[0], x.data.simd[0]);
        data.simd[1] = _mm_add_epi16(this->data.simd[1], x.data.simd[1]);
        data.simd[2] = _mm_add_epi16(this->data.simd[2], x.data.simd[2]);
        data.simd[3] = _mm_add_epi16(this->data.simd[3], x.data.simd[3]);
        return *this;
    }

    m512s& operator-=(const m512s& x)
    {
        data.simd[0] = _mm_sub_epi16(this->data.simd[0], x.data.simd[0]);
        data.simd[1] = _mm_sub_epi16(this->data.simd[1], x.data.simd[1]);
        data.simd[2] = _mm_sub_epi16(this->data.simd[2], x.data.simd[2]);
        data.simd[3] = _mm_sub_epi16(this->data.simd[3], x.data.simd[3]);
        return *this;
    }

    m512s& operator*=(const m512s& x)
    {
        data.simd[0] = _mm_mullo_epi16(this->data.simd[0], x.data.simd[0]);
        data.simd[1] = _mm_mullo_epi16(this->data.simd[1], x.data.simd[1]);
        data.simd[2] = _mm_mullo_epi16(this->data.simd[2], x.data.simd[2]);
        data.simd[3] = _mm_mullo_epi16(this->data.simd[3], x.data.simd[3]);
        return *this;
    }

    m512s& operator/=(const m512s& x)
    {
        data.simd[0] = _mm_div_epi16(this->data.simd[0], x.data.simd[0]);
        data.simd[1] = _mm_div_epi16(this->data.simd[1], x.data.simd[1]);
        data.simd[2] = _mm_div_epi16(this->data.simd[2], x.data.simd[2]);
        data.simd[3] = _mm_div_epi16(this->data.simd[3], x.data.simd[3]);
        return *this;
    }

    m512s operator-() const
    {
        const auto zero = _mm_setzero_si128();
        return m512s(_mm_sub_epi16(zero, data.simd[0]),
                     _mm_sub_epi16(zero, data.simd[1]),
                     _mm_sub_epi16(zero, data.simd[2]),
                     _mm_sub_epi16(zero, data.simd[3]));
    }

    // Return a scalar value
    short operator[](unsigned int i) const
    {
        assert(static_cast<size_t>(i) < this->size());
        return data.raw[i];
    }

public:     // Conversion methods

    friend m512s Blend(const m512b& bLow, const m512b& bHigh, const m512s& l, const m512s& r)
    {
        // Again emulating for lack of good SSE intrinsics
        m512s ret;
        for (size_t i = 0; i < 16; ++i)
        {
            ret.data.raw[i] = bLow[i] ? l[i] : r[i];
            ret.data.raw[i+16] = bHigh[i] ? l[i+16] : r[i+16];
        }
        return ret;
    }

    // Converts index 0-15 into a m512f
    friend m512f LowFloats(const m512s& in)
    {
        return m512f(_mm_cvtepi32_ps(_mm_cvtepi16_epi32(in.data.simd[0])),
                     _mm_cvtepi32_ps(_mm_cvtepi16_epi32(_mm_shuffle_epi32(in.data.simd[0], 0xEE))),
                     _mm_cvtepi32_ps(_mm_cvtepi16_epi32(in.data.simd[1])),
                     _mm_cvtepi32_ps(_mm_cvtepi16_epi32(_mm_shuffle_epi32(in.data.simd[1], 0xEE))));
    }

    // Converts index 16-31 into a m512f
    friend m512f HighFloats(const m512s& in)
    {
        return m512f(_mm_cvtepi32_ps(_mm_cvtepi16_epi32(in.data.simd[2])),
                     _mm_cvtepi32_ps(_mm_cvtepi16_epi32(_mm_shuffle_epi32(in.data.simd[2], 0xEE))),
                     _mm_cvtepi32_ps(_mm_cvtepi16_epi32(in.data.simd[3])),
                     _mm_cvtepi32_ps(_mm_cvtepi16_epi32(_mm_shuffle_epi32(in.data.simd[3], 0xEE))));
    }
    // Converts index 0-15 into a m512f
    friend m512i LowInts(const m512s& in)
    {
        return m512i(_mm_cvtepi16_epi32(in.data.simd[0]),
                     _mm_cvtepi16_epi32(_mm_shuffle_epi32(in.data.simd[0], 0xEE)),
                     _mm_cvtepi16_epi32(in.data.simd[1]),
                     _mm_cvtepi16_epi32(_mm_shuffle_epi32(in.data.simd[1], 0xEE)));
    }

    // Converts index 16-31 into a m512f
    friend m512i HighInts(const m512s& in)
    {
        return m512i(_mm_cvtepi16_epi32(in.data.simd[2]),
                     _mm_cvtepi16_epi32(_mm_shuffle_epi32(in.data.simd[2], 0xEE)),
                     _mm_cvtepi16_epi32(in.data.simd[3]),
                     _mm_cvtepi16_epi32(_mm_shuffle_epi32(in.data.simd[3], 0xEE)));
    }


public:     // Non-member (friend) functions

    friend m512s operator & (const m512s &l, const m512s &r)
    {
        return m512s(_mm_and_si128(l.data.simd[0], r.data.simd[0]),
                     _mm_and_si128(l.data.simd[1], r.data.simd[1]),
                     _mm_and_si128(l.data.simd[2], r.data.simd[2]),
                     _mm_and_si128(l.data.simd[3], r.data.simd[3]));
    }

    friend m512s operator | (const m512s &l, const m512s &r)
    {
        return m512s(_mm_or_si128(l.data.simd[0], r.data.simd[0]),
                     _mm_or_si128(l.data.simd[1], r.data.simd[1]),
                     _mm_or_si128(l.data.simd[2], r.data.simd[2]),
                     _mm_or_si128(l.data.simd[3], r.data.simd[3]));
    }

    friend m512s operator + (const m512s &l, const m512s &r)
    {
        return m512s(_mm_add_epi16(l.data.simd[0], r.data.simd[0]),
                     _mm_add_epi16(l.data.simd[1], r.data.simd[1]),
                     _mm_add_epi16(l.data.simd[2], r.data.simd[2]),
                     _mm_add_epi16(l.data.simd[3], r.data.simd[3]));
    }

    friend m512s operator - (const m512s &l, const m512s &r)
    {
        return m512s(_mm_sub_epi16(l.data.simd[0], r.data.simd[0]),
                     _mm_sub_epi16(l.data.simd[1], r.data.simd[1]),
                     _mm_sub_epi16(l.data.simd[2], r.data.simd[2]),
                     _mm_sub_epi16(l.data.simd[3], r.data.simd[3]));
    }

    /// Multiply: overflow returns the low 16 bits of the 32-bit result.
    friend m512s operator * (const m512s &l, const m512s &r)
    {
        return m512s(_mm_mullo_epi16(l.data.simd[0], r.data.simd[0]),
                     _mm_mullo_epi16(l.data.simd[1], r.data.simd[1]),
                     _mm_mullo_epi16(l.data.simd[2], r.data.simd[2]),
                     _mm_mullo_epi16(l.data.simd[3], r.data.simd[3]));
    }

    friend m512s operator / (const m512s &l, const m512s &r)
    {
        return m512s(_mm_div_epi16(l.data.simd[0], r.data.simd[0]),
                     _mm_div_epi16(l.data.simd[1], r.data.simd[1]),
                     _mm_div_epi16(l.data.simd[2], r.data.simd[2]),
                     _mm_div_epi16(l.data.simd[3], r.data.simd[3]));
    }

    friend m512s min(const m512s& a, const m512s&b)
    {
        return m512s(_mm_min_epi16(a.data.simd[0], b.data.simd[0]),
                     _mm_min_epi16(a.data.simd[1], b.data.simd[1]),
                     _mm_min_epi16(a.data.simd[2], b.data.simd[2]),
                     _mm_min_epi16(a.data.simd[3], b.data.simd[3]));
    }

    friend m512s max(const m512s& a, const m512s&b)
    {
        return m512s(_mm_max_epi16(a.data.simd[0], b.data.simd[0]),
                     _mm_max_epi16(a.data.simd[1], b.data.simd[1]),
                     _mm_max_epi16(a.data.simd[2], b.data.simd[2]),
                     _mm_max_epi16(a.data.simd[3], b.data.simd[3]));
    }

    static m512b help(__m128i a, __m128i b)
    {
        __m128 low1 = _mm_castsi128_ps(_mm_cvtepi16_epi32(a));
        __m128 high1 = _mm_castsi128_ps(_mm_cvtepi16_epi32(_mm_bsrli_si128(a, 8)));
        __m128 low2 = _mm_castsi128_ps(_mm_cvtepi16_epi32(b));
        __m128 high2 = _mm_castsi128_ps(_mm_cvtepi16_epi32(_mm_bsrli_si128(b, 8)));

        return m512b{low1, high1, low2, high2};
    }

    friend std::pair<m512b, m512b> operator!=(const m512s& a, const m512s& b)
    {
        std::pair<m512b, m512b> ret;
        ret.first = help(~_mm_cmpeq_epi16(a.data.simd[0], b.data.simd[0]),
                         ~_mm_cmpeq_epi16(a.data.simd[1], b.data.simd[1]));
        ret.second= help(~_mm_cmpeq_epi16(a.data.simd[2], b.data.simd[2]),
                         ~_mm_cmpeq_epi16(a.data.simd[3], b.data.simd[3]));
        return ret;
    }
    friend std::pair<m512b, m512b> operator==(const m512s& a, const m512s& b)
    {
        std::pair<m512b, m512b> ret;
        ret.first = help(_mm_cmpeq_epi16(a.data.simd[0], b.data.simd[0]),
                         _mm_cmpeq_epi16(a.data.simd[1], b.data.simd[1]));
        ret.second= help(_mm_cmpeq_epi16(a.data.simd[2], b.data.simd[2]),
                         _mm_cmpeq_epi16(a.data.simd[3], b.data.simd[3]));
        return ret;
    }
    friend std::pair<m512b, m512b> operator>(const m512s& a, const m512s& b)
    {
        std::pair<m512b, m512b> ret;
        ret.first = help(_mm_cmpgt_epi16(a.data.simd[0], b.data.simd[0]),
                         _mm_cmpgt_epi16(a.data.simd[1], b.data.simd[1]));
        ret.second= help(_mm_cmpgt_epi16(a.data.simd[2], b.data.simd[2]),
                         _mm_cmpgt_epi16(a.data.simd[3], b.data.simd[3]));
        return ret;
    }
    friend std::pair<m512b, m512b> operator<(const m512s& a, const m512s& b)
    {
        std::pair<m512b, m512b> ret;
        ret.first = help(_mm_cmplt_epi16(a.data.simd[0], b.data.simd[0]),
                         _mm_cmplt_epi16(a.data.simd[1], b.data.simd[1]));
        ret.second= help(_mm_cmplt_epi16(a.data.simd[2], b.data.simd[2]),
                         _mm_cmplt_epi16(a.data.simd[3], b.data.simd[3]));
        return ret;
    }
    friend std::pair<m512b, m512b> operator<=(const m512s& a, const m512s& b)
    {
        auto tmp = (a > b);
        return {!tmp.first, !tmp.second};
    }
    friend std::pair<m512b, m512b> operator>=(const m512s& a, const m512s& b)
    {
        auto tmp = (a < b);
        return {!tmp.first, !tmp.second};
    }
};

}}      // namespace PacBio::Simd

#endif  // mongo_common_simd_m512s_SSE_H_
