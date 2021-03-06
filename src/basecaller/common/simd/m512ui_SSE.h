#ifndef mongo_common_simd_m512ui_SSE_H_
#define mongo_common_simd_m512ui_SSE_H_

// Copyright (c) 2020, Pacific Biosciences of California, Inc.
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
/// \file   m512ui_SSE.h
/// \brief  SIMD 512-bit int vector (16 packed 32-bit int values) using SSE.

#if !defined(__SSE2__)
#error This type requires SSE2 intrinsics.
#endif

#ifndef __SSE4_2__
#error This type requires SSE4 intrinsics.
#endif

#include <cassert>
#include <immintrin.h>
#include <limits>
#include <ostream>
#include <smmintrin.h>

#include "m512i_SSE.h"
#include "mm_blendv_si128.h"
#include "xcompile.h"

namespace PacBio {
namespace Simd {

// SIMD 512-bit int vector (16 packed 32-bit int values).
CLASS_ALIGNAS(16) m512ui
{
public:     // Types
    typedef m512ui type;

public:     // Static constants
    /// The number of floats represented by one instance.
    static constexpr size_t size()
    {
        return sizeof(m512ui) / sizeof(uint32_t);
    }

private:    // Implementation
    using ImplType = __m128i;
    static const size_t implOffsetElems = sizeof(ImplType) / sizeof(uint32_t);

private:
    union
    {
        ImplType simd[4];
        uint32_t raw[16];
    } data;

public:     // Structors
    // Purposefully do not initialize v.
    m512ui() = default;

    // Replicate scalar x across v.
    m512ui(uint32_t x)
        : data{{_mm_set1_epi32(x)
              , _mm_set1_epi32(x)
              , _mm_set1_epi32(x)
              , _mm_set1_epi32(x)}}
    {}

    // Load x from pointer px. px must be aligned to 16 bytes.
    m512ui(const uint32_t *px)
        : data{{_mm_load_si128(reinterpret_cast<const ImplType*>(px))
              , _mm_load_si128(reinterpret_cast<const ImplType*>(px +   implOffsetElems))
              , _mm_load_si128(reinterpret_cast<const ImplType*>(px + 2*implOffsetElems))
              , _mm_load_si128(reinterpret_cast<const ImplType*>(px + 3*implOffsetElems))}}
    {}

    // Copy constructor
    m512ui(const m512ui& x) = default;

    // Construct from native vector type
    m512ui(ImplType v1, ImplType v2, ImplType v3, ImplType v4)
        : data{{v1, v2, v3, v4}}
    {}

    // Construct from m128b vector type
    explicit m512ui(const m512b& x)
        : data{{_mm_castps_si128(x.data1())
              , _mm_castps_si128(x.data2())
              , _mm_castps_si128(x.data3())
              , _mm_castps_si128(x.data4())}} {}

    explicit m512ui(const m512i& x)
        : data{{x.data1()
              , x.data2()
              , x.data3()
              , x.data4()}} {}

    explicit operator m512i() const
    {
        return m512i(data1(), data2(), data3(), data4());
    }

public:     // Export

    const ImplType& data1() const
    { return data.simd[0]; }

    const ImplType& data2() const
    { return data.simd[1]; }

    const ImplType& data3() const
    { return data.simd[2]; }

    const ImplType& data4() const
    { return data.simd[3]; }


public:     // Assignment
    m512ui& operator=(const m512ui& x) = default;

    // Assignment from scalar value
    m512ui& operator=(int x)
    {
        data.simd[0] = _mm_set1_epi32(x);
        data.simd[1] = _mm_set1_epi32(x);
        data.simd[2] = _mm_set1_epi32(x);
        data.simd[3] = _mm_set1_epi32(x);
        return *this;
    }

    // Compound assignment operators
    m512ui& operator+=(const m512ui& x)
    {
        data.simd[0] = _mm_add_epi32(this->data.simd[0], x.data.simd[0]);
        data.simd[1] = _mm_add_epi32(this->data.simd[1], x.data.simd[1]);
        data.simd[2] = _mm_add_epi32(this->data.simd[2], x.data.simd[2]);
        data.simd[3] = _mm_add_epi32(this->data.simd[3], x.data.simd[3]);
        return *this;
    }

    m512ui& operator-=(const m512ui& x)
    {
        data.simd[0] = _mm_sub_epi32(this->data.simd[0], x.data.simd[0]);
        data.simd[1] = _mm_sub_epi32(this->data.simd[1], x.data.simd[1]);
        data.simd[2] = _mm_sub_epi32(this->data.simd[2], x.data.simd[2]);
        data.simd[3] = _mm_sub_epi32(this->data.simd[3], x.data.simd[3]);
        return *this;
    }

    m512ui& operator*=(const m512ui& x)
    {
        data.simd[0] = _mm_mullo_epi32(this->data.simd[0], x.data.simd[0]);
        data.simd[1] = _mm_mullo_epi32(this->data.simd[1], x.data.simd[1]);
        data.simd[2] = _mm_mullo_epi32(this->data.simd[2], x.data.simd[2]);
        data.simd[3] = _mm_mullo_epi32(this->data.simd[3], x.data.simd[3]);
        return *this;
    }

    m512ui& operator/=(const m512ui& x)
    {
        data.simd[0] = _mm_div_epu32(this->data.simd[0], x.data.simd[0]);
        data.simd[1] = _mm_div_epu32(this->data.simd[1], x.data.simd[1]);
        data.simd[2] = _mm_div_epu32(this->data.simd[2], x.data.simd[2]);
        data.simd[3] = _mm_div_epu32(this->data.simd[3], x.data.simd[3]);
        return *this;
    }

    m512ui& operator &= (const m512ui& x)
    {
        data.simd[0] = _mm_and_si128(data.simd[0], x.data.simd[0]);
        data.simd[1] = _mm_and_si128(data.simd[1], x.data.simd[1]);
        data.simd[2] = _mm_and_si128(data.simd[2], x.data.simd[2]);
        data.simd[3] = _mm_and_si128(data.simd[3], x.data.simd[3]);
        return *this;
    }

    m512ui& operator |= (const m512ui& x)
    {
        data.simd[0] = _mm_or_si128(data.simd[0], x.data.simd[0]);
        data.simd[1] = _mm_or_si128(data.simd[1], x.data.simd[1]);
        data.simd[2] = _mm_or_si128(data.simd[2], x.data.simd[2]);
        data.simd[3] = _mm_or_si128(data.simd[3], x.data.simd[3]);
        return *this;
    }

    m512ui& operator ^= (const m512ui& x)
    {
        data.simd[0] = _mm_xor_si128(data.simd[0], x.data.simd[0]);
        data.simd[1] = _mm_xor_si128(data.simd[1], x.data.simd[1]);
        data.simd[2] = _mm_xor_si128(data.simd[2], x.data.simd[2]);
        data.simd[3] = _mm_xor_si128(data.simd[3], x.data.simd[3]);
        return *this;
    }


    // Return a scalar value
    uint32_t operator[](unsigned int i) const
    {
        assert(static_cast<size_t>(i) < this->size());
        return data.raw[i];
    }

public:     // Non-member (friend) functions

    friend std::ostream& operator << (std::ostream& stream, const m512ui& vec)
    {
        stream << vec[0] << "\t" << vec[1] << "\t" << vec[2] << "\t" << vec[3]
               << "\t" << vec[4] << "\t" << vec[5] << "\t" << vec[6] << "\t"
               << vec[7] << "\t" << vec[8] << "\t" << vec[9] << "\t" << vec[10]
               << "\t" << vec[11] << "\t" << vec[12] << "\t" << vec[13] << "\t"
               << vec[14] << "\t" << vec[15];

        return stream;
    }

    friend m512ui operator + (const m512ui &l, const m512ui &r)
    {
        return m512ui(_mm_add_epi32(l.data.simd[0], r.data.simd[0]),
                      _mm_add_epi32(l.data.simd[1], r.data.simd[1]),
                      _mm_add_epi32(l.data.simd[2], r.data.simd[2]),
                      _mm_add_epi32(l.data.simd[3], r.data.simd[3]));
    }

    friend m512ui operator - (const m512ui &l, const m512ui &r)
    {
        return m512ui(_mm_sub_epi32(l.data.simd[0], r.data.simd[0]),
                      _mm_sub_epi32(l.data.simd[1], r.data.simd[1]),
                      _mm_sub_epi32(l.data.simd[2], r.data.simd[2]),
                      _mm_sub_epi32(l.data.simd[3], r.data.simd[3]));
    }

    /// Multiply: overflow returns the low bits of the 32-bit result.
    friend m512ui operator * (const m512ui &l, const m512ui &r)
    {
        return m512ui(_mm_mullo_epi32(l.data.simd[0], r.data.simd[0]),
                      _mm_mullo_epi32(l.data.simd[1], r.data.simd[1]),
                      _mm_mullo_epi32(l.data.simd[2], r.data.simd[2]),
                      _mm_mullo_epi32(l.data.simd[3], r.data.simd[3]));
    }

    friend m512ui operator / (const m512ui &l, const m512ui &r)
    {
        return m512ui(_mm_div_epu32(l.data.simd[0], r.data.simd[0]),
                      _mm_div_epu32(l.data.simd[1], r.data.simd[1]),
                      _mm_div_epu32(l.data.simd[2], r.data.simd[2]),
                      _mm_div_epu32(l.data.simd[3], r.data.simd[3]));
    }

    friend m512ui operator & (const m512ui& l, const m512ui& r)
    {
        return m512ui(_mm_and_si128(l.data.simd[0], r.data.simd[0]),
                      _mm_and_si128(l.data.simd[1], r.data.simd[1]),
                      _mm_and_si128(l.data.simd[2], r.data.simd[2]),
                      _mm_and_si128(l.data.simd[3], r.data.simd[3]));
    }

    friend m512ui operator | (const m512ui& l, const m512ui& r)
    {
        return m512ui(_mm_or_si128(l.data.simd[0], r.data.simd[0]),
                      _mm_or_si128(l.data.simd[1], r.data.simd[1]),
                      _mm_or_si128(l.data.simd[2], r.data.simd[2]),
                      _mm_or_si128(l.data.simd[3], r.data.simd[3]));
    }

    friend m512ui operator ^ (const m512ui& l, const m512ui& r)
    {
        return m512ui(_mm_xor_si128(l.data.simd[0], r.data.simd[0]),
                     _mm_xor_si128(l.data.simd[1], r.data.simd[1]),
                     _mm_xor_si128(l.data.simd[2], r.data.simd[2]),
                     _mm_xor_si128(l.data.simd[3], r.data.simd[3])
                );
    }

    m512ui lshift(const uint8_t count) const
    {
        return m512ui(_mm_slli_epi32(data.simd[0], count),
                     _mm_slli_epi32(data.simd[1], count),
                     _mm_slli_epi32(data.simd[2], count),
                     _mm_slli_epi32(data.simd[3], count)
                );
    }

    m512ui lshift(const m512ui& count) const
    {
        // This is the desired intrinsic for _m128 data, but for whatever
        // reason it didn't make it into the simd instruciton set until AVX2
        //return m512ui(_mm_sllv_epi32(data[1], count),
        //             _mm_sllv_epi32(data[2], count),
        //             _mm_sllv_epi32(data[3], count),
        //             _mm_sllv_epi32(data[4], count)
        //        );
        m512ui ret;
        for (size_t i = 0; i < size(); ++i)
        {
            auto val = static_cast<uint32_t>((*this)[i]);
            ret.data.raw[i] = static_cast<int32_t>(val << count[i]);
        }
        return ret;
    }

    m512ui rshift(const uint8_t count) const
    {
        return m512ui(_mm_srli_epi32(data.simd[0], count),
                     _mm_srli_epi32(data.simd[1], count),
                     _mm_srli_epi32(data.simd[2], count),
                     _mm_srli_epi32(data.simd[3], count)
                );
    }

    m512ui rshift(const m512ui& count) const
    {
        // This is the desired intrinsic for _m128 data, but for whatever
        // reason it didn't make it into the simd instruciton set until AVX2
        //return m512ui(_mm_srlv_epi32(data[1], count),
        //             _mm_srlv_epi32(data[2], count),
        //             _mm_srlv_epi32(data[3], count),
        //             _mm_srlv_epi32(data[4], count)
        //        );
        m512ui ret;
        for (size_t i = 0; i < size(); ++i)
        {
            auto val = static_cast<uint32_t>((*this)[i]);
            ret.data.raw[i] = static_cast<int32_t>(val >> count[i]);
        }
        return ret;
    }

    // SSE is lacking the necessary comparison operators for unsigned ints.
    // However we can just treat the incoming uint16_t vector as int16_t and
    // subtract off the lowest int16_t value. With rollovers considered, this
    // will put 0 back at the lowest value, making the result of signed int16_t
    // comparisons give us what we need
    static __m128i CompPrep(__m128i v)
    {
        auto min = _mm_set1_epi32(std::numeric_limits<int32_t>::lowest());
        return _mm_sub_epi32(v, min);
    }

    friend m512b operator < (const m512ui &l, const m512ui &r)
    {
        return m512b(_mm_castsi128_ps(_mm_cmplt_epi32(CompPrep(l.data.simd[0]), CompPrep(r.data.simd[0]))),
                     _mm_castsi128_ps(_mm_cmplt_epi32(CompPrep(l.data.simd[1]), CompPrep(r.data.simd[1]))),
                     _mm_castsi128_ps(_mm_cmplt_epi32(CompPrep(l.data.simd[2]), CompPrep(r.data.simd[2]))),
                     _mm_castsi128_ps(_mm_cmplt_epi32(CompPrep(l.data.simd[3]), CompPrep(r.data.simd[3]))));
    }

    friend m512b operator <= (const m512ui &l, const m512ui &r)
    {
        return !(l > r);
    }

    friend m512b operator > (const m512ui &l, const m512ui &r)
    {
        return m512b(_mm_castsi128_ps(_mm_cmpgt_epi32(CompPrep(l.data.simd[0]), CompPrep(r.data.simd[0]))),
                     _mm_castsi128_ps(_mm_cmpgt_epi32(CompPrep(l.data.simd[1]), CompPrep(r.data.simd[1]))),
                     _mm_castsi128_ps(_mm_cmpgt_epi32(CompPrep(l.data.simd[2]), CompPrep(r.data.simd[2]))),
                     _mm_castsi128_ps(_mm_cmpgt_epi32(CompPrep(l.data.simd[3]), CompPrep(r.data.simd[3]))));
    }

    friend m512b operator >= (const m512ui &l, const m512ui &r)
    {
        return !(l < r);
    }

    friend m512b operator == (const m512ui &l, const m512ui &r)
    {
        return m512b(_mm_castsi128_ps(_mm_cmpeq_epi32(l.data.simd[0], r.data.simd[0])),
                     _mm_castsi128_ps(_mm_cmpeq_epi32(l.data.simd[1], r.data.simd[1])),
                     _mm_castsi128_ps(_mm_cmpeq_epi32(l.data.simd[2], r.data.simd[2])),
                     _mm_castsi128_ps(_mm_cmpeq_epi32(l.data.simd[3], r.data.simd[3])));
    }

    friend m512b operator != (const m512ui &l, const m512ui &r)
    {
        return !(l == r);
    }

    friend m512ui min(const m512ui& a, const m512ui&b)
    {
        return m512ui(_mm_min_epu32(a.data.simd[0], b.data.simd[0]),
                      _mm_min_epu32(a.data.simd[1], b.data.simd[1]),
                      _mm_min_epu32(a.data.simd[2], b.data.simd[2]),
                      _mm_min_epu32(a.data.simd[3], b.data.simd[3]));
    }

    friend m512ui max(const m512ui& a, const m512ui&b)
    {
        return m512ui(_mm_max_epu32(a.data.simd[0], b.data.simd[0]),
                      _mm_max_epu32(a.data.simd[1], b.data.simd[1]),
                      _mm_max_epu32(a.data.simd[2], b.data.simd[2]),
                      _mm_max_epu32(a.data.simd[3], b.data.simd[3]));
    }

    friend int reduceMax(const m512ui& a)
    {
        uint32_t ret = std::numeric_limits<uint32_t>::min();
        for (size_t i = 0; i < size(); ++i)
        {
            ret = std::max(ret, a[i]);
        }
        return ret;
    }

    friend int reduceMin(const m512ui& a)
    {
        uint32_t ret = std::numeric_limits<uint32_t>::max();
        for (size_t i = 0; i < size(); ++i)
        {
            ret = std::min(ret, a[i]);
        }
        return ret;
    }

    friend m512ui Blend(const m512b& mask, const m512ui& success, const m512ui& failure)
    {
        const m512ui m = m512ui(mask);
        return m512ui(_mm_blendv_si128(failure.data.simd[0], success.data.simd[0], m.data.simd[0]),
                      _mm_blendv_si128(failure.data.simd[1], success.data.simd[1], m.data.simd[1]),
                      _mm_blendv_si128(failure.data.simd[2], success.data.simd[2], m.data.simd[2]),
                      _mm_blendv_si128(failure.data.simd[3], success.data.simd[3], m.data.simd[3]));
    }

    friend m512ui inc(const m512ui& a, const m512b& maskB)
    {
        const m512ui b = a + 1u;
        return Blend(maskB, b, a);
    }
};

}}      // namespace PacBio::Simd

#endif  // mongo_common_simd_m512ui_SSE_H_
