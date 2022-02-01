#ifndef mongo_common_simd_m512i_AVX512_H_
#define mongo_common_simd_m512i_AVX512_H_

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
/// \file   m512i_AVX512.h
/// \brief  SIMD 512-bit int vector (16 packed 32-bit int values) using AVX-512.

#if !defined(PB_CORE_AVX512)
#error This type requires AVX-512 intrinsics.
#endif

#include <cassert>
#include <immintrin.h>
#include <ostream>

#include "m512b_AVX512.h"
#include "xcompile.h"

//#include <Eigen/Core>

namespace PacBio {
namespace Simd {

// SIMD 512-bit int vector (16 packed 32-bit int values).
CLASS_ALIGNAS(64) m512i //: Eigen::NumTraits<float>
{
public:     // Types
    typedef m512i type;

public:     // Static constants
    /// The number of floats represented by one instance.
    static constexpr size_t size()
    { return sizeof(m512i) / sizeof(int); }

private:    // Implementation
    using ImplType = __m512i;

    ImplType v;

public:     // Structors
    // Purposefully do not initialize v.
    m512i() = default;

    // Replicate scalar x across v
    m512i(int x) : v(_mm512_set1_epi32(x)) {}

    // Load x from pointer px. px must be aligned to 16 bytes.
    m512i(const int *px) : v(_mm512_load_si512(reinterpret_cast<const __m512i*>(px))) {}

    // Copy constructor
    m512i(const m512i& x) = default;

    // Construct from native vector type
    m512i(ImplType v_) : v(v_) {}

public:     // Assignment
    m512i& operator=(const m512i& x) = default;

    // Assignment from scalar value
    m512i& operator =(int x) { v = _mm512_set1_epi32(x); return *this; }

    // Compound assignment operators
    m512i& operator += (const m512i& x) { v = _mm512_add_epi32(this->v, x.v);   return *this; }
    m512i& operator -= (const m512i& x) { v = _mm512_sub_epi32(this->v, x.v);   return *this; }
    m512i& operator *= (const m512i& x) { v = _mm512_mullo_epi32(this->v, x.v); return *this; }
    m512i& operator /= (const m512i& x) { v = _mm512_div_epi32(this->v, x.v);   return *this; }

    m512i operator - () const { return m512i(_mm512_sub_epi32(_mm512_setzero_si512(), v)); }

    // Return a scalar value
    int operator[](unsigned int i) const
    {
        assert(static_cast<size_t>(i) < this->size());
        const auto mask = static_cast<uint16_t>(1 << i);
        return _mm512_mask_reduce_add_epi32(mask, v);
    }

    const ImplType& data() const { return v; }

public:     // Non-member (friend) functions

    friend m512i operator + (const m512i& l, const m512i& r) { return m512i(_mm512_add_epi32(l.v, r.v)); }
    friend m512i operator - (const m512i& l, const m512i& r) { return m512i(_mm512_sub_epi32(l.v, r.v)); }
    friend m512i operator * (const m512i& l, const m512i& r) { return m512i(_mm512_mullo_epi32(l.v, r.v)); }
    friend m512i operator / (const m512i& l, const m512i& r) { return m512i(_mm512_div_epi32(l.v, r.v)); }

    friend m512i operator & (const m512i& l, const m512i& r) { return m512i(_mm512_and_si512(l.v, r.v)); }
    friend m512i operator | (const m512i& l, const m512i& r) { return m512i(_mm512_or_si512(l.v, r.v)); }
    friend m512i operator ^ (const m512i& l, const m512i& r) { return m512i(_mm512_xor_si512(l.v, r.v)); }

    friend m512b operator < (const m512i &l, const m512i &r)
    {
        return m512b(_mm512_cmplt_epi32_mask(l.v, r.v));
    }

    friend m512b operator <= (const m512i &l, const m512i &r)
    {
        return m512b(_mm512_cmple_epi32_mask(l.v, r.v));
    }

    friend m512b operator > (const m512i &l, const m512i &r)
    {
        return m512b(_mm512_cmpgt_epi32_mask(l.v, r.v));
    }

    friend m512b operator >= (const m512i &l, const m512i &r)
    {
        return m512b(_mm512_cmpge_epi32_mask(l.v, r.v));
    }

    friend m512b operator == (const m512i &l, const m512i &r)
    {
        return m512b(_mm512_cmpeq_epi32_mask(l.v, r.v));
    }

    friend m512b operator != (const m512i &l, const m512i &r)
    {
        return m512b(_mm512_cmpneq_epi32_mask(l.v, r.v));
    }

    m512i lshift(const uint8_t count) const
    {
        return m512i(_mm512_slli_epi32(v, count));
    }

    m512i lshift(const m512i& count) const
    {
        return m512i(_mm512_sllv_epi32(v, count.v));
    }

    m512i rshift(const uint8_t count) const
    {
        return m512i(_mm512_srli_epi32(v, count));
    }

    m512i rshift(const m512i& count) const
    {
        return m512i(_mm512_srlv_epi32(v, count.v));
    }

    friend int reduceMax(const m512i& a) { return _mm512_reduce_max_epi32(a.v); }
    friend int reduceMin(const m512i& a) { return _mm512_reduce_min_epi32(a.v); }

    friend m512i min(const m512i& a, const m512i&b) {   return m512i(_mm512_min_epi32(a.v, b.v)); }
    friend m512i max(const m512i& a, const m512i&b) { return m512i(_mm512_max_epi32(a.v, b.v)); }

    friend m512i Blend(const m512b& mask, const m512i& success, const m512i& failure)
    { return m512i(_mm512_mask_blend_epi32(mask.data(), failure.v, success.v)); }

    friend m512i inc (const m512i& a, const m512b& mask)
    { return m512i(_mm512_mask_add_epi32(a.v, mask.data(), a.v, m512i(1).v)); }

public:     // stream
    friend std::ostream& operator << (std::ostream& stream, const m512i& vec)
    {
        stream << vec[0] << "\t" << vec[1] << "\t" << vec[2] << "\t" << vec[3]
               << "\t" << vec[4] << "\t" << vec[5] << "\t" << vec[6] << "\t"
               << vec[7] << "\t" << vec[8] << "\t" << vec[9] << "\t" << vec[10]
               << "\t" << vec[11] << "\t" << vec[12] << "\t" << vec[13] << "\t"
               << vec[14] << "\t" << vec[15];

        return stream;
    }
};

}}      // namespace PacBio::Simd

#endif  // mongo_common_simd_m512i_AVX512_H_
