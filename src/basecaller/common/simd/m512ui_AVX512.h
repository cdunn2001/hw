#ifndef mongo_common_simd_m512ui_AVX512_H_
#define mongo_common_simd_m512ui_AVX512_H_

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
/// \file   m512ui_AVX512.h
/// \brief  SIMD 512-bit int vector (16 packed 32-bit int values) using AVX-512.

#if !defined(PB_CORE_AVX512)
#error This type requires AVX-512 intrinsics.
#endif

#include <cassert>
#include <immintrin.h>
#include <ostream>

#include "m512b_AVX512.h"
#include "m512i_AVX512.h"
#include "xcompile.h"

//#include <Eigen/Core>

namespace PacBio {
namespace Simd {

// SIMD 512-bit int vector (16 packed 32-bit int values).
CLASS_ALIGNAS(64) m512ui //: Eigen::NumTraits<float>
{
public:     // Types
    typedef m512ui type;

public:     // Static constants
    /// The number of floats represented by one instance.
    static constexpr size_t size()
    { return sizeof(m512ui) / sizeof(uint32_t); }

private:    // Implementation
    using ImplType = __m512i;

    ImplType v;

public:     // Structors
    // Purposefully do not initialize v.
    m512ui() = default;

    // Replicate scalar x across v
    m512ui(uint32_t x)
    {
        v = _mm512_set1_epi32(x);
    }

    // Load x from pointer px. px must be aligned to 16 bytes.
    m512ui(const uint32_t *px) : v(_mm512_load_si512(reinterpret_cast<const __m512i*>(px))) {}

    // Copy constructor
    m512ui(const m512ui& x) = default;

    // Construct from native vector type
    m512ui(ImplType v_) : v(v_) {}

    explicit m512ui(const m512i& x)
        : v(x.data())
    {}

    explicit operator m512i() const
    {
        return m512i(v);
    }

public:     // Assignment
    m512ui& operator=(const m512ui& x) = default;

    // Assignment from scalar value
    m512ui& operator =(int x) { v = _mm512_set1_epi32(x); return *this; }

    // Compound assignment operators
    m512ui& operator += (const m512ui& x) { v = _mm512_add_epi32(this->v, x.v);   return *this; }
    m512ui& operator -= (const m512ui& x) { v = _mm512_sub_epi32(this->v, x.v);   return *this; }
    m512ui& operator *= (const m512ui& x) { v = _mm512_mullo_epi32(this->v, x.v); return *this; }
    m512ui& operator /= (const m512ui& x) { v = _mm512_div_epu32(this->v, x.v);   return *this; }

    // Return a scalar value
    uint32_t operator[](unsigned int i) const
    {
        assert(static_cast<size_t>(i) < this->size());
        const auto mask = static_cast<uint16_t>(1 << i);
        return _mm512_mask_reduce_add_epi32(mask, v);
    }

    const ImplType& data() const { return v; }

public:     // Non-member (friend) functions

    friend m512ui operator + (const m512ui& l, const m512ui& r) { return m512ui(_mm512_add_epi32(l.v, r.v)); }
    friend m512ui operator - (const m512ui& l, const m512ui& r) { return m512ui(_mm512_sub_epi32(l.v, r.v)); }
    friend m512ui operator * (const m512ui& l, const m512ui& r) { return m512ui(_mm512_mullo_epi32(l.v, r.v)); }
    friend m512ui operator / (const m512ui& l, const m512ui& r) { return m512ui(_mm512_div_epu32(l.v, r.v)); }

    friend m512ui operator & (const m512ui& l, const m512ui& r) { return m512ui(_mm512_and_si512(l.v, r.v)); }
    friend m512ui operator | (const m512ui& l, const m512ui& r) { return m512ui(_mm512_or_si512(l.v, r.v)); }
    friend m512ui operator ^ (const m512ui& l, const m512ui& r) { return m512ui(_mm512_xor_si512(l.v, r.v)); }

    friend m512b operator < (const m512ui &l, const m512ui &r)
    {
        return m512b(_mm512_cmplt_epu32_mask(l.v, r.v));
    }

    friend m512b operator <= (const m512ui &l, const m512ui &r)
    {
        return m512b(_mm512_cmple_epu32_mask(l.v, r.v));
    }

    friend m512b operator > (const m512ui &l, const m512ui &r)
    {
        return m512b(_mm512_cmpgt_epu32_mask(l.v, r.v));
    }

    friend m512b operator >= (const m512ui &l, const m512ui &r)
    {
        return m512b(_mm512_cmpge_epu32_mask(l.v, r.v));
    }

    friend m512b operator == (const m512ui &l, const m512ui &r)
    {
        return m512b(_mm512_cmpeq_epu32_mask(l.v, r.v));
    }

    friend m512b operator != (const m512ui &l, const m512ui &r)
    {
        return m512b(_mm512_cmpneq_epu32_mask(l.v, r.v));
    }

    friend int reduceMax(const m512ui& a) { return _mm512_reduce_max_epu32(a.v); }
    friend int reduceMin(const m512ui& a) { return _mm512_reduce_min_epu32(a.v); }

    friend m512ui min(const m512ui& a, const m512ui&b) { return m512ui(_mm512_min_epu32(a.v, b.v)); }
    friend m512ui max(const m512ui& a, const m512ui&b) { return m512ui(_mm512_max_epu32(a.v, b.v)); }

    friend m512ui Blend(const m512b& mask, const m512ui& success, const m512ui& failure)
    { return m512ui(_mm512_mask_blend_epi32(mask.data(), failure.v, success.v)); }

    friend m512ui inc (const m512ui& a, const m512b& mask)
    { return m512ui(_mm512_mask_add_epi32(a.v, mask.data(), a.v, m512ui(1).v)); }

};

}}      // namespace PacBio::Simd

#endif  // mongo_common_simd_m512ui_AVX512_H_
