#ifndef mongo_common_simd_m512b_AVX512_H_
#define mongo_common_simd_m512b_AVX512_H_

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
/// \file   m512b_AVX512.h
/// \brief  SIMD 512-bit bool vector (16 packed 1-bit bool values) for AVX-512.

#if !defined(PB_CORE_AVX512)
#error This type requires AVX-512 intrinsics.
#endif

#include <cassert>
#include <immintrin.h>
#include <iostream>
#include <ostream>

#include "xcompile.h"

namespace PacBio {
namespace Simd {

// SIMD 512-bit bool vector (16 packed 1-bit bool values).
CLASS_ALIGNAS(64) m512b
{
public:     // Types
    typedef m512b type;

    using Iterator      =       bool*;
    using ConstIterator = const bool*;

public:     // Static constants
    /// The number of floats represented by one instance.
    static constexpr size_t size() { return 16; }

private:    // Implementation
    using ImplType = __mmask16;
    ImplType v;

public:     // Structors
    // Purposefully do not initialize v.
    m512b() = default;

    // Copy constructor
    m512b(const m512b& x) = default;

    // Construct from native vector type
    explicit m512b(ImplType v_) : v(v_) {}

    // Construct from 16 bools
    m512b(bool f0, bool f1, bool f2, bool f3, bool f4, bool f5, bool f6,
          bool f7, bool f8, bool f9, bool f10, bool f11, bool f12,
          bool f13, bool f14, bool f15)
            :
            v(_mm512_int2mask(
                            (f0?0x0001:0) |
                            (f1?0x0002:0) |
                            (f2?0x0004:0) |
                            (f3?0x0008:0) |
                            (f4?0x0010:0) |
                            (f5?0x0020:0) |
                            (f6?0x0040:0) |
                            (f7?0x0080:0) |
                            (f8?0x0100:0) |
                            (f9?0x0200:0) |
                            (f10?0x0400:0) |
                            (f11?0x0800:0) |
                            (f12?0x1000:0) |
                            (f13?0x2000:0) |
                            (f14?0x4000:0) |
                            (f15?0x8000:0)
            )) {}

    explicit m512b(bool x) : v(_mm512_int2mask(x ? 0xFFFF : 0x0)) {}

    explicit m512b(int x) : v(_mm512_int2mask(x ? 0xFFFF : 0x0)) {}

public:     // Assignment
    m512b& operator=(const m512b& x) = default;

    m512b& operator=(const bool x)
    {
        v = _mm512_int2mask(x ? 0xFFFF : 0x0);
        return *this;
    }

    // TODO: There's probably a more efficient way to define the compound
    // assignment operators.

    m512b& operator&=(const m512b& rhs)
    { return *this = (*this & rhs); }

    m512b& operator|=(const m512b& rhs)
    { return *this = (*this | rhs); }

public:     // Scalar access
    // Return a scalar value
    bool operator[](unsigned int i) const
    {
        assert(static_cast<size_t>(i) < this->size());
        return (v >> i) & 1;
    }

    const ImplType& data() const { return v; }

public:  // member operators

    m512b operator !()
    {
        __mmask16 ones = _mm512_cmpeq_ps_mask(_mm512_setzero(),_mm512_setzero());
        return m512b(_mm512_kxor(v, ones));
    }

public:     // Non-member (friend) function
    friend m512b operator ! (const m512b &x)                 { return m512b(_mm512_knot(x.v)); }
    friend m512b operator & (const m512b &l, const m512b &r) { return m512b(_mm512_kand(l.v, r.v)); }
    friend m512b operator | (const m512b &l, const m512b &r) { return m512b(_mm512_kor(l.v, r.v)); }
    friend m512b operator ^ (const m512b &l, const m512b &r) { return m512b(_mm512_kxor(l.v, r.v)); }
    friend m512b operator ==(const m512b &l, const m512b &r) { return m512b(_mm512_knot(_mm512_kxor(l.v, r.v))); }

    friend bool any (const m512b& mask) { return _mm512_mask2int(mask.v) != 0; }
    friend bool all (const m512b& mask) { return _mm512_mask2int(mask.v) == 0xffff; }
    friend bool none(const m512b& mask) { return _mm512_mask2int(mask.v) == 0; }

public: // stream
    friend std::ostream& operator << (std::ostream& stream, const m512b& vec)
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

#endif  // mongo_common_simd_m512b_AVX512_H_
