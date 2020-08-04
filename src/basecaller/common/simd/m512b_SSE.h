#ifndef mongo_common_simd_m512b_SSE_H_
#define mongo_common_simd_m512b_SSE_H_

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
/// \file   m512b_SSE.h
/// \brief  SIMD 512-bit bool vector (16 packed 1-bit bool values) for SSE.

#if !defined(__SSE2__)
#error This type requires SSE2 intrinsics.
#endif

#include <cassert>
#include <immintrin.h>
#include <iostream>
#include <ostream>

#include "xcompile.h"

namespace PacBio {
namespace Simd {

// SIMD 512-bit bool vector (16 packed 1-bit bool values).
CLASS_ALIGNAS(16) m512b
{
public:     // Types
    typedef m512b type;

    using Iterator = float*;
    using ConstIterator = const float*;

public:     // Static constants
    /// The number of floats represented by one instance.
    static constexpr size_t size()
    {
        return sizeof(m512b) / sizeof(float);
    }

private:    // Implementation
    using ImplType = __m128;

    union
    {
        ImplType simd[4];
        float raw[16];
    } data;

public:     // Structors
    // Purposefully do not initialize v.
    m512b() = default;
    
    // Copy constructor
    m512b(const m512b& x) = default;

    // Construct from native vector type
    m512b(ImplType v1, ImplType v2, ImplType v3, ImplType v4)
        : data{{v1, v2, v3, v4}}
    {}

    explicit m512b(bool x)
        : data{{(x ? _mm_cmpeq_ps(_mm_setzero_ps(), _mm_setzero_ps()) : _mm_set1_ps(0))
              , (x ? _mm_cmpeq_ps(_mm_setzero_ps(), _mm_setzero_ps()) : _mm_set1_ps(0))
              , (x ? _mm_cmpeq_ps(_mm_setzero_ps(), _mm_setzero_ps()) : _mm_set1_ps(0))
              , (x ? _mm_cmpeq_ps(_mm_setzero_ps(), _mm_setzero_ps()) : _mm_set1_ps(0))}}
    {}

    inline static const uint32_t& allones()
    {
        static const uint32_t allones = 0xFFFFFFFF;
        return allones;
    }
    inline static float FloatTrue()
    {
        return *(const float*)&allones();
    }

    inline static float BoolToFloat(bool b)
    {
        return b?FloatTrue():0;
    }

    // Construct from 16 bools
    m512b(bool f0, bool f1, bool f2, bool f3, bool f4, bool f5, bool f6,
          bool f7, bool f8, bool f9, bool f10, bool f11, bool f12,
          bool f13, bool f14, bool f15)
            : data{{_mm_setr_ps(BoolToFloat(f0),  BoolToFloat(f1),  BoolToFloat(f2),  BoolToFloat(f3))
                  , _mm_setr_ps(BoolToFloat(f4),  BoolToFloat(f5),  BoolToFloat(f6),  BoolToFloat(f7))
                  , _mm_setr_ps(BoolToFloat(f8),  BoolToFloat(f9),  BoolToFloat(f10), BoolToFloat(f11))
                  , _mm_setr_ps(BoolToFloat(f12), BoolToFloat(f13), BoolToFloat(f14), BoolToFloat(f15))}}
    {}
    // Construct from int
    // m512b(int intMask) : v(_mm512_int2mask(intMask)) {}

public:     // Assignment
    m512b& operator=(const m512b& x) = default;
    m512b& operator=(const bool b) { return *this = m512b(b); }

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
        return static_cast<bool>(data.raw[i]);
    }

    const ImplType& data1() const
    { return data.simd[0]; }

    const ImplType& data2() const
    { return data.simd[1]; }

    const ImplType& data3() const
    { return data.simd[2]; }

    const ImplType& data4() const
    { return data.simd[3]; }

public:  //member operators 

    m512b operator !()  const
    {
        //Not sure if there is a more efficient way to negate...
        __m128i ones = _mm_cmpeq_epi32(_mm_setzero_si128(), _mm_setzero_si128());
        return m512b(_mm_castsi128_ps(_mm_xor_si128(ones, _mm_castps_si128(data.simd[0]))),
                     _mm_castsi128_ps(_mm_xor_si128(ones, _mm_castps_si128(data.simd[1]))),
                     _mm_castsi128_ps(_mm_xor_si128(ones, _mm_castps_si128(data.simd[2]))),
                     _mm_castsi128_ps(_mm_xor_si128(ones, _mm_castps_si128(data.simd[3]))));
    }

public:     // Non-member (friend) functions
    friend std::ostream& operator << (std::ostream& stream, const m512b& vec)
    {
        stream << (bool) vec[0] << "\t" << (bool) vec[1] << "\t" 
               << (bool) vec[2] << "\t" << (bool) vec[3] << "\t" 
               << (bool) vec[4] << "\t" << (bool) vec[5] << "\t" 
               << (bool) vec[6] << "\t" << (bool) vec[7] << "\t" 
               << (bool) vec[8] << "\t" << (bool) vec[9] << "\t" 
               << (bool) vec[10] << "\t" << (bool) vec[11] << "\t"
               << (bool) vec[12] << "\t" << (bool) vec[13] << "\t" 
               << (bool) vec[14] << "\t" << (bool) vec[15];

        return stream;
    }

    friend m512b operator & (const m512b &l, const m512b &r)
    {
    	return m512b(_mm_and_ps(l.data.simd[0], r.data.simd[0]),
    	             _mm_and_ps(l.data.simd[1], r.data.simd[1]),
    	             _mm_and_ps(l.data.simd[2], r.data.simd[2]),
    	             _mm_and_ps(l.data.simd[3], r.data.simd[3]));
    }

    friend m512b operator | (const m512b &l, const m512b &r)
    { 
    	return m512b(_mm_or_ps(l.data.simd[0], r.data.simd[0]),
    	             _mm_or_ps(l.data.simd[1], r.data.simd[1]),
    	             _mm_or_ps(l.data.simd[2], r.data.simd[2]),
    	             _mm_or_ps(l.data.simd[3], r.data.simd[3]));
    }

    friend m512b operator ^ (const m512b &l, const m512b &r)
    { 
    	return m512b(_mm_xor_ps(l.data.simd[0], r.data.simd[0]),
    	             _mm_xor_ps(l.data.simd[1], r.data.simd[1]),
    	             _mm_xor_ps(l.data.simd[2], r.data.simd[2]),
    	             _mm_xor_ps(l.data.simd[3], r.data.simd[3]));
    }

    friend m512b operator==(const m512b& l, const m512b& r)
    {
        return m512b(_mm_castsi128_ps(_mm_cmpeq_epi32(_mm_castps_si128(l.data.simd[0]),
                                                      _mm_castps_si128(r.data.simd[0]))),
                     _mm_castsi128_ps(_mm_cmpeq_epi32(_mm_castps_si128(l.data.simd[1]),
                                                      _mm_castps_si128(r.data.simd[1]))),
                     _mm_castsi128_ps(_mm_cmpeq_epi32(_mm_castps_si128(l.data.simd[2]),
                                                      _mm_castps_si128(r.data.simd[2]))),
                     _mm_castsi128_ps(_mm_cmpeq_epi32(_mm_castps_si128(l.data.simd[3]),
                                                      _mm_castps_si128(r.data.simd[3]))));
    }

    friend bool any(const m512b& mask)
    { return (_mm_movemask_ps(mask.data.simd[0]) & 0xF) != 0x0
              || (_mm_movemask_ps(mask.data.simd[1]) & 0xF) != 0x0
              || (_mm_movemask_ps(mask.data.simd[2]) & 0xF) != 0x0
              || (_mm_movemask_ps(mask.data.simd[3]) & 0xF) != 0x0;}

    friend bool all(const m512b& mask)
    { return (_mm_movemask_ps(mask.data.simd[0]) & 0xF) == 0xF
              && (_mm_movemask_ps(mask.data.simd[1]) & 0xF) == 0xF
              && (_mm_movemask_ps(mask.data.simd[2]) & 0xF) == 0xF
              && (_mm_movemask_ps(mask.data.simd[3]) & 0xF) == 0xF;}

    friend bool none(const m512b& mask)
    { return (_mm_movemask_ps(mask.data.simd[0]) & 0xF) == 0x0
              && (_mm_movemask_ps(mask.data.simd[1]) & 0xF) == 0x0
              && (_mm_movemask_ps(mask.data.simd[2]) & 0xF) == 0x0
              && (_mm_movemask_ps(mask.data.simd[3]) & 0xF) == 0x0;}
};

}}      // namepsace PacBio::Simd

#endif  // mongo_common_simd_m512b_SSE_H_
