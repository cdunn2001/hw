#ifndef mongo_common_simd_m512i_SSE_H_
#define mongo_common_simd_m512i_SSE_H_

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
/// \file   m512i_SSE.h
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

#include "m512f_SSE.h"
#include "mm_blendv_si128.h"
#include "xcompile.h"

namespace PacBio {
namespace Simd {

// SIMD 512-bit int vector (16 packed 32-bit int values).
CLASS_ALIGNAS(16) m512i
{
public:     // Types
    typedef m512i type;

    using Iterator = int*;
    using ConstIterator = const int*;

public:     // Static constants
    /// The number of floats represented by one instance.
    static constexpr size_t size()
    {
        return sizeof(m512i) / sizeof(int);
    }

private:    // Implementation
    using ImplType = __m128i;
    static const size_t implOffsetElems = sizeof(ImplType) / sizeof(int);

    union
    {
        ImplType simd[4];
        int raw[16];
    } data;

public:     // Structors
    // Purposefully do not initialize v.
    m512i() {}

    // Replicate scalar x across v.
    m512i(int x)
        : data{{_mm_set1_epi32(x)
              , _mm_set1_epi32(x)
              , _mm_set1_epi32(x)
              , _mm_set1_epi32(x)}}
    {}

    // Load x from pointer px. px must be aligned to 16 bytes.
    m512i(const int *px)
        : data{{_mm_load_si128(reinterpret_cast<const ImplType*>(px))
              , _mm_load_si128(reinterpret_cast<const ImplType*>(px +   implOffsetElems))
              , _mm_load_si128(reinterpret_cast<const ImplType*>(px + 2*implOffsetElems))
              , _mm_load_si128(reinterpret_cast<const ImplType*>(px + 3*implOffsetElems))}}
    {}

    // Copy constructor
    m512i(const m512i& x) = default;

    // Construct from native vector type
    m512i(ImplType v1, ImplType v2, ImplType v3, ImplType v4)
        : data{{v1, v2, v3, v4}}
    {}

    // Construct from m128f vector type
    explicit m512i(const m512f& x)
        : data{{_mm_cvtps_epi32(_mm_round_ps(x.data1(),_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC))
              , _mm_cvtps_epi32(_mm_round_ps(x.data2(),_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC))
              , _mm_cvtps_epi32(_mm_round_ps(x.data3(),_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC))
              , _mm_cvtps_epi32(_mm_round_ps(x.data4(),_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC))}} {}

    // Construct from m128b vector type
    explicit m512i(const m512b& x) 
        : data{{_mm_castps_si128(x.data1())
              , _mm_castps_si128(x.data2())
              , _mm_castps_si128(x.data3())
              , _mm_castps_si128(x.data4())}} {}

public:     // Export
    m512f AsFloat() const
    {
        return m512f(_mm_cvtepi32_ps(data.simd[0]),
                     _mm_cvtepi32_ps(data.simd[1]),
                     _mm_cvtepi32_ps(data.simd[2]),
                     _mm_cvtepi32_ps(data.simd[3]));
    }

    operator m512f() const
    {
        return m512f(_mm_cvtepi32_ps(data.simd[0]),
                     _mm_cvtepi32_ps(data.simd[1]),
                     _mm_cvtepi32_ps(data.simd[2]),
                     _mm_cvtepi32_ps(data.simd[3]));
    }

public:     // Assignment
    m512i& operator=(const m512i& x) = default;

    // Assignment from scalar value
    m512i& operator=(int x)
    {
        data.simd[0] = _mm_set1_epi32(x);
        data.simd[1] = _mm_set1_epi32(x);
        data.simd[2] = _mm_set1_epi32(x);
        data.simd[3] = _mm_set1_epi32(x);
        return *this;
    }

    // Compound assignment operators
    m512i& operator+=(const m512i& x)
    {
        data.simd[0] = _mm_add_epi32(this->data.simd[0], x.data.simd[0]);
        data.simd[1] = _mm_add_epi32(this->data.simd[1], x.data.simd[1]);
        data.simd[2] = _mm_add_epi32(this->data.simd[2], x.data.simd[2]);
        data.simd[3] = _mm_add_epi32(this->data.simd[3], x.data.simd[3]);
        return *this;
    }

    m512i& operator-=(const m512i& x)
    {
        data.simd[0] = _mm_sub_epi32(this->data.simd[0], x.data.simd[0]);
        data.simd[1] = _mm_sub_epi32(this->data.simd[1], x.data.simd[1]);
        data.simd[2] = _mm_sub_epi32(this->data.simd[2], x.data.simd[2]);
        data.simd[3] = _mm_sub_epi32(this->data.simd[3], x.data.simd[3]);
        return *this;
    }

    m512i& operator*=(const m512i& x)
    {
        data.simd[0] = _mm_mullo_epi32(this->data.simd[0], x.data.simd[0]);
        data.simd[1] = _mm_mullo_epi32(this->data.simd[1], x.data.simd[1]);
        data.simd[2] = _mm_mullo_epi32(this->data.simd[2], x.data.simd[2]);
        data.simd[3] = _mm_mullo_epi32(this->data.simd[3], x.data.simd[3]);
        return *this;
    }

    // Return a scalar value
    int operator[](unsigned int i) const
    {
        assert(static_cast<size_t>(i) < this->size());
        return data.raw[i];
    }

public:     // Functor types

    struct minOp
    {
        m512i operator() (const m512i& a, const m512i& b)
        {
            return m512i(_mm_min_epi32(a.data.simd[0], b.data.simd[0]),
                         _mm_min_epi32(a.data.simd[1], b.data.simd[1]),
                         _mm_min_epi32(a.data.simd[2], b.data.simd[2]),
                         _mm_min_epi32(a.data.simd[3], b.data.simd[3]));
        }
    };

    struct maxOp
    {
        m512i operator() (const m512i& a, const m512i& b)
        {
            return m512i(_mm_max_epi32(a.data.simd[0], b.data.simd[0]),
                         _mm_max_epi32(a.data.simd[1], b.data.simd[1]),
                         _mm_max_epi32(a.data.simd[2], b.data.simd[2]),
                         _mm_max_epi32(a.data.simd[3], b.data.simd[3]));
        }
    };

    struct plus
    {
        m512i operator() (const m512i& a, const m512i& b)
        {
            return m512i(_mm_add_epi32(a.data.simd[0], b.data.simd[0]),
                         _mm_add_epi32(a.data.simd[1], b.data.simd[1]),
                         _mm_add_epi32(a.data.simd[2], b.data.simd[2]),
                         _mm_add_epi32(a.data.simd[3], b.data.simd[3]));
        }
    };
    struct minus
    {
        m512i operator() (const m512i& a, const m512i& b)
        {
            return m512i(_mm_sub_epi32(a.data.simd[0], b.data.simd[0]),
                         _mm_sub_epi32(a.data.simd[1], b.data.simd[1]),
                         _mm_sub_epi32(a.data.simd[2], b.data.simd[2]),
                         _mm_sub_epi32(a.data.simd[3], b.data.simd[3]));
        }
    };

public:     // Non-member (friend) functions

    friend std::ostream& operator << (std::ostream& stream, const m512i& vec)
    {
        stream << vec[0] << "\t" << vec[1] << "\t" << vec[2] << "\t" << vec[3]
               << "\t" << vec[4] << "\t" << vec[5] << "\t" << vec[6] << "\t"
               << vec[7] << "\t" << vec[8] << "\t" << vec[9] << "\t" << vec[10]
               << "\t" << vec[11] << "\t" << vec[12] << "\t" << vec[13] << "\t"
               << vec[14] << "\t" << vec[15];

        return stream;
    }

    friend m512i operator + (const m512i &l, const m512i &r)
    {
        return m512i(_mm_add_epi32(l.data.simd[0], r.data.simd[0]),
                     _mm_add_epi32(l.data.simd[1], r.data.simd[1]),
                     _mm_add_epi32(l.data.simd[2], r.data.simd[2]),
                     _mm_add_epi32(l.data.simd[3], r.data.simd[3]));
    }

    friend m512i operator - (const m512i &l, const m512i &r)
    {
        return m512i(_mm_sub_epi32(l.data.simd[0], r.data.simd[0]),
                     _mm_sub_epi32(l.data.simd[1], r.data.simd[1]),
                     _mm_sub_epi32(l.data.simd[2], r.data.simd[2]),
                     _mm_sub_epi32(l.data.simd[3], r.data.simd[3]));
    }

    /// Multiply: overflow returns the low bits of the 32-bit result.
    friend m512i operator * (const m512i &l, const m512i &r)
    {
        return m512i(_mm_mullo_epi32(l.data.simd[0], r.data.simd[0]),
                     _mm_mullo_epi32(l.data.simd[1], r.data.simd[1]),
                     _mm_mullo_epi32(l.data.simd[2], r.data.simd[2]),
                     _mm_mullo_epi32(l.data.simd[3], r.data.simd[3]));
    }

    friend m512i operator & (const m512i& l, const m512i& r)
    {
        return m512i(_mm_and_si128(l.data.simd[0], r.data.simd[0]),
                     _mm_and_si128(l.data.simd[1], r.data.simd[1]),
                     _mm_and_si128(l.data.simd[2], r.data.simd[2]),
                     _mm_and_si128(l.data.simd[3], r.data.simd[3]));
    }

    friend m512i operator | (const m512i& l, const m512i& r)
    {
        return m512i(_mm_or_si128(l.data.simd[0], r.data.simd[0]),
                     _mm_or_si128(l.data.simd[1], r.data.simd[1]),
                     _mm_or_si128(l.data.simd[2], r.data.simd[2]),
                     _mm_or_si128(l.data.simd[3], r.data.simd[3]));
    }
    
    friend m512i operator ^ (const m512i& l, const m512i& r)
    {
        return m512i(_mm_xor_si128(l.data.simd[0], r.data.simd[0]),
                     _mm_xor_si128(l.data.simd[1], r.data.simd[1]),
                     _mm_xor_si128(l.data.simd[2], r.data.simd[2]),
                     _mm_xor_si128(l.data.simd[3], r.data.simd[3])
                );
    }

    m512i lshift(const uint8_t count) const
    {
        return m512i(_mm_slli_epi32(data.simd[0], count),
                     _mm_slli_epi32(data.simd[1], count),
                     _mm_slli_epi32(data.simd[2], count),
                     _mm_slli_epi32(data.simd[3], count)
                );
    }

    m512i lshift(const m512i& count) const
    {
        // This is the desired intrinsic for _m128 data, but for whatever
        // reason it didn't make it into the simd instruciton set until AVX2
        //return m512i(_mm_sllv_epi32(data[1], count),
        //             _mm_sllv_epi32(data[2], count),
        //             _mm_sllv_epi32(data[3], count),
        //             _mm_sllv_epi32(data[4], count)
        //        );
        m512i ret;
        for (size_t i = 0; i < size(); ++i)
        {
            auto val = static_cast<uint32_t>((*this)[i]);
            ret.data.raw[i] = static_cast<int32_t>(val << count[i]);
        }
        return ret;
    }

    m512i rshift(const uint8_t count) const
    {
        return m512i(_mm_srli_epi32(data.simd[0], count),
                     _mm_srli_epi32(data.simd[1], count),
                     _mm_srli_epi32(data.simd[2], count),
                     _mm_srli_epi32(data.simd[3], count)
                );
    }

    m512i rshift(const m512i& count) const
    {
        // This is the desired intrinsic for _m128 data, but for whatever
        // reason it didn't make it into the simd instruciton set until AVX2
        //return m512i(_mm_srlv_epi32(data[1], count),
        //             _mm_srlv_epi32(data[2], count),
        //             _mm_srlv_epi32(data[3], count),
        //             _mm_srlv_epi32(data[4], count)
        //        );
        m512i ret;
        for (size_t i = 0; i < size(); ++i)
        {
            auto val = static_cast<uint32_t>((*this)[i]);
            ret.data.raw[i] = static_cast<int32_t>(val >> count[i]);
        }
        return ret;
    }

    friend m512b operator < (const m512i &l, const m512i &r)
    {
        return m512b(_mm_castsi128_ps(_mm_cmplt_epi32(l.data.simd[0], r.data.simd[0])),
                     _mm_castsi128_ps(_mm_cmplt_epi32(l.data.simd[1], r.data.simd[1])),
                     _mm_castsi128_ps(_mm_cmplt_epi32(l.data.simd[2], r.data.simd[2])),
                     _mm_castsi128_ps(_mm_cmplt_epi32(l.data.simd[3], r.data.simd[3])));
    }

    friend m512b operator <= (const m512i &l, const m512i &r)
    {
        return !(l > r);
    }

    friend m512b operator > (const m512i &l, const m512i &r)
    {
        return m512b(_mm_castsi128_ps(_mm_cmpgt_epi32(l.data.simd[0], r.data.simd[0])),
                     _mm_castsi128_ps(_mm_cmpgt_epi32(l.data.simd[1], r.data.simd[1])),
                     _mm_castsi128_ps(_mm_cmpgt_epi32(l.data.simd[2], r.data.simd[2])),
                     _mm_castsi128_ps(_mm_cmpgt_epi32(l.data.simd[3], r.data.simd[3])));
    }

    friend m512b operator >= (const m512i &l, const m512i &r)
    {
        return !(l < r);
    }

    friend m512b operator == (const m512i &l, const m512i &r)
    {
        return m512b(_mm_castsi128_ps(_mm_cmpeq_epi32(l.data.simd[0], r.data.simd[0])),
                     _mm_castsi128_ps(_mm_cmpeq_epi32(l.data.simd[1], r.data.simd[1])),
                     _mm_castsi128_ps(_mm_cmpeq_epi32(l.data.simd[2], r.data.simd[2])),
                     _mm_castsi128_ps(_mm_cmpeq_epi32(l.data.simd[3], r.data.simd[3])));
    }

    friend m512b operator != (const m512i &l, const m512i &r)
    {
        return !(l == r);
    }

    friend m512i min(const m512i& a, const m512i&b)
    {
        return m512i(_mm_min_epi32(a.data.simd[0], b.data.simd[0]),
                     _mm_min_epi32(a.data.simd[1], b.data.simd[1]),
                     _mm_min_epi32(a.data.simd[2], b.data.simd[2]),
                     _mm_min_epi32(a.data.simd[3], b.data.simd[3]));
    }

    friend m512i max(const m512i& a, const m512i&b)
    {
        return m512i(_mm_max_epi32(a.data.simd[0], b.data.simd[0]),
                     _mm_max_epi32(a.data.simd[1], b.data.simd[1]),
                     _mm_max_epi32(a.data.simd[2], b.data.simd[2]),
                     _mm_max_epi32(a.data.simd[3], b.data.simd[3]));
    }

    friend int reduceMax(const m512i& a)
    {
        int ret = std::numeric_limits<int>::min();
        for (size_t i = 0; i < size(); ++i)
        {
            ret = std::max(ret, a[i]);
        }
        return ret;
    }

    friend int reduceMin(const m512i& a)
    {
        int ret = std::numeric_limits<int>::max();
        for (size_t i = 0; i < size(); ++i)
        {
            ret = std::min(ret, a[i]);
        }
        return ret;
    }

    friend m512i IndexOfMax(const m512f& nextVal, const m512i& nextIdx, m512f* curVal, const m512i& curIdx)
    {
        const auto maskf = m512f(nextVal > *curVal);
        const auto maski = m512i(nextVal > *curVal);

        *curVal = m512f(_mm_blendv_ps(curVal->data1(), nextVal.data1(), maskf.data1()),
                        _mm_blendv_ps(curVal->data2(), nextVal.data2(), maskf.data2()),
                        _mm_blendv_ps(curVal->data3(), nextVal.data3(), maskf.data3()),
                        _mm_blendv_ps(curVal->data4(), nextVal.data4(), maskf.data4()));

        return m512i(_mm_blendv_si128(curIdx.data.simd[0], nextIdx.data.simd[0], maski.data.simd[0]),
                        _mm_blendv_si128(curIdx.data.simd[1], nextIdx.data.simd[1], maski.data.simd[1]),
                        _mm_blendv_si128(curIdx.data.simd[2], nextIdx.data.simd[2], maski.data.simd[2]),
                        _mm_blendv_si128(curIdx.data.simd[3], nextIdx.data.simd[3], maski.data.simd[3]));
    }

    friend void IndexOfMin(const m512f& nextVal, const m512i& nextIdx,
                           const m512f& curVal, const m512i& curIdx,
                           const m512i& nextMaxIdx, const m512i& curMaxIdx,
                           m512i* newIdx, m512i* newMaxIdx)
    {
        const auto mask = m512i(nextVal <= curVal);
        *newIdx = m512i(_mm_blendv_si128(curIdx.data.simd[0], nextIdx.data.simd[0], mask.data.simd[0]),
                            _mm_blendv_si128(curIdx.data.simd[1], nextIdx.data.simd[1], mask.data.simd[1]),
                            _mm_blendv_si128(curIdx.data.simd[2], nextIdx.data.simd[2], mask.data.simd[2]),
                            _mm_blendv_si128(curIdx.data.simd[3], nextIdx.data.simd[3], mask.data.simd[3]));
        *newMaxIdx = m512i(_mm_blendv_si128(curMaxIdx.data.simd[0], nextMaxIdx.data.simd[0], mask.data.simd[0]),
                            _mm_blendv_si128(curMaxIdx.data.simd[1], nextMaxIdx.data.simd[1], mask.data.simd[1]),
                            _mm_blendv_si128(curMaxIdx.data.simd[2], nextMaxIdx.data.simd[2], mask.data.simd[2]),
                            _mm_blendv_si128(curMaxIdx.data.simd[3], nextMaxIdx.data.simd[3], mask.data.simd[3]));
    }

    friend m512i Blend(const m512b& mask, const m512i& success, const m512i& failure)
    {
        const m512i m = m512i(mask);
        return m512i(_mm_blendv_si128(failure.data.simd[0], success.data.simd[0], m.data.simd[0]),
                     _mm_blendv_si128(failure.data.simd[1], success.data.simd[1], m.data.simd[1]),
                     _mm_blendv_si128(failure.data.simd[2], success.data.simd[2], m.data.simd[2]),
                     _mm_blendv_si128(failure.data.simd[3], success.data.simd[3], m.data.simd[3]));
    }

    friend m512i inc(const m512i& a, const m512b& maskB)
    {
        const m512i b = a + 1;
        return Blend(maskB, b, a);
    }
};

inline m512i floorCastInt(const m512f& f)
{
    return m512i(_mm_cvtps_epi32(_mm_floor_ps(f.data1()))
                ,_mm_cvtps_epi32(_mm_floor_ps(f.data2()))
                ,_mm_cvtps_epi32(_mm_floor_ps(f.data3()))
                ,_mm_cvtps_epi32(_mm_floor_ps(f.data4())));
}

}}      // namespace PacBio::Simd

#endif  // mongo_common_simd_m512i_SSE_H_
