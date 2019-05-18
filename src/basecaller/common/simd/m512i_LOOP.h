#ifndef mongo_common_simd_m512i_LOOP_H_
#define mongo_common_simd_m512i_LOOP_H_

// Copyright (c) 2017-2019, Pacific Biosciences of California, Inc.
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
/// \file   m512i_LOOP.h
/// \brief  raw loop based (non-simd) implementation for 16 packed 32 bit ints

#include <cassert>
#include <cmath>
#include <cstring>
#include <immintrin.h>
#include <ostream>
#include <smmintrin.h>

#include "m512f_LOOP.h"
#include "m512s_LOOP.h"
#include "xcompile.h"

namespace PacBio {
namespace Simd {

// SIMD 512-bit int vector (16 packed 32-bit int values).
CLASS_ALIGNAS(EIGEN_SIMD_SIZE) m512i
{
    static constexpr size_t vecLen = 16;
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

public:    // Implementation

    int data[vecLen];

public:     // Structors
    // Purposefully do not initialize data.
    m512i() {}

    // Replicate scalar x across data.
    m512i(int x)
    {
        std::fill(data, data+vecLen, x);
    }

    // Load x from pointer px. px must be aligned to 16 bytes.
    m512i(const int *px)
    {
        std::memcpy(data, px, vecLen*sizeof(int));
    }

    m512i(int f0, int f1, int f2, int f3, int f4, int f5, int f6,
          int f7, int f8, int f9, int f10, int f11, int f12,
          int f13, int f14, int f15)
    {
        data[0] = f0;
        data[1] = f1;
        data[2] = f2;
        data[3] = f3;
        data[4] = f4;
        data[5] = f5;
        data[6] = f6;
        data[7] = f7;
        data[8] = f8;
        data[9] = f9;
        data[10] = f10;
        data[11] = f11;
        data[12] = f12;
        data[13] = f13;
        data[14] = f14;
        data[15] = f15;
    }

    // Copy constructor
    m512i(const m512i& x) = default;

    // Construct from m128f vector type
    explicit m512i(const m512f& x)
    {
        for (unsigned int i = 0; i < vecLen; ++i)
        {
            data[i] = static_cast<int>(std::round(x.data[i]));
        }
    }

public:     // Export
    m512f AsFloat() const
    {
        m512f ret;
        for (unsigned int i = 0; i < vecLen; ++i)
        {
            ret.data[i] = static_cast<float>(data[i]);
        }
        return ret;
    }

    explicit operator m512f() const
    {
        return this->AsFloat();
    }

public:     // Assignment
    m512i& operator=(const m512i& x) = default;

    // Assignment from scalar value
    m512i& operator=(int x)
    {
        std::fill(data, data+vecLen, x);
        return *this;
    }

    // Compound assignment operators
    m512i& operator+=(const m512i& x)
    {
        for (unsigned int i = 0; i < vecLen; ++i)
        {
            data[i] += x.data[i];
        }
        return *this;
    }

    m512i& operator-=(const m512i& x)
    {
        for (unsigned int i = 0; i < vecLen; ++i)
        {
            data[i] -= x.data[i];
        }
        return *this;
    }

    m512i& operator*=(const m512i& x)
    {
        for (unsigned int i = 0; i < vecLen; ++i)
        {
            data[i] *= x.data[i];
        }
        return *this;
    }

    // Return a scalar value
    int operator[](unsigned int i) const
    {
        assert(static_cast<size_t>(i) < this->size());
        return data[i];
    }

public:     // Functor types

    struct minOp
    {
        m512i operator() (const m512i& a, const m512i& b)
        { return min(a, b); }
    };

    struct maxOp
    {
        m512i operator() (const m512i& a, const m512i& b)
        { return max(a, b); }
    };

    struct plus
    {
        m512i operator() (const m512i& a, const m512i& b)
        { return a + b; }
    };

    struct minus
    {
        m512i operator() (const m512i& a, const m512i& b)
        { return a - b; }
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
        m512i ret;
        for (unsigned int i = 0; i < vecLen; ++i)
        {
            ret.data[i] = l.data[i] + r.data[i];
        }
        return ret;
    }

    friend m512i operator - (const m512i &l, const m512i &r)
    {
        m512i ret;
        for (unsigned int i = 0; i < vecLen; ++i)
        {
            ret.data[i] = l.data[i] - r.data[i];
        }
        return ret;
    }

    /// Multiply: overflow returns the low bits of the 32-bit result.
    friend m512i operator * (const m512i &l, const m512i &r)
    {
        m512i ret;
        for (unsigned int i = 0; i < vecLen; ++i)
        {
            ret.data[i] = l.data[i] * r.data[i];
        }
        return ret;
    }

    friend m512i operator & (const m512i& l, const m512i& r)
    {
        m512i ret;
        for (unsigned int i = 0; i < vecLen; ++i)
        {
            ret.data[i] = l.data[i] & r.data[i];
        }
        return ret;
    }

    friend m512i operator | (const m512i& l, const m512i& r)
    {
        m512i ret;
        for (unsigned int i = 0; i < vecLen; ++i)
        {
            ret.data[i] = l.data[i] | r.data[i];
        }
        return ret;
    }

    friend m512i operator ^ (const m512i& l, const m512i& r)
    {
        m512i ret;
        for (unsigned int i = 0; i < vecLen; ++i)
        {
            ret.data[i] = l.data[i] ^ r.data[i];
        }
        return ret;
    }

    m512i lshift(const uint8_t count) const
    {
        m512i ret;
        for (unsigned int i = 0; i < vecLen; ++i)
        {
            ret.data[i] = data[i] << count;
        }
        return ret;
    }

    m512i lshift(const m512i& count) const
    {
        m512i ret;
        for (unsigned int i = 0; i < vecLen; ++i)
        {
            ret.data[i] = data[i] << count.data[i];
        }
        return ret;
    }

    m512i rshift(const uint8_t count) const
    {
        m512i ret;
        for (unsigned int i = 0; i < vecLen; ++i)
        {
            ret.data[i] = static_cast<uint32_t>(data[i]) >> count;
        }
        return ret;
    }

    m512i rshift(const m512i& count) const
    {
        m512i ret;
        for (unsigned int i = 0; i < vecLen; ++i)
        {
            ret.data[i] = static_cast<uint32_t>(data[i]) >> count.data[i];
        }
        return ret;
    }

    friend m512b operator < (const m512i &l, const m512i &r)
    {
        m512b ret;
        for (unsigned int i = 0; i < vecLen; ++i)
        {
            ret[i] = l.data[i] < r.data[i];
        }
        return ret;
    }

    friend m512b operator <= (const m512i &l, const m512i &r)
    {
        return !(l > r);
    }

    friend m512b operator > (const m512i &l, const m512i &r)
    {
        m512b ret;
        for (unsigned int i = 0; i < vecLen; ++i)
        {
            ret[i] = l.data[i] > r.data[i];
        }
        return ret;
    }

    friend m512b operator >= (const m512i &l, const m512i &r)
    {
        return !(l < r);
    }

    friend m512b operator == (const m512i &l, const m512i &r)
    {
        m512b ret;
        for (unsigned int i = 0; i < vecLen; ++i)
        {
            ret[i] = l.data[i] == r.data[i];
        }
        return ret;
    }

    friend m512b operator != (const m512i &l, const m512i &r)
    {
        return !(l == r);
    }

    friend m512i min(const m512i& a, const m512i&b)
    {
        m512i ret;
        for (unsigned int i = 0; i < vecLen; ++i)
        {
            ret.data[i] = std::min(a.data[i], b.data[i]);
        }
        return ret;
    }

    friend m512i max(const m512i& a, const m512i&b)
    {
        m512i ret;
        for (unsigned int i = 0; i < vecLen; ++i)
        {
            ret.data[i] = std::max(a.data[i], b.data[i]);
        }
        return ret;
    }

    friend int reduceMax(const m512i& a)
    {
        int ret = std::numeric_limits<int>::min();
        for (size_t i = 0; i < size(); ++i)
        {
            ret = std::max(ret, a.data[i]);
        }
        return ret;
    }

    friend int reduceMin(const m512i& a)
    {
        int ret = std::numeric_limits<int>::max();
        for (size_t i = 0; i < size(); ++i)
        {
            ret = std::min(ret, a.data[i]);
        }
        return ret;
    }

    friend m512i IndexOfMax(const m512f& nextVal, const m512i& nextIdx, m512f* curVal, const m512i& curIdx)
    {
        m512i ret;
        for (unsigned int i = 0; i < vecLen; ++i)
        {
            if (nextVal.data[i] > curVal->data[i])
            {
                ret.data[i] = nextIdx.data[i];
                curVal->data[i] = nextVal.data[i];
            } else {
                ret.data[i] = curIdx.data[i];
            }
        }
        return ret;
    }

    friend void IndexOfMin(const m512f& nextVal, const m512i& nextIdx,
                           const m512f& curVal, const m512i& curIdx,
                           const m512i& nextMaxIdx, const m512i& curMaxIdx,
                           m512i* newIdx, m512i* newMaxIdx)
    {
        for (unsigned int i = 0; i < vecLen; ++i)
        {
            if (nextVal.data[i] <= curVal.data[i])
            {
                newIdx->data[i] = nextIdx.data[i];
                newMaxIdx->data[i] = nextMaxIdx.data[i];
            } else {
                newIdx->data[i] = curIdx.data[i];
                newMaxIdx->data[i] = curMaxIdx.data[i];
            }
        }
    }

    friend m512i Blend(const m512b& mask, const m512i& success, const m512i& failure)
    {
        m512i ret;
        for (unsigned int i = 0; i < vecLen; ++i)
        {
            if (mask[i])
            {
                ret.data[i] = success.data[i];
            } else {
                ret.data[i] = failure.data[i];
            }
        }
        return ret;
    }

    friend m512i inc(const m512i& a, const m512b& maskB)
    {
        m512i ret = a;
        for (unsigned int i = 0; i < vecLen; ++i)
        {
            if(maskB[i])
                ret.data[i] += 1;
        }
        return ret;
    }

    // Creates an m512s that has the m512i data duplicated.  The replication is
    // compact (16 original followed by the 16 replica)
    friend m512s Replicate(const m512i& a)
    {
        m512s ret;
        static_assert(2 * size() == m512s::size(), "Unexpected size mismatch");
        for (unsigned int i = 0; i < size(); ++i)
        {
            assert(a[i] <= std::numeric_limits<short>::max());
            ret.data[i] = ret.data[i + size()] = static_cast<short>(a[i]);
        }
        return ret;
    }

};

inline m512i floorCastInt(const m512f& f)
{
    m512i ret;
    for (unsigned int i = 0; i < m512i::size(); ++i)
    {
        ret.data[i] = static_cast<int>(std::floor(f.data[i]));
    }
    return ret;
}

}}      // namespace PacBio::Simd

#endif  // mongo_common_simd_m512i_LOOP_H_
