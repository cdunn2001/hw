#ifndef mongo_common_simd_m512f_LOOP_H_
#define mongo_common_simd_m512f_LOOP_H_

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
/// \file   m512f_LOOP.h
/// \brief  raw loop based (non-simd) implementation for 16 packed floats


#include <cassert>
#include <cstring>
#include <immintrin.h>
#include <limits>
#include <ostream>

#include "m512b_LOOP.h"
#include "xcompile.h"

namespace PacBio {
namespace Simd {

// SIMD 512-bit float vector (16 packed 32-bit float values).
CLASS_ALIGNAS(EIGEN_SIMD_SIZE) m512f
{
    static constexpr size_t vecLen = 16;
public:     // Types
    typedef m512f type;

    using Iterator = float*;
    using ConstIterator = const float*;

public:     // static constants
    // Number of floats in one m512f.
    static constexpr size_t size()
    {
        return sizeof(m512f) / sizeof(float);
    }

public:    // Implementation
    float data[vecLen];

public:     // Structors
    // Purposefully do not initialize data.
    m512f() {}
        
    // Replicate scalar x across data.
    m512f(float x)
    {
        std::fill(data, data+vecLen, x);
    }

    // Convenient for initializing from an simple integer, e.g., m512f(0).
    // Eigen 3.2.5 uses such expressions, which are ambiguous without this constructor.
    m512f(int x) : m512f(static_cast<float>(x))
    {}

    // Load x from pointer px. px must be aligned to 16 bytes.
    m512f(const float *px)
    {
        std::memcpy(data, px, vecLen*sizeof(float));
    }

    // Copy constructor
    m512f(const m512f& v) = default;

    // Construct from 16 floats
    m512f(float f0, float f1, float f2, float f3, float f4, float f5, float f6,
          float f7, float f8, float f9, float f10, float f11, float f12,
          float f13, float f14, float f15)
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

public:     // Assignment
    m512f& operator=(const m512f& x) = default;

    // Assignment from scalar value
    m512f& operator=(float fv)
    {
        std::fill(data, data+vecLen, fv);
        return *this;
    }

    // Compound assignment operators
    m512f& operator+=(const m512f& x)
    {
        for (unsigned int i = 0; i < vecLen; ++i)
        {
            data[i] += x.data[i];
        }
        return *this;
    }

    m512f& operator-=(const m512f& x)
    {
        for (unsigned int i = 0; i < vecLen; ++i)
        {
            data[i] -= x.data[i];
        }
        return *this;
    }

    m512f& operator*=(const m512f& x)
    {
        for (unsigned int i = 0; i < vecLen; ++i)
        {
            data[i] *= x.data[i];
        }
        return *this;
    }

    m512f& operator/=(const m512f& x)
    {
        for (unsigned int i = 0; i < vecLen; ++i)
        {
            data[i] /= x.data[i];
        }
        return *this;
    }

    // Unary minus operator.
    m512f operator-() const
    {
        m512f ret(0);
        return ret - *this;
    }

    // Return a scalar value
    float operator[](unsigned int i) const
    {
        assert(static_cast<size_t>(i) < this->size());
        return data[i];
    }

public:     // Functor types 

    struct minOp
    {
        m512f operator() (const m512f& a, const m512f& b)
        {
            m512f ret;
            for (unsigned int i = 0; i < vecLen; ++i)
            {
                ret.data[i] = std::min(a.data[i], b.data[i]);
            }
            return ret;
        }
    };

    struct maxOp
    {
        m512f operator() (const m512f& a, const m512f& b)
        {
            m512f ret;
            for (unsigned int i = 0; i < vecLen; ++i)
            {
                ret.data[i] = std::max(a.data[i], b.data[i]);
            }
            return ret;
        }
    };

public:     // Non-member (friend) functions

    friend std::ostream& operator << (std::ostream& stream, const m512f& vec)
    {
        stream << vec[0] << "\t" << vec[1] << "\t" << vec[2] << "\t" << vec[3]
               << "\t" << vec[4] << "\t" << vec[5] << "\t" << vec[6] << "\t"
               << vec[7] << "\t" << vec[8] << "\t" << vec[9] << "\t" << vec[10]
               << "\t" << vec[11] << "\t" << vec[12] << "\t" << vec[13] << "\t"
               << vec[14] << "\t" << vec[15];

        return stream;
    }

    friend m512b operator < (const m512f &l, const m512f &r)
    {
        m512b ret;
        for (unsigned int i = 0; i < vecLen; ++i)
        {
            ret[i] = l.data[i] < r.data[i];
        }
        return ret;
    }

    friend m512b operator <= (const m512f &l, const m512f &r)
    {
        return !(l > r);
    }

    friend m512b operator > (const m512f &l, const m512f &r)
    {
        m512b ret;
        for (unsigned int i = 0; i < vecLen; ++i)
        {
            ret[i] = l.data[i] > r.data[i];
        }
        return ret;
    }

    friend m512b operator >= (const m512f &l, const m512f &r)
    {
        return !(l < r);
    }

    friend m512b operator == (const m512f &l, const m512f &r)
    {
        m512b ret;
        for (unsigned int i = 0; i < vecLen; ++i)
        {
            ret[i] = l.data[i] == r.data[i];
        }
        return ret;
    }

    friend m512b operator != (const m512f &l, const m512f &r)
    {
        return !(l == r);
    }

    friend m512f min(const m512f& a, const m512f&b)
    {
        m512f ret;
        for (unsigned int i = 0; i < vecLen; ++i)
        {
            ret.data[i] = std::min(a.data[i], b.data[i]);
        }
        return ret;
    }

    friend m512f max(const m512f& a, const m512f&b)
    {
        m512f ret;
        for (unsigned int i = 0; i < vecLen; ++i)
        {
            ret.data[i] = std::max(a.data[i], b.data[i]);
        }
        return ret;
    }

    friend float reduceMax(const m512f& a)
    {
        float ret = -std::numeric_limits<float>::max();
        for (size_t i = 0; i < size(); ++i)
        {
            ret = std::max(ret, a.data[i]);
        }
        return ret;
    }

    friend float reduceMin(const m512f& a)
    {
        float ret = std::numeric_limits<float>::max();
        for (size_t i = 0; i < size(); ++i)
        {
            ret = std::min(ret, a.data[i]);
        }
        return ret;
    }

    friend m512f operator + (const m512f &l, const m512f &r)
    {
        m512f ret;
        for (unsigned int i = 0; i < vecLen; ++i)
        {
            ret.data[i] = l.data[i] + r.data[i];
        }
        return ret;
    }

    friend m512f operator - (const m512f &l, const m512f &r)
    {
        m512f ret;
        for (unsigned int i = 0; i < vecLen; ++i)
        {
            ret.data[i] = l.data[i] - r.data[i];
        }
        return ret;
    }

    friend m512f operator * (const m512f &l, const m512f &r)
    {
        m512f ret;
        for (unsigned int i = 0; i < vecLen; ++i)
        {
            ret.data[i] = l.data[i] * r.data[i];
        }
        return ret;
    }

    friend m512f operator / (const m512f &l, const m512f &r)
    {
        m512f ret;
        for (unsigned int i = 0; i < vecLen; ++i)
        {
            ret.data[i] = l.data[i] / r.data[i];
        }
        return ret;
    }

    friend m512b isnan(const m512f& x)
    { 
        m512b ret;
        for (unsigned int i = 0; i < vecLen; ++i)
        {
            ret[i] = std::isnan(x.data[i]);
        }
        return ret;
    }

    friend m512b isnotnan(const m512f& x)
    {
        return !isnan(x);
    }

    friend m512b isinf(const m512f& x)
    {
        return !isfinite(x);
    }

    friend m512b isfinite(const m512f& x)
    {
        m512b ret;
        for (unsigned int i = 0; i < vecLen; ++i)
        {
            ret[i] = std::isfinite(x.data[i]);
        }
        return ret;
    }

    friend m512b isnormal(const m512f& x)
    {
        m512b ret;
        for (unsigned int i = 0; i < vecLen; ++i)
        {
            ret[i] = std::isnormal(x.data[i]);
        }
        return ret;
    }

    friend m512f sqrt(const m512f& x)
    {
        m512f ret;
        for (unsigned int i = 0; i < vecLen; ++i)
        {
            ret.data[i] = std::sqrt(x.data[i]);
        }
        return ret;
    }

    friend m512f square(const m512f& x)
    { return x * x; }

    friend m512f cos(const m512f& x)
    {
        m512f ret;
        for (unsigned int i = 0; i < vecLen; ++i)
        {
            ret.data[i] = std::cos(x.data[i]);
        }
        return ret;
    }

    friend m512f sin(const m512f& x)
    {
        m512f ret;
        for (unsigned int i = 0; i < vecLen; ++i)
        {
            ret.data[i] = std::sin(x.data[i]);
        }
        return ret;
    }

    friend m512f atan2(const m512f &y, const m512f &x)
    {
        m512f ret;
        for (unsigned int i = 0; i < vecLen; ++i)
        {
            ret.data[i] = std::atan2(y.data[i], x.data[i]);
        }
        return ret;
    }

    friend m512f log(const m512f &x)
    {
        m512f ret;
        for (unsigned int i = 0; i < vecLen; ++i)
        {
            ret.data[i] = std::log(x.data[i]);
        }
        return ret;
    }

    friend m512f log2(const m512f &x)
    {
        m512f ret;
        for (unsigned int i = 0; i < vecLen; ++i)
        {
            ret.data[i] = std::log2(x.data[i]);
        }
        return ret;
    }

    friend m512f exp(const m512f &x)
    {
        m512f ret;
        for (unsigned int i = 0; i < vecLen; ++i)
        {
            ret.data[i] = std::exp(x.data[i]);
        }
        return ret;
    }

    friend m512f exp2(const m512f& x)
    {
        m512f ret;
        for (unsigned int i = 0; i < vecLen; ++i)
        {
            ret.data[i] = std::exp2(x.data[i]);
        }
        return ret;
    }

    friend m512f expSuperFast(const m512f& x)
    {

        return exp(x);
    }

    friend m512f erf(const m512f &x)
    {
        m512f ret;
        for (unsigned int i = 0; i < vecLen; ++i)
        {
            ret.data[i] = std::erf(x.data[i]);
        }
        return ret;
    }

    /// Complementary error function.
    friend m512f erfc(const m512f& x)
    {
        m512f ret;
        for (unsigned int i = 0; i < vecLen; ++i)
        {
            ret.data[i] = std::erfc(x.data[i]);
        }
        return ret;
    }

    friend m512f erfDiff(const m512f &x, const m512f &y)
    {
        return erf(x) - erf(y);
    }

    friend m512f singleEntropy(const m512f &x)
    {
        const auto zero = m512f(0);
        const auto mask = x == zero;
        // const auto mask = m512f((x == zero) & m512f(-1));
        const auto ll = x * log(x);
        // Is this ordered correctly?
        return Blend(mask, ll, zero);
    }

    friend m512f Blend(const m512b& mask, const m512f& success, const m512f& failure)
    {
        m512f ret;
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

    friend m512f fmadd(const m512f& mul1, const m512f& mul2, const m512f& add, const m512b& mask)
    {
        const auto zero      = m512f(0);
        return Blend(mask, mul1 * mul2, zero) + add;
    }

    friend m512f add(const m512f& a, const m512f& b, const m512b& mask)
    {
        return Blend(mask, a + b, a);
    }

    friend m512f inc(const m512f& a, const m512b& mask)
    {
        return Blend(mask, a + m512f(1) , a);
    }
};

}}      // namespace PacBio::Simd

#endif  // mongo_common_simd_m512f_LOOP_H_
