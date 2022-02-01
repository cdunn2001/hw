#ifndef mongo_common_simd_m32s_H_
#define mongo_common_simd_m32s_H_

//  Copyright (c) 2011-2015, Pacific Biosciences of California, Inc.
//
//  All rights reserved.
//
//  Redistribution and use in source and binary forms, with or without
//  modification, are permitted provided that the following conditions are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of Pacific Biosciences nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.
//
//  NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
//  THIS LICENSE.  THIS SOFTWARE IS PROVIDED BY PACIFIC BIOSCIENCES AND ITS
//  CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
//  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
//  PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL PACIFIC BIOSCIENCES OR
//  ITS CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
//  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
//  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
//  BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
//  IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
//  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
//  POSSIBILITY OF SUCH DAMAGE.
//
//  Description:
/// \file  m32s.h
/// \brief A type representing two int16_t values packed into a 32-bit word,
///        suitable for representing the two pixels from a ZMW; provides
///        virtual "SIMD" functionality like the other true SIMD types

#include <cassert>

#include "m32f.h"
#include "SimdTypeTraits.h"
#include "xcompile.h"

namespace PacBio {
namespace Simd {

CLASS_ALIGNAS(4) m32s
{
private: // Data
    using ImplType = int16_t;
    int16_t p[2];

public: // Types
    typedef m32s type;

    using Iterator = int16_t*;
    using ConstIterator = const int16_t*;


public: // Static constants
    /// The number of shorts represented by one instance.
    static constexpr size_t size()
    {
        return sizeof(m32s) / sizeof(int16_t);
    }

public:  // Structors
    // (does not initialize)
    m32s() {}

    // Replicate a scalar value
    m32s(int16_t x) : p { x, x }  {}

    // Load from memory
    m32s(const int16_t *px) : p { *px, *(px + 1) } {}

    // Copy
    m32s(const m32s& x) = default;

    // Instantiate from pixel values
    // TODO: do we need "const short&" args here?
    m32s(int16_t even, int16_t odd) :  p { even, odd }  {}


public:  // Assignment
    m32s& operator=(const m32s& x) = default;

    // Assignment from scalar value
    m32s& operator=(int16_t x)
    {
        p[0] = x;
        p[1] = x;
        return *this;
    }

    // unsaturated subtraction
    m32s& AddWithRollOver(const m32s& x)
    {
        p[0] += x.p[0];
        p[1] += x.p[1];
        return *this;
    }

public:  // Scalar access
    // Reference a scalar value
    int16_t& operator[](unsigned int i)
    {
        assert(static_cast<size_t>(i) < this->size());
        return p[i];
    }

    // Return a scalar value
    int16_t operator[](unsigned int i) const
    {
        assert(static_cast<size_t>(i) < this->size());
        return p[i];
    }


public:  // Iterators
    Iterator begin()
    {
        return p;
    }

    Iterator end()
    {
        return begin() + this->size();
    }

    ConstIterator begin() const
    {
        return p;
    }

    ConstIterator end() const
    {
        return begin() + this->size();
    }

    ConstIterator cbegin()
    {
        return p;
    }

    ConstIterator cend()
    {
        return begin() + this->size();
    }


public:  // Functor types

    // Performs min on even elements and max on odd
    struct minmax
    {
        m32s operator() (const m32s& a, const m32s& b)
        {
            return m32s(std::min(a.p[0], b.p[0]),
                        std::max(a.p[1], b.p[1]));        }
    };

    // Performs max on even elements and min on odd
    struct maxmin
    {
        m32s operator() (const m32s& a, const m32s& b)
        {
            return m32s(std::max(a.p[0], b.p[0]),
                        std::min(a.p[1], b.p[1]));        }
    };

    struct minOp
    {
        m32s operator() (const m32s& a, const m32s& b)
        {
            return m32s(std::min(a.p[0], b.p[0]),
                        std::min(a.p[1], b.p[1]));        }
    };

    struct maxOp
    {
        m32s operator() (const m32s& a, const m32s& b)
        {
            return m32s(std::max(a.p[0], b.p[0]),
                        std::max(a.p[1], b.p[1]));        }
    };


public:  // Conversion methods

    friend float Channel0(const m32s& in)
    { return static_cast<float>(in.p[0]); }

    friend float Channel1(const m32s& in)
    { return static_cast<float>(in.p[1]); }

    friend float LowHalf(const m32s& in)
    { return static_cast<float>(in.p[0]); }

    friend float HighHalf(const m32s& in)
    { return static_cast<float>(in.p[1]); }

public:  // Non-member (friend) functions

    friend std::ostream& operator << (std::ostream& stream, const m32s& v)
    {
        stream << v.p[0] << ":" << v.p[1] << "\t";
        return stream;
    }

    friend m32s min(const m32s& l, const m32s&r)
    {
        return m32s(std::min(l.p[0], r.p[0]),
                    std::min(l.p[1], r.p[1]));
    }

    friend m32s max(const m32s& l, const m32s&r)
    {
        return m32s(std::max(l.p[0], r.p[0]),
                    std::max(l.p[1], r.p[1]));
    }

    static int16_t sadd16(int16_t a, int16_t b)
    {
        int32_t x = static_cast<int32_t>(a) + static_cast<int32_t>(b);
        if (x < -32768) x = -32768;
        else if (x > 32767) x = 32767;

        return static_cast<int16_t>(x);
    }
};


static_assert(sizeof(PacBio::Simd::m32s) == 4,
              "m32s is not being packed correctly by the compiler");

template<>
struct SimdTypeTraits<m32s>
{
    typedef short scalar_type;
    typedef float float_conv;
    typedef m32s index_conv;
    static const size_t width = 2;
};

}}      // namespace PacBio::Simd

#endif  // mongo_common_simd_m32s_H_
