#ifndef mongo_common_simd_m512b_LOOP_H_
#define mongo_common_simd_m512b_LOOP_H_

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
/// \file   m512b_LOOP.h
/// \brief  raw loop based implementation (non-sse) for 16 packed bools

#include <cassert>
#include <immintrin.h>
#include <iostream>
#include <ostream>

#include "xcompile.h"

namespace PacBio {
namespace Simd {

// SIMD 512-bit bool vector (16 packed 1-bit bool values).
CLASS_ALIGNAS(EIGEN_SIMD_SIZE) m512b
{
    static constexpr size_t vecLen = 16;
public:     // Types
    typedef m512b type;

    using Iterator = bool*;
    using ConstIterator = const bool*;

public:     // Static constants
    /// The number of floats represented by one instance.
    static constexpr size_t size()
    {
        return vecLen;
    }

private:    // Implementation
    bool data[vecLen];

public:     // Structors
    // Purposefully do not initialize data.
    m512b() {}
    
    // Copy constructor
    m512b(const m512b& x) = default;


    m512b(bool x)
    {
        std::fill(data, data+vecLen, x);
    }

    //// Construct from 16 bools
    m512b(bool f0, bool f1, bool f2, bool f3, bool f4, bool f5, bool f6,
          bool f7, bool f8, bool f9, bool f10, bool f11, bool f12,
          bool f13, bool f14, bool f15)
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
    m512b& operator=(const m512b& x) = default;

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
        return data[i];
    }
    // Reference version not supplied by other versions of m512b and is not
    // suitable for use in general code. This is provided just for the other
    // m512*_LOOP.h classes, who may need to individually set elements.
    bool& operator[](unsigned int i) 
    {
        assert(static_cast<size_t>(i) < this->size());
        return data[i];
    }

public:  //member operators 

    m512b operator !()  const
    {
        m512b ret;
        for (unsigned int i = 0; i < vecLen; ++i)
        {
            ret.data[i] = !data[i];
        }
        return ret;
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
        m512b ret;
        for (unsigned int i = 0; i < vecLen; ++i)
        {
            ret.data[i] = l.data[i] & r.data[i];
        }
        return ret;
    }

    friend m512b operator | (const m512b &l, const m512b &r)
    { 
        m512b ret;
        for (unsigned int i = 0; i < vecLen; ++i)
        {
            ret.data[i] = l.data[i] | r.data[i];
        }
        return ret;
    }

    friend m512b operator ^ (const m512b &l, const m512b &r)
    { 
        m512b ret;
        for (unsigned int i = 0; i < vecLen; ++i)
        {
            ret.data[i] = l.data[i] ^ r.data[i];
        }
        return ret;
    }

    friend m512b operator==(const m512b& l, const m512b& r)
    {
        m512b ret;
        for (unsigned int i = 0; i < vecLen; ++i)
        {
            ret.data[i] = l.data[i] == r.data[i];
        }
        return ret;
    }

    friend bool any(const m512b& mask)
    {
        bool ret = false;
        for (unsigned int i = 0; i < vecLen; ++i)
        {
            ret |= mask.data[i];
        }
        return ret;
    }


    friend bool all(const m512b& mask)
    {
        bool ret = true;
        for (unsigned int i = 0; i < vecLen; ++i)
        {
            ret &= mask.data[i];
        }
        return ret;
    }

    friend bool none(const m512b& mask)
    {
        bool ret = true;
        for (unsigned int i = 0; i < vecLen; ++i)
        {
            ret &= !mask.data[i];
        }
        return ret;
    }
};

}}      // namespace PacBio::Simd

#endif  // mongo_common_simd_m512b_LOOP_H_
