#ifndef mongo_common_simd_m512s_LOOP_H_
#define mongo_common_simd_m512s_LOOP_H_

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
/// \file   m512s_LOOP.h
/// \brief  raw loop based (non-simd) implementation for 32 packed 16 bit shorts

#include <cassert>
#include <cstring>
#include <immintrin.h>
#include <ostream>

#include "m512f_LOOP.h"
#include "xcompile.h"

namespace PacBio {
namespace Simd {

// SIMD 512-bit short vector (32 packed 16-bit short int values).
CLASS_ALIGNAS(EIGEN_SIMD_SIZE) m512s
{
    static constexpr size_t vecLen = 32;
public:     // Types
    typedef m512s type;

    using Iterator = short*;
    using ConstIterator = const short*;

public:     // Static constants
    /// The number of floats represented by one instance.
    static constexpr size_t size()
    {
        return sizeof(m512s) / sizeof(short);
    }

    short data[vecLen];

public:     // Structors
    // Purposefully do not initialize data.
    m512s() {}

    // Replicate scalar x across data.
    m512s(short x)
    {
        std::fill(data, data+vecLen, x);
    }

    // Load x from pointer px. px must be aligned to 16 bytes.
    m512s(const short *px)
    {
        std::memcpy(data, px, vecLen*sizeof(short));
    }

    // Copy constructor
    m512s(const m512s& x) = default;

    m512s(const m512f& even, const m512f& odd)
    {
        for (unsigned int i = 0; i < vecLen / 2; ++i)
        {
            data[2*i] = static_cast<short>(even.data[i]);
            data[2*i+1] = static_cast<short>(odd.data[i]);
        }
    }

public:     // Assignment
    m512s& operator=(const m512s& x) = default;

    // Assignment from scalar value
    m512s& operator=(short x)
    {
        std::fill(data, data+vecLen, x);
        return *this;
    }

    // unsaturated addition
    m512s& AddWithRollOver(const m512s& x)
    {
        for (unsigned int i = 0; i < vecLen; ++i)
        {
            data[i] += x.data[i];
        }
        return *this;
    }

    // Return a scalar value
    short operator[](unsigned int i) const
    {
        assert(static_cast<size_t>(i) < this->size());
        return data[i];
    }

public:     // Functor types

    // Performs min on the first 16 elements and max on the last 16
    struct minmax
    {
        m512s operator() (const m512s& a, const m512s& b)
        {
            m512s ret;
            for (size_t i = 0; i < vecLen/2; ++i)
            {
                ret.data[i] = std::min(a[i], b[i]);
            }
            for (size_t i = vecLen/2; i < vecLen; ++i)
            {
                ret.data[i] = std::max(a[i], b[i]);
            }
            return ret;
        }
    };

    // Performs max on the first 16 elements and min on the last 16
    struct maxmin
    {
        m512s operator() (const m512s& a, const m512s& b)
        {
            m512s ret;
            for (size_t i = 0; i < vecLen/2; ++i)
            {
                ret.data[i] = std::max(a[i], b[i]);
            }
            for (size_t i = vecLen/2; i < vecLen; ++i)
            {
                ret.data[i] = std::min(a[i], b[i]);
            }
            return ret;
        }
    };

    struct minOp
    {
        m512s operator() (const m512s& a, const m512s& b)
        {
            m512s ret;
            for (unsigned int i = 0; i < vecLen; i++)
            {
                ret.data[i] = std::min(a.data[i], b.data[i]);
            }
            return ret;
        }
    };

    struct maxOp
    {
        m512s operator() (const m512s& a, const m512s& b)
        {
            m512s ret;
            for (unsigned int i = 0; i < vecLen; i++)
            {
                ret.data[i] = std::max(a.data[i], b.data[i]);
            }
            return ret;
        }
    };

public:     // Conversion methods

    /// Convert the even channel of an interleaved 2-channel layout to float.
    friend m512f Channel0(const m512s& in)
    {
        m512f ret;
        for (unsigned int i = 0; i < vecLen / 2; ++i)
        {
            ret.data[i] = in.data[2*i];
        }
        return ret;
    }

    /// Convert the odd channel of an interleaved 2-channel layout to float.
    friend m512f Channel1(const m512s& in)
    {
        // Convert the 32-bit representation of channel 0 to float
        m512f ret;
        for (unsigned int i = 0; i < vecLen / 2; ++i)
        {
            ret.data[i] = in.data[2*i+1];
        }
        return ret;
    }

    // Converts index 0-15 into a m512f
    friend m512f LowHalf(const m512s& in)
    {
        m512f ret;
        for (unsigned int i = 0; i < vecLen/2; ++i)
        {
            ret.data[i] = in.data[i];
        }
        return ret;
    }

    // Converts index 16-31 into a m512f
    friend m512f HighHalf(const m512s& in)
    {
        m512f ret;
        for (unsigned int i = 0; i < vecLen/2; ++i)
        {
            const auto j = i + 16;
            ret.data[i] = in.data[j];
        }
        return ret;
    }

public:     // Non-member (friend) functions

    friend std::ostream& operator << (std::ostream& stream, const m512s& vec)
    {
        const auto ch0 = Channel0(vec);
        const auto ch1 = Channel1(vec);

        stream << ch0[0] << ":" << ch1[0] << "\t" << ch0[1] << ":" << ch1[1]
               << "\t" << ch0[2] << ":" << ch1[2] << "\t" << ch0[3] << ":"
               << ch1[3] << "\t" << ch0[4] << ":" << ch1[4] << "\t" << ch0[5]
               << ":" << ch1[5] << "\t" << ch0[6] << ":" << ch1[6] << "\t"
               << ch0[7] << ":" << ch1[7] << "\t" << ch0[8] << ":" << ch1[8]
               << "\t" << ch0[9] << ":" << ch1[9] << "\t" << ch0[10] << ":"
               << ch1[10] << "\t" << ch0[11] << ":" << ch1[11] << "\t"
               << ch0[12] << ":" << ch1[12] << "\t" << ch0[13] << ":" << ch1[13]
               << "\t" << ch0[14] << ":" << ch1[14] << "\t" << ch0[15] << ":"
               << ch1[15];

        return stream;
    }

    friend m512s min(const m512s& a, const m512s&b)
    {
        m512s ret;
        for (unsigned int i = 0; i < vecLen; i++)
        {
            ret.data[i] = std::min(a.data[i], b.data[i]);
        }
        return ret;
    }

    friend m512s max(const m512s& a, const m512s&b)
    {
        m512s ret;
        for (unsigned int i = 0; i < vecLen; i++)
        {
            ret.data[i] = std::max(a.data[i], b.data[i]);
        }
        return ret;
    }

};

}}      // namespace PacBio::Simd

#endif  // mongo_common_simd_m512s_LOOP_H_
