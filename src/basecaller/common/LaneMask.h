#ifndef mongo_common_LaneMask_H_
#define mongo_common_LaneMask_H_

// Copyright (c) 2019, Pacific Biosciences of California, Inc.
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
//  Defines class template LaneMask.

namespace PacBio {
namespace Mongo {

/// A fixed-size array of boolean values with typical logical operators.
template <unsigned int N = laneSize>
class LaneMask : public boost::bitwise<LaneMask<N>>
{
    // Static assertions that enable efficient SIMD and CUDA implementations.
    static_assert(N != 0, "Template argument cannot be 0.");

    // This is a nuisance for unit tests.
//    static constexpr auto laneUnit = std::max<unsigned int>(cudaThreadsPerWarp,
//                                                            Simd::SimdTypeTraits<Simd::m512b>::width);
//    static_assert(N % laneUnit == 0u, "Bad LaneArray size.");

public:     // Structors and assignment
    LaneMask() = default;

    LaneMask(const LaneMask& tf) = default;

    LaneMask& operator=(const LaneMask& tf) = default;

    /// Broadcasting constructor supports implicit conversion of scalar value
    /// to uniform vector.
    LaneMask(bool tf)
    {
        std::fill(data_, data_+N, tf);
    }

public:     // Scalar access
    bool operator[](unsigned int i) const
    {
        assert(i < N);
        return data_[i];
    }

    bool& operator[](unsigned int i)
    {
        assert(i < N);
        return data_[i];
    }

public:     // Compound assignment
    // Boost provides the associated binary operators.
    LaneMask& operator|=(const LaneMask& a)
    {
        for (unsigned int i = 0; i < N; ++i)
        {
            this->data_[i] |= a[i];
        }
        return *this;
    }

    LaneMask& operator&=(const LaneMask& a)
    {
        for (unsigned int i = 0; i < N; ++i)
        {
            this->data_[i] &= a[i];
        }
        return *this;
    }

    LaneMask& operator^=(const LaneMask& a)
    {
        for (unsigned int i = 0; i < N; ++i)
        {
            this->data_[i] ^= a[i];
        }
        return *this;
    }

public:
    /// Returns a copy with each element negated.
    LaneMask operator!() const
    {
        LaneMask ret;
        for (unsigned int i = 0; i < N; ++i)
        {
            ret[i] = !data_[i];
        }
        return ret;
    }

public:     // Reductions
    friend bool all(const LaneMask& tf)
    {
        bool ret = true;
        for (unsigned int i = 0; i < N; ++i)
        {
            ret = ret && tf[i];
        }
        return ret;
    }

    friend bool any(const LaneMask& tf)
    {
        bool ret = false;
        for (unsigned int i = 0; i < N; ++i)
        {
            ret = ret || tf[i];
        }
        return ret;
    }

    friend bool none(const LaneMask& tf)
    {
        bool ret = true;
        for (unsigned int i = 0; i < N; ++i)
        {
            ret = ret && !tf[i];
        }
        return ret;
    }

private:
    bool data_[N];
};

}}      // namespace PacBio::Mongo

#endif  // mongo_common_LaneMask_H_
