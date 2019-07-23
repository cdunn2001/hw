#ifndef mongo_common_LaneArrayRef_H_
#define mongo_common_LaneArrayRef_H_

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
//  Defines class templates LaneArrayRef and ConstLaneArrayRef.

#include <algorithm>
#include <cmath>
#include <limits>

#include "LaneMask.h"
#include "MongoConstants.h"

namespace PacBio {
namespace Mongo {

// The class-inheritance design of LaneArray, LaneArrayRef, and
// ConstLaneArrayRef was inspired by the Boost MultiArray library.

/// A statically-sized arithmetic array adapter.
/// Provides the non-mutating LaneArray interface over an externally managed
/// contiguous array of elements.
template <typename T, unsigned int N = laneSize>
class ConstLaneArrayRef
{
    // Static assertions that enable efficient SIMD and CUDA implementations.
    static_assert(sizeof(T) == 2u || sizeof(T) == 4u,
                  "Bad element size.");
    static_assert(N != 0, "Second template argument cannot be 0.");

    // This is a nuisance for unit tests.
//    static constexpr auto laneUnit = sizeof(int) / sizeof(T)
//            * std::max<unsigned int>(cudaThreadsPerWarp,
//                                     Simd::SimdTypeTraits<Simd::m512i>::width);
//    static_assert(N % laneUnit == 0u, "Bad LaneArray size.");

public:     // Types
    using ElementType = T;
    using SizeType = unsigned int;
    using ConstReference = const ElementType&;
    using ConstPointer = const ElementType*;
    using ConstIterator = const ElementType*;

public:     // Static functions
    static SizeType Size()
    { return N; }

public:     // Structors and assignment
    ConstLaneArrayRef() = delete;

    /// Create wrapper referring to \a data.
    ConstLaneArrayRef(ConstPointer data) : data_ (data) { }

    /// Create a wrapper referring to the same data as \a that.
    ConstLaneArrayRef(const ConstLaneArrayRef& that) = default;

    ~ConstLaneArrayRef() = default;

    /// No assignment.
    ConstLaneArrayRef& operator=(const ConstLaneArrayRef& that) = delete;

public:     // Random-access iterators
    ConstIterator begin() const { return data_; }
    ConstIterator end() const { return data_ + N; }

public:     // Element access
    ConstReference operator[](unsigned int i) const
    {
        assert(i < N);
        return data_[i];
    }

    ConstPointer Data() const
    { return data_; }

public:     // Element-wise comparison operators
    friend LaneMask<N> operator==(const ConstLaneArrayRef& lhs, const ConstLaneArrayRef& rhs)
    {
        LaneMask<N> ret;
        for (unsigned int i = 0; i < N; ++i)
        {
            ret[i] = lhs[i] == rhs[i];
        }
        return ret;
    }

    friend LaneMask<N> operator!=(const ConstLaneArrayRef& lhs, const ConstLaneArrayRef& rhs)
    {
        return !(lhs == rhs);
    }

    friend LaneMask<N> operator<(const ConstLaneArrayRef& lhs, const ConstLaneArrayRef& rhs)
    {
        LaneMask<N> ret;
        for (unsigned int i = 0; i < N; ++i)
        {
            ret[i] = lhs[i] < rhs[i];
        }
        return ret;
    }

    friend LaneMask<N> operator<=(const ConstLaneArrayRef& lhs, const ConstLaneArrayRef& rhs)
    {
        return !(lhs > rhs);
    }

    friend LaneMask<N> operator>(const ConstLaneArrayRef& lhs, const ConstLaneArrayRef& rhs)
    {
        LaneMask<N> ret;
        for (unsigned int i = 0; i < N; ++i)
        {
            ret[i] = lhs[i] > rhs[i];
        }
        return ret;
    }

    friend LaneMask<N> operator>=(const ConstLaneArrayRef& lhs, const ConstLaneArrayRef& rhs)
    {
        return !(lhs < rhs);
    }


public:     // Miscellaneous friend functions.
    friend LaneMask<N> isnan(const ConstLaneArrayRef& a)
    {
        LaneMask<N> ret(false);
        if (std::is_floating_point<T>::value)
        {
            for (unsigned int i = 0; i < N; ++i)
            {
                ret[i] = std::isnan(a[i]);
            }
        }
        return ret;
    }

    friend LaneMask<N> isfinite(const ConstLaneArrayRef& a)
    {
        LaneMask<N> r (true);
        if (std::is_floating_point<T>::value)
        {
            for (unsigned int i = 0; i < N; ++i) r[i] = std::isfinite(a[i]);
        }
        return r;
    }

    friend ElementType reduceMin(const ConstLaneArrayRef& a)
    {
        auto r = std::numeric_limits<ElementType>::max();
        for (unsigned int i = 0; i < N; ++i)
        {
            r = std::min(r, a[i]);
        }
        return r;
    }

    friend ElementType reduceMax(const ConstLaneArrayRef& a)
    {
        auto r = std::numeric_limits<ElementType>::lowest();
        for (unsigned int i = 0; i < N; ++i)
        {
            r = std::max(r, a[i]);
        }
        return r;
    }

protected:
    void SetBasePointer(ConstPointer dataPtr)
    { data_ = dataPtr; }

    // Only to be used by subclass LaneArrayRef.
    // When used, data_ must ultimately be a copy of a T*.
    T* MutableData()
    { return const_cast<T*>(data_); }

private:
    ConstPointer data_;
};


// TODO: Provide a conversion from LaneArrayRef<const T> to ConstLaneArrayRef<T>.

/// A statically-sized arithmetic array adapter.
/// Provides the LaneArray interface over an externally managed contiguous
/// array of elements.
/// \note Though the user cannot mutate the referenced data via a const
/// LaneArrayRef, a const instance can be copied to a non-const one. For
/// const correctness, use ConstLaneArrayRef.
template <typename T, unsigned int N = laneSize>
class LaneArrayRef : public ConstLaneArrayRef<T, N>
{
public:     // Types
    using BaseConstRef = ConstLaneArrayRef<T, N>;
    using ElementType = T;
    using Reference = ElementType&;
    using ConstReference = const ElementType&;
    using Pointer = ElementType*;
    using ConstPointer = const ElementType*;
    using Iterator = ElementType*;
    using ConstIterator = const ElementType*;

public:     // Structors and assignment
    LaneArrayRef() = delete;

    /// Create a LaneArrayRef instance that refers to externally managed data
    /// located at \a data.
    // Note that this is where the Pointer is stored as ConstPointer in the
    // super object. We use Super:MutableData to get non-const access.
    LaneArrayRef(Pointer data) : BaseConstRef(data) { }

    /// Create a wrapper referring to the same data as \a that.
    /// \note Cannot create a LaneArrayRef from a ConstLaneArrayRef.
    LaneArrayRef(const LaneArrayRef& that) = default;

    ~LaneArrayRef() noexcept = default;

    /// Assign contained elements.
    LaneArrayRef& operator=(const LaneArrayRef& that)
    {
        std::copy(that.begin(), that.end(), begin());
        return *this;
    }

    /// Assign contained elements.
    template <typename U>
    LaneArrayRef& operator=(const ConstLaneArrayRef<U, N>& that)
    {
        std::copy(that.begin(), that.end(), begin());
        return *this;
    }

    LaneArrayRef& operator=(const ElementType& val)
    {
        std::fill(begin(), end(), val);
        return *this;
    }

public:     // Random-access iterators
    Iterator begin()  { return BaseConstRef::MutableData(); }
    Iterator end()  { return begin() + N; }

    ConstIterator begin() const  { return BaseConstRef::begin(); }
    ConstIterator end() const  { return BaseConstRef::end(); }

    ConstIterator cbegin() const  { return BaseConstRef::begin(); }
    ConstIterator cend() const  { return BaseConstRef::end(); }

public:     // Export
//    Cuda::Utility::CudaArray<T, N> AsCudaArray() const
//    {
//        return Cuda::Utility::CudaArray<T, N>(data_);
//    }

public:     // Element access
    Reference operator[](unsigned int i)
    {
        assert(i < N);
        return BaseConstRef::MutableData()[i];
    }

    using BaseConstRef::operator[];

public:
    // TODO: Do we want to return Pointer& in order to allow "rebinding".
    Pointer Data()
    { return BaseConstRef::MutableData(); }

public:     // Compound assigment
    LaneArrayRef& operator+=(const ElementType& a)
    {
        for (auto& x : *this) x += a;
        return *this;
    }

    LaneArrayRef& operator+=(const BaseConstRef& a)
    {
        for (unsigned int i = 0; i < N; ++i)
        {
            BaseConstRef::MutableData()[i] += a[i];
        }
        return *this;
    }

    LaneArrayRef& operator-=(const ElementType& a)
    {
        for (auto& x : *this) x -= a;
        return *this;
    }

    LaneArrayRef& operator-=(const BaseConstRef& a)
    {
        for (unsigned int i = 0; i < N; ++i)
        {
            BaseConstRef::MutableData()[i] -= a[i];
        }
        return *this;
    }

    LaneArrayRef& operator*=(const ElementType& a)
    {
        for (auto& x : *this) x *= a;
        return *this;
    }

    LaneArrayRef& operator*=(const BaseConstRef& a)
    {
        for (unsigned int i = 0; i < N; ++i)
        {
            BaseConstRef::MutableData()[i] *= a[i];
        }
        return *this;
    }

    LaneArrayRef& operator/=(const ElementType& a)
    {
        for (auto& x : *this) x /= a;
        return *this;
    }

    LaneArrayRef& operator/=(const BaseConstRef& a)
    {
        for (unsigned int i = 0; i < N; ++i)
        {
            BaseConstRef::MutableData()[i] /= a[i];
        }
        return *this;
    }

    LaneArrayRef& operator|=(const BaseConstRef& a)
    {
        auto* p = BaseConstRef::MutableData();
        for (unsigned int i = 0; i < N; ++i) p[i] |= a[i];
        return *this;
    }
};

}}      // namespace PacBio::Mongo

#endif  // mongo_common_LaneArrayRef_H_
