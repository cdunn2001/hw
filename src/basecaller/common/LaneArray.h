#ifndef mongo_common_LaneArray_H_
#define mongo_common_LaneArray_H_

#include <algorithm>
#include <type_traits>
#include <boost/operators.hpp>

#include <common/MongoConstants.h>
#include <common/simd/SimdConvTraits.h>
#include <common/simd/SimdTypeTraits.h>
#include <common/cuda/utility/CudaArray.h>

#include "LaneArrayRef.h"
#include "LaneMask.h"

namespace PacBio {
namespace Mongo {

/// A fixed-size array type that supports elementwise arithmetic operations.
template <typename T, unsigned int N = laneSize>
class LaneArray : public LaneArrayRef<T, N>
{
    // Static assertions that enable efficient SIMD and CUDA implementations.
    static_assert(std::is_same<T, short>::value
                  || std::is_same<T, int>::value
                  || std::is_same<T, float>::value,
                  "First template argument must be short, int, or float.");
    static constexpr auto laneUnit = std::max<unsigned int>(cudaThreadsPerWarp,
                                                            Simd::SimdTypeTraits<Simd::m512i>::width) * 4u / sizeof(T);
    static_assert(N != 0, "Second template argument cannot be 0.");
    static_assert(N % laneUnit == 0u, "Bad LaneArray size.");

public:     // Types
    using Super = LaneArrayRef<T, N>;
    using Super2 = typename Super::Super;

public:     // Structors and assignment
    LaneArray() : Super(nullptr)
    { Super::SetBasePointer(data_); }

    LaneArray(const LaneArray& other)
        : Super(nullptr)
    {
        Super::SetBasePointer(data_);
        std::copy(other.begin(), other.end(), begin());
    }

    explicit LaneArray(const Super2& other)
        : Super(nullptr)
    {
        this->SetBasePointer(data_);
        std::copy(other.begin(), other.end(), this->begin());
    }

    /// Broadcasting constructor supports implicit conversion of scalar value
    /// to uniform vector.
    LaneArray(const T& val)
        : Super(nullptr)
    {
        this->SetBasePointer(data_);
        std::fill(begin(), end(), val);
    }

    LaneArray& operator=(const LaneArray& that)
    {
        Super::operator=(that);
        return *this;
    }

    LaneArray& operator=(const Super2& that)
    {
        Super::operator=(that);
        return *this;
    }

public:     // Iterators
    using Super::begin;
    using Super::end;

public:     // Export
    LaneArray<float, N> AsFloat() const
    {
        LaneArray<float, N> ret;
        for (unsigned int i = 0; i < N; ++i)
        {
            ret[i] = static_cast<float>(data_[i]);
        }
        return ret;
    }

    LaneArray<short, N> AsShort() const
    {
        LaneArray<short, N> ret;
        for (unsigned int i = 0; i < N; ++i)
        {
            ret[i] = static_cast<short>(data_[i]);
        }
        return ret;
    }

    Cuda::Utility::CudaArray<T, N> AsCudaArray() const
    {
        return Cuda::Utility::CudaArray<T, N>(data_);
    }

public:     // Comparison operators
    friend LaneMask<N> operator==(const LaneArray& lhs, const LaneArray& rhs)
    {
        LaneMask<N> ret;
        for (unsigned int i = 0; i < N; ++i)
        {
            ret[i] = lhs[i] == rhs[i];
        }
        return ret;
    }

    friend LaneMask<N> operator<(const LaneArray& lhs, const LaneArray& rhs)
    {
        LaneMask<N> ret;
        for (unsigned int i = 0; i < N; ++i)
        {
            ret[i] = lhs[i] < rhs[i];
        }
        return ret;
    }

    friend LaneMask<N> operator<=(const LaneArray& lhs, const LaneArray& rhs)
    {
        return !(lhs > rhs);
    }

    friend LaneMask<N> operator>(const LaneArray& lhs, const LaneArray& rhs)
    {
        LaneMask<N> ret;
        for (unsigned int i = 0; i < N; ++i)
        {
            ret[i] = lhs[i] > rhs[i];
        }
        return ret;
    }

    friend LaneMask<N> operator>=(const LaneArray& lhs, const LaneArray& rhs)
    {
        return !(lhs < rhs);
    }

public:     // Named binary operators
    friend LaneArray max(const LaneArray& a, const LaneArray& b)
    {
        LaneArray ret;
        for (unsigned int i = 0; i < N; ++i)
        {
            ret.data_[i] = std::max(a[i], b[i]);
        }
        return ret;
    }

    friend LaneArray min(const LaneArray& a, const LaneArray& b)
    {
        LaneArray ret;
        for (unsigned int i = 0; i < N; ++i)
        {
            ret.data_[i] = std::min(a[i], b[i]);
        }
        return ret;
    }

public:     // Functor types
    struct minOp
    {
        LaneArray operator()(const LaneArray& a, const LaneArray& b)
        { return min(a, b); }
    };

    struct maxOp
    {
        LaneArray operator()(const LaneArray& a, const LaneArray& b)
        { return max(a, b); }
    };

public:
    friend LaneMask<N> isnan(const LaneArray& a)
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

private:
    T data_[N];
};


// Binary operators
template <typename T, unsigned int N>
LaneArray<T, N> operator+(const LaneArrayRef<T, N>& lhs, const T& rhs)
{
    LaneArray<T, N> nrv (lhs);
    nrv += rhs;
    return nrv;
}

template <typename T, unsigned int N>
LaneArray<T, N> operator+(const LaneArrayRef<T, N>& lhs, const LaneArrayRef<T, N>& rhs)
{
    LaneArray<T, N> nrv (lhs);
    nrv += rhs;
    return nrv;
}

template <typename T, unsigned int N>
LaneArray<T, N> operator-(const LaneArrayRef<T, N>& lhs, const T& rhs)
{
    LaneArray<T, N> nrv (lhs);
    nrv -= rhs;
    return nrv;
}

template <typename T, unsigned int N>
LaneArray<T, N> operator-(const LaneArrayRef<T, N>& lhs, const LaneArrayRef<T, N>& rhs)
{
    LaneArray<T, N> nrv (lhs);
    nrv -= rhs;
    return nrv;
}

template <typename T, unsigned int N>
LaneArray<T, N> operator*(const LaneArrayRef<T, N>& lhs, const T& rhs)
{
    LaneArray<T, N> nrv (lhs);
    nrv *= rhs;
    return nrv;
}

template <typename T, unsigned int N>
LaneArray<T, N> operator*(const LaneArrayRef<T, N>& lhs, const LaneArrayRef<T, N>& rhs)
{
    LaneArray<T, N> nrv (lhs);
    nrv *= rhs;
    return nrv;
}

template <typename T, unsigned int N>
LaneArray<T, N> operator/(const LaneArrayRef<T, N>& lhs, const LaneArrayRef<T, N>& rhs)
{
    LaneArray<T, N> nrv (lhs);
    nrv /= rhs;
    return nrv;
}

template <typename T, unsigned int N>
typename std::enable_if<std::is_integral<T>::value, LaneArray<T, N>>::type
min(const LaneArray<float, N>& a, const LaneArray<T, N>& b)
{
    LaneArray<T, N> ret;
    for (unsigned int i = 0; i < N; ++i)
    {
        ret[i] = static_cast<T>(std::min(a[i], static_cast<float>(b[i])));
    }
    return ret;
}


template <typename T, unsigned int N>
typename std::enable_if<std::is_integral<T>::value, LaneArray<T, N>>::type
max(const LaneArray<float, N>& a, const LaneArray<T, N>& b)
{
    LaneArray<T, N> ret;
    for (unsigned int i = 0; i < N; ++i)
    {
        ret[i] = static_cast<T>(std::max(a[i], static_cast<float>(b[i])));
    }
    return ret;
}

template <typename T, unsigned int N>
LaneArray<T,N> Blend(const LaneMask<N>& tf, const LaneArray<T,N>& success, const LaneArray<T,N>& failure)
{
    LaneArray<T,N> ret;
    for (unsigned int i = 0; i < N; ++i)
    {
        ret[i] = tf[i] ? success[i] : failure[i];
    }
    return ret;
}

}}      // namespace PacBio::Mongo


// TODO: Seems like we might have a namespace wrinkle to iron out.
namespace PacBio {
namespace Simd {

template <typename T, unsigned int N>
struct SimdConvTraits<Mongo::LaneArray<T,N>>
{
    typedef Mongo::LaneMask<N> bool_conv;
    typedef Mongo::LaneArray<float,N> float_conv;
    typedef Mongo::LaneArray<int,N> index_conv;
    typedef Mongo::LaneArray<short,N> pixel_conv;
    typedef Mongo::LaneArray<T,N> union_conv;
};

}}   // namespace PacBio::Simd

#endif  // mongo_common_LaneArray_H_
