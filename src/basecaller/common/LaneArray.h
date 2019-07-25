#ifndef mongo_common_LaneArray_H_
#define mongo_common_LaneArray_H_

#include <algorithm>
#include <type_traits>
#include <boost/operators.hpp>

#include <common/MongoConstants.h>
#include <common/cuda/utility/CudaArray.h>
#include <common/simd/SimdConvTraits.h>
#include <common/simd/SimdTypeTraits.h>

#include "LaneArrayRef.h"
#include "LaneMask.h"

namespace PacBio {
namespace Mongo {

/// A fixed-size array type that supports elementwise arithmetic operations.
template <typename T, unsigned int N = laneSize>
class LaneArray : public LaneArrayRef<T, N>
{
public:     // Types
    using BaseRef = LaneArrayRef<T, N>;
    using BaseConstRef = typename BaseRef::BaseConstRef;
    using ElementType = T;

public:     // Static constants
    static constexpr unsigned int size = N;

public:     // Structors and assignment
    LaneArray() : BaseRef(nullptr)
    { BaseRef::SetBasePointer(data_); }

    LaneArray(const LaneArray& other)
        : LaneArray()
    { std::copy(other.begin(), other.end(), begin()); }

    template <typename U>
    explicit LaneArray(const ConstLaneArrayRef<U, N>& other)
        : LaneArray()
    { std::copy(other.begin(), other.end(), this->begin()); }

    /// Broadcasting constructor supports implicit conversion of scalar value
    /// to uniform vector.
    LaneArray(const T& val)
        : LaneArray()
    { std::fill(begin(), end(), val); }

    template <typename InputIterT>
    LaneArray(const InputIterT& first, const InputIterT& last)
        : LaneArray()
    {
        assert(std::distance(first, last) == N);
        std::copy(first, last, begin());
    }

    // This could be accomplished with LaneArray(ConstLaneArrayRef(ca.data())),
    // but this convenience seems worth the dependency on CudaArray.h.
    // It's also safer since we ensure that ca has the correct size.
    // TODO: Should we enable implicit conversion (i.e., remove the explicit qualifier)?
    // TODO: Make the element type of CudaArray a template parameter.
    explicit LaneArray(const Cuda::Utility::CudaArray<T, N>& ca)
        : LaneArray(ca.begin(), ca.end())
    { }

    LaneArray& operator=(const LaneArray& that)
    {
        BaseRef::operator=(that);
        return *this;
    }

    LaneArray& operator=(const ElementType& val)
    {
        BaseRef::operator=(val);
        return *this;
    }

    template <typename U>
    LaneArray& operator=(const ConstLaneArrayRef<U, N>& that)
    {
        BaseRef::template operator=<U>(that);
        return *this;
    }

public:     // Iterators
    using BaseRef::begin;
    using BaseRef::end;
    using BaseRef::cbegin;
    using BaseRef::cend;

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

public:     // Named unary operators
    /// Square root
    friend LaneArray sqrt(const BaseConstRef& x)
    {
        LaneArray ret;
        for (unsigned int i = 0; i < N; ++i)
        {
            ret[i] = std::sqrt(x[i]);
        }
        return ret;
    }

    /// Absolute value
    friend LaneArray abs(const BaseConstRef& x)
    {
        LaneArray ret;
        for (unsigned int i = 0; i < N; ++i)
        {
            ret[i] = std::abs(x[i]);
        }
        return ret;
    }

    /// Natural exponential
    friend LaneArray exp(const BaseConstRef& x)
    {
        LaneArray ret;
        for (unsigned int i = 0; i < N; ++i) ret[i] = std::exp(x[i]);
        return ret;
    }

    /// Base-2 exponential
    friend LaneArray exp2(const BaseConstRef& x)
    {
        LaneArray ret;
        for (unsigned int i = 0; i < N; ++i)
        {
            ret[i] = std::exp2(x[i]);
        }
        return ret;
    }

    /// Natural logarithm
    friend LaneArray log(const BaseConstRef& x)
    {
        LaneArray ret;
        for (unsigned int i = 0; i < N; ++i) ret[i] = std::log(x[i]);
        return ret;
    }

    /// Base-2 logarithm
    friend LaneArray log2(const BaseConstRef& x)
    {
        LaneArray ret;
        for (unsigned int i = 0; i < N; ++i) ret[i] = std::log2(x[i]);
        return ret;
    }

    /// Complementary error function
    friend LaneArray erfc(const BaseConstRef& x)
    {
        LaneArray ret;
        for (unsigned int i = 0; i < N; ++i) ret[i] = std::erfc(x[i]);
        return ret;
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


/// Unary negation.
template <typename T, unsigned int N>
LaneArray<T, N> operator-(const ConstLaneArrayRef<T, N>& a)
{
    LaneArray<T, N> nrv;
    for (unsigned int i = 0; i < N; ++i) nrv[i] = -a[i];
    return nrv;
}

// Binary operators with uniform element type.
// TODO: Enable nonuniform types.
template <typename T, unsigned int N>
LaneArray<T, N> operator+(const ConstLaneArrayRef<T, N>& lhs, const T& rhs)
{
    LaneArray<T, N> nrv (lhs);
    nrv += rhs;
    return nrv;
}

template <typename T, unsigned int N>
LaneArray<T, N> operator+(const T& lhs, const ConstLaneArrayRef<T, N>& rhs)
{
    LaneArray<T, N> nrv (rhs);
    nrv += lhs;
    return nrv;
}

template <typename T, unsigned int N>
LaneArray<T, N> operator+(const ConstLaneArrayRef<T, N>& lhs, const ConstLaneArrayRef<T, N>& rhs)
{
    LaneArray<T, N> nrv (lhs);
    nrv += rhs;
    return nrv;
}

template <typename T, unsigned int N>
LaneArray<T, N> operator-(const ConstLaneArrayRef<T, N>& lhs, const T& rhs)
{
    LaneArray<T, N> nrv (lhs);
    nrv -= rhs;
    return nrv;
}

template <typename T, unsigned int N>
LaneArray<T, N> operator-(const T& lhs, const ConstLaneArrayRef<T, N>& rhs)
{
    LaneArray<T, N> nrv (lhs);
    nrv -= rhs;
    return nrv;
}

template <typename T, unsigned int N>
LaneArray<T, N> operator-(const ConstLaneArrayRef<T, N>& lhs, const ConstLaneArrayRef<T, N>& rhs)
{
    LaneArray<T, N> nrv (lhs);
    nrv -= rhs;
    return nrv;
}

template <typename T, unsigned int N>
LaneArray<T, N> operator*(const ConstLaneArrayRef<T, N>& lhs, const T& rhs)
{
    LaneArray<T, N> nrv (lhs);
    nrv *= rhs;
    return nrv;
}

template <typename T, unsigned int N>
LaneArray<T, N> operator*(const T& lhs, const ConstLaneArrayRef<T, N>& rhs)
{
    LaneArray<T, N> nrv (rhs);
    nrv *= lhs;
    return nrv;
}

template <typename T, unsigned int N>
LaneArray<T, N> operator*(const ConstLaneArrayRef<T, N>& lhs, const ConstLaneArrayRef<T, N>& rhs)
{
    LaneArray<T, N> nrv (lhs);
    nrv *= rhs;
    return nrv;
}

template <typename T, unsigned int N>
LaneArray<T, N> operator/(const ConstLaneArrayRef<T, N>& lhs, const T& rhs)
{
    LaneArray<T, N> nrv (lhs);
    nrv /= rhs;
    return nrv;
}

template <typename T, unsigned int N>
LaneArray<T, N> operator/(const T& lhs, const ConstLaneArrayRef<T, N>& rhs)
{
    LaneArray<T, N> nrv (lhs);
    nrv /= rhs;
    return nrv;
}

template <typename T, unsigned int N>
LaneArray<T, N> operator/(const ConstLaneArrayRef<T, N>& lhs, const ConstLaneArrayRef<T, N>& rhs)
{
    LaneArray<T, N> nrv (lhs);
    nrv /= rhs;
    return nrv;
}

template <typename T, unsigned int N>
LaneArray<T, N> operator|(const ConstLaneArrayRef<T, N>& lhs, const ConstLaneArrayRef<T, N>& rhs)
{
    LaneArray<T, N> nrv (lhs);
    nrv |= rhs;
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
LaneArray<T,N> Blend(const LaneMask<N>& tf,
                     const ConstLaneArrayRef<T,N>& success,
                     const ConstLaneArrayRef<T,N>& failure)
{
    LaneArray<T,N> ret;
    for (unsigned int i = 0; i < N; ++i)
    {
        ret[i] = tf[i] ? success[i] : failure[i];
    }
    return ret;
}

template <typename T, unsigned int N> inline
LaneArray<int, N> floorCastInt(const ConstLaneArrayRef<T, N>& f)
{
    static_assert(std::is_floating_point<T>::value, "T must be floating-point type.");
    LaneArray<int, N> ret;
    for (unsigned int i = 0; i < N; ++i)
    {
        ret[i] = static_cast<int>(std::floor(f[i]));
    }
    return ret;
}

template <typename T, unsigned int N> inline
LaneArray<T, N> inc(const ConstLaneArrayRef<T, N>& a, const LaneMask<N>& mask)
{
    const LaneArray<T, N> ap1 = a + T(1);
    return Blend(mask, ap1 , a);
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

template <typename T, unsigned int N>
struct SimdTypeTraits<Mongo::LaneArray<T,N>>
{
    typedef T scalar_type;
    static const uint16_t width = N;
};

}}   // namespace PacBio::Simd

#endif  // mongo_common_LaneArray_H_
