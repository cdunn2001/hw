#ifndef mongo_common_LaneArray_H_
#define mongo_common_LaneArray_H_

#include <algorithm>
#include <type_traits>
#include <boost/operators.hpp>

#include <common/MongoConstants.h>
#include <common/simd/SimdConvTraits.h>
#include <common/simd/SimdTypeTraits.h>

namespace PacBio {
namespace Mongo {

/// A fixed-size array of boolean values.
template <unsigned int N>
class LaneMask : public boost::bitwise<LaneMask<N>>
{
    // Static assertions that enable efficient SIMD and CUDA implementations.
    static constexpr auto laneUnit = std::max<unsigned int>(cudaThreadsPerWarp,
                                                            Simd::SimdTypeTraits<Simd::m512b>::width);
    static_assert(N != 0, "Template argument cannot be 0.");
    static_assert(N % laneUnit == 0u, "Bad LaneArray size.");

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


/// A fixed-size array type that supports elementwise arithmetic operations.
template <typename T, unsigned int N>
class LaneArray : public boost::arithmetic<LaneArray<T, N>>
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

public:     // Structors and assignment
    LaneArray() = default;

    /// Broadcasting constructor supports implicit conversion of scalar value
    /// to uniform vector.
    LaneArray(const T& val)
    {
        std::fill(data_, data_+N, val);
    }

public:     // Scalar access
    T operator[](unsigned int i) const
    {
        assert(i < N);
        return data_[i];
    }

    T& operator[](unsigned int i)
    {
        assert(i < N);
        return data_[i];
    }

public:
    T* Data()
    { return data_; }

    const T* Data() const
    { return data_; }

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

public:     // Compound assigment
    // Boost provides the associated binary operators.
    LaneArray& operator+=(const LaneArray& a)
    {
        for (unsigned int i = 0; i < N; ++i)
        {
            this->data_[i] += a[i];
        }
        return *this;
    }

    LaneArray& operator-=(const LaneArray& a)
    {
        for (unsigned int i = 0; i < N; ++i)
        {
            this->data_[i] -= a[i];
        }
        return *this;
    }

    LaneArray& operator*=(const LaneArray& a)
    {
        for (unsigned int i = 0; i < N; ++i)
        {
            this->data_[i] *= a[i];
        }
        return *this;
    }

    LaneArray& operator/=(const LaneArray& a)
    {
        for (unsigned int i = 0; i < N; ++i)
        {
            this->data_[i] /= a[i];
        }
        return *this;
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

template <typename T, unsigned int N>
LaneArray<T,N> Blend(const LaneMask<N>& tf, const LaneArray<T,N>& success, const LaneArray<T,N>& failure)
{
    LaneArray<T,N> ret;
    for (unsigned int i = 0; i < N; ++i)
    {
        ret[i] = tf[i] ? success[i] : failure[i];
    }
    return failure;
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
