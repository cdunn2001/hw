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

// TODO: Add to interface to model the concept prototyped by m512b.
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
    /// Broadcasting constructor supports implicit conversion of scalar value
    /// to uniform vector.
    LaneMask(bool tf)
    {
        // TODO
    }

public:     // Compound assignment
    // Boost provides the associated binary operators.
    LaneMask& operator|=(const LaneMask& a)
    {
        // TODO
        return *this;
    }

    LaneMask& operator&=(const LaneMask& a)
    {
        // TODO
        return *this;
    }

    LaneMask& operator^(const LaneMask& a)
    {
        // TODO
        return *this;
    }

public:
    /// Returns a copy with each element negated.
    LaneMask operator!()
    {
        // TODO
        return *this;
    }

public:     // Reductions
    friend bool all(const LaneMask& tf)
    {
        // TODO
        return false;
    }
};


// TODO: Add to interface to model the concepts prototyped by m512f and m512i.
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
    { }

public:     // Comparison operators
    friend LaneMask<N> operator==(const LaneArray& lhs, const LaneArray& rhs)
    {
        // TODO
        return LaneMask<N>(false);
    }

    friend LaneMask<N> operator<(const LaneArray& lhs, const LaneArray& rhs)
    {
        // TODO
        return LaneMask<N>(false);
    }

    friend LaneMask<N> operator<=(const LaneArray& lhs, const LaneArray& rhs)
    {
        // TODO: Optimize
        return (lhs < rhs) | (lhs == rhs);
    }

    friend LaneMask<N> operator>(const LaneArray& lhs, const LaneArray& rhs)
    {
        // TODO: Optimize
        return !(lhs <= rhs);
    }

    friend LaneMask<N> operator>=(const LaneArray& lhs, const LaneArray& rhs)
    {
        // TODO: Optimize
        return !(lhs < rhs);
    }

public:     // Compound assigment
    // Boost provides the associated binary operators.
    LaneArray& operator+=(const LaneArray& a)
    {
        // TODO
        return *this;
    }

    LaneArray& operator-=(const LaneArray& a)
    {
        // TODO
        return *this;
    }

    LaneArray& operator*=(const LaneArray& a)
    {
        // TODO
        return *this;
    }

    LaneArray& operator/=(const LaneArray& a)
    {
        // TODO
        return *this;
    }

public:     // Named binary operators
    friend LaneArray max(const LaneArray& a, const LaneArray& b)
    {
        // TODO
        return LaneArray();
    }

    friend LaneArray min(const LaneArray& a, const LaneArray& b)
    {
        // TODO
        return LaneArray();
    }

public:     // Tests for special floating-point values
    friend LaneMask<N> isnan(const LaneArray& a)
    {
        // TODO
        return LaneMask<N>(false);
    }
};


template <typename T, unsigned int N>
LaneArray<T, N> Blend(const LaneMask<N>& tf, const LaneArray<T, N>& success, const LaneArray<T, N>& failure)
{
    // TODO
    return failure;
}

}}      // namespace PacBio::Mongo


// TODO: Seems like we might have a namespace wrinkle to iron out.
namespace PacBio {
namespace Simd {

template <typename T, unsigned int N>
struct SimdConvTraits<Mongo::LaneArray<T, N>>
{
    typedef Mongo::LaneMask<N> bool_conv;
    typedef Mongo::LaneArray<float, N> float_conv;
    typedef Mongo::LaneArray<int, N> index_conv;
    typedef Mongo::LaneArray<short, 2*N> pixel_conv;   // TODO: ?!
    typedef Mongo::LaneArray<T, N> union_conv;
};

}}   // namespace PacBio::Simd

#endif  // mongo_common_LaneArray_H_
