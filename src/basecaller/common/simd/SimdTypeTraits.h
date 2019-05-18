#ifndef mongo_common_simd_SimdTypeTraits_H_
#define mongo_common_simd_SimdTypeTraits_H_

/// Traits for types based on SIMD intrinsics.
#include <cstddef>
#include <cstdint>


namespace PacBio {
namespace Simd {

template <typename _V>
struct SimdTypeTraits
{
    typedef _V      scalar_type;
    static const uint8_t width = 1;
};


// TODO: Audit places where we are using "short".  Should we be using m32s?
template <>
struct SimdTypeTraits<short>
{
    typedef short scalar_type;
    static const uint8_t width = 1;
};

template <>
struct SimdTypeTraits<float>
{
    typedef float scalar_type;
    static const uint8_t width = 1;
};

template <typename V>
using ScalarType = typename SimdTypeTraits<V>::scalar_type;


}}      // namespace PacBio::Simd

#endif  // mongo_common_simd_SimdTypeTraits_H_
