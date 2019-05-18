#ifndef mongo_common_simd_m512b_H_
#define mongo_common_simd_m512b_H_

// 16-bit boolean simd-vector type
// 16 x 32 bit float for SSE

#include <bitset>

#include "SimdTypeTraits.h"

#if     defined(PB_LOOP)
#include "m512b_LOOP.h"
#elif   defined(PB_CORE_AVX512)
#include "m512b_AVX512.h"
#elif   defined(__SSE2__)
#include "m512b_SSE.h"
#else
#error m512b type is not supported by the available instruction set.
#endif

namespace PacBio {
namespace Simd {

template<>
struct SimdTypeTraits<m512b>
{
    typedef bool scalar_type;
    static const uint8_t width = 16;
};


/// The number of "true" elements in \a tf.
inline unsigned short count(const m512b& tf)
{
    unsigned short r = 0;
    for (unsigned int i = 0; i < SimdTypeTraits<m512b>::width; ++i)
    {
        if (tf[i]) ++r;
    }
    return r;
}


/// Returns 1 if tf == true; otherwise, 0.
inline unsigned short count(bool tf)
{
    return static_cast<unsigned short>(tf ? 1u : 0u);
}


// TODO: Can this be optimized?
/// Convert to std::bitset.
template <typename VecBool>
inline std::bitset<SimdTypeTraits<VecBool>::width>
toBitset(const VecBool& value)
{
    constexpr unsigned int n = SimdTypeTraits<VecBool>::width;
    std::bitset<n> r;
    for (unsigned int i = 0; i < n; ++i) r[i] = value[i];
    return r;
}

template <>
inline std::bitset<1u> toBitset(const bool& value)
{
    std::bitset<1u> r;
    r[0] = value;
    return r;
}

}}      // namespace PacBio::Simd

#endif  // mongo_common_simd_m512b_H_
