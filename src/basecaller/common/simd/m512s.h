#ifndef mongo_common_simd_m512s_H_
#define mongo_common_simd_m512s_H_

// 512-bit short simd-vector type (32 packed 16-bit short int values)

#include "SimdTypeTraits.h"

#if     defined(PB_LOOP)
#include "m512s_LOOP.h"
#elif   defined(PB_CORE_AVX512)
#include "m512s_AVX512.h"
#elif   defined(__SSE2__)
#include "m512s_SSE.h"
#else
#error m512s type is not supported by the available instruction set.
#endif

namespace PacBio {
namespace Simd {

template <typename T>
std::conditional_t<std::is_same<T, float>::value, m512f, m512i>Low(const m512s& in);

template <>
inline m512f Low<float>(const m512s& in)
{
    return LowFloats(in);
}
template <>
inline m512i Low<int>(const m512s& in)
{
    return LowInts(in);
}

template <typename T>
std::conditional_t<std::is_same<T, float>::value, m512f, m512i>High(const m512s& in);

template <>
inline m512f High<float>(const m512s& in)
{
    return HighFloats(in);
}
template <>
inline m512i High<int>(const m512s& in)
{
    return HighInts(in);
}

template<>
struct SimdTypeTraits<m512s>
{
    typedef short scalar_type;
    static const uint8_t width = 32;
};

}}      // namespace PacBio::Simd

#endif  // mongo_common_simd_m512s_H_
