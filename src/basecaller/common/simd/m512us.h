#ifndef mongo_common_simd_m512us_H_
#define mongo_common_simd_m512us_H_

// 512-bit short simd-vector type (32 packed 16-bit short int values)

#include "SimdTypeTraits.h"

#if     defined(PB_LOOP)
//#include "m512us_LOOP.h"
#error LOOP not supported
#elif   defined(PB_CORE_AVX512)
#include "m512us_AVX512.h"
#elif   defined(__SSE2__)
#include "m512us_SSE.h"
#else
#error m512us type is not supported by the available instruction set.
#endif

namespace PacBio {
namespace Simd {

template <typename T>
std::conditional_t<std::is_same<T, float>::value, m512f, m512ui> Low(const m512us& in);

template <>
inline std::conditional_t<std::is_same<float, float>::value, m512f, m512ui> Low<float>(const m512us& in)
{
    return LowFloats(in);
}
template <>
inline std::conditional_t<std::is_same<uint32_t, float>::value, m512f, m512ui> Low<uint32_t>(const m512us& in)
{
    return LowInts(in);
}

template <typename T>
std::conditional_t<std::is_same<T, float>::value, m512f, m512ui>High(const m512us& in);

template <>
inline std::conditional_t<std::is_same<float, float>::value, m512f, m512ui>High<float>(const m512us& in)
{
    return HighFloats(in);
}
template <>
inline std::conditional_t<std::is_same<uint32_t, float>::value, m512f, m512ui>High<uint32_t>(const m512us& in)
{
    return HighInts(in);
}

template<>
struct SimdTypeTraits<m512us>
{
    typedef uint16_t scalar_type;
    static const uint8_t width = 32;
};

}}      // namespace PacBio::Simd

#endif  // mongo_common_simd_m512us_H_
