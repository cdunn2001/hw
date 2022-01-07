#ifndef mongo_common_simd_m512f_H_
#define mongo_common_simd_m512f_H_

// 512-bit float simd-vector type (16 packed 32-bit float values)

#include <cmath>

#include "ArrayUnion.h"
#include "SimdTypeTraits.h"

#if     defined(PB_LOOP)
#include "m512f_LOOP.h"
#elif   defined(PB_CORE_AVX512)
#include "m512f_AVX512.h"
#elif   defined(__SSE2__)
#include "m512f_SSE.h"
#else
#error m512f type is not supported by the available instruction set.
#endif

namespace PacBio {
namespace Simd {

template<>
struct SimdTypeTraits<m512f>
{
    typedef float scalar_type;
    static const uint8_t width = 16;
};

/// Computes the value of \a base raised to the power \a exp.
inline m512f pow(const m512f& base, const m512f& exp)
{
    ArrayUnion<m512f> x;
    for (uint8_t i = 0; i < SimdTypeTraits<m512f>::width; ++i)
    {
        x[i] = std::pow(base[i], exp[i]);
    }
    return x;
}


}}      // namespace PacBio::Simd

#endif  // mongo_common_simd_m512f_H_
