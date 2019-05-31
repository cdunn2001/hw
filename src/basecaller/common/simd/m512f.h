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

#include "m512i.h"

namespace PacBio {
namespace Simd {

template<>
struct SimdTypeTraits<m512f>
{
    typedef float scalar_type;
    static const uint8_t width = 16;
};


// TODO: Vectorize this.
inline m512f round(const m512f& xx)
{
    ArrayUnion<m512f> x(xx);
    for (unsigned int i = 0; i < SimdTypeTraits<m512f>::width; ++i)
    {
        x[i] = std::round(x[i]);
    }
    return x;
}

/// Logistic function
inline m512f expit(const m512f& x)
{
    const auto y = exp(x);
    return y / (1 + y);
}

/// Inverse of the logistic function (expit)
inline m512f logit(const m512f& x)
{
    return log(x / (1 - x));
}

}}      // namespace PacBio::Simd

#endif  // mongo_common_simd_m512f_H_
