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

template<>
struct SimdTypeTraits<m512s>
{
    typedef short scalar_type;
    static const uint8_t width = 32;
};

}}      // namespace PacBio::Simd

#endif  // mongo_common_simd_m512s_H_
