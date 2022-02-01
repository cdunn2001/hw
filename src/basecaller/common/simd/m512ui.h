#ifndef mongo_common_simd_m512ui_H_
#define mongo_common_simd_m512ui_H_

#include <limits>

// 512-bit int simd-vector type (16 packed 32-bit int values)

#include "ArrayUnion.h"
#include "SimdTypeTraits.h"

#if     defined(PB_LOOP)
//#include "m512ui_LOOP.h"
#error LOOP not supported
#elif   defined(PB_CORE_AVX512)
#include "m512ui_AVX512.h"
#elif   defined(__SSE2__)
#include "m512ui_SSE.h"
#else
#error m512ui type is not supported by the available instruction set.
#endif

namespace PacBio {
namespace Simd {

template<>
struct SimdTypeTraits<m512ui>
{
    typedef uint32_t scalar_type;
    static const uint8_t width = 16;
};

}}      // namespace PacBio::Simd

#endif  // mongo_common_simd_m512ui_H_
