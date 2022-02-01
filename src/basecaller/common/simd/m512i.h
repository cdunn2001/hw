#ifndef mongo_common_simd_m512i_H_
#define mongo_common_simd_m512i_H_

#include <limits>

// 512-bit int simd-vector type (16 packed 32-bit int values)

#include "ArrayUnion.h"
#include "SimdTypeTraits.h"

#if     defined(PB_LOOP)
#include "m512i_LOOP.h"
#elif   defined(PB_CORE_AVX512)
#include "m512i_AVX512.h"
#elif   defined(__SSE2__)
#include "m512i_SSE.h"
#else
#error m512i type is not supported by the available instruction set.
#endif

namespace PacBio {
namespace Simd {

template<>
struct SimdTypeTraits<m512i>
{
	typedef int scalar_type;
    static const uint8_t width = 16;
};

}}      // namespace PacBio::Simd

#endif  // mongo_common_simd_m512i_H_
