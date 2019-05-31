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

/// Construct an m512i instance from an unsigned int.
inline m512i make_m512i(unsigned int a)
{
    // Ensure that the value of a is in the range of ScalarType<m512i>.
    using ScalarInt = typename SimdTypeTraits<m512i>::scalar_type;
    assert(a <= static_cast<unsigned int>(std::numeric_limits<ScalarInt>::max()));
    return m512i(static_cast<ScalarInt>(a));
}

// TODO: There must be a more efficient way to do this.
/// Prefix increment.
inline m512i& operator++(m512i& a)
{ return a += 1; }

inline m512i operator/(const m512i& a, const m512i& b)
{
    // TODO: Does the compiler vectorize the loop?
    ArrayUnion<m512i> c;
    for (unsigned int i = 0; i < SimdTypeTraits<m512i>::width; ++i)
    {
        c[i] = a[i] / b[i];
    }
    return c;
}

}}      // namespace PacBio::Simd

#endif  // mongo_common_simd_m512i_H_
