#ifndef mongo_common_simd_SimdVectorTypes_H_
#define mongo_common_simd_SimdVectorTypes_H_

// Copyright (c) 2015, Pacific Biosciences of California, Inc.
//
// All rights reserved.
//
// THIS SOFTWARE CONSTITUTES AND EMBODIES PACIFIC BIOSCIENCES' CONFIDENTIAL
// AND PROPRIETARY INFORMATION.
//
// Disclosure, redistribution and use of this software is subject to the
// terms and conditions of the applicable written agreement(s) between you
// and Pacific Biosciences, where "you" refers to you or your company or
// organization, as applicable.  Any other disclosure, redistribution or
// use is prohibited.
//
// THIS SOFTWARE IS PROVIDED BY PACIFIC BIOSCIENCES AND ITS CONTRIBUTORS "AS
// IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
// THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL PACIFIC BIOSCIENCES OR ITS
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
// OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
// WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
// OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
// ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
//  Description:
/// \file   SimdVectorTypes.h
/// \brief  Comprehensive include file for all SIMD vector types.

#include <cmath>
#include <type_traits>

#include "SimdConvTraits.h"
#include "SimdTypeTraits.h"

// Support for use of scalar types in SIMD-enabled filters.
namespace PacBio {
namespace Simd {

template <typename T>
struct StdOps
{
    static std::plus<T> plus_;
    static std::minus<T> minus_;
    static std::multiplies<T> multiplies_;
    static std::divides<T> divides_;

    struct minOp
    {
        const T& operator() (const T& a, const T& b)
        { return std::min(a, b); }
    };

    struct maxOp
    {
        const T& operator() (const T& a, const T& b)
        { return std::max(a, b); }
    };

    struct assign { T operator() (const T& v) { return v; } };

    struct plus { T operator() (const T& a, const T&b) { return plus_(a, b); } };
    struct minus { T operator() (const T& a, const T&b) { return minus_(a, b); } };

    struct multiplies { T operator() (const T& a, const T&b) { return multiplies_(a, b); } };
    struct divides { T operator() (const T& a, const T&b) { return divides_(a, b); } };
};

/// SIMD absolute value.
template <typename SimdType> inline
SimdType abs(const SimdType& x)
{
    using std::max;
    return max(x, -x);
}

// Overload SIMD reductions for scalar types
inline float reduceMax(float v) { return v; }
inline int reduceMax(int v) { return v; }
inline float reduceMin(float v) { return v; }
inline int reduceMin(int v) { return v; }

/// Overload SIMD Blend for scalar types.
/// Simply wraps ternary operator (?:).
template <typename T> inline
T Blend(bool tf, const T& a, const T& b)
{ return tf ? a : b; }

/// Overload for logicial reduction all that allows exchanging float for SIMD types.
/// \returns \a tf.
inline bool all(bool tf)
{ return tf; }

/// Overload for logicial reduction any that allows exchanging float for SIMD types.
/// \returns \a tf.
inline bool any(bool tf)
{ return tf; }

/// Overload for logicial reduction none that allows exchanging float for SIMD types.
/// \returns !tf.
inline bool none(bool tf)
{ return !tf; }


// Define similar template specializations as needed for int, int16_t, etc.

template <typename T>
using ArithOps = typename std::conditional<std::is_fundamental<T>::value, StdOps<T>, T>::type;

}}      // namespace PacBio::Simd

#endif // mongo_common_simd_SimdVectorTypes_H_
