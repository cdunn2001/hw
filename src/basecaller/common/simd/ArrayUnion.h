#ifndef mongo_common_simd_ArrayUnion_H_
#define mongo_common_simd_ArrayUnion_H_

// Copyright (c) 2017, Pacific Biosciences of California, Inc.
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
//  Brief:
//  Wrapper type that allows array access to SIMD types (Or scalar types if
//  they have the appropriate simd_type_traits specialization).
//
//  Description:
//  It is designed to be as transparent as possible, so implicit conversions
//  allow *most* operations/functions to treat these as the underlying type.
//  This class will provide operator[] allowing reference semantics for element
//  access.  However, a simd variable contained in a union is slower than one
//  by itself, so it is very much by design that operations using these wrapper
//  types result in the underlying type rather than another wrapper type.
//  In general element access is to be discouraged.

#include <array>

#include "SimdTypeTraits.h"

namespace PacBio {
namespace Simd {

template <typename T>
union ArrayUnion
{
    using Scalar_T = ScalarType<T>;
    static_assert(sizeof(T) == SimdTypeTraits<T>::width * sizeof(Scalar_T),
                  "Incorrect argument to ArrayUnion template parameter");

    ArrayUnion() : simd{} {}
    // Implicit ctor by design.
    ArrayUnion(const T& t) : simd{t} {}
    ArrayUnion& operator=(const T& t) { simd = t; return *this; }

    // Implicit casts by design.
    operator const T&() const { return simd; }
    operator T&() {return simd; }
    // Named conversion for the few times implicit casts are insufficient.
    T& Simd() { return simd; }
    const T& Simd() const { return simd; }

    // Compound operators are not provided via the implicit conversion.  Provide
    // them here for convenience.  Though warning, this is a bit odd if the
    // underlying type is `bool`
    T operator+=(const T& val) { return simd += val; }
    T operator-=(const T& val) { return simd -= val; }
    T operator*=(const T& val) { return simd *= val; }
    T operator/=(const T& val) { return simd /= val; }

    // Another convenience operator.  Again odd if T is a bool
    T operator-() const { return -simd; }

    // Element access via array member of union
    Scalar_T& operator[](size_t idx) { return elements[idx]; }
    const Scalar_T& operator[](size_t idx) const { return elements[idx]; }

    Scalar_T* begin() { return elements.data(); };
    Scalar_T* end() { return begin() + SimdTypeTraits<T>::width; }
    const Scalar_T* cbegin() const { return elements.data(); }
    const Scalar_T* cend() const { return cbegin() + SimdTypeTraits<T>::width; }
private:
    T simd;
    std::array<Scalar_T, SimdTypeTraits<T>::width> elements;
};

}}      // namespace PacBio::Simd

#endif // mongo_common_simd_ArrayUnion_H_
