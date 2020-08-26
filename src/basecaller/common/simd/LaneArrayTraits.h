// Copyright (c) 2020, Pacific Biosciences of California, Inc.
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
// This file is used to try and isolate away the bulk of the TMP programming
// magic.  Here be dragons.  I've tried to document what I've done so
// it's less opaque, but I've also tried to name the usable types
// meaninfully enough that they can be treated as black boxes and still
// understand what the consuming code is doing.
//
// For anyone trying to read this code a few things to keep in mind:
// 1. A lot of this relies on SFINAE.  Read up on that if you're not
//    familiar
// 2. This has a fair bit of "procedural programming with types", which
//    people might not be used to reading.  E.g there might be a template
//    list that has the following form:
//    template <typename Arg1, typename Arg2, typename Arg3,
//              typename Arg5 = Somefunction<Arg1, Arg2>,
//              typename Arg6 = AnotherFunction<Arg5, Arg3>
//             >
//    Here we're basically using the rules of template argument substitution
//    to write a basic procedural program.  You can do similar things with
//    a list of types/aliases declared inside a templated class.

//    In these cases you can go a long way by just forgetting that we're
//    dealing with types, and threat this like a normal function.
//    Args 1,2 and 3 are the inputs, and Arg5 and Arg6 are const variables
//    you set up as functions of inputs/variables that come before.

#ifndef mongo_common_Simd_LaneArrayTraits_H_
#define mongo_common_Simd_LaneArrayTraits_H_

#include <common/LaneArray_fwd.h>

#include <common/cuda/utility/CudaArray.h>
#include <common/simd/m512b.h>
#include <common/simd/m512f.h>
#include <common/simd/m512i.h>
#include <common/simd/m512s.h>
#include <common/simd/SimdConvTraits.h>
#include <common/simd/SimdTypeTraits.h>

namespace PacBio {
namespace Simd {

// Need a traits class to convert from a scalar
// type to the appropriate m512 type
template <typename T>
struct vec_type;

template<> struct vec_type<float>    { using type = m512f;  };
template<> struct vec_type<int32_t>  { using type = m512i;  };
template<> struct vec_type<uint32_t> { using type = m512ui; };
template<> struct vec_type<bool>     { using type = m512b;  };
template<> struct vec_type<int16_t>  { using type = m512s;  };
template<> struct vec_type<uint16_t> { using type = m512us; };

template <typename T>
using vec_type_t = typename vec_type<T>::type;

//--------------------------------------------------------------

template <typename T>
struct len_trait
{
    static constexpr size_t SimdCount = 0;
};
template <typename T, size_t Len>
struct len_trait<LaneArray<T, Len>>
{
    static_assert(Len % vec_type_t<T>::size() == 0, "Invalid length");
    static constexpr size_t SimdCount = Len / vec_type_t<T>::size();
};
template <size_t Len>
struct len_trait<LaneMask<Len>>
{
    static_assert(Len % vec_type_t<bool>::size() == 0, "Invalid length");
    static constexpr size_t SimdCount = Len / vec_type_t<bool>::size();
};
template <typename T>
struct len_trait<ArrayUnion<T>> : public len_trait<T> {};

//---------------------------------------------------------------------

template <typename T> struct IsLaneArray
{ static constexpr bool value = false; };
template <typename T, size_t N> struct IsLaneArray<LaneArray<T, N>>
{ static constexpr bool value = true; };
template <typename T> struct IsLaneArray<ArrayUnion<T>>
{ static constexpr bool value = true; };

// We're going to do some TMP to extend the notion
// of "common type" to LaneArrays.  That is, the
// common type of float and int is float, and
// similarly the common type of LaneArray<float> and
// LaneArray<int> should be LaneArray<float>.  For the
// most part we're just going to use type traits to decay
// LaneArray types to their associated scalar types and
// leverage std::common_type.  However, in some limited cases
// we want to ignore one of the inputs.  For instance
// LaneArray<short> and int should have the common type of
// LaneArray<short> rather than LaneArray<int>, simply to
// avoid unecessary conversions as it's really hard to keep
// around true 16 bit scalars.  So whenever we want to ignore
// a type we're substitute it with this.
struct IgnoredType {};

// Accepts a type T and finds the associated scalar type,
// after filtering occurs.  The result will either be
// - void (there is no sensible scalar type, e.g. for arbitrary
//         class Foo)
// - IgnoredType (For whatever reason there is a sensible scalar
//                but we want to disregard it)
// - int/float/etc
template <typename T>
class FindFilteredScalar
{
    // Don't let 32 bit integral scalars affect the return type.  This is primarily
    // to combat things like LaneArray<short> + int = LaneArray<int>, as
    // it's painfully difficult to type short literals
    using IgnoreInts = std::conditional_t<std::is_same<T, int32_t>::value
                                          || std::is_same<T, uint32_t>::value,
                                          IgnoredType, ScalarType<T>>;

    // intel is stupid and requires this...  our final result eventually gets used in
    // `std::common_type`, and intel barfs if certain eigen types make it there
    // This is just a crude filter to replace any non-arithmetic types with void,
    // which for our purposes is just as good as there will be no common type
    using ForceArithmetic = std::conditional_t<std::is_arithmetic<IgnoreInts>::value
                                               || std::is_same<IgnoredType, IgnoreInts>::value,
                                               IgnoreInts, void>;
    // 32 bit scalars get a pass because it's hard to write and maintain 16 bit integers.  The same
    // exception does *not* apply for 64 bit integers and *definitely* does not apply for
    // doubles.  Detect them and turn them into void, so that users are forced to cast them
    // to a sensible type and take responsibility for overflow/etc.
    // Note: We have to do some gymnastics here though as we can't take `sizeof(void)`.  Temporarily
    //       transform voids into bools for the sake of the next step.
    using helper = std::conditional_t<std::is_same<void, ForceArithmetic>::value, bool, ForceArithmetic>;

public:
    using type = std::conditional_t<sizeof(helper) == 8, void, ForceArithmetic>;
};
// Need a quick specialization to help with ArrayUnions of LaneArrays.
template <typename T>
struct FindFilteredScalar<ArrayUnion<T>>
{
    using type = typename FindFilteredScalar<T>::type;
};
template <typename T>
using FindFilteredScalar_t = typename FindFilteredScalar<T>::type;

//--------------------------------------------------------------------

// Similar in spirit to std::common_type but augmented for LaneArrays.
// Ts... will be a list of types and this class can be used to determine
// a sensible "common type" for mixed type operations.  The resulting
// common type (if it exists) will still be a scalar type.
template <typename...Ts>
class CommonArrayType
{
    // We have to deal with the presence of IgnoredType, If that makes it
    // into std::common_type then it will report back that there is no
    // common type.  Instead we'll replace it with a "default", which here
    // we'll identify as the first non-IgnoredType entry in Ts... (because
    // that was convenient and easy).  because std::common_type<T, T> is T,
    // this will have the same effect as just removing all IgnoredType
    // entries from Ts...
    static constexpr size_t IdxOfDefault()
    {
        bool isNoop[sizeof...(Ts)] {std::is_same<IgnoredType, FindFilteredScalar_t<Ts>>::value...};
        for (size_t i = 0; i < sizeof...(Ts); ++i)
        {
            if (!isNoop[i]) return i;
        }
        // Should be unreachable.
        return std::numeric_limits<size_t>::max();
    }

    // Check if Ts... has both signed and unsigned integrals.  We have to
    // jump through a couple hoops though as std::is_signed<float> == true
    // and std::is_signed<bool> == false;
    static constexpr bool NoSignedMismatch()
    {
        bool isIntegral[sizeof...(Ts)] { std::is_integral<ScalarType<Ts>>::value...};
        bool isSigned[sizeof...(Ts)]   { std::is_signed<ScalarType<Ts>>::value...};
        bool isBool[sizeof...(Ts)]     { std::is_same<ScalarType<Ts>, bool>::value...};
        bool isNoop[sizeof...(Ts)]     {std::is_same<IgnoredType, ScalarType<Ts>>::value...};

        bool anyUnsignedInt = false;
        bool anySignedInt = false;
        for (size_t i = 0; i < sizeof...(Ts); ++i)
        {
            anyUnsignedInt |= !(isSigned[i] | isBool[i] | isNoop[i]);
            anySignedInt |= (isSigned[i] & isIntegral[i]);
        }

        return !(anySignedInt & anyUnsignedInt);
    }

    using DefType = std::tuple_element_t<IdxOfDefault(), std::tuple<Ts...>>;

public:
    // This abomination is to keep us as a dependant type for SFINAE failures.  Intel was having
    // issues so I had to get slightly more convoluted than should have been necessary
    //
    // Both a Signed/Unsigned mismatch as well as invalid arguments to common_type will
    // cause a SFINAE.  This makes this class useful as part of the function set when
    // defining a generic overload set for arithmetic operations.
    template <typename ...Us>
    auto Helper() ->
        std::enable_if_t<NoSignedMismatch(),
                         std::common_type_t<std::conditional_t<std::is_same<FindFilteredScalar_t<Us>, IgnoredType>::value,
                                                               FindFilteredScalar_t<DefType>,
                                                               FindFilteredScalar_t<Us>>...>>;
};
template <typename...Ts>
using CommonArrayType_t = decltype(std::declval<CommonArrayType<Ts...>>().template Helper<Ts...>());

//-------------------------------------------------------------------------

// This is a helper alias that automatically does SFINAE on generic
// binary operations.  It is intended to be used on inline friend
// operators, where either (or both depending on ADL!) of the
// arguments are different fromt the class it's defined in.
// There are three mandatory arguments:
// - Arg1 : The first operand to the binary operator
// - Arg2 : The second operand to the binary operator
// - T    : The class in which the operator is defined.  It is not
//          guaranteed to be either Arg1 or Arg2
//          (e.g. ArrayUnion<LaneArray<int>> * int is ultimately
//          defined in LaneArray<int>)
// - RetRequest (optional) : The actual desired return value.  If
//                           not specified it will default to
//                           the common type.
//
// This class does two main things:
// - Makes sure it's a valid binary op.  Multiplying LaneArray<float> * Foo
//   will result in an immediate compilation error saying that operation does
//   not exist, rather than some random error deeper in the bowels of this library
// - Makes sure that inline friend functions are not multiply defined.
//   LaneArray<int> * LaneArray<float> can only be defined in one of the two classes,
//   else there will be an error.  In general the chosen class will be the same as
//   the common type (LaneArray<float> in that example), but there are exceptions.
//   For instance LaneArray<short> * int = LaneArray<int> must be defined inside
//   LaneArray<short>, because that is the only place ADL can find it.
template <
    typename Arg1, typename Arg2, typename T, typename RetRequest = void,  // public API
    // Everything from now on is doing procedural programming with types!
    // SFINAE check to see if there is a sensible common type
    typename common = CommonArrayType_t<Arg1, Arg2>,
    // If both args are LaneArray, then we need to be careful to not end up
    // with multiple definitions for the associated function.
    bool SingleLaneArray = IsLaneArray<Arg1>::value xor IsLaneArray<Arg2>::value,
    // Is T the common type?  If so it will be chosen for containing the definition
    bool IsCommon = std::is_same<ScalarType<T>, common>::value,
    //
    size_t VecLen = std::max((uint16_t)SimdTypeTraits<Arg1>::width, (uint16_t)SimdTypeTraits<Arg2>::width),
    // Here's the result of our "computation".  If we've not had a SFINAE instance already, then our
    // chosen return type will either be the common type, or RetRequest, depending on if
    // RetRequest was even specified.
    typename Ret = std::conditional_t<std::is_same<void, RetRequest>::value, LaneArray<common, VecLen>, RetRequest>
    >
using InlineFriendReturn = std::enable_if_t<SingleLaneArray || IsCommon, Ret>;

// SFINAE helper for compound operators.  Compound operators have the
// obvious constraint where the result type must be the same as the
// original type (e.g. int += float doesn't work because we don't
// allow implicit truncations)
template <typename T1, typename T2>
using CompoundOpIsValid = std::enable_if_t<std::is_same<ScalarType<T1>, CommonArrayType_t<T1, T2>>::value>;


template <typename T, size_t N>
struct SimdConvTraits<Mongo::LaneArray<T,N>>
{
    typedef Mongo::LaneMask<N> bool_conv;
    typedef Mongo::LaneArray<float,N> float_conv;
    typedef Mongo::LaneArray<int,N> index_conv;
    typedef Mongo::LaneArray<short,N> pixel_conv;
    typedef ArrayUnion<Mongo::LaneArray<T,N>> union_conv;
};

template <typename T, size_t N>
struct SimdTypeTraits<Mongo::LaneArray<T,N>>
{
    typedef T scalar_type;
    static const uint16_t width = N;
};

template <size_t N>
struct SimdTypeTraits<Mongo::LaneMask<N>>
{
    typedef bool scalar_type;
    static const uint16_t width = N;
};

}}   // namespace PacBio::Simd

#endif //mongo_common_Simd_LaneArrayTraits_H_
