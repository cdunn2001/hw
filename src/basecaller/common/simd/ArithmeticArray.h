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

#ifndef mongo_common_simd_ArithmeticArray_H_
#define mongo_common_simd_ArithmeticArray_H_

#include <type_traits>

#include <common/simd/BaseArray.h>
#include <common/simd/LaneMaskImpl.h>
#include <common/simd/m512b.h>
#include <common/simd/m512i.h>
#include <common/simd/m512f.h>
#include <common/simd/m512s.h>
#include <common/simd/m512ui.h>
#include <common/simd/m512us.h>

namespace PacBio {
namespace Simd {

// Fixing a slight impedance mismatch here.  Doing a blend with 16 bit types
// requires two m512b values, which the BaseArray code will produce as a
// special PairRef type (to avoid making coppies).  The actual m512 types
// however just expect two standalong m512b arguments.

// Could potentially make the PairRef concept a little more visible so
// the m512 types could accept it directly, but since this was the only
// pain point I didn't quite feel the need to go that far.
inline m512s Blend(const PairRef<const m512b>& b, const m512s& l, const m512s& r)
{
    return Blend(b.first, b.second, l, r);
}
inline m512us Blend(const PairRef<const m512b>& b, const m512us& l, const m512us& r)
{
    return Blend(b.first, b.second, l, r);
}

/// This class provides a set expected arithmetic operations
/// common to many primitive types.  Concrete types of LaneArray
/// will inherit from this (and potentially provide additional
/// operators/functions specific to that type.)
///
/// On the surface this is just providing a simple set of operator
/// overloads to make LaneArrays behave like a mathematical type.
/// However there is some carefully crafted SFINAE template
/// magic going on to make sure that this can:
/// 1. Seamlessly handle mixed type operations. It should invisbly
///    handle all combinations of mixed arithmetic types as well as mixed
///    LaneArray/Scalar.  It does so by doing per-element conversions
///    rather than more costly things like an upfront conversion of
///    a scalar float to a LaneArray<float>
/// 2. Mimics as closely as possible the standard type conversions
///    (e.g. int + float = float) while also disabling operators
///    that make no sense (LaneArray<int> += LaneArray<float>
///    cannot work unless you were willing to suffer truncation)
/// 3. Successfully avoid multiple definitions of friend operators.
///    For Example LaneArray<int> + LaneArray<float> is only defined
///    inside the float version, and would be an error if it were
///    defined in both.  However LaneArray<int> + float is defined
///    inside the int version because that's the only place ADL
///    would find the operator.
///
/// As mentioned this tries to mimic how normal scalar types
/// behave, but there are a few notible exceptions:
/// 1. LaneArray<short> + LaneArray<short> = LaneArray<short>
///    Normal shorts get promoted to int in such a case but
///    we very much don't want that for our types.
/// 2. While LaneArray<short> * LaneArray<int> will result in
///    LaneArray<int>, LaneArray<short> * int will not.  This
///    is because it's very difficult to type 16 bit literals,
///    and even if you have a 16 bit variable it's too easy to
///    accidentally bump it up to 32 bit.  Without this exception
///    we'll be either constantly casting back to 16 bit types
///    or else suffering from unecessary (and somewhat expensive)
///    conversion to arrays of 32 bit types.
///    2b. Scalar int and unsigned int are the *only* exceptions
///        to this rule.  LaneArray<int> * float will still result
///        in LaneArray<float>
/// 3. No implicit "demotions".  You can't do something like
///    LaneArray<int> += float as that requires truncation.  Maybe
///    this is my personal preferences bleeding through but enabling
///    that did not seem a good idea even if primive types can
///    behave that way.
/// 4. All mixed signed operations are disabled.  The compiler
///    can warn you if you do things like a scalar signed/unsigned
///    comparison but that seemed difficult for me to emulate.
///    So these operators are all undefined, and if a user wants
///    to do something mixed type they will have to explicitly
///    cast one of their arguments.
template <typename T, size_t SimdCount_, typename Derived>
class ArithmeticArray : public BaseArray<T, SimdCount_, Derived>
{
    static_assert(std::is_arithmetic<ScalarType<T>>::value, "Arithmetic array requires an arithematic POD or simd type");
    static_assert(!std::is_same<ScalarType<T>, bool>::value, "Arithmetic array cannot be of bools");
    using Base = BaseArray<T, SimdCount_, Derived>;
public:
    // These are already public in our base class, but
    // these are used in the implementation so doing
    // this helps avoid a lot of extra template/typename
    // keywords.
    using Base::data;
    using Base::Base;
    using Base::operator=;
    using Base::Update;
    using Base::Reduce;
    static constexpr auto SimdCount = Base::SimdCount;
    static constexpr auto ScalarCount = Base::ScalarCount;

public: // Compound operators

    // CompoundOpIsValid is a helper trait that relies
    // on SFINAE to disable certain mixed operations
    // like int += float.
    template <typename Other, typename valid = CompoundOpIsValid<Derived, Other>>
    Derived& operator+=(const Other& other)
    {
        // Update is a BaseArray function used to
        // modify the current object.  It accepts
        // a lambda as well as an arbitrary list of arguments.
        // The lambda will be called for each element of
        // any arrays involved, with the first argument
        // being a reference to the element from *this;
        return Update([](auto&& l, auto&& r) { l += r; }, other);
    }
    template <typename Other, typename valid = CompoundOpIsValid<Derived, Other>>
    Derived& operator-=(const Other& other)
    {
        return Update([](auto&& l, auto&& r) { l -= r; }, other);
    }
    template <typename Other, typename valid = CompoundOpIsValid<Derived, Other>>
    Derived& operator/=(const Other& other)
    {
        return Update([](auto&& l, auto&& r) { l /= r; }, other);
    }
    template <typename Other, typename valid = CompoundOpIsValid<Derived, Other>>
    Derived& operator*=(const Other& other)
    {
        return Update([](auto&& l, auto&& r) { l *= r; }, other);
    }

    // Only for integer types.
    template <typename Other, typename valid = CompoundOpIsValid<Derived, Other>>
    Derived& operator%=(const Other& other)
    {
        static_assert(std::is_integral_v<ScalarType<T>>);
        static_assert(std::is_integral_v<ScalarType<Other>>);
        return Update([](auto&& l, auto&& r) { l %= r; }, other);
    }

public: // Arithmetic friend operators
    friend Derived operator -(const Derived& c)
    {
        return Derived(
            [](auto&& d){ return -d;},
            c);
    }

    // InlineFriendReturn is a SFINAE helper trait.  The main thing to
    // take away here is that it's type will be the "common type"
    // of the two inputs.  (e.g. float * int = float)
    //
    // Beyond that, it's responsible for handling all of the
    // the magic regarding mixed types, ensuring single definitions,
    // disabling undesirable type combinations, etc.  It's definition
    // should be well commented if you wish to understand it, but
    // I've really tried to write things such that all SFINAE logic
    // can just be treated as a magic black box.
    template <typename T1, typename T2>
    friend auto operator -(const T1& l, const T2& r) -> InlineFriendReturn<T1, T2, Derived>
    {
        // Constructor defined in BaseArray that accepts a lambda
        // and an arbitrary set of arguments.  The array entries
        // in *this* are initialized by calling this lambda
        // for each element in the arrays in the arguments.
        return InlineFriendReturn<T1, T2, Derived>(
            [](auto&& l2, auto&& r2){ return l2 - r2;},
            l, r);
    }

    template <typename T1, typename T2>
    friend auto operator *(const T1& l, const T2& r) -> InlineFriendReturn<T1, T2, Derived>
    {
        return InlineFriendReturn<T1, T2, Derived>(
            [](auto&& l2, auto&& r2){ return l2 * r2;},
            l, r);
    }

    template <typename T1, typename T2>
    friend auto operator /(const T1& l, const T2& r) -> InlineFriendReturn<T1, T2, Derived>
    {
        return InlineFriendReturn<T1, T2, Derived>(
            [](auto&& l2, auto&& r2){ return l2 / r2;},
            l, r);
    }

    template <typename T1, typename T2>
    friend auto operator +(const T1& l, const T2& r) -> InlineFriendReturn<T1, T2, Derived>
    {
        return InlineFriendReturn<T1, T2, Derived>(
            [](auto&& l2, auto&& r2){ return l2 + r2;},
            l, r);
    }

    // Only for integer types.
    template <typename T1, typename T2>
    friend auto operator%(const T1& l, const T2& r) -> InlineFriendReturn<T1, T2, Derived>
    {
        static_assert(std::is_integral_v<ScalarType<T1>>);
        static_assert(std::is_integral_v<ScalarType<T2>>);
        return InlineFriendReturn<T1, T2, Derived>(
            [](auto&& l2, auto&&r2){ return l2 % r2; },
            l, r);
    }

public: // Logical friend operators

    // Similar to the above, save that an optional 4th argument to
    // InlineFriendReturn is specified.  This is how we set the
    // return type to explicitly be LaneMask, rather than
    // the common type of the arguments.
    template <typename T1, typename T2>
    friend auto operator >=(const T1& l, const T2& r) -> InlineFriendReturn<T1, T2, Derived, LaneMask<ScalarCount>>
    {
        return LaneMask<ScalarCount>(
            [](auto&& l2, auto&& r2){ return l2 >= r2;},
            l, r);
    }

    template <typename T1, typename T2>
    friend auto operator >(const T1& l, const T2& r) -> InlineFriendReturn<T1, T2, Derived, LaneMask<ScalarCount>>
    {
        return LaneMask<ScalarCount>(
            [](auto&& l2, auto&& r2){ return l2 > r2;},
            l, r);
    }

    template <typename T1, typename T2>
    friend auto operator <=(const T1& l, const T2& r) -> InlineFriendReturn<T1, T2, Derived, LaneMask<ScalarCount>>
    {
        return LaneMask<ScalarCount>(
            [](auto&& l2, auto&& r2){ return l2 <= r2;},
            l, r);
    }

    template <typename T1, typename T2>
    friend auto operator <(const T1& l, const T2& r) -> InlineFriendReturn<T1, T2, Derived, LaneMask<ScalarCount>>
    {
        return LaneMask<ScalarCount>(
            [](auto&& l2, auto&& r2){ return l2 < r2;},
            l, r);
    }

    template <typename T1, typename T2>
    friend auto operator ==(const T1& l, const T2& r) -> InlineFriendReturn<T1, T2, Derived, LaneMask<ScalarCount>>
    {
        return LaneMask<ScalarCount>(
            [](auto&& l2, auto&& r2){ return l2 == r2;},
            l, r);
    }

    template <typename T1, typename T2>
    friend auto operator !=(const T1& l, const T2& r) -> InlineFriendReturn<T1, T2, Derived, LaneMask<ScalarCount>>
    {
        return LaneMask<ScalarCount>(
            [](auto&& l2, auto&& r2){ return l2 != r2;},
            l, r);
    }

public: // more friend functions (not operators)
    friend Derived min(const Derived& l, const Derived& r)
    {
        return Derived(
            [](auto&& l2, auto&& r2){ return min(l2, r2);},
            l, r);
    }
    friend Derived max(const Derived& l, const Derived& r)
    {
        return Derived(
            [](auto&& l2, auto&& r2){ return max(l2, r2);},
            l, r);
    }

    // TODO: Seems like this does not really need to be "friend".
    /// Returns the value of v constrained by limits lo and hi.
    friend Derived clamp(const Derived& v, const Derived& lo, const Derived& hi)
    {
        assert(all(lo <= hi));
        return min(max(v, lo), hi);
    }

    friend Derived abs(const Derived& a)
    {
        if constexpr (std::is_signed_v<ScalarType<T>>) return max(a, -a);
        else return a;
    }

    friend Derived pow2(const Derived& d)
    {
        return Derived([](auto&& x) { return x*x; }, d);
    }

    friend Derived pow(const Derived& l, const Derived& r)
    {
        return Derived(
            [](auto&& l2, auto&& r2){ return pow(l2, r2);},
            l, r);
    }

    friend ScalarType<T> reduceMax(const Derived& c)
    {
        auto init = std::numeric_limits<ScalarType<T>>::lowest();
        return Reduce([](auto&& l, auto&& r) { l = std::max(l, reduceMax(r)); }, init, c);
    }

    friend ScalarType<T> reduceMin(const Derived& c)
    {
        auto init = std::numeric_limits<ScalarType<T>>::max();
        return Reduce([](auto&& l, auto&& r) { l = std::min(l, reduceMin(r)); }, init, c);
    }

    friend Derived Blend(const LaneMask<ScalarCount>& b, const Derived& c1, const Derived& c2)
    {
        return Derived(
            [](auto&& b2, auto&& l, auto&& r){ return Blend(b2, l, r); },
            b, c1, c2);
    }

    friend Derived inc(const Derived& in, const LaneMask<ScalarCount>& mask)
    {
        return Blend(mask, in + static_cast<ScalarType<T>>(1), in);
    }

    struct minOp
    {
        Derived operator()(const Derived& a, const Derived& b)
        { return min(a, b); }
    };

    struct maxOp
    {
        Derived operator()(const Derived& a, const Derived& b)
        { return max(a, b); }
    };
};

}}

#endif // mongo_common_simd_ArithmeticArray_H_
