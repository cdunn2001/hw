#ifndef mongo_common_IntInterval_H_
#define mongo_common_IntInterval_H_

// Copyright (c) 2018-2021, Pacific Biosciences of California, Inc.
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
//  Defines a type that represents an interval or range of integers.

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <type_traits>

#include <boost/numeric/conversion/cast.hpp>

namespace PacBio::Mongo {

// TODO: Eliminate the multitude of representations of the empty set?

// TODO: Current design does not permit intervals that contain the largest value
// representable by ElementType.

// See _Introduction to Interval Analysis_,
// by Ramon E. Moore, R. Baker Kearfott, and Michael J. Cloud,
// SIAM, 2009.

/// A class template that represents an interval (or range) of integers.
/// Arithmetic operators represent elementwise arithmetic.
template <typename IntType>
class IntInterval
{
    static_assert(std::is_integral<IntType>::value,
                  "Template argument must be an integer type.");

    // This constraint is primarily motivated by a desire to keep the definition
    // of Center() simple (for now).
    static_assert(sizeof(IntType) <= 4u,
                  "IntInterval only supports types up to four bytes in size.");

public:     // types
    using ElementType = IntType;
    using SizeType = std::make_unsigned_t<IntType>;

public:     // structors
    // Should be ok, but can cause error with clang 12 for default constructed
    // const instances.
    // IntInterval() = default;
    IntInterval() {}

    IntInterval(const IntType& lower, const IntType& upper)
        : lower_ {lower}
        , upper_ {upper}
    {
        assert(lower_ <= upper_);
    }

    /// Create an interval [lower, upper).
    /// Requires lower <= upper.
    template <typename T>
    explicit IntInterval(const T& lower, const T& upper)
        : lower_ {boost::numeric_cast<IntType>(lower)}
        , upper_ {boost::numeric_cast<IntType>(upper)}
    {
        static_assert(std::is_integral_v<T>,
                      "Argument type must be integral.");
        assert(lower_ <= upper_);
    }

public:     // const methods
    bool Empty() const
    {
        assert(lower_ <= upper_);
        return lower_ == upper_;
    }

    SizeType Size() const
    {
        assert(lower_ <= upper_);
        // Assumes two's complement representation of signed integers.
        const auto lu = static_cast<SizeType>(lower_);
        const auto uu = static_cast<SizeType>(upper_);
        return uu - lu;
    }

    /// The smallest value in the interval.
    IntType Lower() const
    { return lower_; }

    /// One more than the largest value in the interval.
    IntType Upper() const
    { return upper_; }

    // TODO: Would be nice to make this more generic and more airtight.
    // Eliminating this function would allow IntType to be wider than 4 bytes.
    double Center() const
    {
        if (Empty()) return std::nan("");
        // Possible loss of precision here.
        const double ld = Lower();
        const double um1d = Upper() - 1;
        return 0.5 * (ld + um1d);
    }

    /// Smallest integer >= the real center of a non-empty interval.
    IntType CenterInt() const
    {
        assert(!Empty());
        return Lower() + static_cast<IntType>(Size()/2u);
    }

public:     // modifying methods
    /// Sets *this to the empty interval.
    IntInterval& Clear() noexcept
    {
        lower_ = upper_ = 0;
        return *this;
    }

public:     // compound assignment operators
    /// The interval defined by adding \a a to all the elements in this interval.
    /// Overflow results in undefined behavior.
    IntInterval& operator+=(const IntType a)
    {
        assert(lower_ <= upper_);
        lower_ += a;
        upper_ += a;
        assert(lower_ <= upper_);
        return *this;
    }

    /// The interval defined by subtracting \a a from all the elements in this
    /// interval.
    /// Overflow results in undefined behavior.
    IntInterval& operator-=(const IntType a)
    {
        assert(lower_ <= upper_);
        lower_ -= a;
        upper_ -= a;
        assert(lower_ <= upper_);
        return *this;
    }

private:
    // The bounds of the interval.
    IntType lower_ {};  // Inclusive lower bound
    IntType upper_ {};  // Exclusive upper bound
};


//  Namespace-scope operators and relations

template <typename IntType> inline
bool operator==(const IntInterval<IntType>& a, const IntInterval<IntType>& b)
{
    if (a.Empty() && b.Empty()) return true;
    return a.Lower() == b.Lower() && a.Upper() == b.Upper();
}

template <typename IntType> inline
bool operator!=(const IntInterval<IntType>& a, const IntInterval<IntType>& b)
{ return !(a == b); }

/// Is Intersection(a, b) empty?
template <typename IntType> inline
bool Disjoint(const IntInterval<IntType>& a, const IntInterval<IntType>& b)
{
    return a.Empty() || b.Empty()
            || a.Upper() <= b.Lower()
            || b.Upper() <= a.Lower();
}

/// Is the union of a and b connected (i.e., a single interval)?
template <typename IntType> inline
bool IsUnionConnected(const IntInterval<IntType>& a, const IntInterval<IntType>& b)
{
    if (a.Empty() || b.Empty()) return true;
    return a.Upper() >= b.Lower() && b.Upper() >= a.Lower();
}

/// Are a and b adjacent (i.e., Disjoint and IsUnionConnected are both true) and
/// in increasing order (i.e., all elements in a are < all elements in b).
/// If either a or b is empty, returns true.
template <typename IntType> inline
bool AreOrderedAdjacent(const IntInterval<IntType>& a, const IntInterval<IntType>& b)
{
    if (a.Empty() || b.Empty()) return true;
    return a.Upper() == b.Lower();
}

template <typename IntType>
inline IntInterval<IntType>
Intersection(const IntInterval<IntType>& a, const IntInterval<IntType>& b)
{
    if (Disjoint(a, b)) return IntInterval<IntType>{};
    using std::max;
    using std::min;
    return IntInterval<IntType>(max(a.Lower(), b.Lower()),
                                min(a.Upper(), b.Upper()));
}

/// The smallest interval that contains a and b.
/// \note If IsUnionConnected(a, b), then Hull(a, b) is the union of a and b.
template <typename IntType>
inline IntInterval<IntType>
Hull(const IntInterval<IntType>& a, const IntInterval<IntType>& b)
{
    if (a.Empty()) return b;
    if (b.Empty()) return a;

    using std::max;
    using std::min;
    return IntInterval<IntType>(min(a.Lower(), b.Lower()),
                                max(a.Upper(), b.Upper()));
}

// TODO: Could define conversion from IntType to IntInterval<IntType>.

/// The interval defined by adding \a n to all the elements in \a ii.
template <typename IntType>
inline IntInterval<IntType>
operator+(const IntInterval<IntType>& ii, IntType n)
{
    IntInterval<IntType> r {ii};
    r += n;
    return r;
}

/// The interval defined by adding \a n to all the elements in \a ii.
template <typename IntType>
inline IntInterval<IntType>
operator+(IntType n, const IntInterval<IntType>& ii)
{
    IntInterval<IntType> r {ii};
    r += n;
    return r;
}

// TODO: Define more arithmetic operators.

}   // namespace PacBio::mongo

#endif // mongo_common_IntInterval_H_
