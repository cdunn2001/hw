#ifndef Sequel_Basecaller_Common_IntInterval_H_
#define Sequel_Basecaller_Common_IntInterval_H_

// Copyright (c) 2018, Pacific Biosciences of California, Inc.
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

#include <assert.h>
#include <algorithm>
#include <limits>
#include <string>
#include <type_traits>

#include <pacbio/PBException.h>

namespace PacBio {
namespace Primary {

// TODO: Eliminate the multitude of representations of the empty set?

// See _Introduction to Interval Analysis_,
// by Ramon E. Moore, R. Baker Kearfott, and Michael J. Cloud,
// SIAM, 2009.

/// A class template that represents an interval (or range) of integers.
template <typename IntType>
class IntInterval
{
    static_assert(std::is_integral<IntType>::value && !std::is_signed<IntType>::value,
                  "Template argument must be an unsigned integer type.");

public:     //types
    using ElementType = IntType;
    using SizeType = std::make_unsigned_t<IntType>;

public:     // structors
    /// Default constructor creates an empty interval.
    constexpr IntInterval()
        : lower_ {0}
        , upper_ {0}
    { }

    /// Create an interval [lower, upper).
    /// Requires lower <= upper.
    constexpr IntInterval(IntType lower, IntType upper)
        : lower_ {lower}
        , upper_ {upper}
    {
        if (upper < lower)
        {
            using std::to_string;
            throw PBException("Bad arguments for constructor: IntInterval("
                              + to_string(lower) + ", " + to_string(upper)
                              + ").  Second argument may not be less than first.");
        }
    }

public:     // const methods
    bool Empty() const
    { return lower_ == upper_; }

    SizeType Size() const
    { return static_cast<SizeType>(upper_ - lower_); }
    // TODO: To avoid overflow, this would need to be a bit more complicated.
    // When we adopt C++17, we can effectively partially specialize this.
    // For the time being, we only support unsigned integers.

    /// The smallest value in the interval.
    IntType Lower() const
    { return lower_; }

    /// One more than the largest value in the interval.
    IntType Upper() const
    { return upper_; }

public:     // modifying methods
    IntInterval& AddWithSaturation(IntType a)
    {
        // TODO: Restore this when support for signed IntType is added.
        // if (a < 0) return SubtractWithSaturation(-a);

        constexpr auto highest = std::numeric_limits<IntType>::max();
        lower_ = (lower_ > highest - a ? highest : lower_ + a);
        upper_ = (upper_ > highest - a ? highest : upper_ + a);
        return *this;
    }

    /// Like operator-=, but guards against overflow.
    IntInterval& SubtractWithSaturation(IntType a)
    {
        // TODO: Restore this when support for signed IntType is added.
        // if (a < 0) return AddWithSaturation(-a);

        constexpr auto lowest = std::numeric_limits<IntType>::min();
        lower_ = (lower_ < lowest + a ? lowest : lower_ - a);
        upper_ = (upper_ < lowest + a ? lowest : upper_ - a);
        return *this;
    }

public:     // compound assignment operators
    /// Over- or underflow results in undefined behavior.
    IntInterval& operator+=(const IntType a)
    {
        lower_ += a;
        upper_ += a;
        assert(lower_ <= upper_);
        return *this;
    }

    /// Over- or underflow results in undefined behavior.
    IntInterval& operator-=(const IntType a)
    {
        lower_ -= a;
        upper_ -= a;
        assert(lower_ <= upper_);
        return *this;
    }

private:
    // The bounds of the interval.
    IntType lower_;     // Inclusive lower bound
    IntType upper_;     // Exclusive upper bound
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

// TODO: Define conversion from IntType to IntInterval<IntType>.

// TODO: Define relationships between IntType and IntInterval<IntType>.

// TODO: Define arithmetic operators.

}}  // PacBio::Primary

#endif // Sequel_Basecaller_Common_IntInterval_H_
