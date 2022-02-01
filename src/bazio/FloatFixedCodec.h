#ifndef Sequel_Common_FloatFixedCodec_H_
#define Sequel_Common_FloatFixedCodec_H_

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
//  Description:
//  Defines class FloatFixedCodec.

#include <assert.h>
#include <cmath>
#include <limits>
#include <type_traits>

namespace PacBio {
namespace Primary {

/// FloatFixedCodec scales and rounds floating-point values to map them to and
/// from a (typically narrower) fixed-point integer representation. The
/// greatest and least integer values are used to represent special
/// floating-point values (NaN, +Inf, -Inf). Very large finite values may be
/// "clamped". If isfinite(x) && x < MinFiniteRepresentable(),
/// Decode(Encode(x)) == MinFiniteRepresentable(). A similar rule applies if x
/// > MaxFiniteRepresentable().
/// \note All member functions are declared constexpr.
template <typename FloatT, typename IntT>
class FloatFixedCodec
{
    static_assert(std::is_floating_point<FloatT>::value,
                  "First template parameter must be a floating-point type.");
    static_assert(std::is_integral<IntT>::value && std::is_signed<IntT>::value,
                  "Second template parameter must be signed integral type.");
    static_assert(sizeof(IntT) < sizeof(long),
                  "Second template parameter must be smaller than long int.");

public:     // Types
    using FloatType = FloatT;
    using IntType = IntT;

public:     // Structors and assignment
    /// Constructs a codec instance for which \a precisionUnit is represented
    /// as IntType(1).
    constexpr FloatFixedCodec(FloatType precisionUnit)
        : unit_ (precisionUnit)
    { }

public:     // Functions
    /// Maps from floating-point to fixed-point representation.
    constexpr IntType Encode(FloatType value) const
    {
        if (std::isnan(value)) return repNaN;
        if (std::isinf(value)) return (value < 0 ? repNegInf : repPosInf);

        assert(std::isfinite(value));

        // Candidate integer for representation.
        const long r = std::round(value / unit_);

        // We could return rep{neg,pos}Inf in these cases. This clamping policy
        // is more consistent with legacy behavior in Sequel PA BAZ I/O.
        if (r < repMinFinite) return repMinFinite;
        if (r > repMaxFinite) return repMaxFinite;

        return static_cast<IntType>(r);
    }

    /// Maps from fixed-point representation to floating-point.
    constexpr FloatType Decode(IntType repval) const
    {
        if (repval == repNaN) return std::numeric_limits<FloatType>::quiet_NaN();
        if (repval == repPosInf) return std::numeric_limits<FloatType>::infinity();
        if (repval == repNegInf) return -std::numeric_limits<FloatType>::infinity();

        assert(repval >= repMinFinite && repval <= repMaxFinite);

        return unit_ * static_cast<FloatType>(repval);
    }

    /// The least finite FloatType value that is exactly representable.
    constexpr FloatType MinFiniteRepresentable() const
    { return unit_ * static_cast<FloatType>(repMinFinite); }

    /// The greatest finite FloatType value that is exactly representable.
    constexpr FloatType MaxFiniteRepresentable() const
    { return unit_ * static_cast<FloatType>(repMaxFinite); }

    /// The smallest positive exactly representable value.
    constexpr FloatType PrecisionUnit() const
    { return unit_; }

private:     // Static constants
    static constexpr IntType repNaN = std::numeric_limits<IntType>::min();
    static constexpr IntType repNegInf = repNaN + 1;
    static constexpr IntType repPosInf = std::numeric_limits<IntType>::max();
    static constexpr IntType repMaxFinite = repPosInf - 1;
    static constexpr IntType repMinFinite = repNegInf + 1;

private:    // Data
    FloatType unit_;
};

}}      // namespace PacBio::Primary

#endif  // Sequel_Common_FloatFixedCodec_H_
