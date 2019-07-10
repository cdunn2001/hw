#ifndef mongo_common_StatAccumulator_H_
#define mongo_common_StatAccumulator_H_

// Copyright (c) 2017-2019, Pacific Biosciences of California, Inc.
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
/// \file StatAccumulator.h
/// Defines class template StatAccumulator.

#include <cmath>
#include <pacbio/logging/Logger.h>
#include <common/simd/SimdVectorTypes.h>
#include "NumericUtil.h"
#include "StatAccumState.h"

namespace PacBio {
namespace Mongo {

// TODO: Add declaration decorators to enable use on CUDA device.

// TODO: Use a type-traits-like technique to enable accepting
// [Const]LaneArrayRef as function parameters instead of [const] LaneArray&.

/// \brief A bundle of moment statistics.
/// \details
/// Provides mean and variance statistics on augmentable dataset.
/// Offset is defined at construction, but can be modified.
/// The closer Offset() is to Mean(), the greater the precision of Variance().
/// \tparam VF A LaneArray type.
template <typename VF>
class StatAccumulator
{
public:     // Types
    using VBool = typename Simd::SimdConvTraits<VF>::bool_conv;

public:     // Structors
    explicit StatAccumulator(const VF& offset = VF(0))
        : offset_ {offset}
        , m0_ {0}
        , m1_ {0}
        , m2_ {0}
    { }

    StatAccumulator(const StatAccumulator& that) = default;

    StatAccumulator(const StatAccumState& state)
        : offset_ {state.offset}
        , m0_ {state.moment0}
        , m1_ {state.moment1}
        , m2_ {state.moment2}
    { }
    /// Constructs an instance from previously computed moments.
    /// m0 and m2 cannot be negative.
    StatAccumulator(const VF& m0, const VF& m1, const VF& m2,
                    const VF& offset = VF(0))
        : offset_ {offset}
        , m0_ {m0}
        , m1_ {m1}
        , m2_ {m2}
    {
        assert(all(m0_ >= 0.0f));
        assert(all(m2_ >= 0.0f));
    }

public:     // Const methods
    /// Number of samples aggregated.
    const VF& Count() const
    { return m0_; }

    /// The mean of the aggregated samples. NaN if Count() == 0.
    VF Mean() const
    {
        return m1_/m0_ + offset_;
    }

    /// The unbiased sample variance of the aggregated samples.
    /// NaN if Count() < 2.
    VF Variance() const
    {
        using std::max;
        static const VF nan {VF(std::numeric_limits<float>::quiet_NaN())};
        auto var = m1_ * m1_ / m0_;
        var = (m2_ - var) / (m0_ - 1.0f);
        var = max(var, 0.0f);
        return Blend(m0_ > 1.0f, var, nan);
    }

    const VF& Offset() const
    { return offset_; }

    /// Returns a copy of this with all moments scaled by \a s.
    StatAccumulator operator*(float s) const
    {
        StatAccumulator r {offset_};
        r.m0_ = m0_ * s;
        r.m1_ = m1_ * s;
        r.m2_ = m2_ * s;
        return r;
    }

    StatAccumState GetState() const
    {
        return StatAccumState
        {
            {offset_.cbegin(), offset_.cend()},
            {m0_.cbegin(), m0_.cend()},
            {m1_.cbegin(), m1_.cend()},
            {m2_.cbegin(), m2_.cend()}
        };
    }

    const VF& M1() const
    { return m1_; }

    const VF& M2() const
    { return m2_; }

public:     // Mutating methods
    StatAccumulator& operator=(const StatAccumulator& that) = default;

    /// \brief Merge in another StatAccumulator object.
    /// \details
    /// The offset of the result will be a weighted average of this->Offset()
    /// and other.Offset().
    /// If other.Count() == 0, returns *this (SIMD-wise).
    StatAccumulator& operator+=(StatAccumulator other)
    {
        // TODO: Since the alignment of other might be 64B, do we need to pass
        // by const reference and explicitly copy?
        if (all(other.m0_ == 0.0f))
        {
            return *this;
        }

        if (any(isnan(offset_)))
        {
            // TODO: Really don't have enough context information to know what to do here.
            // Should probably throw or just assert(none(isnan(offset_))).
            PBLOG_ERROR << "Encountered NaN in StatAccumulator. Resetting.";
            Reset(isnan(offset_));
            offset_ = Blend(isnan(offset_), 0.0f, offset_);
        }

        const VF w = other.m0_ / (m0_ + other.m0_);
        VF offsetNew = (1.0f - w)*offset_ + w*other.offset_;
        offsetNew = Blend(isnan(offsetNew), offset_, offsetNew);
        other.Offset(offsetNew);
        this->Offset(offsetNew);
        Merge(other);
        return *this;
    }

    /// Scales all moments by \a s, as if s * Count() samples had been processed.
    StatAccumulator& operator*=(float s)
    {
        using std::round;
        m0_ = m0_ * s;
        m1_ *= s;
        m2_ *= s;
        return *this;
    }

    /// Aggregate another data sample into the moments.
    void AddSample(const VF& value)
    {
        const auto x = value - offset_;
        m0_ += 1.0f;
        m1_ += x;
        m2_ += x*x;
    }

    /// Aggregate another data sample into the moments if corresponding mask
    /// element is true.
    void AddSample(const VF& value, const VBool& mask)
    {
        const auto x = value - offset_;
        m0_ = Blend(mask, m0_ + 1.0f, m0_);
        m1_ = Blend(mask, m1_ + x, m1_);
        m2_ = Blend(mask, m2_ + x*x, m2_);
    }

    /// Merge another instance with the same offset into this one.
    StatAccumulator& Merge(const StatAccumulator& other)
    {
        using PacBio::Simd::all;
        assert(all(other.offset_ == offset_));
        m0_ += other.m0_;
        m1_ += other.m1_;
        m2_ += other.m2_;
        return *this;
    }

    /// Reset just the SIMD elements for which \a mask is true.
    void Reset(const VBool& mask)
    {
        m0_ = Blend(mask, VF(0), m0_);
        m1_ = Blend(mask, VF(0), m1_);
        m2_ = Blend(mask, VF(0), m2_);
    }

    /// Reset all elements
    void Reset()
    { m0_ = m1_ = m2_ = VF(0); }

    /// \brief Adjust moments to account for a uniform shift of the samples already added.
    /// \details
    /// This is equivalent to shifting the offset by the same amount without
    /// modifying the moments.
    void Shift(const VF& shift)
    {
        offset_ += shift;
    }


    /// Set the offset. Adjust moments to preserve mean and variance.
    void Offset(const VF& value)
    {
        if (all(value == offset_)) return;

        const auto m1new = m1_ + m0_*(offset_ - value);
        const auto m2new = m2_ + (pow2(m1new) - pow2(m1_)) / m0_;

        offset_ = value;
        m1_ = m1new;
        // Guard against NaN.
        m2_ = Blend(m0_ == VF(0), VF(0), m2new);
    }

    /// Set the internal moments to arbitrary values.
    /// m0 and m2 cannot be negative.
    void Moments(const VF& m0, const VF& m1, const VF& m2,
                 const VF& offset = VF(0))
    {
        assert(all(m0 >= 0.0f));
        assert(all(m2 >= 0.0f));
        m0_ = m0;
        m1_ = m1;
        m2_ = m2;
        offset_ = offset;
    }

private:    // Data
    // Data offset.
    VF offset_;

    // The number of data samples included in the moment statistics.
    // Floating-point to support scaling smoothly.
    VF m0_;

    // First moment.
    VF m1_;

    // Second moment.
    VF m2_;
};

}}     // PacBio::Mongo

#endif  // mongo_common_StatAccumulator_H_
