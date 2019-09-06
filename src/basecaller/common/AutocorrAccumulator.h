#ifndef mongo_common_AutocorrAccumulator_H_
#define mongo_common_AutocorrAccumulator_H_

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
/// \file AutocorrAccumulator.h
/// Defines class template AutocorrAccumulator.

#include <common/simd/SimdConvTraits.h>

#include "AlignedCircularBuffer.h"
#include "AutocorrAccumState.h"
#include "NumericUtil.h"
#include "StatAccumulator.h"

namespace PacBio {
namespace Mongo {

// TODO: Add declaration decorators to enable use on CUDA device.

/// \brief A progressive estimate of the normalized autocorrelation.
/// \details
/// Assumes that data samples are derived from a stationary process and added
/// in order.
/// \tparam T The floating-point data type. SIMD vector types (e.g., m512f)
/// are supported.
/// Two modes: Initially, can add samples and get statistics.
/// \note If the object is produced by operator* or has been modified by
/// operator*= or operator+=, no more samples can be added.
template <typename T>
class AutocorrAccumulator
{
public:     // Types
    /// Boolean type that is vectorized as T.
    using BoolType = Simd::BoolConv<T>;

public:     // Structors
    /// Construct an instance with specified offset.
    explicit AutocorrAccumulator(const T& offset = T(0));

    /// Copy constructor.
    AutocorrAccumulator(const AutocorrAccumulator& that) = default;

    AutocorrAccumulator(const AutocorrAccumState& state)
        : stats_ {state.basicStats}
        , m1First_ {state.moment1First}
        , m1Last_ {state.moment1Last}
        , m2_ {state.moment2}
        , canAddSample_ {false}
    { }

public:     // Const methods
    AutocorrAccumState GetState() const
    {
        return AutocorrAccumState
        {
            stats_.GetState(),
            {m1First_.cbegin(), m1First_.cend()},
            {m1Last_.cbegin(), m1Last_.cend()},
            {m2_.cbegin(), m2_.cend()},
        };
    }

    const T& M1First() const
    { return m1First_; }

    const T& M1Last() const
    { return m1Last_; }

    const T& M2() const
    { return m2_; }

    /// Number of samples accumulated.
    const T& Count() const
    { return stats_.Count(); }

    /// The mean of accumulated samples. NaN if Count() == 0.
    T Mean() const
    { return stats_.Mean(); }

    /// The unbiased sample variance of the accumulated samples.
    /// NaN if Count() < 2.
    T Variance() const
    { return stats_.Variance(); }

    /// The autocorrelation coefficient of the accumulated samples at the
    /// lag AutocorrAccumState::lag.
    /// NaN if Count() < AutocorrAccumState::lag + 1.
    /// -1.0 <= Autocorrelation() <= 1.0.
    T Autocorrelation() const;

    /// The current offset.
    const T& Offset() const
    { return stats_.Offset(); }

    /// Returns a copy of *this with all moments scaled by \a s.
    AutocorrAccumulator operator*(float s) const;

    /// Whether *this is in a state that allows addition of more sample data.
    bool CanAddSample() const
    { return canAddSample_; }

public:     // Mutating methods
    /// Copy assignment.
    AutocorrAccumulator& operator=(const AutocorrAccumulator& that) = default;

    /// \brief Merge in another AutocorrAccumulator object.
    /// \details
    /// If *this is empty, assign that to *this.
    /// Otherwise, lag and offset of that must be consistent with this.
    /// \note If T is SIMD type, behavior undefined if any element of Count()
    /// is nonzero and lags are not equal.
    /// \returns *this.
    AutocorrAccumulator& operator+=(const AutocorrAccumulator& that);

    /// Scales all moments by \a s, as if s * Count() samples had been accumulated.
    /// \returns *this.
    AutocorrAccumulator& operator*=(float s);

    /// Accumulate another data sample into the moments.
    void AddSample(const T& value);

    /// Accumulate a sequence of data samples into the moments.
    /// \tparam ForwardIt A forward iterator that refers to a value of type T.
    template <typename ForwardIt>
    void AddSamples(ForwardIt first, ForwardIt last)
    {
        while (first != last) AddSample(*first++);
    }

    /// Merge another instance with the same lag and offset into this one.
    /// \returns *this.
    AutocorrAccumulator& Merge(const AutocorrAccumulator& that);

    /// Updates the object to account for a uniform shift of the samples
    /// already added.
    void Shift(const T& shift)
    { stats_.Shift(shift); }

    /// Reset to initial state (conserve offset).
    void Reset()
    {
        stats_.Reset();
        lagBuf_.clear();
        m1First_ = m1Last_ = m2_ = T(0);
        canAddSample_ = true;
    }

private:    // Data
    StatAccumulator<T> stats_;

    // TODO: Pretty sure that AlignedCircularBuffer uses heap allocation.
    // Replace with something based on something like boost::static_vector.
    AlignedCircularBuffer<T> lagBuf_;

    T m1First_;     // First moment of the first Count() - AutocorrAccumState::lag samples.
    T m1Last_;     // First moment of the last Count() - AutocorrAccumState::lag samples.
    T m2_;      // Generalized second moment. Sum of x_i * x_i+AutocorrAccumState::lag.
    bool canAddSample_;
};

}} // PacBio::Mongo

#endif  // mongo_common_AutocorrAccumulator_H_
