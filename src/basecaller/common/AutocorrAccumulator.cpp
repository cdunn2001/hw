
#include "AutocorrAccumulator.h"

#include <algorithm>
#include <limits>
#include <boost/numeric/conversion/cast.hpp>

#include <common/simd/SimdVectorTypes.h>

#include "LaneArray.h"
#include "NumericUtil.h"

namespace PacBio {
namespace Mongo {

template <typename T>
AutocorrAccumulator<T>::AutocorrAccumulator(const T& offset)
    : stats_ {offset}
    , m1First_ {0}
    , m1Last_ {0}
    , m2_ {0}
    , canAddSample_ {true}
{
    static_assert(AutocorrAccumState::lag > 0, "Invalid lag value");

    frontBuf_.reserve(AutocorrAccumState::lag);
    backBuf_.set_capacity(AutocorrAccumState::lag);
}

template <typename T>
void AutocorrAccumulator<T>::AddSample(const T& value)
{
    assert (canAddSample_);
    const auto offsetVal = value - Offset();
    // back buf
    if (backBuf_.full())
    {
        m1First_ += backBuf_.front();
        m1Last_ += offsetVal;
        m2_ += backBuf_.front() * offsetVal;
    } else {
        frontBuf_.push_back(offsetVal);
    }
    backBuf_.push_back(offsetVal);
    stats_.AddSample(value);   // StatAccumulator will subtract the offset itself.
}

template <typename T>
T AutocorrAccumulator<T>::Autocorrelation() const
{
    using U = Simd::ScalarType<T>;
    static const T nan {std::numeric_limits<U>::quiet_NaN()};
    const auto nmk = stats_.Count() - boost::numeric_cast<U>(AutocorrAccumState::lag);
#if 1
    // If we define R(k) = \sum_{i=0}^{n-k-1} (x_i - m10_/(n-k)) (x_{i+k} - m1k_/(n-k)) / [(n-k)*Variance()]
    // As of 2017-09-12, this definition appears to be more accurate than the
    // alternative below, based on preliminary testing.
    T ac = m1First_ * m1Last_ / nmk;
    ac = (m2_ - ac) / (nmk * stats_.Variance());
#else
    // Approximation: m10_ == m1k_ == nmk*Mean().
    T ac = m2_ / nmk - pow2(sa_.Mean());
    ac /= sa_.Variance();
#endif
    // Ensure range bounds.
    using std::max;
    using std::min;
    const auto mnan = isnan(ac);
    ac = max(ac, -1.0f);
    ac = min(ac, 1.0f);

    // If insufficient data, return NaN.
    // Also, restore NaN that might have been dropped in max or min above.
    using Simd::Blend;
    return Blend((nmk < T(1.0f)) | mnan, nan, ac);
}

template <typename T>
AutocorrAccumulator<T> AutocorrAccumulator<T>::operator*(float s) const
{
    AutocorrAccumulator r {*this};
    r *= s;
    return r;
}

template <typename T>
AutocorrAccumulator<T>& AutocorrAccumulator<T>::operator*=(float s)
{
    stats_ *= s;
    m1First_ *= s;
    m1Last_ *= s;
    m2_ *= s;
    canAddSample_ = false;
    return *this;
}

template <typename T>
AutocorrAccumulator<T>&
AutocorrAccumulator<T>::Merge(const AutocorrAccumulator& that)
{
    auto lag_ = AutocorrAccumState::lag;
    // The operation is now not commutative.  *this is the front and "that" is the back
    if (that.backBuf_.full() || !canAddSample_ || !that.CanAddSample())
    {
        stats_.Merge(that.stats_);
        m1First_ += that.m1First_;
        m1Last_ += that.m1Last_;
        m2_ += that.m2_;
        // "that" has more than lag_ samples i.e. the typical case
        if (canAddSample_ && that.CanAddSample())
        {
            auto backBufIter = backBuf_.begin();
            auto thatFrontBufIter = that.frontBuf_.begin();
            // skip (lag_ - backBuf_.size()) entries to preserve lag
            thatFrontBufIter += (lag_ - backBuf_.size());
            for (;
                backBufIter != backBuf_.end() && thatFrontBufIter != that.frontBuf_.end();
                ++backBufIter, ++thatFrontBufIter)
            {
                // add the front
                m1Last_ += *thatFrontBufIter;
                m1First_ += *backBufIter;
                m2_ += (*backBufIter) * (*thatFrontBufIter);
            }
            // The "that" output back buffer is appended to the back buffer of "this"
            for (auto thatBackBufVal : that.backBuf_)
            {
                backBuf_.push_back(thatBackBufVal);
            }
            // 0 <= nCopyStartBuf <= lag_; it is zero if frontBuf_ is full,
            // and the loop below is not executed
            // otherwise, copy the first nCopyStartBuf values from that.frontBuf
            unsigned nCopyStartBuf = lag_ - frontBuf_.size();
            // append nCopyStartBuf entries from that.frontBuf_ to this->frontBuf_
            unsigned iCopy = 0;
            for (thatFrontBufIter = that.frontBuf_.begin();
                iCopy < nCopyStartBuf;
                ++iCopy, ++thatFrontBufIter)
            {
                frontBuf_.push_back(*thatFrontBufIter);
            }
        } else
        {
            canAddSample_ = false;
        }
    } else
    {
        // "that" sequence is less than lag_ entries long so we can
        // simply add the data from that to this using AddSample()
      const auto offsetVal = Offset();
      for (auto thatFrontBufVal : that.frontBuf_)
        {
            // add offset value before adding the sample, since the buffer contains
            // offset adjusted values
            AddSample(thatFrontBufVal+offsetVal);
        }
    }
    canAddSample_ &= that.CanAddSample();

    return *this;
}

template <typename T>
AutocorrAccumulator<T>&
AutocorrAccumulator<T>::operator+=(const AutocorrAccumulator& that)
{
    const auto m = (Count() == T(0));
    // TODO: Handle the nonuniform SIMD case.
    using PacBio::Simd::all;
    assert(all(m));
    if (all(m)) *this = that;
    else Merge(that);
    return *this;
}


// Explicit instantiation.
// template class AutocorrAccumulator<float>;
template class AutocorrAccumulator<LaneArray<float, laneSize>>;

}}      // namespace PacBio::Mongo
