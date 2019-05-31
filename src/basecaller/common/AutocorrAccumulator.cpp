
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
AutocorrAccumulator<T>::AutocorrAccumulator(unsigned int lag, const T& offset)
    : stats_ {offset}
    , lagBuf_ {lag}
    , m1First_ {0}
    , m1Last_ {0}
    , m2_ {0}
    , lag_ {lag}
    , canAddSample_ {true}
{
    assert(lag > 0);
}

template <typename T>
void AutocorrAccumulator<T>::AddSample(const T& value)
{
    assert (canAddSample_);
    const auto offsetVal = value - Offset();
    if (lagBuf_.full())
    {
        m1First_ += lagBuf_.front();
        m1Last_ += offsetVal;
        m2_ += lagBuf_.front() * offsetVal;
    }
    lagBuf_.push_back(offsetVal);
    stats_.AddSample(value);   // StatAccumulator will subtract the offset itself.
}

template <typename T>
T AutocorrAccumulator<T>::Autocorrelation() const
{
    using U = Simd::ScalarType<T>;
    static const T nan {std::numeric_limits<U>::quiet_NaN()};
    const auto nmk = stats_.Count() - boost::numeric_cast<U>(Lag());
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
    return Blend((nmk < 1.0f) | mnan, nan, ac);
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
    // TODO: If s == 0, could clear lagBuf_ and set canAddSample_ = true.
    // TODO: If s == 1, could preserve canAddSample_.
    return *this;
}


template <typename T>
AutocorrAccumulator<T>&
AutocorrAccumulator<T>::Merge(const AutocorrAccumulator& that)
{
    assert(lag_ == that.lag_);
    stats_.Merge(that.stats_);
    // TODO: If CanAddSample(), could exactly merge m1First_.
    // TODO: If that.CanAddSample(), could exactly merge m1Last_.
    // TODO: Would need to add "front lag buffer", to be able exactly merge m2_.
    m1First_ += that.m1First_;
    m1Last_ += that.m1Last_;
    m2_ += that.m2_;
    canAddSample_ = false;
    return *this;
}

template <typename T>
AutocorrAccumulator<T>&
AutocorrAccumulator<T>::operator+=(const AutocorrAccumulator& that)
{
    const auto m = (Count() == 0);
    // TODO: Handle the nonuniform SIMD case.
    using PacBio::Simd::all;
    assert(all(m) || lag_ == that.lag_);
    if (all(m)) *this = that;
    else Merge(that);
    return *this;
}


// Explicit instantiation.
template class AutocorrAccumulator<float>;
template class AutocorrAccumulator<Simd::m512f>;
template class AutocorrAccumulator<LaneArray<float, 64u>>;

}}      // namespace PacBio::Mongo
