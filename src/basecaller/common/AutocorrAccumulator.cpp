
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
    , m2_  {0}
    , fbi_ {0}
    , bbi_ {0}
{
    static_assert(lag_ > 0, "Invalid lag value");

    for (auto k = 0u; k < lag_; ++k) { fBuf_[k] = bBuf_[k] = T(0); }
}

template <typename T>
void AutocorrAccumulator<T>::AddSample(const T& value)
{
    auto valLessOffset = value - Offset();
    if (fbi_ < lag_)
    {
        fBuf_[fbi_++] = valLessOffset;
    }
    m2_  += bBuf_[bbi_%lag_] * valLessOffset;
    bBuf_[bbi_++%lag_] = valLessOffset; bbi_ %= lag_;
    stats_.AddSample(value);   // StatAccumulator subtracts the offset itself.
}

template <typename T>
T AutocorrAccumulator<T>::Autocorrelation() const
{
    using U = Simd::ScalarType<T>;
    const auto nmk = Count() - boost::numeric_cast<U>(lag_);

    // Define R(k) = \frac{1}{[(n-k)*Variance()]}*\sum_{i=0}^{n-k-1} (x_i - \frac{m10_}{(n-k)}) (x_{i+k} - \frac{m1k_}{(n-k)})
    // Reference python code (with offset = 0):
    // a, l = np.sin(1.7*np.arange(100)), 4
    // mu = np.mean(a)
    // ac = mu*(np.sum(a[:-l] + a[l:]) - len(a[l:])*mu)
    // autocorr_l4 = (np.sum(a[:-l]*a[l:]) - ac) / len(a[l:]) / np.var(a, ddof=1)
    // autocorr_l4 # 0.8617625800897488
    auto mu = Mean() - Offset();
    auto m1x2 = 2*stats_.M1();
    for (auto k = 0u; k < lag_; ++k) { m1x2 -= fBuf_[k] + bBuf_[k]; }
    auto ac = mu*(m1x2 - nmk*mu);
    ac = (m2_ - ac) / ((nmk - 1) * stats_.Variance());

    // Ensure range bounds and if insufficient data, return NaN.
    // Also, restore NaN that might have been dropped in max or min above.
    using Simd::Blend;
    static const T nan(std::numeric_limits<U>::quiet_NaN());

    ac = min(max(ac, -1.0f), 1.0f);  // clamp ac in [-1..1]
    return Blend((nmk < T(1.0f)) | isnan(ac), nan, ac);
}


template <typename T>
AutocorrAccumulator<T>&
AutocorrAccumulator<T>::Merge(const AutocorrAccumulator& that)
{
    // !!! The operation is not commutative !!!
    // "this" is the left accumulator and "that" is the right one

    // Merge common statistics before processing tails
    stats_.Merge(that.stats_);
    m2_  += that.m2_;

    auto n1 = lag_ - that.fbi_;  // that fBuf may be not filled up
    for (uint16_t k = 0; k < lag_ - n1; k++)
    {
        // Sum of muls of overlapping elements
        m2_  += bBuf_[(bbi_+k)%lag_] * that.fBuf_[k];
        // Accept the whole back buffer
        bBuf_[(bbi_+k)%lag_] = that.bBuf_[(that.bbi_+n1+k)%lag_];
    }

    auto n2 = lag_ - fbi_;      // this fBuf may be not filled up
    for (uint16_t k = 0; k < n2; ++k)
    {
        // No need to adjust m2_ as excessive values were mul by 0
        fBuf_[fbi_+k] = that.fBuf_[k];
    }

    // Advance buffer indices
    fbi_ += n2;
    bbi_ += (lag_-n1); bbi_ %= lag_;

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
    if (all(m)) 
        *this = that;
    else 
        Merge(that);
    return *this;
}


// Explicit instantiation.
// template class AutocorrAccumulator<float>;
template class AutocorrAccumulator<LaneArray<float, laneSize>>;

}}      // namespace PacBio::Mongo
