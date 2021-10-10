
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
    , m1L_ {0}
    , m1R_ {0}
    , m2_  {0}
    , lbi_ {0}
    , rbi_ {0}
    , canAddSample_ {true}
{
    static_assert(lag_ > 0, "Invalid lag value");

    auto i=lag_; while (i--) { lBuf_[i] = rBuf_[i] = T(0); }
}

template <typename T>
void AutocorrAccumulator<T>::AddSample(const T& value)
{
    assert (canAddSample_);
    auto offlessVal = value - Offset();
    auto m1RTerm = offlessVal;
    if (lbi_ < lag_)
    {
        lBuf_[lbi_++] = offlessVal;
        m1RTerm = 0;
    }
    m1L_ += rBuf_[rbi_%lag_];
    m1R_ += m1RTerm;
    m2_  += rBuf_[rbi_%lag_] * offlessVal;
    rBuf_[rbi_++%lag_] = offlessVal; rbi_ %= lag_;
    stats_.AddSample(value);   // StatAccumulator subtracts the offset itself.
}

template <typename T>
T AutocorrAccumulator<T>::Autocorrelation() const
{
    using U = Simd::ScalarType<T>;
    const auto nmk = stats_.Count() - boost::numeric_cast<U>(lag_);

    // Define R(k) = \frac{1}{[(n-k)*Variance()]}*\sum_{i=0}^{n-k-1} (x_i - \frac{m10_}{(n-k)}) (x_{i+k} - \frac{m1k_}{(n-k)})
    // Reference python code:
    // a, l = np.sin(1.7*np.arange(100)), 4
    // m1 = np.mean(a)
    // ac = m1*np.sum(a[:-l] + a[l:]) - len(a[l:])*m1*m1
    // autocorr_l4 = (np.sum(a[:-l]*a[l:]) - ac) / len(a[l:]) / np.var(a, ddof=1)
    // autocorr_l4 # 0.8617625800897488
#if 0
    auto mu = stats_.Mean();

    // m1L_ and m1R_ can be obtained without storing them
    // T fb_m1(0); T bb_m1(0);
    // auto i=lag_; while (i--) { fb_m1 += lBuf_[i]; bb_m1 += rBuf_[i]; }
    // T m1L_ = stats_.M1() - bb_m1; // Sum of first n-l elements
    // T m1R_ = stats_.M1() - fb_m1; // Sum of  last n-l elements
    // T ac = mu*((m1L_+m1R_) - nmk*mu);

    // Further, only m1L_ + m1R_ is needed, so tail_m1 == m1L_ + m1R_
    T m1x2 = 2*stats_.M1();
    T tail_m1(0); i=lag_; while (i--) { tail_m1 += lBuf_[i] + rBuf_[i]; }
    T ac = mu*((m1x2 - tail_m1) - nmk*mu);
#else
    T ac = m1L_ * m1R_ / nmk;   // TODO: change this low degree formula to the one above
#endif
    ac = (m2_ - ac) / (nmk * stats_.Variance());
    
    // Ensure range bounds and if insufficient data, return NaN.
    // Also, restore NaN that might have been dropped in max or min above.
    using Simd::Blend;
    static const T nan(std::numeric_limits<U>::quiet_NaN());

    ac = min(max(ac, -1.0f), 1.0f);  // clamp ac in [-1..1]
    return Blend((nmk < T(1.0f)) | isnan(ac), nan, ac);
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
    m1L_   *= s;
    m1R_   *= s;
    m2_    *= s;
    canAddSample_ = false; // the only case when it gets false
    return *this;
}

template <typename T>
AutocorrAccumulator<T>&
AutocorrAccumulator<T>::Merge(const AutocorrAccumulator& that)
{
    assert(canAddSample_ && that.CanAddSample());

    // !!! The operation is not commutative !!!
    // "this" is the left accumulator and "that" is the right one

    // Merge common statistics before processing tails
    stats_.Merge(that.stats_);
    m1L_ += that.m1L_;
    m1R_ += that.m1R_;
    m2_  += that.m2_;

    auto n1 = lag_ - that.lbi_;  // that.lBuf may be not filled up
    for (uint16_t k = 0; k < lag_ - n1; k++)
    {
        m1L_ += rBuf_[(rbi_+k)%lag_];
        m1R_ += that.lBuf_[k];
        // Sum of muls of overlapping elements
        m2_  += rBuf_[(rbi_+k)%lag_] * that.lBuf_[k];
        // Accept the whole right buffer
        rBuf_[(rbi_+k)%lag_] = that.rBuf_[(that.rbi_+n1+k)%lag_];
    }

    auto n2 = lag_ - lbi_;      // this->lBuf may be not filled up
    for (uint16_t k = 0; k < n2; ++k)
    {
        m1R_ -= that.lBuf_[k]; // Remove excessively overlapped values
        // No need to adjust m2_ as excessive values were mul by 0
        lBuf_[lbi_+k] = that.lBuf_[k];
    }

    // Advance buffer indices
    lbi_ += n2;
    rbi_ += (lag_-n1); rbi_ %= lag_;

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
