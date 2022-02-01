#ifndef mongo_common_NumericUtil_H_
#define mongo_common_NumericUtil_H_

#include <algorithm>
#include <array>
#include <numeric>
#include <cmath>
#include <type_traits>
#include <iterator>

#include <boost/numeric/conversion/cast.hpp>
#include <boost/math/distributions/chi_squared.hpp>

#include <common/simd/SimdVectorTypes.h>

namespace PacBio {
namespace Mongo {

// Constants
constexpr double pi = 3.141592653589793238;
constexpr float pi_f = 3.1415926536f;
constexpr float log2pi_f = 1.8378770664f;


// Function templates
/// Round to nearest integer and cast to integer type.
template <typename IntT, typename FloatT>
IntT round_cast(FloatT x)
{
    static_assert (std::is_integral<IntT>::value,
                   "First template parameter must be integral type.");
    static_assert (std::is_floating_point<FloatT>::value,
                   "Second template parameter must be floating-point type.");
    return boost::numeric_cast<IntT>(std::round(x));
}

/// Saturated linear activation function.
/// A fuzzy threshold that ramps from 0 at a to 1 at b.
/// \returns (x - a)/(b - a) clamped to [0, 1] range.
template <typename FP>
inline FP satlin(const FP& a, const FP& b, const FP& x)
{
    using std::min;
    using std::max;
    const auto r = (x - a) / (b - a);
    return min(max(r, FP(0)), FP(1));
}


/// Squares the argument.
template <typename NumT>
NumT pow2(const NumT& x)
{ return x*x; }


/// Cubes the argument.
template <typename NumT>
NumT pow3(const NumT& x)
{ return x*x*x; }


/// An alternative to exp(x) that might be faster.
template <typename NumT>
NumT expAlt(const NumT& x)
{
    static const float log2e = std::log2(std::exp(1.0f));
    return exp2(x * log2e);
}

template <typename NumT>
NumT atan2(const std::array<NumT, 2>& x)
{
    using std::atan2;
    return atan2(x[1], x[0]);
}


template <typename NumT>
std::array<NumT, 2> cosSin(NumT angle)
{
    using std::cos;
    using std::sin;
    return {{cos(angle), sin(angle)}};
}

template <typename InputIt>
typename std::iterator_traits<InputIt>::value_type
mean(InputIt first, InputIt last)
{
    using ValueType = typename std::iterator_traits<InputIt>::value_type;
    const auto sum = std::accumulate(first, last, ValueType(0));
    const auto n = std::distance(first, last);
    return sum / boost::numeric_cast<ValueType>(n);
}

/// The cumulative probability of the standard normal distribution (mean=0,
/// variance=1).
template <typename FP> inline
FP normalStdCdf(FP x)
{
    using std::sqrt;
    using std::erfc;
    static const FP s = FP(1) / sqrt(FP(2));
    x *= s;
    const FP r = boost::numeric_cast<FP>(0.5f * erfc(-x));
    assert(all(((r >= 0.0f) & (r <= 1.0f)) | isnan(x)));
    return r;
}

/// The cumulative probability at \a x of the normal distribution with \mean
/// and standard deviation \a stdDev.
/// \tparam FP a floating-point numeric type (including m512f).
template <typename FP> inline
FP normalCdf(FP x, FP mean = 0.0f, FP stdDev = 1.0f)
{
    assert(all(stdDev > 0));
    const FP y = (x - mean) / stdDev;
    return normalStdCdf(y);
}

/// The complement of the cumulative probability at \a x of the normal
/// distribution with \mean and standard deviation \a stdDev.
/// normalCdfComp(x, m, sigma) is approximately equal to FP(1) - normalCdf(x, m, sigma),
/// but will be more accurate when normalCdf(x, m, sigma) is close to 1.
/// \tparam FP a floating-point numeric type (including m512f).
template <typename FP> inline
FP normalCdfComp(FP x, FP mean = 0.0f, FP stdDev = 1.0f)
{
    assert(all(stdDev > 0));
    const FP y = (x - mean) / stdDev;
    return normalStdCdf(-y);
}

/// A functor class that represents the logarithm of a univariate normal
/// distribution with fixed mean and variance parameters.
/// \tparam FP A floating-point type. SIMD floating-point types, such as m512f,
/// are supported.
template <typename FP>
class alignas(FP) NormalLog
{
public:     // Structors
    NormalLog(const FP& mean = 0.0f, const FP& variance = 1.0f)
        : mean_ {mean}
    { Variance(variance); }

public:     // Properties
    const FP& Mean() const
    { return mean_; }

    /// Sets Mean to new value.
    /// \returns *this.
    NormalLog& Mean(const FP& value)
    {
        mean_ = value;
        return *this;
    }

    FP Variance() const
    { return 0.5f / scaleFactor_; }

    /// Sets Variance to new value, which must be positive.
    /// \returns *this.
    NormalLog& Variance(const FP& value)
    {
        using std::log;
        assert(all(value > 0.0f));
        normTerm_ = -0.5f * (log2pi_f + log(value));
        scaleFactor_ = 0.5f / value;
        return *this;
    }

public:
    /// Evaluate the distribution at \a x.
    FP operator()(const FP& x) const
    { return normTerm_ - pow2(x - mean_) * scaleFactor_; }

private:    // Data
    FP mean_;
    FP normTerm_;
    FP scaleFactor_;
};


/// The complement of the cumulative probability at \a x of the chi-square
/// distribution with \a dof degrees of freedom.
/// \tparam FP a floating-point numeric type (including m512f).
template <typename FP> inline
FP chi2CdfComp(FP x, int dof)
{
    // If FP is double, still get only single-precision accuracy from
    // chi2CdfComp because of implicit conversion from double to float for
    // calling Blend, which is used to support m512f.
    // TODO: Add template specialization for chi2CdfComp<double>.

    using std::isfinite;
    using std::isnan;
    using Simd::all;
    assert(all(dof > 0));
    assert(all(isfinite(dof)));
    boost::math::chi_squared_distribution<Simd::ScalarType<FP>> csd (dof);
    const auto maskNaN = isnan(x);
    const auto maskFinite = isfinite(x);
    x = Blend(maskFinite, x, FP(0));
    const auto y = Simd::MakeUnion(x);
    Simd::UnionConv<FP> p;
    for (unsigned int i = 0; i < Simd::SimdTypeTraits<FP>::width; ++i)
    {
        p[i] = cdf(complement(csd, y[i]));
    }
    p = Blend(maskFinite, p, FP(0));
    p = Blend(maskNaN, FP(NAN), p);
    return p;
}

}}  // PacBio::Mongo

#endif // mongo_common_NumericUtil_H_
