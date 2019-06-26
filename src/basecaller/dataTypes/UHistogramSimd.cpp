
// Copyright (c) 2017, Pacific Biosciences of California, Inc.
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
//  Defines members of class UHistogramSimd.

#include "UHistogramSimd.h"

#include <algorithm>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <boost/numeric/conversion/cast.hpp>
#include <pacbio/PBException.h>
#include <common/simd/SimdVectorTypes.h>

using std::ostringstream;
using boost::numeric_cast;

namespace PacBio {
namespace Mongo {
namespace Data {

using namespace Simd;

template <typename DataT, typename CountT>
UHistogramSimd<DataT, CountT>::UHistogramSimd(unsigned int numBins,
                                              const DataType& lowerBound,
                                              const DataType& upperBound)
    : binSize_ {0}
    , nLowOutliers_ {0}
    , nHighOutliers_ {0}
    , binStart_ (numBins + 1u)
    , binCount_ (numBins + 1u, CountType{0})
    , numBins_ (numBins)
{
    // Check parameter values.
    if (numBins == 0)
    {
        ostringstream msg;
        msg << "numBins must be greater than zero.  numBins = "
            << numBins << ".";
        throw std::invalid_argument(msg.str());
    }

    const auto& badBounds = (lowerBound >= upperBound);
    if (any(badBounds))
    {
        ostringstream msg;
        msg << "lowerBound must be less than upperBound.";
        msg << " lowerBound = " << lowerBound;
        msg << ". upperBound = " << upperBound << ".";
        throw std::invalid_argument(msg.str());
    }

    // Check for underflow of bin size.
    static const std::string errUnderflowMsg = "Bin size underflow.";
    binSize_ = (upperBound - lowerBound) / static_cast<float>(numBins);
    if (any(binSize_ <= float(0)))
    {
        ostringstream msg;
        msg << errUnderflowMsg
            << "  Bin size = "
            << binSize_ << '.';
        throw std::underflow_error(msg.str());
    }

    // Set up bin starts (i.e., lower bound of each bin).
    binStart_.front() = lowerBound;
    for (size_t i = 1; i < binStart_.size(); ++i)
    {
        binStart_[i] = binSize_ * numeric_cast<float>(i) + lowerBound;
        if (any(binStart_[i-1] == binStart_[i]))
        {
            throw std::underflow_error(errUnderflowMsg);
        }
    }
    binStart_.back() = upperBound;
}


// TODO: If this remains very simple, move the definition into the class
// definition once the dust settles.
template <typename DataT, typename CountT>
typename UHistogramSimd<DataT, CountT>::CountType
UHistogramSimd<DataT, CountT>::Count(ScalarIndexType first,
                                     ScalarIndexType last) const
{
    if (first >= last) return CountType(0);
    assert (last <= NumBins());
    const auto start = binCount_.begin() + first;
    const auto stop = binCount_.begin() + last;
    return std::accumulate(start, stop, CountType(0));
}

template <typename DataT, typename CountT>
typename UHistogramSimd<DataT, CountT>::CountType
UHistogramSimd<DataT, CountT>::CountNonuniform(IndexType first,
                                               IndexType last) const
{
    assert (all(last <= NumBins()));
    auto i = std::max<ScalarIndexType>(0, reduceMin(first));
    const auto j = std::min(NumBins(), reduceMax(last));
    CountType sum = 0;
    while (i < j)
    {
        sum += Blend((i >= first) & (i < last), binCount_[i], 0);
        ++i;
    }
    return sum;
}

template <typename DataT, typename CountT>
typename UHistogramSimd<DataT, CountT>::DataType
UHistogramSimd<DataT, CountT>::Fractile(FloatType frac) const
{
    if (any(frac < 0.0f))
    {
        std::ostringstream msg;
        msg << "Fractile argument must be >= 0: "
            << frac << '.';
        throw PBException(msg.str());
    }
    if (any(frac > 1.0f))
    {
        std::ostringstream msg;
        msg << "Fractile argument must be <= 1: "
            << frac << '.';
        throw PBException(msg.str());
    }

    assert(NumBins() > 0);

    using ScalarDataType = ScalarType<DataType>;
    static constexpr auto inf = std::numeric_limits<ScalarDataType>::infinity();

    UnionConv<DataType> ret;
    const auto nf = MakeUnion(frac * FloatType(TotalCount()));
    for (unsigned int z = 0; z < SimdTypeTraits<DataType>::width; ++z)
    {
        // Find the critical bin.
        float nfz = nf[z];
        auto n = MakeUnion(LowOutlierCount())[z];
        if (n > 0 && n >= nfz)
        {
            // The precise fractile is in the low-outlier bin.
            // Since this bin is unbounded below, ...
            ret[z] = -inf;
            continue;
        }

        ScalarIndexType i = 0;
        while ((n == 0 || n < nfz) && i < NumBins())
        {
            n += MakeUnion(BinCount(i++))[z];
        }

        if (n < nfz)
        {
            // The precise fractile is in the high-outlier bin.
            // Since this bin is unbounded above, ...
            ret[z] = +inf;
            continue;
        }

        // Otherwise, the precise fractile is in a normal bin.
        // Interpolate within the critical bin.
        assert(i > 0);
        assert(n >= nfz);
        i -= 1;     // Back up to the bin that pushed us over the target.
        auto x0 = MakeUnion(BinStart(i))[z];
        const auto niz = MakeUnion(BinCount(i))[z];
        ScalarType<CountType> m = n - niz;
        assert(m < nfz || (m == 0 && nfz == 0));
        ret[z] = x0 + MakeUnion(BinSize())[z] * (nfz - m) / (niz + 1);
    }

    return ret;
}


template <typename DataT, typename CountT>
typename UHistogramSimd<DataT, CountT>::FloatType
UHistogramSimd<DataT, CountT>::CumulativeCount(DataType x) const
{
    const auto mNan = isnan(x);
    const auto mLow = (x < LowerBound());
    const auto mHigh = (x >= UpperBound());
    const auto mInBounds = !(mNan | mLow | mHigh);

    UnionConv<IndexType> xbin (BinIndex(x));

    // Avoid out-of-bounds array indexes.
    xbin = Blend(mInBounds, xbin, 0);

    // Interpolation in bin containing x.
    const auto& xx = MakeUnion(x);
    UnionConv<DataType> xrem;
    UnionConv<FloatType> xbinCount;
    using ScalarFloatType = ScalarType<FloatType>;
    for (unsigned int i = 0; i < SimdTypeTraits<DataType>::width; ++i)
    {
        xrem[i] = (xx[i] - MakeUnion(BinStart(xbin[i]))[i]);
        xbinCount[i] = numeric_cast<ScalarFloatType>(binCount_[xbin[i]][i]);
    }

    // Tally the cumulative count.
    auto cc = FloatType(LowOutlierCount());
    cc += FloatType(CountNonuniform(IndexType(0), xbin));
    cc += xbinCount * xrem / BinSize();

    cc = Blend(mLow, FloatType(LowOutlierCount()), cc);
    cc = Blend(mHigh, FloatType(LowOutlierCount() + InRangeCount()), cc);
    cc = Blend(mNan, NAN, cc);

    return cc;
}


// Explicit instantiations
template class UHistogramSimd<float>;
template class UHistogramSimd<m512f>;

}}}     // namespace PacBio::Mongo::Data
