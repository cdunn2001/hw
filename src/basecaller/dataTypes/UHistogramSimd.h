#ifndef mongo_dataTypes_UHistogramSimd_H_
#define mongo_dataTypes_UHistogramSimd_H_

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
/// \file UHistogramSimd.h
/// Defines class UHistogramSimd.h.

#include <numeric>
#include <vector>
#include <pacbio/PBException.h>
#include <pacbio/PBAssert.h>
#include <common/AlignedVector.h>
#include <common/simd/SimdConvTraits.h>
#include <common/simd/SimdTypeTraits.h>
#include <dataTypes/PoolHistogram.h>

namespace PacBio {
namespace Mongo {
namespace Data {

// TODO: Make this LaneArray specific.

/// A histogram with uniform bin boundaries that supports SIMD data types.
/// Similar to class template UHistogram.
/// \tparam DataT A floating-point SIMD type. Currently, supports float,
/// PacBio::Simd::m512f, and PacBio::LaneArray<float>.
/// \note The alignment requirement of UHistogramSimd<DataT> is at least as
/// large as that of DataT. When creating on the heap, this may require use of
/// a special allocator.
template <typename DataT, typename CountT>
class alignas(DataT) UHistogramSimd
{
public:     // Types
    /// The type of the data.
    using DataType = DataT;

    /// The type of the bin counts.
    using CountType = CountT;

    using FloatType = Simd::FloatConv<DataType>;

    /// The SIMD type for bin indexes.
    using IndexType = Simd::IndexConv<DataType>;

    /// Scalar type for bin indexes.
    using ScalarIndexType = Simd::ScalarType<IndexType>;

    /// Scalar type for data.
    using ScalarDataType = Simd::ScalarType<DataType>;

    /// Scalar type for bin counts.
    using ScalarCountType = Simd::ScalarType<CountType>;

    /// The type returned by predicates (e.g., InRange()).
    using BoolType = Simd::BoolConv<DataType>;

    template <typename T>
    using ArrayType = AlignedVector<T>;

public:     // Structors
    /// Constructs a histogram with \a numBins bins that span the data range
    /// from \a lowerBound to \a upperBound.
    UHistogramSimd(unsigned int numBins,
                   const DataType& lowerBound, const DataType& upperBound);

    /// Constructs a histogram by loading data from an instance of PoolHistogram.
    explicit UHistogramSimd(const LaneHistogram<ScalarDataType, ScalarCountType>& laneHist);

public:     // Const methods
    /// \returns The number of bins composing the histogram.
    ScalarIndexType NumBins() const
    { return numBins_; }

    /// \brief Total number of data populating the histogram, including outliers.
    /// \details
    /// Asymptotic complexity same as InRangeCount().
    CountType TotalCount() const
    { return InRangeCount() + OutlierCount(); }

    /// \returns Total number of data populating the histogram bins.
    /// \details
    /// Asymptotic growth proportional to NumBins().
    CountType InRangeCount() const
    { return Count(0u, NumBins()); }

    /// \returns Number of data less than the lower limit of the lowest bin.
    const CountType& LowOutlierCount() const
    { return nLowOutliers_; }

    /// \returns Number of data greater than the upper limit of the highest bin.
    const CountType& HighOutlierCount() const
    { return nHighOutliers_; }

    /// \returns LowOutlierCount() + HighOutlierCount().
    CountType OutlierCount() const
    { return LowOutlierCount() + HighOutlierCount(); }

    /// \returns The width of each bin.
    const DataType& BinSize() const
    { return binSize_; }

    /// \returns The center value of the i-th bin.
    DataType BinCenter(size_t i) const
    {
        assert (i < binStart_.size());
        return binStart_[i] + 0.5f*binSize_;
    }

    /// \returns The lower bound of the i-th bin.
    /// BinStart(NumBins()) == UpperBound().
    const DataType& BinStart(size_t i) const
    { return binStart_[i]; }

    /// An array of boundaries (or edges) defining the bins of the histogram.
    /// \returns Reference to array with NumBins() + 1 elements.
    const ArrayType<DataType>& BinBoundaries() const
    { return binStart_; }

    /// \returns The number of data in the interval of the i-th bin.
    CountType BinCount(size_t i) const
    { return binCount_[i]; }

    /// \returns Number of data populating bins in the range [first, last).
    CountType Count(ScalarIndexType first, ScalarIndexType last) const
    {
        if (first >= last) return CountType(0);
        assert (last <= NumBins());
        const auto start = binCount_.begin() + first;
        const auto stop = binCount_.begin() + last;
        return std::accumulate(start, stop, CountType(0));
    }


    /// Vectorized version of Count(ScalarIndexType, ScalarIndexType) const;
    CountType CountNonuniform(IndexType first, IndexType last) const;

    // TODO: Is Mean(...) really needed?
    /// \returns Mean of bins in range [first, last).
    DataType Mean(size_t first, size_t last) const;

    /// \returns Mean of all in-range bins.
    DataType Mean() const;

    /// \returns The value that is larger than a portion \a frac and smaller than a
    /// portion 1 - \a frac of the data represented in the histogram.
    /// The value is determined by linear interpolation in the bin containing
    /// the desired fractile.
    /// 0 <= \a frac <= 1.
    /// If LowOutlierCount() > 0, Fractile(0) = -Inf.
    /// If HighOutlierCount() > 0, Fractile(1) = +Inf.
    DataType Fractile(FloatType frac) const;

    /// Estimates the number of data less than \a x.
    /// Linearly interpolates within the bin containing \a x.
    /// If \a x < LowerBound(), returns LowOutlierCount().
    /// If \a x >= UpperBound(), returns LowOutlierCount() + InRangeCount().
    FloatType CumulativeCount(DataType x) const;

    /// \returns The lower bound of the histogram range.
    const DataType& LowerBound() const
    { return binStart_.front(); }

    /// \returns The upper bound of the histogram range.
    const DataType& UpperBound() const
    { return binStart_.back(); }

    /// \returns if \a x in the range [LowerBound(), UpperBound())?
    BoolType InRange(DataType x) const
    { return (x >= LowerBound()) & (x < UpperBound()); }

    /// \returns The index of the bin that contains \a x.  If the input data is *not* in
    /// the histogram range, the produced index will be out of range as well!
    /// Bounds checking is the responsibility of the calling code
    IndexType BinIndex(DataType x) const
    {
        DataType y = (x - LowerBound()) / binSize_;
        return floorCastInt(y);
    }

public:     // Modifying methods
    /// Adds elements of \a x only when correspond elements of \a select are true.
    UHistogramSimd& AddDatum(const DataType& x, const BoolType& select = BoolType(true))
    {
        using PacBio::inc;

        // This function can return an index that is out of bounds
        auto bin = Simd::MakeUnion(BinIndex(x));

        // Handle outliers.
        const auto maskLow  = select & (bin < 0);
        nLowOutliers_  = inc(nLowOutliers_, maskLow);
        const auto maskHigh = select & (bin >= static_cast<int32_t>(numBins_));
        nHighOutliers_ = inc(nHighOutliers_, maskHigh);

        // Increment bin counts for in-range elements.
        const auto maskSkip = (!select) | (maskLow | maskHigh);
        // Toss out of bounds values into extra bin at the end that will be 
        // ignored.  It's cheaper than putting a conditional in the loop below.
        bin = Blend(maskSkip, IndexType(numBins_), bin);
        for (unsigned int i = 0; i < Simd::SimdTypeTraits<DataType>::width; ++i)
        {
            ++binCount_[bin[i]][i];
        }

        return *this;
    }

    /// Add a sequence of data to the histogram.
    /// \returns *this.
    template<typename ReadOnlyInputIter>
    UHistogramSimd& AddData(ReadOnlyInputIter first, ReadOnlyInputIter last)
    {
        while (first != last) AddDatum(*first++);
        return *this;
    }

private:    // Data
    DataType binSize_;
    CountType nLowOutliers_;
    CountType nHighOutliers_;
    ArrayType<DataType> binStart_;
    ArrayType<Simd::UnionConv<CountType>> binCount_;
    unsigned int numBins_;
};

}}}     // namespace PacBio::Mongo::Data

#endif  // mongo_dataTypes_UHistogramSimd_H_
