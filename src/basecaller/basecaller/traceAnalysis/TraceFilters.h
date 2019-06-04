#ifndef mongo_basecaller_traceAnalysis_TraceFilters_H
#define mongo_basecaller_traceAnalysis_TraceFilters_H


#include "WindowBuffer.h"

#include <boost/range/iterator_range.hpp>

#include <common/simd/SimdVectorTypes.h>
#include <dataTypes/TraceBatch.h>

namespace PacBio {
namespace Mongo {
namespace Basecaller {

template <typename InIter, typename Arith>
class MovingAverageFilter
{
    using DiffType = typename std::iterator_traits<InIter>::difference_type;
    using ValueType = typename std::iterator_traits<InIter>::value_type;
    using RangeType = typename boost::iterator_range<InIter>;
public:
    size_t operator()(InIter in, const DiffType n, WindowBuffer<ValueType>& buf) const;
};

template <typename InIter, typename OutIter, typename F>
class ExtremumFilterHgw
{
    using DiffType = typename std::iterator_traits<InIter>::difference_type;
    using ValueType = typename std::iterator_traits<InIter>::value_type;
    using RangeType = typename boost::iterator_range<InIter>;
public:
    size_t operator()(const RangeType& rng, OutIter out, WindowBuffer<ValueType>& wb,
                      DiffType stride = 1) const;
};

template <typename T>
using MovingAvg = MovingAverageFilter<T*, Simd::ArithOps<T>>;

// TODO: Need to decide on iterator types below

/*
template <typename T>
using ErodeHgw = ExtremumFilterHgw<typename Data::BlockView<T>::const_iterator,
                                   typename Data::BlockView<T>::iterator,
                                   typename Simd::ArithOps<T>::minOp>;

template <typename T>
using DilateHgw = ExtremumFilterHgw<typename Data::BlockView<T>::const_iterator,
                                    typename Data::BlockView<T>::iterator,
                                    typename Simd::ArithOps<T>::maxOp>;
*/

}}}     // namespace PacBio::Mongo::Basecaller

#endif // mongo_basecaller_traceAnalysis_TraceFilters_H
