#ifndef mongo_basecaller_traceAnalysis_TraceFilters_H
#define mongo_basecaller_traceAnalysis_TraceFilters_H


#include "WindowBuffer.h"

#include <boost/range/iterator_range.hpp>

#include <common/simd/SimdVectorTypes.h>
#include <common/LaneArray.h>
#include <dataTypes/TraceBatch.h>

namespace PacBio {
namespace Mongo {
namespace Basecaller {

template <typename InIter, typename OutIter, typename F>
class ExtremumFilterHgw
{
public:
    using DiffType = typename InIter::DiffType;
    using ValueType = typename InIter::ValueType;
public:
    size_t operator()(const InIter& start, const InIter& end, OutIter out, WindowBuffer<ValueType>& wb,
                      DiffType stride = 1) const;
};

template <typename T>
using ErodeHgw = ExtremumFilterHgw<typename Data::BlockView<T>::ConstIterator,
                                   typename Data::BlockView<T>::Iterator,
                                   typename Simd::ArithOps<LaneArray<T, laneSize>>::minOp>;

template <typename T>
using DilateHgw = ExtremumFilterHgw<typename Data::BlockView<T>::ConstIterator,
                                    typename Data::BlockView<T>::Iterator,
                                    typename Simd::ArithOps<LaneArray<T, laneSize>>::maxOp>;


}}}     // namespace PacBio::Mongo::Basecaller

#endif // mongo_basecaller_traceAnalysis_TraceFilters_H
