#include "TraceFilters.h"

#include <iterator>
#include <numeric>

namespace PacBio {
namespace Mongo {
namespace Basecaller {

template <typename RandInIter, typename RandOutIter, typename ExtremumFunc>
size_t extremumFilterHgw(RandInIter p, RandInIter in1, RandOutIter out,
                         WindowBuffer<typename ExtremumFilterHgw<RandInIter, RandOutIter, ExtremumFunc>::ValueType>& r,
                         typename ExtremumFilterHgw<RandInIter, RandOutIter, ExtremumFunc>::DiffType stride/*=1*/)
{
    static ExtremumFunc extremOp;		// The pairwise min or max operation.

    auto out0 = out;
    size_t width = r.capacity();

    using ValueType = typename ExtremumFilterHgw<RandInIter, RandOutIter, ExtremumFunc>::ValueType;
    // Assumption is that the boundary condition is already applied for the first chunk,
    // with a half-window's worth of values placed in the window buffer using PushBack().
    // Fill in all but the last needed to compute the first value.
    while (r.size() < width &&  p != in1)
    {
        r.PushBack(ValueType(*p));
        p += std::min(stride, RandInIter::distance(p, in1));
    }

    auto s = r.GetHoldoverValue();

    using IterDiffType = typename RandInIter::DiffType;
    const auto dist = RandInIter::distance(p, in1);
    for (IterDiffType i = 0; i < dist; i += stride)
    {
        if (r.Counter() == 0)
        {
            // Compute R(r);
            for (int j = width - 1; j > 0; --j)
            {
                r[j - 1] = extremOp(r[j], r[j - 1]);
            }
            // In this case, s is re-assigned to be the input value at
            // the end of the last window, which is at the back of r.
            s = r.back();
        }
        else
        {
            s = extremOp(s, r.back());
        }

        // Order matters here, since we allow the input and output buffers to be the
        // same address space:  push the current value before assigning the output.
        //
        auto front = r.front();
        r.PushBack(ValueType(p[i]));
        *out++ = extremOp(s, front);
    }

    // Store the current value of s
    r.SetHoldoverValue(s);

    return RandInIter::distance(out0, out);
}


template <typename InIter, typename OutIter, typename ExtremumFunc>
size_t ExtremumFilterHgw<InIter, OutIter, ExtremumFunc>::operator()(const InIter& start,
                                                                    const InIter& end,
                                                                    OutIter out,
                                                                    WindowBuffer<ValueType>& wb,
                                                                    DiffType stride/*=1*/) const
{
    return extremumFilterHgw<InIter, OutIter, ExtremumFunc>(start, end, out, wb, stride);
}

//
// Explicit Instantiation
//

#define Template_ErodeHgw(_type_) template class ExtremumFilterHgw< \
    typename Data::BlockView<_type_>::ConstIterator,                \
    typename Data::BlockView<_type_>::Iterator,                     \
    typename Simd::ArithOps<LaneArray<_type_, laneSize>>::minOp>

#define Template_DilateHgw(_type_) template class ExtremumFilterHgw<\
    typename Data::BlockView<_type_>::ConstIterator,                \
    typename Data::BlockView<_type_>::Iterator,                     \
    typename Simd::ArithOps<LaneArray<_type_, laneSize>>::maxOp>

Template_ErodeHgw(short);
Template_DilateHgw(short);

}}}     // namespace PacBio::Mongo::Basecaller
