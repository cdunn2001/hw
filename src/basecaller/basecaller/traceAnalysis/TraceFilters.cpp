#include "TraceFilters.h"

#include <iterator>
#include <numeric>

namespace PacBio {
namespace Mongo {
namespace Basecaller {

/// A block-streaming moving average (top-hat) filter
template <typename InIter, typename OutIter, typename Arith>
size_t movingAvgFilter(InIter p,
                       const typename MovingAverageFilter<InIter, Arith>::DiffType n,
                       OutIter out,
                       WindowBuffer<typename MovingAverageFilter<InIter, Arith>::ValueType>& buf)
{
    // Arithmetic functors
    static typename Arith::assign assign;
    static typename Arith::plus plus;
    static typename Arith::minus minus;
    static typename Arith::divides divides;

    auto in0 = p;
    auto out0 = out;

    auto in1 = p;
    std::advance(in1, n);

    size_t iwidth = buf.capacity();
    auto width = assign(buf.capacity());

    // Assumption is that the boundary condition is already applied for the
    // first chunk, with a half-window's worth of values placed in the window
    // buffer.  Fill in all but the last needed to compute the first value.
    while (buf.size() < iwidth - 1)
    {
        buf.push_back(*p++);
    }

    // Running sum over the current window
    auto rsum = assign(0);

    // First chunk?
    if (buf.size() < iwidth)
    {
        // On the first chunk, compute the initial running sum.
        assert(buf.size() == iwidth - 1);
        buf.push_back(*p++);
        rsum = std::accumulate(buf.begin(), buf.end(), rsum, plus);
    }
    else
    {
        // Otherwise, pick off the running sum from the front
        rsum = buf.front();

        // Compute the first element -- the front of the window
        // has already been subtracted from the cached rsum.
        rsum = plus(rsum, *p);

        // Normalize the buffer to contain elements [-(w-1), -(w-2), ..., -1, 0]
        // with respect to the current chunk starting at 0, and move to the next sample
        buf.push_back(*p++);
    }

    // Output the first element.
    *out++ = divides(rsum, width);

    for (; p != in1; ++p)
    {
        rsum = plus(minus(rsum, buf.front()), *p);
        buf.push_back(*p);

        // Output the next element
        *out++ = divides(rsum, width);
    }

    // Subtract the next value from rsum and replace it
    rsum = minus(rsum, buf.front());
    buf.pop_front();
    buf.push_front(rsum);

    return (out - out0);
}

template <typename InIter, typename Arith>
size_t MovingAverageFilter<InIter, Arith>::operator()(InIter in0, const DiffType n, WindowBuffer<ValueType>& buf) const
{
    return movingAvgFilter<InIter, InIter, Arith>(in0, n, in0, buf);
}

template <typename RandInIter, typename RandOutIter, typename ExtremumFunc>
size_t extremumFilterHgw(RandInIter p, RandInIter in1, RandOutIter out,
                         WindowBuffer<typename ExtremumFilterHgw<RandInIter, RandOutIter, ExtremumFunc>::ValueType>& r,
                         typename ExtremumFilterHgw<RandInIter, RandOutIter, ExtremumFunc>::DiffType stride/*=1*/)
{
    static ExtremumFunc extremOp;		// The pairwise min or max operation

    auto out0 = out;
    size_t width = r.capacity();

    // Assumption is that the boundary condition is already applied for the first chunk,
    // with a half-window's worth of values placed in the window buffer using PushBack().
    // Fill in all but the last needed to compute the first value.
    while (r.size() < width &&  p != in1)
    {
        r.PushBack(*p);
        std::advance(p, std::min(stride, std::distance(p, in1)));
    }

    auto s = r.GetHoldoverValue();

    using IterDiffType = typename std::iterator_traits<RandInIter>::difference_type;
    const auto dist = std::distance(p, in1);
    for (IterDiffType i = 0; i < dist; i += stride)
    {
        if (r.Counter() == 0)
        {
            // ComputeR(r);
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
        r.PushBack(p[i]);
        *out++ = extremOp(s, front);
    }

    // Store the current value of s
    r.SetHoldoverValue(s);

    return std::distance(out0, out);
}


template <typename InIter, typename OutIter, typename ExtremumFunc>
size_t ExtremumFilterHgw<InIter, OutIter, ExtremumFunc>::operator()(const RangeType& rng, OutIter out,
                                                                    WindowBuffer<ValueType>& wb,
                                                                    DiffType stride/*=1*/) const
{
    return extremumFilterHgw<InIter, OutIter, ExtremumFunc>(boost::begin(rng), boost::end(rng), out, wb, stride);
}

//
// Explicit Instantiation
//
// TODO: figure out which below

/*

#define Template_ErodeHgw(_type_) template class ExtremumFilterHgw<\
    typename Data::BlockView<_type_>::const_iterator,\
    typename Data::BlockView<_type_>::iterator,\
    typename ArithOps<_type_>::minOp>

#define Template_DilateHgw(_type_) template class ExtremumFilterHgw<\
    typename Data::BlockView<_type_>::const_iterator, \
    typename Data::BlockView<_type_>::iterator, \
    typename ArithOps<_type_>::maxOp>

#define Template_DilateErodeHgw(_type_) template class ExtremumFilterHgw<\
    typename Data::BlockView<_type_>::const_iterator,\
    typename Data::BlockView<_type_>::iterator,\
    typename ArithOps<_type_>::minmax>

#define Template_ErodeDilateHgw(_type_) template class ExtremumFilterHgw<\
    typename Data::BlockView<_type_>::const_iterator, \
    typename Data::BlockView<_type_>::iterator, \
    typename ArithOps<_type_>::maxmin>

*/

}}}     // namespace PacBio::Mongo::Basecaller
