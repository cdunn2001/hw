#include "BlockFilterStage.h"
#include "TraceFilters.h"

namespace PacBio {
namespace Mongo {
namespace Basecaller {

template <typename V, typename F>
Data::BlockView<V>* BlockFilterStage<V, F>::operator()(Data::BlockView<V>* pInput)
{
    // Note that output is an input alias: filtering is in-place.
    Data::BlockView<V>& input = *pInput;
    Data::BlockView<V>& output = input;

    // Iterators initially pointing at start and end of block,
    // these will be adjusted below as necessary to account
    // for the stride.
    auto filterRangeStart = input.begin();
    auto filterRangeEnd = input.end();

    auto inSize = input.NumFrames();
    assert (inSize > 0);

    // LHS boundary condition: Load LH values into window buffer
    if (winbuf_.size() == 0)
    {
        // This should be the start of the series,
        // and there should be no down-sampling skip.
        assert(winbuf_.GetStrideSkip() == 0);

        // Load the window buffer entirely instead of just the left width.
        for (unsigned int i = 0; i < width_; ++i)
        {
            winbuf_.PushBack(*filterRangeStart);
        }
    }

    // Set the appropriate skip to use for the next interval.
    auto sskipIn = winbuf_.GetStrideSkip();
    assert(sskipIn < inSize);
    std::advance(filterRangeStart, sskipIn);
    auto ovrhang = (std::distance(filterRangeStart, filterRangeEnd) - 1) % stride_;
    winbuf_.SetStrideSkip(stride_ - ovrhang - 1);

    // Run the filter on the available range of input data.
    // The filter runs in-place, and the buffer stays in use.
    using RangeType = typename F::RangeType;
    RangeType inRange(filterRangeStart, filterRangeEnd);
    filter_(inRange, output.begin(), winbuf_, stride_);

    return &output;
}

///
/// Explicit Instantiations
///

template class BlockFilterStage<short, ErodeHgw<short>>;
template class BlockFilterStage<short, DilateHgw<short>>;

}}}     // namespace PacBio::Mongo::Basecaller
