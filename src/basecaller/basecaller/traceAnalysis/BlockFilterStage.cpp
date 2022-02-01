#include "BlockFilterStage.h"
#include "TraceFilters.h"

namespace PacBio {
namespace Mongo {
namespace Basecaller {

template <typename T, typename Filter>
Data::BlockView<T>* BlockFilterStage<T, Filter>::operator()(Data::BlockView<T>* pInput)
{
    // Note that output is an input alias: filtering is in-place.
    Data::BlockView<T>& input = *pInput;
    Data::BlockView<T>& output = input;

    assert (pInput->NumFrames() > 0);

    auto filterRangeStart = input.CBegin();
    auto filterRangeEnd   = input.CBegin() + pInput->NumFrames() / nf_; 

    // Run the filter on the available range of input data.
    // The filter runs in-place, and the buffer stays in use.
    filter_(filterRangeStart, filterRangeEnd, output.Begin(), winbuf_, stride_);

    return &output;
}

///
/// Explicit Instantiations
///

template class BlockFilterStage<short, ErodeHgw<short>>;
template class BlockFilterStage<short, DilateHgw<short>>;

}}}     // namespace PacBio::Mongo::Basecaller
