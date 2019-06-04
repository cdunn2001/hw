#include "BlockFilterStage.h"

namespace PacBio {
namespace Mongo {
namespace Basecaller {

template <typename V, typename F>
Data::BlockView<V>* BlockFilterStage<V, F>::operator()(Data::BlockView<V>* pInput)
{
    // Note that output is an input alias: filtering is in-place.
    Data::BlockView<V>& input = *pInput;
    Data::BlockView<V>& output = input;

    // TODO: fill in

    return &output;
}


}}}     // namespace PacBio::Mongo::Basecaller
