
#include "TraceHistogramAccumulator.h"

namespace PacBio {
namespace Mongo {
namespace Basecaller {

// TODO: Where should these come from.
const unsigned int numLanes = 12;
const unsigned int numBins = 100;

// static
void TraceHistogramAccumulator::Configure(
        const Data::BasecallerTraceHistogramConfig& histConfig,
        const Data::MovieConfig& movConfig)
{
    // TODO
}

TraceHistogramAccumulator::TraceHistogramAccumulator(uint32_t poolId)
    : poolId_ (poolId)
    , poolHist_ (poolId, numLanes)
{

}

}}}     // namespace PacBio::Mongo::Basecaller
