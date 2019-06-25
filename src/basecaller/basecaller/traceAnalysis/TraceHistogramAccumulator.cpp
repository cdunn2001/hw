
#include "TraceHistogramAccumulator.h"

namespace PacBio {
namespace Mongo {
namespace Basecaller {

// static
void TraceHistogramAccumulator::Configure(
        const Data::BasecallerTraceHistogramConfig& histConfig,
        const Data::MovieConfig& movConfig)
{
    // TODO
}

TraceHistogramAccumulator::TraceHistogramAccumulator(uint32_t poolId, unsigned int poolSize)
    : poolId_ (poolId)
    , poolHist_ (poolId, poolSize)
{

}

}}}     // namespace PacBio::Mongo::Basecaller
