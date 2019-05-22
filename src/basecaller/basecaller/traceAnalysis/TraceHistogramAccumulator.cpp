
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

TraceHistogramAccumulator::TraceHistogramAccumulator(uint32_t poolId)
    : poolId_ (poolId)
{

}

}}}     // namespace PacBio::Mongo::Basecaller
