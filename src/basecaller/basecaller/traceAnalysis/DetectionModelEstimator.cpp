
#include "DetectionModelEstimator.h"

namespace PacBio {
namespace Mongo {
namespace Basecaller {

// static
void DetectionModelEstimator::Configure(const Data::BasecallerDmeConfig& dmeConfig,
                                        const Data::MovieConfig& movConfig)
{
    // TODO
}

DetectionModelEstimator::DetectionModelEstimator(uint32_t poolId)
    : poolId_ (poolId)
{
    // TODO
}

}}}     // namespace PacBio::Mongo::Basecaller
