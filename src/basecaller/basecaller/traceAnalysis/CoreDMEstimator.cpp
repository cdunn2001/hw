#include "CoreDMEstimator.h"

#include <common/LaneArray.h>
#include <common/StatAccumulator.h>
#include <dataTypes/configs/BasecallerDmeConfig.h>
#include <dataTypes/configs/AnalysisConfig.h>

namespace PacBio {
namespace Mongo {
namespace Basecaller {

// Static constants
constexpr unsigned short CoreDMEstimator::nModelParams;
constexpr unsigned int CoreDMEstimator::nFramesMin;
constexpr float CoreDMEstimator::shotVarCoeff;

// static
PacBio::Logging::PBLogger CoreDMEstimator::logger_ (boost::log::keywords::channel = "DetectionModelEstimator");

CoreDMEstimator::CoreDMEstimator(uint32_t poolId, unsigned int poolSize)
    : poolId_ (poolId)
    , poolSize_ (poolSize)
{
}

}}}     // namespace PacBio::Mongo::Basecaller