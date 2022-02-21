#include "CoreDMEstimator.h"

#include <cmath>

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

// static
CoreDMEstimator::Configuration CoreDMEstimator::config_;

CoreDMEstimator::CoreDMEstimator(uint32_t poolId, unsigned int poolSize)
    : poolId_ (poolId)
    , poolSize_ (poolSize)
{
}

// static
void CoreDMEstimator::Configure(const Data::AnalysisConfig& analysisConfig)
{
    config_.fallbackBaselineMean = 0.0f;
    config_.fallbackBaselineVariance = 100.0f;

    const float ss = analysisConfig.movieInfo.photoelectronSensitivity;
    assert(std::isfinite(ss) && ss > 0.0f);
    config_.signalScaler = ss;
}

}}}     // namespace PacBio::Mongo::Basecaller
