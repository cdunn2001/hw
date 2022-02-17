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
const float CoreDMEstimator::fallbackBaselineMean_ = 0.0f;
const float CoreDMEstimator::fallbackBaselineVariance_ = 100.0f;
float CoreDMEstimator::signalScaler_ = 1.0f;

CoreDMEstimator::CoreDMEstimator(uint32_t poolId, unsigned int poolSize)
    : poolId_ (poolId)
    , poolSize_ (poolSize)
{
}

// static
void CoreDMEstimator::Configure(const Data::AnalysisConfig& analysisConfig)
{
    signalScaler_ = analysisConfig.movieInfo.photoelectronSensitivity;
    assert(std::isfinite(signalScaler_) && signalScaler_ > 0.0f);
}

}}}     // namespace PacBio::Mongo::Basecaller
