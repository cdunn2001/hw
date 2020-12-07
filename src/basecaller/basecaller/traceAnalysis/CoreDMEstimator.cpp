#include "CoreDMEstimator.h"

#include <common/LaneArray.h>
#include <common/StatAccumulator.h>
#include <dataTypes/configs/BasecallerDmeConfig.h>
#include <dataTypes/configs/MovieConfig.h>

namespace PacBio {
namespace Mongo {
namespace Basecaller {

// static
Cuda::Utility::CudaArray<Data::AnalogMode, numAnalogs>
CoreDMEstimator::analogs_;

PacBio::Logging::PBLogger CoreDMEstimator::logger_ (boost::log::keywords::channel = "DetectionModelEstimator");

float CoreDMEstimator::refSnr_;
bool CoreDMEstimator::fixedBaselineParams_ = false;
float CoreDMEstimator::fixedBaselineMean_ = 0;
float CoreDMEstimator::fixedBaselineVar_ = 0;

// static
void CoreDMEstimator::Configure(const Data::BasecallerDmeConfig& dmeConfig,
                                        const Data::MovieConfig& movConfig)
{
    refSnr_ = movConfig.refSnr;
    for (size_t i = 0; i < movConfig.analogs.size(); i++)
    {
        analogs_[i] = movConfig.analogs[i];
    }

    if (dmeConfig.Method == Data::BasecallerDmeConfig::MethodName::Fixed &&
        dmeConfig.SimModel.useSimulatedBaselineParams == true)
    {
        fixedBaselineParams_ = true;
        fixedBaselineMean_ = dmeConfig.SimModel.baselineMean;
        fixedBaselineVar_ = dmeConfig.SimModel.baselineVar;
    }
}

CoreDMEstimator::CoreDMEstimator(uint32_t poolId, unsigned int poolSize)
    : poolId_ (poolId)
    , poolSize_ (poolSize)
{
}

}}}     // namespace PacBio::Mongo::Basecaller
