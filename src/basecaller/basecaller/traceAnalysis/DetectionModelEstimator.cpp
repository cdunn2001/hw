
#include <common/LaneArray.h>
#include <common/StatAccumulator.h>

#include "DetectionModelEstimator.h"

namespace PacBio {
namespace Mongo {
namespace Basecaller {

// static
std::array<Data::AnalogMode, 4> DetectionModelEstimator::analogs_;
float DetectionModelEstimator::refSnr_;

// static
void DetectionModelEstimator::Configure(const Data::BasecallerDmeConfig& dmeConfig,
                                        const Data::MovieConfig& movConfig)
{
    // TODO

    // TODO: These values are bogus. They should be extracted from movConfig.
    refSnr_ = 20.0f;

    analogs_[0].baseLabel = 'G';
    analogs_[0].relAmplitude = 0.20f;
    analogs_[0].excessNoiseCV = 0.10f;
    analogs_[0].interPulseDistance = 0.1f;
    analogs_[0].pulseWidth = 0.1f;
    analogs_[0].pw2SlowStepRatio = 0.1f;
    analogs_[0].ipd2SlowStepRatio = 0.0f;

    analogs_[1].baseLabel = 'T';
    analogs_[1].relAmplitude = 0.40f;
    analogs_[1].excessNoiseCV = 0.10f;
    analogs_[1].interPulseDistance = 0.1f;
    analogs_[1].pulseWidth = 0.1f;
    analogs_[1].pw2SlowStepRatio = 0.1f;
    analogs_[1].ipd2SlowStepRatio = 0.0f;

    analogs_[2].baseLabel = 'A';
    analogs_[2].relAmplitude = 0.70f;
    analogs_[2].excessNoiseCV = 0.10f;
    analogs_[2].interPulseDistance = 0.1f;
    analogs_[2].pulseWidth = 0.1f;
    analogs_[2].pw2SlowStepRatio = 0.1f;
    analogs_[2].ipd2SlowStepRatio = 0.0f;

    analogs_[3].baseLabel = 'C';
    analogs_[3].relAmplitude = 1.00f;
    analogs_[3].excessNoiseCV = 0.10f;
    analogs_[3].interPulseDistance = 0.1f;
    analogs_[3].pulseWidth = 0.1f;
    analogs_[3].pw2SlowStepRatio = 0.1f;
    analogs_[3].ipd2SlowStepRatio = 0.0f;
}

DetectionModelEstimator::DetectionModelEstimator(uint32_t poolId, unsigned int poolSize)
    : poolId_ (poolId)
{
    // TODO
}

void DetectionModelEstimator::InitDetModel(unsigned int lane,
                                           const Data::BaselineStats<laneSize>& blStats)
{
    // TODO: Convert blStats from CudaArray to LaneArrayRef.

    // TODO: Initialize baseline stat accumulator.
    StatAccumulator<LaneArray<float>> bStats;

    // TODO: Initialize the lane detection model.
}


}}}     // namespace PacBio::Mongo::Basecaller
