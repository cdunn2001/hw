
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

void DetectionModelEstimator::InitDetModel(const Data::BaselineStats<laneSize>& blStats,
                                           LaneDetModel& ldm)
{
    using ElementType = typename Data::BaselineStats<laneSize>::ElementType;
    using LaneArr = LaneArray<ElementType>;
    using Clar = ConstLaneArrayRef<ElementType>;

    Clar mom0 (blStats.m0_.data());
    Clar mom1 (blStats.m1_.data());
    Clar mom2 (blStats.m2_.data());

    StatAccumulator<LaneArr> bStats (LaneArr{mom0}, LaneArr{mom1}, LaneArr{mom2});

    using std::copy;
    const auto& blMean = bStats.Mean();
    std::copy(blMean.begin(), blMean.end(), ldm.BaselineMode().means.data());
    const auto& blVar = bStats.Variance();
    const auto& blSigma = sqrt(blVar);
    std::copy(blVar.begin(), blVar.end(), ldm.BaselineMode().vars.data());
    assert(numAnalogs <= analogs_.size());
    const auto refSignal = refSnr_ * blSigma;
    for (unsigned int a = 0; a < numAnalogs; ++a)
    {
        const auto aMean = blMean + analogs_[a].relAmplitude * refSignal;
        copy(aMean.begin(), aMean.end(), ldm.AnalogMode(a).means.data());

        // This noise model assumes that the trace data have been converted to
        // photoelectron units.
        const auto aVar = blVar + aMean + analogs_[a].excessNoiseCV * pow2(aMean);
        copy(aVar.begin(), aVar.end(), ldm.AnalogMode(a).vars.data());
    }
}


}}}     // namespace PacBio::Mongo::Basecaller
