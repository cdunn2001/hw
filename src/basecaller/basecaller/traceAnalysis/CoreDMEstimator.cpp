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

// static
LaneArray<float> CoreDMEstimator::ModelSignalCovar(
        const Data::AnalogMode& analog,
        const LaneArray<float>& signalMean,
        const LaneArray<float>& baselineVar)
{
    LaneArray<float> r {baselineVar};
    r += signalMean;
    r += pow2(analog.excessNoiseCV * signalMean);
    return r;
}

CoreDMEstimator::CoreDMEstimator(uint32_t poolId, unsigned int poolSize)
    : poolId_ (poolId)
    , poolSize_ (poolSize)
{
}


CoreDMEstimator::PoolDetModel
CoreDMEstimator::InitDetectionModels(const PoolBaselineStats& blStats) const
{
    PoolDetModel pdm (poolSize_, Cuda::Memory::SyncDirection::HostWriteDeviceRead, SOURCE_MARKER());

    auto pdmHost = pdm.GetHostView();
    const auto& blStatsHost = blStats.GetHostView();
    for (unsigned int lane = 0; lane < poolSize_; ++lane)
    {
        InitLaneDetModel(blStatsHost[lane], pdmHost[lane]);
    }

    return pdm;
}


void CoreDMEstimator::InitLaneDetModel(const Data::BaselinerStatAccumState& blStats,
                                       LaneDetModel& ldm) const
{
    using ElementType = typename Data::BaselinerStatAccumState::StatElement;
    using LaneArr = LaneArray<ElementType>;

    StatAccumulator<LaneArr> blsa (blStats.baselineStats);

    const auto& blMean = fixedBaselineParams_ ? fixedBaselineMean_ : blsa.Mean();
    const auto& blVar = fixedBaselineParams_ ? fixedBaselineVar_ : blsa.Variance();
    // TODO: We want to pull the total number of frames from the blStats object but the
    // fullAutocorrState is not yet populated by the GPU baseliner. For now, we just use the
    // minimum frames for estimating.
    //const auto& blWeight = CLanArrRef(blStats.NumBaselineFrames()) / CLanArrRef(blStats.TotalFrames());
    const auto& blWeight = LaneArr(blStats.NumBaselineFrames()) / LaneArr(blStats.fullAutocorrState.basicStats.moment0);

    ldm.BaselineMode().means = blMean;
    ldm.BaselineMode().vars = blVar;
    ldm.BaselineWeight() = blWeight;
    assert(numAnalogs <= analogs_.size());
    const auto refSignal = refSnr_ * sqrt(blVar);
    for (unsigned int a = 0; a < numAnalogs; ++a)
    {
        const auto aMean = blMean + analogs_[a].relAmplitude * refSignal;
        auto& aMode = ldm.AnalogMode(a);
        aMode.means = aMean;

        // This noise model assumes that the trace data have been converted to
        // photoelectron units.
        aMode.vars = ModelSignalCovar(Analog(a), aMean, blVar);
    }
}

}}}     // namespace PacBio::Mongo::Basecaller
