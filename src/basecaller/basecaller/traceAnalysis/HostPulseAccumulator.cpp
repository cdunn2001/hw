#include "HostPulseAccumulator.h"

#include <tbb/task_arena.h>
#include <tbb/parallel_for.h>

#include <dataTypes/configs/AnalysisConfig.h>

namespace PacBio {
namespace Mongo {
namespace Basecaller {

template <typename LabelManager>
std::unique_ptr<LabelManager> HostPulseAccumulator<LabelManager>::manager_;

template <typename LabelManager>
void HostPulseAccumulator<LabelManager>::Configure(const Data::AnalysisConfig& analysisConfig,
                                                   const Data::BasecallerPulseAccumConfig& pulseConfig)
{
    const auto hostExecution = true;
    PulseAccumulator::InitFactory(hostExecution, pulseConfig);

    Cuda::Utility::CudaArray<Data::Pulse::NucleotideLabel, numAnalogs> analogMap;

    for(size_t i = 0; i < analogMap.size(); i++)
    {
        analogMap[i] = Data::mapToNucleotideLabel(analysisConfig.movieInfo.analogs[i].baseLabel);
    }

    manager_ = std::make_unique<LabelManager>(analogMap);
}

template <typename LabelManager>
void HostPulseAccumulator<LabelManager>::Finalize()
{
    manager_.release();
    PulseAccumulator::Finalize();
}

template <typename LabelManager>
HostPulseAccumulator<LabelManager>::HostPulseAccumulator(uint32_t poolId, uint32_t lanesPerBatch)
    : PulseAccumulator(poolId)
    , startSegmentByLane(lanesPerBatch)
    { }

template <typename LabelManager>
HostPulseAccumulator<LabelManager>::~HostPulseAccumulator() = default;

template <typename LabelManager>
std::pair<Data::PulseBatch, Data::PulseDetectorMetrics>
HostPulseAccumulator<LabelManager>::Process(Data::LabelsBatch labels, const PoolModelParameters& models)
{
    auto ret = batchFactory_->NewBatch(labels.Metadata(), labels.StorageDims());

    auto modelView = models.GetHostView();

    tbb::task_arena().execute([&] {
        tbb::parallel_for(size_t{0}, labels.LanesPerBatch(), [&](size_t laneIdx) {
            const auto& blockLabels = labels.GetBlockView(laneIdx);
            const auto& blockLatTrace = labels.LatentTrace().GetBlockView(laneIdx);
            const auto& currTrace = labels.TraceData().GetBlockView(laneIdx);

            auto lanePulses = ret.first.Pulses().LaneView(laneIdx);
            lanePulses.Reset();

            LabelsSegment& currSegment = startSegmentByLane[laneIdx];
            // Last analog channel is the darkest (lowest amplitude)
            FloatArray minMean(modelView[laneIdx].AnalogMode(numAnalogs-1).means);

            auto baselineStats = BaselineStats{};
            auto blIter = blockLabels.CBegin();
            for (size_t relativeFrameIndex = 0;
                 relativeFrameIndex < blockLabels.NumFrames() &&
                 blIter != blockLabels.CEnd();
                 ++relativeFrameIndex, ++blIter)
            {
                EmitFrameLabels(currSegment, lanePulses, baselineStats,
                                blIter.Extract(), blockLatTrace, currTrace, minMean,
                                relativeFrameIndex, relativeFrameIndex + labels.GetMeta().FirstFrame());
            }

            ret.second.baselineStats.GetHostView()[laneIdx] = baselineStats.GetState();
        });
    });

    return ret;
}
template <typename LabelManager>
void HostPulseAccumulator<LabelManager>::EmitFrameLabels(LabelsSegment& currSegment,
                                                         Data::LaneVectorView<Data::Pulse>& pulses,
                                                         BaselineStats& baselineStats,
                                                         const LabelArray& label,
                                                         const SignalBlockView& blockLatTrace,
                                                         const SignalBlockView& currTrace,
                                                         const FloatArray& meanAmp,
                                                         size_t relativeFrameIndex,
                                                         uint32_t absFrameIndex)
{
    auto signal = Signal(relativeFrameIndex, blockLatTrace, currTrace);

    auto boundaryMask = currSegment.IsNewSegment(label);
    auto pulseMask = currSegment.IsPulse();

    const auto& emitPulse = boundaryMask & pulseMask;
    if (any(emitPulse))
    {
        for (size_t z = 0; z < laneSize; ++z)
        {
            if (emitPulse[z])
            {
                pulses.push_back(z, currSegment.ToPulse(absFrameIndex, z, MakeUnion(meanAmp)[z], *manager_));
            }
        }
    }

    currSegment.ResetSegment(boundaryMask, absFrameIndex, label, signal);
    currSegment.AddSignal(!boundaryMask, signal);

    // TODO: The below excludes the baseline frame succeeding a pulse
    // but we also want to exclude the baseline frame preceding a pulse
    // to not contaminate the baseline statistics by frames that
    // might be partially influenced by an adjacent pulse frame.
    const auto& isBaseline = (!pulseMask) & (!boundaryMask);
    if (any(isBaseline))
    {
        baselineStats.AddSample(FloatArray{signal}, isBaseline);
    }
}

template class HostPulseAccumulator<SubframeLabelManager>;

}}} // namespace PacBio::Mongo::Basecaller
