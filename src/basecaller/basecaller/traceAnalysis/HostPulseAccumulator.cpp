#include "HostPulseAccumulator.h"

#include <tbb/parallel_for.h>

#include <dataTypes/configs/MovieConfig.h>
#include "SubframeLabelManager.h"

namespace PacBio {
namespace Mongo {
namespace Basecaller {

template <typename LabelManager>
std::unique_ptr<LabelManager> HostPulseAccumulator<LabelManager>::manager_;

template <typename LabelManager>
void HostPulseAccumulator<LabelManager>::Configure(const Data::MovieConfig& movieConfig,
                                                   const Data::BasecallerPulseAccumConfig& pulseConfig)
{
    const auto hostExecution = true;
    PulseAccumulator::InitFactory(hostExecution, pulseConfig);

    Cuda::Utility::CudaArray<Data::Pulse::NucleotideLabel, numAnalogs> analogMap;

    for(size_t i = 0; i < analogMap.size(); i++)
    {
        analogMap[i] = Data::mapToNucleotideLabel(movieConfig.analogs[i].baseLabel);
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
HostPulseAccumulator<LabelManager>::Process(Data::LabelsBatch labels)
{
    auto ret = batchFactory_->NewBatch(labels.Metadata(), labels.StorageDims());

    tbb::parallel_for(size_t{0}, labels.LanesPerBatch(), [&](size_t laneIdx)
    {
        const auto& blockLabels = labels.GetBlockView(laneIdx);
        const auto& blockLatTrace = labels.LatentTrace().GetBlockView(laneIdx);
        const auto& currTrace = labels.TraceData().GetBlockView(laneIdx);

        auto lanePulses = ret.first.Pulses().LaneView(laneIdx);
        lanePulses.Reset();

        LabelsSegment& currSegment = startSegmentByLane[laneIdx];

        auto baselineStats = BaselineStats{};
        auto blIter = blockLabels.CBegin();
        for (size_t relativeFrameIndex = 0;
             relativeFrameIndex < blockLabels.NumFrames() &&
             blIter != blockLabels.CEnd();
             ++relativeFrameIndex, ++blIter)
        {
            EmitFrameLabels(currSegment, lanePulses, baselineStats,
                            blIter.Extract(), blockLatTrace, currTrace,
                            relativeFrameIndex, relativeFrameIndex + labels.GetMeta().FirstFrame());
        }

        ret.second.baselineStats.GetHostView()[laneIdx] = baselineStats.GetState();
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
                                                         size_t relativeFrameIndex,
                                                         uint32_t absFrameIndex)
{
    auto signal = Signal(relativeFrameIndex, blockLatTrace, currTrace);

    auto boundaryMask = currSegment.IsNewSegment(label);
    auto pulseMask = currSegment.IsPulse();

    const auto& emitPulse = boundaryMask & pulseMask;
    if (any(emitPulse))
    {
        for (size_t i = 0; i < laneSize; ++i)
        {
            if (emitPulse[i])
            {
                pulses.push_back(i, currSegment.ToPulse(absFrameIndex, i, *manager_));
            }
        }
    }

    currSegment.ResetSegment(boundaryMask, absFrameIndex, label, signal);
    currSegment.AddSignal(!boundaryMask, signal);

    const auto& isBaseline = (!pulseMask) & (!boundaryMask);
    if (any(isBaseline))
    {
        baselineStats.AddSample(AsFloat(signal), isBaseline);
    }
}

template class HostPulseAccumulator<SubframeLabelManager>;

}}} // namespace PacBio::Mongo::Basecaller
