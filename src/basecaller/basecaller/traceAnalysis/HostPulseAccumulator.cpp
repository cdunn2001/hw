#include "HostPulseAccumulator.h"
#include "SubframeLabelManager.h"

namespace PacBio {
namespace Mongo {
namespace Basecaller {

template <typename LabelManager>
std::unique_ptr<LabelManager> HostPulseAccumulator<LabelManager>::manager_;

template <typename LabelManager>
void HostPulseAccumulator<LabelManager>::Configure(size_t maxCallsPerZmw)
{
    const auto hostExecution = true;
    PulseAccumulator::InitAllocationPools(hostExecution, maxCallsPerZmw);

    Cuda::Utility::CudaArray<Data::Pulse::NucleotideLabel, numAnalogs> analogMap;
    // TODO once ready, this needs to be plumbed in from the movie configuration.
    analogMap[0] = Data::Pulse::NucleotideLabel::A;
    analogMap[1] = Data::Pulse::NucleotideLabel::C;
    analogMap[2] = Data::Pulse::NucleotideLabel::G;
    analogMap[3] = Data::Pulse::NucleotideLabel::T;
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
Data::PulseBatch HostPulseAccumulator<LabelManager>::Process(Data::LabelsBatch labels)
{
    auto ret = batchFactory_->NewBatch(labels.Metadata());

    for (size_t laneIdx = 0; laneIdx < labels.LanesPerBatch(); ++laneIdx)
    {
        const auto& blockLabels = labels.GetBlockView(laneIdx);
        const auto& blockLatTrace = labels.LatentTrace().GetBlockView(laneIdx);
        const auto& currTrace = labels.TraceData().GetBlockView(laneIdx);

        auto lanePulses = ret.Pulses().LaneView(laneIdx);
        lanePulses.Reset();

        LabelsSegment& currSegment = startSegmentByLane[laneIdx];

        auto blIter = blockLabels.CBegin();
        for (size_t relativeFrameIndex = 0;
             relativeFrameIndex < blockLabels.NumFrames() &&
             blIter != blockLabels.CEnd();
             ++relativeFrameIndex, ++blIter)
        {
            EmitFrameLabels(currSegment, lanePulses, *blIter, blockLatTrace, currTrace,
                            relativeFrameIndex, relativeFrameIndex + labels.GetMeta().FirstFrame());
        }
    }

    return ret;
}
template <typename LabelManager>
void HostPulseAccumulator<LabelManager>::EmitFrameLabels(LabelsSegment& currSegment, Data::LaneVectorView<Data::Pulse>& pulses,
                                           const ConstLabelArrayRef& label, const SignalBlockView& blockLatTrace,
                                           const SignalBlockView& currTrace, size_t relativeFrameIndex,
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
}

template class HostPulseAccumulator<SubframeLabelManager>;

}}} // namespace PacBio::Mongo::Basecaller
