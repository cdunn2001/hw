#include "HostPulseAccumulator.h"

namespace PacBio {
namespace Mongo {
namespace Basecaller {

void HostPulseAccumulator::Configure(size_t maxCallsPerZmw)
{
    const auto hostExecution = true;
    PulseAccumulator::InitAllocationPools(hostExecution, maxCallsPerZmw);
}

void HostPulseAccumulator::Finalize()
{
    PulseAccumulator::Finalize();
}

HostPulseAccumulator::HostPulseAccumulator(uint32_t poolId, uint32_t lanesPerBatch)
    : PulseAccumulator(poolId)
    , startSegmentByLane(lanesPerBatch)
    { }

HostPulseAccumulator::~HostPulseAccumulator() = default;

Data::PulseBatch HostPulseAccumulator::Process(Data::LabelsBatch labels)
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

void HostPulseAccumulator::EmitFrameLabels(LabelsSegment& currSegment, Data::LaneVectorView<Data::Pulse>& pulses,
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
                pulses.push_back(i, currSegment.ToPulse(absFrameIndex, i));
            }
        }
    }

    currSegment.ResetSegment(boundaryMask, absFrameIndex, label, signal);
    currSegment.AddSignal(!boundaryMask, signal);
}


}}} // namespace PacBio::Mongo::Basecaller