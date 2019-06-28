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

HostPulseAccumulator::HostPulseAccumulator(uint32_t poolId)
        : PulseAccumulator(poolId)
{ }

HostPulseAccumulator::~HostPulseAccumulator() = default;

Data::PulseBatch HostPulseAccumulator::Process(Data::LabelsBatch labels)
{
    auto ret = batchFactory_->NewBatch(labels.Metadata());

    for (size_t laneIdx = 0; laneIdx < labels.LanesPerBatch(); ++laneIdx)
    {
        auto blockLabels = labels.GetBlockView(laneIdx);
        auto lanePulses = ret.Pulses().LaneView(laneIdx);
        lanePulses.Reset();

    }

    return ret;
}



}}} // namespace PacBio::Mongo::Basecaller