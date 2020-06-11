#include "HostSimulatedPulseAccumulator.h"

namespace PacBio {
namespace Mongo {
namespace Basecaller {

void HostSimulatedPulseAccumulator::Configure(const Data::BasecallerPulseAccumConfig& pulseConfig)
{
    const auto hostExecution = true;
    PulseAccumulator::InitFactory(hostExecution, pulseConfig);
}

void HostSimulatedPulseAccumulator::Finalize()
{
    PulseAccumulator::Finalize();
}

HostSimulatedPulseAccumulator::HostSimulatedPulseAccumulator(uint32_t poolId)
    : PulseAccumulator(poolId)
    { }

HostSimulatedPulseAccumulator::~HostSimulatedPulseAccumulator() = default;

std::pair<Data::PulseBatch, Data::PulseDetectorMetrics>
HostSimulatedPulseAccumulator::Process(Data::LabelsBatch labels)
{
    auto ret = batchFactory_->NewBatch(labels.Metadata(), labels.StorageDims());

    for (size_t laneIdx = 0; laneIdx < labels.LanesPerBatch(); ++laneIdx)
    {
        auto lanePulses = ret.first.Pulses().LaneView(laneIdx);
        lanePulses.Reset();
        for (size_t zmwIdx = 0; zmwIdx < labels.LaneWidth(); ++zmwIdx)
        {
            for (uint32_t pulseNum = 0; pulseNum < ret.first.Pulses().MaxLen(); ++pulseNum)
            {
                lanePulses.push_back(zmwIdx, GeneratePulse(pulseNum));
            }
        }
    }

    return ret;
}

Data::Pulse HostSimulatedPulseAccumulator::GeneratePulse(uint32_t pulseNum)
{
    using NucleotideLabel = Data::Pulse::NucleotideLabel;
    // Repeating sequence of ACGT.
    const NucleotideLabel labels[] =
        { NucleotideLabel::A, NucleotideLabel::C, NucleotideLabel::G, NucleotideLabel::T };

    // Associated values
    const std::array<float, 4> meanSignals { { 20.0f, 10.0f, 16.0f, 8.0f } };
    const std::array<float, 4> midSignals { { 21.0f, 11.0f, 17.0f, 9.0f } };
    const std::array<float, 4> maxSignals { { 21.0f, 11.0f, 17.0f, 9.0f } };

    size_t iL = pulseNum % 4;

    auto pulse = Data::Pulse();

    pulse.Label(labels[iL]);
    pulse.Start(1).Width(3);
    pulse.MeanSignal(meanSignals[iL]).
        MidSignal(midSignals[iL]).
        MaxSignal(maxSignals[iL]).
        SignalM2(meanSignals[iL] * meanSignals[iL]).
        IsReject(false);

    return pulse;
}

}}} // namespace PacBio::Mongo::Basecaller
