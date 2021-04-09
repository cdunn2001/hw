#include "HostSimulatedPulseAccumulator.h"

namespace PacBio {
namespace Mongo {
namespace Basecaller {

void HostSimulatedPulseAccumulator::Configure(const Data::BasecallerPulseAccumConfig& pulseConfig)
{
    const auto hostExecution = true;
    config_ = pulseConfig.simConfig;
    PulseAccumulator::InitFactory(hostExecution, pulseConfig);
}

void HostSimulatedPulseAccumulator::Finalize()
{
    PulseAccumulator::Finalize();
}

Data::SimulatedPulseConfig HostSimulatedPulseAccumulator::config_;

HostSimulatedPulseAccumulator::HostSimulatedPulseAccumulator(uint32_t poolId, size_t numLanes)
    : PulseAccumulator(poolId)
    , nextPulse_(numLanes*laneSize)
    , pulseId_(numLanes*laneSize)
    {
        for (size_t i = 0; i < nextPulse_.size(); ++i)
        {
            nextPulse_[i] = GeneratePulse(i);
            pulseId_[i] = 1;
        }
    }

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
            size_t id = zmwIdx + laneIdx*laneSize;
            while (nextPulse_[id].Stop() < labels.Metadata().LastFrame())
            {
                lanePulses.push_back(zmwIdx, nextPulse_[id]);
                nextPulse_[id] = GeneratePulse(id);
                pulseId_[id]++;
            }
        }
    }

    return ret;
}

Data::Pulse HostSimulatedPulseAccumulator::GeneratePulse(size_t zmw) const
{
    using NucleotideLabel = Data::Pulse::NucleotideLabel;

    auto makeGenerator = [&](const auto& vec)
    {
        return [&]()
        {
            auto idx = pulseId_[zmw] % vec.size();
            return vec[idx];
        };
    };

    auto basecall = makeGenerator(config_.basecalls);
    auto ipd = makeGenerator(config_.ipds);
    auto pw = makeGenerator(config_.pws);
    auto meanSignal = makeGenerator(config_.meanSignals);
    auto midSignal = makeGenerator(config_.midSignals);
    auto maxSignal = makeGenerator(config_.maxSignals);

    auto pulse = Data::Pulse();

    pulse.Label(Data::mapToNucleotideLabel(basecall()));
    pulse.Start(nextPulse_[zmw].Stop() + ipd());
    pulse.Width(pw());
    pulse.MeanSignal(meanSignal());
    pulse.MidSignal(midSignal());
    pulse.MaxSignal(maxSignal());
    pulse.SignalM2(pulse.MeanSignal() * pulse.MeanSignal());
    pulse.IsReject(false);

    return pulse;
}

}}} // namespace PacBio::Mongo::Basecaller
