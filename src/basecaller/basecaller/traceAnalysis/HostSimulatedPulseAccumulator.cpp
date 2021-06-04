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
    , mt_(config_.seed + poolId)
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

    auto makeGenerator = [&](const auto& vec, const auto& vars)
    {
        return [&]()
        {
            if (vars.empty())
            {
                auto idx = pulseId_[zmw] % vec.size();
                return vec[idx];
            } else {
                auto mean = vec[zmw % vec.size()];
                auto var = vars[zmw % vars.size()];
                auto beta = var / mean;
                auto alpha = mean / beta;
                auto dist = std::gamma_distribution<float>(alpha, beta);
                return static_cast<decltype(vec[0])>(dist(mt_));
            }
        };
    };

    auto makeIntGenerator = [&](const auto& vec, const auto& vars)
    {
        return [gen = makeGenerator(vec, vars)]()
        {
            return static_cast<int>(std::round(gen()));
        };
    };

    decltype(config_.basecalls) tmp{};
    auto basecall = makeGenerator(config_.basecalls,tmp);
    auto ipd = makeIntGenerator(config_.ipds, config_.ipdVars);
    auto pw = makeIntGenerator(config_.pws, config_.pwVars);
    auto meanSignal = makeGenerator(config_.meanSignals, config_.meanSignalsVars);
    auto midSignal = makeGenerator(config_.midSignals, config_.midSignalsVars);
    auto maxSignal = makeGenerator(config_.maxSignals, config_.maxSignalsVars);

    auto pulse = Data::Pulse();

    pulse.Label(Data::mapToNucleotideLabel(basecall()));
    pulse.Start(nextPulse_[zmw].Stop() + ipd());
    pulse.Width(pw());
    pulse.MeanSignal(meanSignal());
    pulse.MidSignal(midSignal());
    pulse.MaxSignal(maxSignal());
    if (pulse.Width() > 2)
        pulse.SignalM2(pulse.MeanSignal() * pulse.MeanSignal() * 2 * (pulse.Width() - 2));
    else
        pulse.SignalM2(0);
    pulse.IsReject(false);

    return pulse;
}

}}} // namespace PacBio::Mongo::Basecaller
