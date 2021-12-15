// Copyright (c) 2015-2018, Pacific Biosciences of California, Inc.
//
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted (subject to the limitations in the
// disclaimer below) provided that the following conditions are met:
//
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//
//  * Redistributions in binary form must reproduce the above
//    copyright notice, this list of conditions and the following
//    disclaimer in the documentation and/or other materials provided
//    with the distribution.
//
//  * Neither the name of Pacific Biosciences nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE
// GRANTED BY THIS LICENSE. THIS SOFTWARE IS PROVIDED BY PACIFIC
// BIOSCIENCES AND ITS CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
// OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL PACIFIC BIOSCIENCES OR ITS
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
// USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
// OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
// SUCH DAMAGE.


#include <algorithm>
#include <cmath>
#include <numeric>

#include <bazio/MetricData.h>
#include <bazio/RegionLabel.h>
#include <bazio/RegionLabelType.h>

#include <postprimary/bam/EventData.h>

#include "ZmwMetrics.h"

namespace PacBio {
namespace Primary {
namespace Postprimary {

using namespace PacBio::Primary;

namespace
{

float Mean(const std::vector<float>& v)
{
    return std::accumulate(v.begin(), v.end(), 0.0f) / static_cast<float>(v.size());
}

// Will do it's work in place, modifying v
float Median(std::vector<float>* v)
{
    assert(v != nullptr);
    assert(!v->empty());

    std::nth_element(v->begin(), v->begin() + v->size() / 2, v->end());
    const float medianInsertLength = v->at(v->size() / 2);
    return medianInsertLength;
}

std::pair<std::vector<float>,std::vector<float>>
RetrieveSubreads(const std::vector<RegionLabel>& adapterRegions,
                 const RegionLabel& hqRegion)
{
    std::vector<float> subreadLengths;
    std::vector<float> interiorLengths;

    // Add in all regions that terminate with an adapter.
    bool prevAdapter = false;
    int regionStart = hqRegion.begin;
    for (const auto& currAdapterRegion : adapterRegions)
    {
        if (prevAdapter)
            interiorLengths.push_back(currAdapterRegion.begin - regionStart);

        subreadLengths.push_back(currAdapterRegion.begin - regionStart);
        prevAdapter = true;
        regionStart = currAdapterRegion.end;
    }
    // And the final subread beginning with an adapter.
    if (prevAdapter)
        subreadLengths.push_back(hqRegion.end - regionStart);

    return std::make_pair(subreadLengths, interiorLengths);
}

float ComputePulseRate(const RegionLabel& hqRegion, const BlockLevelMetrics& metrics)
{
    const auto& data = metrics.NumPulsesAll().GetRegion(metrics.GetMetricRegion(hqRegion));
    auto meanNumPulses = std::accumulate(data.cbegin(), data.cend(), 0.0f) / data.size();
    float blockTime = metrics.NumPulsesAll().BlockTime();

    return (blockTime > 0) ? meanNumPulses / blockTime : 0;
}

float ComputePulseWidth(double frameRateHz, const RegionLabel& hqRegion, const BlockLevelMetrics& metrics)
{
    const auto& metricsRegion = metrics.GetMetricRegion(hqRegion);

    // Copy so we can alter data
    auto pulseWidths = metrics.PulseWidth().GetRegion(metricsRegion).copy();
    const auto& numPulses = metrics.NumPulsesAll().GetRegion(metricsRegion);

    if (pulseWidths.empty())
        return 0;

    assert(numPulses.size() == pulseWidths.size());

    for (size_t p = 0; p < pulseWidths.size(); p++)
    {
        pulseWidths[p] = (numPulses[p] != 0) ? (pulseWidths[p] / numPulses[p]) : numPulses[p];
    }

    return static_cast<float>(Mean(pulseWidths) / frameRateHz);
}

AnalogMetricData<uint32_t> BasesPerChannel(const RegionLabel& hqRegion,
                                           const EventData& events)
{
    assert(hqRegion.Length() > 0);
    AnalogMetricData<uint32_t> baseCounts { 0, 0, 0, 0 };
    const auto& basecalls = events.BaseCalls();
    for (int i = hqRegion.begin; i < hqRegion.end; ++i)
    {
        switch (basecalls[i])
        {
            case 'A': baseCounts.A++; break;
            case 'C': baseCounts.C++; break;
            case 'G': baseCounts.G++; break;
            case 'T': baseCounts.T++; break;
            default: break;
        }
    }

    return baseCounts;
};

float ComputeBaseIpd(double frameRateHz,
                     const RegionLabel& hqRegion,
                     const EventData& events)
{
    const auto& allIpds = events.Ipds();
    const auto& isBase = events.IsBase();

    if (allIpds.size() > 0)
    {
        // Use pulse index of HQ-region here which will be valid for production mode
        // where all pulses are called as bases.
        std::vector<uint32_t> hqIpds;
        for (int i = hqRegion.pulseBegin; i < hqRegion.pulseEnd; i++)
        {
            if (isBase[i])
                hqIpds.push_back(allIpds[i]);
        }

        if (hqIpds.size() > 0)
        {
            // Robust estimate of the mean IPD, in sec:
            // For exponential distribution, median(sample)  = log(2) * mean(sample)
            std::nth_element(hqIpds.begin(), hqIpds.begin() + hqIpds.size() / 2, hqIpds.end());
            double meanIpd = (hqIpds[hqIpds.size() / 2] / (double) log(2)) / frameRateHz;
            return static_cast<float>(meanIpd);
        }
    }

    // This occurs when isBase fails to be set true
    // for any of the pulses within the HQ-region.
    return std::numeric_limits<float>::quiet_NaN();
}

float ComputeBaseRate(const RegionLabel& hqRegion, const BlockLevelMetrics& metrics)
{
    const auto& basesData = metrics.NumBasesAll().GetRegion(metrics.GetMetricRegion(hqRegion));
    float meanNumBasesHq = std::accumulate(
            basesData.cbegin(), basesData.cend(), 0.0f
            ) / basesData.size();

    float blockTime = metrics.NumBasesAll().BlockTime();
    return (blockTime > 0) ? (meanNumBasesHq / blockTime) : 0;
}

float ComputeBaseWidth(double frameRateHz,
                       const RegionLabel& hqRegion,
                       const BlockLevelMetrics& metrics)
{
    const auto& metricsRegion = metrics.GetMetricRegion(hqRegion);

    // intentional copy for in place data manipulation
    auto baseWidths = metrics.BaseWidth().GetRegion(metricsRegion).copy();
    const auto& numBases = metrics.NumBasesAll().GetRegion(metricsRegion);

    assert(!numBases.empty());
    assert(!baseWidths.empty());
    assert(numBases.size() == baseWidths.size());

    for (size_t b = 0; b < baseWidths.size(); b++)
    {
        baseWidths[b] = (numBases[b] != 0) ? (baseWidths[b] / numBases[b]) : numBases[b];
    }

    return static_cast<float>(Mean(baseWidths) / frameRateHz);
}

float BaseLocalRate(double frameRateHz,
                    const RegionLabel& hqRegion,
                    const EventData& events,
                    const BlockLevelMetrics& metrics)
{
    int hqLength = hqRegion.Length();
    double meanWidth = ComputeBaseWidth(frameRateHz, hqRegion, metrics);
    double meanIpd = ComputeBaseIpd(frameRateHz, hqRegion, events);

    double totalTimeEst = hqLength * (meanIpd + meanWidth) - meanIpd;
    double baseLocalRate = (totalTimeEst > 0) ? hqLength / totalTimeEst : 0;
    return static_cast<float>(baseLocalRate);
}

AnalogMetricData<float> ComputeSnr(const FilterMetricData<float>& baselineSD,
                                   const AnalogMetricData<float>& pkmids)
{
    auto baselineSigma = static_cast<AnalogMetricData<float>>(baselineSD);

    AnalogMetricData<float> snr;
    snr.A = (baselineSigma.A > 0) ? pkmids.A / baselineSigma.A : std::numeric_limits<float>::quiet_NaN();
    snr.T = (baselineSigma.T > 0) ? pkmids.T / baselineSigma.T : std::numeric_limits<float>::quiet_NaN();
    snr.G = (baselineSigma.G > 0) ? pkmids.G / baselineSigma.G : std::numeric_limits<float>::quiet_NaN();
    snr.C = (baselineSigma.C > 0) ? pkmids.C / baselineSigma.C : std::numeric_limits<float>::quiet_NaN();

    return snr;
}

AnalogMetricData<float> ComputePkmid(const MetricRegion& region,
                                     const BlockLevelMetrics& metrics)
{
    AnalogMetricData<float> pkmid = { 0, 0, 0, 0 };

    auto computePkmid = [](const VectorView<float>& pkmid,
                           const VectorView<uint32_t>& pkmidFrames)
    {
        assert(pkmid.size() == pkmidFrames.size());

        float meanSignal = 0;
        uint32_t totPkmidFrames = 0;
        for (unsigned int m = 0; m < pkmid.size(); ++m)
        {
            if (pkmidFrames[m] > 0)
            {
                totPkmidFrames += pkmidFrames[m];
                meanSignal += static_cast<float>(pkmidFrames[m]) * pkmid[m];
            }
        }

        if (totPkmidFrames > 0)
            meanSignal /= static_cast<float>(totPkmidFrames);

        return meanSignal;
    };

    const auto& bazPkmid = metrics.PkMid().GetRegion(region);
    const auto& bazPkmidFrames = metrics.PkMidFrames().GetRegion(region);
    pkmid.A = computePkmid(bazPkmid.A, bazPkmidFrames.A);
    pkmid.C = computePkmid(bazPkmid.C, bazPkmidFrames.C);
    pkmid.G = computePkmid(bazPkmid.G, bazPkmidFrames.G);
    pkmid.T = computePkmid(bazPkmid.T, bazPkmidFrames.T);

    return pkmid;
}

FilterMetricData<float> ComputeBaselineSD(const MetricRegion& region,
                                          const BlockLevelMetrics& metrics)
{
    FilterMetricData<float> baselineStd;

    const auto& bazBaselineSD = metrics.BaselineSD().GetRegion(region);

    auto medianNoZeros = [](std::vector<float>&& baselineStd) {
        baselineStd.erase(std::remove_if(baselineStd.begin(), baselineStd.end(),
                                         [](double v) { return v == 0; }),
                          baselineStd.end());
        return (baselineStd.size() > 0) ? Median(&baselineStd) : std::numeric_limits<float>::quiet_NaN();
    };

    baselineStd.red = medianNoZeros(bazBaselineSD.red.copy());
    baselineStd.green = medianNoZeros(bazBaselineSD.green.copy());

    return baselineStd;
}

FilterMetricData<float> ComputeBaseline(const MetricRegion& region,
                                        const BlockLevelMetrics& metrics)
{
    FilterMetricData<float> baseline { 0, 0};
    const auto& bazBaseline = metrics.BaselineMean().GetRegion(region);

    auto medianNoZeros = [](std::vector<float>&& baseline) {
        baseline.erase(std::remove_if(baseline.begin(), baseline.end(),
                                      [](float v) { return v == 0; }),
                       baseline.end());
        return (baseline.size() > 0) ? Median(&baseline) : std::numeric_limits<float>::quiet_NaN();
    };

    baseline.red = medianNoZeros(bazBaseline.red.copy());
    baseline.green = medianNoZeros(bazBaseline.green.copy());
    return baseline;
}

float ComputePausiness(double frameRateHz, const RegionLabel& hqRegion, const EventData& events)
{
    const float pauseIpdThreshSec = 2.5;
    uint32_t threshold = static_cast<uint32_t>(pauseIpdThreshSec * frameRateHz);

    double pauses = 0;
    double numBases = 0;
    int hqLength = hqRegion.Length();
    if (hqLength > 1)
    {
        numBases = hqLength;

        const auto& ipds = events.Ipds();
        const auto& isBase = events.IsBase();

        if (ipds.size() > 0)
        {
            for (int i = hqRegion.pulseBegin; i < hqRegion.pulseEnd; ++i)
            {
                if (isBase[i] && ipds[i] > threshold) ++pauses;
            }
        }
    }
    return static_cast<float>((numBases > 0) ? pauses / numBases : 0);
}

AnalogMetricData<float> DMEFilteredAngleEstimate(const MetricRegion& region,
                                                 const BlockLevelMetrics& metrics)
{
    FilterMetricData<float> filterAngles = {0, 0};
    // Get the angle estimates with respect to the region.
    auto GetAngleEstimate = [](std::vector<float>&& angleEstimates) {

        angleEstimates.erase(std::remove_if(angleEstimates.begin(), angleEstimates.end(),
                                            [](double v) { return isnan(v); }), angleEstimates.end());

        if (angleEstimates.empty())
            return std::numeric_limits<float>::quiet_NaN();
        else
            return Median(&angleEstimates);
    };

    const auto& bazAngle = metrics.Angle().GetRegion(region);

    filterAngles.red = GetAngleEstimate(bazAngle.red.copy());
    filterAngles.green = GetAngleEstimate(bazAngle.green.copy());

    return static_cast<AnalogMetricData<float>>(filterAngles);
}

bool ComputeIsRead(float movieTimeInHrs, const RegionLabel& hqRegion, const EventData& events, const ProductivityInfo& prod)
{
    static const int minBpPerHr = 100;
    const double bpPerChanThresh = minBpPerHr * movieTimeInHrs;

    if (prod.isSequencing && events.NumBases() > (bpPerChanThresh * 4) && hqRegion.Length() > 0)
    {
        const auto basesPerChannel = BasesPerChannel(hqRegion, events);
        return (basesPerChannel.A > bpPerChanThresh &&
                basesPerChannel.C > bpPerChanThresh &&
                basesPerChannel.G > bpPerChanThresh &&
                basesPerChannel.T > bpPerChanThresh);
    }
    return false;
}

FilterMetricData<float> AverageMinSnr(const MetricRegion& region,
                                      const BlockLevelMetrics& metrics)
{
     const auto& chanMinSnr = metrics.ChannelMinSNR().GetRegion(region);
     FilterMetricData<float> minSnr;
     minSnr.green = std::accumulate(
         chanMinSnr.green.cbegin(), chanMinSnr.green.cend(), 0.0f)
         / chanMinSnr.green.size();
     minSnr.red = std::accumulate(
         chanMinSnr.red.cbegin(), chanMinSnr.red.cend(), 0.0f)
         / chanMinSnr.red.size();
     return minSnr;
}

uint32_t TotalPulseCount(const RegionLabel& hqRegion, const BlockLevelMetrics& metrics)
{
    const auto& pulse_counts = metrics.NumPulsesAll().GetRegion(metrics.GetMetricRegion(hqRegion));
    return std::accumulate(pulse_counts.cbegin(), pulse_counts.cend(), 0);
}

} // anon namespace

/// Computes the following insert read stats:
/// 1) an estimate of the insert length using the adapter hits
/// 2) max subread length
/// 3) median insert length using subreads flanked by adapters
/// 4) insert read score (currently uses default)
SubreadMetrics::SubreadMetrics(const RegionLabel& hqRegion, std::vector<RegionLabel> adapterRegions)
{
    if (hqRegion.Length() < 50)
    {
        // This is already handled by the default initializer, but making it
        // more explicit here.
        meanLength_ = 0;
        maxSubreadLength_ = 0;
        medianLength_ = 0;
        readScore_ = 0;
        return;
    }

    // Get only adapter hits that are within the HQ-region.
    adapterRegions.erase(
        std::remove_if(
            adapterRegions.begin(), adapterRegions.end(),
            [&hqRegion](const RegionLabel& a)
            { return (a.end < hqRegion.begin || a.begin > hqRegion.end); }),
        adapterRegions.end());

    // Subreads are regions with at least one adapter hit, interiors are flanked by adapter hits.
    std::vector<float> subreadLengths;
    std::vector<float> interiorLengths;
    std::tie(subreadLengths, interiorLengths) = RetrieveSubreads(adapterRegions, hqRegion);

    if (subreadLengths.empty())
    {
        // No adapters found.
        meanLength_ = 0;
        maxSubreadLength_ = hqRegion.Length();
        medianLength_ = 0;
        readScore_ = 0;
        umy_ = maxSubreadLength_;
        return;
    }
    else if (subreadLengths.size() <= 2)
    {
        // Exactly 1 adapter found.
        const float insertLength = *std::max_element(subreadLengths.begin(), subreadLengths.end());
        meanLength_ = 0;
        maxSubreadLength_ = insertLength;
        medianLength_ = 0;
        readScore_ = 0;
        umy_ = maxSubreadLength_;
        return;
    }
    else
    {
        // 2 or more adapters found.
        assert(!interiorLengths.empty());
        const float insertLength = *std::max_element(subreadLengths.begin(), subreadLengths.end());
        float meanInteriorLengths = Mean(interiorLengths);
        // Doing this second since it will mutate the data (though that wouldn't have major impact on the mean)
        float medianInteriorLengths = Median(&interiorLengths);
        meanLength_ = meanInteriorLengths;
        maxSubreadLength_ = insertLength;
        medianLength_ = medianInteriorLengths;
        readScore_ = 0;
        umy_ = medianLength_;
        return;
    }
}

SignalMetrics::SignalMetrics(const MetricRegion& region,
                             const BlockLevelMetrics& metrics)
    : baseline_(ComputeBaseline(region, metrics))
    , baselineSD_(ComputeBaselineSD(region, metrics))
    , minSnr_(AverageMinSnr(region, metrics))
    , pkMid_(ComputePkmid(region, metrics))
    , snr_(ComputeSnr(baselineSD_, pkMid_))
    , angle_(DMEFilteredAngleEstimate(region, metrics))
{}

ExcludedPulseMetrics::ExcludedPulseMetrics(const std::vector<InsertState>& insertStates)
{
    std::map<InsertState, uint32_t> insertCounts = {
            {InsertState::BASE, 0},
            {InsertState::EX_SHORT_PULSE, 0},
            {InsertState::BURST_PULSE, 0},
            {InsertState::PAUSE_PULSE, 0}};
    std::map<InsertState, uint32_t> insertLengths = {
            {InsertState::BASE, 0},
            {InsertState::EX_SHORT_PULSE, 0},
            {InsertState::BURST_PULSE, 0},
            {InsertState::PAUSE_PULSE, 0}};
    // 'last' should never be ex_short_pulse, because we don't treat it as
    // a state you can enter. We search for the first non-short label as
    // our initial current state (stored in 'last')
    InsertState last;
    for (InsertState label : insertStates)
        if (label != InsertState::EX_SHORT_PULSE)
        {
            last = label;
            break;
        }
    // The counter starts at 0, so when we encounter the first non-short
    // label we don't double count it
    uint32_t lastc = 0;
    for (InsertState label : insertStates)
    {
        // We want to count short pulses, but we don't want to let them
        // break other longer inserts (they can't be overturned by InsertFinder)
        if (label == InsertState::EX_SHORT_PULSE)
        {
            ++insertCounts[InsertState::EX_SHORT_PULSE];
            ++insertLengths[InsertState::EX_SHORT_PULSE];
            continue;
        }
        // We want to keep track of all other states, for sanity checks and
        // useful metrics
        if (label == last)
        {
            ++lastc;
        }
        else
        {
            ++insertCounts[last];
            insertLengths[last] += lastc;
            lastc = 1;
            last = label;
        }
    }
    // Pick up what remained when we ran out of pulses
    if (lastc > 0)
    {
        ++insertCounts[last];
        insertLengths[last] += lastc;
    }

    insertCounts_.base         = insertCounts[InsertState::BASE];
    insertCounts_.exShortPulse = insertCounts[InsertState::EX_SHORT_PULSE];
    insertCounts_.burstPulse   = insertCounts[InsertState::BURST_PULSE];
    insertCounts_.pausePulse   = insertCounts[InsertState::PAUSE_PULSE];
    insertLengths_.base         = insertLengths[InsertState::BASE];
    insertLengths_.exShortPulse = insertLengths[InsertState::EX_SHORT_PULSE];
    insertLengths_.burstPulse   = insertLengths[InsertState::BURST_PULSE];
    insertLengths_.pausePulse   = insertLengths[InsertState::PAUSE_PULSE];
}

PulseMetrics::PulseMetrics(double frameRateHz,
                           const RegionLabel& hqRegion,
                           const BlockLevelMetrics& metrics)
    : width_(ComputePulseWidth(frameRateHz, hqRegion, metrics))
    , rate_(ComputePulseRate(hqRegion, metrics))
    , totalCount_(TotalPulseCount(hqRegion, metrics))
{}

BaseMetrics::BaseMetrics(double frameRateHz,
                         const RegionLabel& hqRegion,
                         const BlockLevelMetrics& metrics,
                         const EventData& events)
    : width_(ComputeBaseWidth(frameRateHz, hqRegion, metrics))
    , rate_(ComputeBaseRate(hqRegion, metrics))
    , localRate_(BaseLocalRate(frameRateHz, hqRegion, events, metrics))
    , ipd_(ComputeBaseIpd(frameRateHz, hqRegion, events))
    , HQRatio_(static_cast<float>(hqRegion.Length()) / events.NumBases())
    , pausiness_(ComputePausiness(frameRateHz, hqRegion, events))
    , counts_(BasesPerChannel(hqRegion, events))
{}

ReadMetrics::ReadMetrics(float movieTimeInHrs,
                         uint32_t unitFeatures,
                         const RegionLabel& hqRegion,
                         const EventData& events,
                         const ProductivityInfo& prod)
    : unitFeatures_(unitFeatures)
    , readLength_(hqRegion.Length())
    , polyLength_(events.NumBases())
    , holeNumber_(events.ZmwNumber())
    , internal_(events.Internal())
    , isRead_(ComputeIsRead(movieTimeInHrs, hqRegion, events, prod))
{}

ZmwMetrics::ZmwMetrics(float movieTimeInHrs,
                       float frameRateHz,
                       uint32_t unitFeatures,
                       const RegionLabel& hqRegion,
                       const std::vector<RegionLabel>& adapters,
                       const BlockLevelMetrics& metrics,
                       const EventData& events,
                       const ProductivityInfo& prod,
                       const struct ControlMetrics& control,
                       const AdapterMetrics& adapterMetrics)
    : zmwSignalMetrics_(metrics.GetFullRegion(), metrics)
    , excludedPulseMetrics_(events.InsertStates())
    , readMetrics_(movieTimeInHrs, unitFeatures, hqRegion, events, prod)
    , prodMetrics_(prod)
    , controlMetrics_(control)
    , adapterMetrics_(adapterMetrics)
{
    if (hqRegion.Length() > 0 && !metrics.NumBasesAll().GetRegion(metrics.GetMetricRegion(hqRegion)).empty())
    {
        pulseMetrics_ = Postprimary::PulseMetrics(frameRateHz, hqRegion, metrics);
        hqSignalMetrics_ = SignalMetrics(metrics.GetMetricRegion(hqRegion), metrics);
        baseMetrics_ = Postprimary::BaseMetrics(frameRateHz, hqRegion, metrics, events);
        subreadMetrics_ = Postprimary::SubreadMetrics(hqRegion, adapters);
    }
}

ZmwMetrics::ZmwMetrics(const RegionLabel& hqRegion,
                       const std::vector<RegionLabel>& adapters,
                       const ProductivityInfo& prod,
                       const struct ControlMetrics& control,
                       const AdapterMetrics& adapterMetrics,
                       bool computeInsertStats)
    : prodMetrics_(prod)
    , controlMetrics_(control)
    , adapterMetrics_(adapterMetrics)
{
    if(computeInsertStats)
    {
        subreadMetrics_ = Postprimary::SubreadMetrics(hqRegion, adapters);
    }
}

}}}
