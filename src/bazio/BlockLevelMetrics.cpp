// Copyright (c) 2018, Pacific Biosciences of California, Inc.
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

#include "DataParsing.h"
#include "BlockLevelMetrics.h"

#include <cmath>
#include <half.hpp>

namespace PacBio {
namespace Primary {

BlockLevelMetrics::BlockLevelMetrics(const RawMetricData& rawMetrics,
                                     uint32_t metricFrames,
                                     double frameRateHz,
                                     const std::vector<float> relAmps,
                                     const std::string& baseMap,
                                     bool internal)
{
    internal_ = internal;
    //assert(fh.MetricFieldScaling(MetricFieldName::NUM_PULSES, frequency) == 1);

    // The below assumes that the metrics are all of the same size.
    const size_t expectedSize = [&]() -> size_t
    {
        for (const auto& metric : rawMetrics.IntMetrics())
        {
            if (!metric.empty()) return metric.size();
        }
        for (const auto& metric : rawMetrics.UIntMetrics())
        {
            if (!metric.empty()) return metric.size();
        }
        for (const auto& metric : rawMetrics.FloatMetrics())
        {
            if (!metric.empty()) return metric.size();
        }
        return 0;
    }();

    // We technically allow empty metrics right now, but this should be changed.
    if (rawMetrics.Empty())
    {
        empty_ = true;
        return;
    }

    size_t framesPerBlock = metricFrames;
    const auto frameRate = static_cast<float>(frameRateHz);

    const auto toFloatVec = [](const auto& data) {
        std::vector<float> ret;
        ret.reserve(data.size());
        for (const auto& v : data) ret.push_back(static_cast<float>(v));
        return ret;
    };

    if (rawMetrics.HasMetric(MetricFieldName::NUM_FRAMES))
    {
        numFrames_             = SingleMetric<uint32_t>(rawMetrics.UIntMetric(MetricFieldName::NUM_FRAMES),
                                                        frameRate, framesPerBlock);
    }
    else
    {
        numFrames_             = SingleMetric<uint32_t>(std::vector<uint32_t>(expectedSize, framesPerBlock),
                                                        frameRate, framesPerBlock);
    }

    numPulsesAll_          = SingleMetric<uint32_t>(rawMetrics.UIntMetric(MetricFieldName::NUM_PULSES),
                                                    frameRate, framesPerBlock);
    pulseWidth_            = SingleMetric<float>(toFloatVec(rawMetrics.UIntMetric(MetricFieldName::PULSE_WIDTH)),
                                                 frameRate, framesPerBlock);
    baseWidth_             = SingleMetric<float>(toFloatVec(rawMetrics.UIntMetric(MetricFieldName::BASE_WIDTH)),
                                                 frameRate, framesPerBlock);

    if (rawMetrics.HasMetric(MetricFieldName::NUM_SANDWICHES))
        numSandwiches_         = SingleMetric<uint32_t>(rawMetrics.UIntMetric(MetricFieldName::NUM_SANDWICHES),
                                                        frameRate, framesPerBlock);
    if (rawMetrics.HasMetric(MetricFieldName::NUM_HALF_SANDWICHES))
        numHalfSandwiches_     = SingleMetric<uint32_t>(rawMetrics.UIntMetric(MetricFieldName::NUM_HALF_SANDWICHES),
                                                        frameRate, framesPerBlock);
    if (rawMetrics.HasMetric(MetricFieldName::NUM_PULSE_LABEL_STUTTERS))
        numPulseLabelStutters_ = SingleMetric<uint32_t>(rawMetrics.UIntMetric(MetricFieldName::NUM_PULSE_LABEL_STUTTERS),
                                                        frameRate, framesPerBlock);
    if (rawMetrics.HasMetric(MetricFieldName::PULSE_DETECTION_SCORE))
        pulseDetectionScore_   = SingleMetric<float>(rawMetrics.FloatMetric(MetricFieldName::PULSE_DETECTION_SCORE),
                                                     frameRate, framesPerBlock);
    if (rawMetrics.HasMetric(MetricFieldName::TRACE_AUTOCORR))
        traceAutocorr_         = SingleMetric<float>(rawMetrics.FloatMetric(MetricFieldName::TRACE_AUTOCORR),
                                                     frameRate, framesPerBlock);

    pixelChecksum_ = SingleMetric<int32_t>(rawMetrics.IntMetric(MetricFieldName::PIXEL_CHECKSUM),
                                           frameRate, framesPerBlock);
    dmeStats_      = SingleMetric<uint32_t>(rawMetrics.UIntMetric(MetricFieldName::DME_STATUS),
                                            frameRate, framesPerBlock);

    // pkzvar and bpzvar below can contain NaNs. We set those to 0 instead
    // as the HQRF downstream expects them to be 0 and not NaN.
    const auto replaceNaNWithZero = [](auto& data) {
        std::replace_if(data.begin(), data.end(), [](float v) { return std::isnan(v); }, 0);
    };

    AnalogMetricData<std::vector<float>> adata;

    if (rawMetrics.HasMetric(MetricFieldName::PKZVAR_A))
    {
        adata.A = rawMetrics.FloatMetric(MetricFieldName::PKZVAR_A);
        replaceNaNWithZero(adata.A);
        adata.C = rawMetrics.FloatMetric(MetricFieldName::PKZVAR_C);
        replaceNaNWithZero(adata.C);
        adata.G = rawMetrics.FloatMetric(MetricFieldName::PKZVAR_G);
        replaceNaNWithZero(adata.G);
        adata.T = rawMetrics.FloatMetric(MetricFieldName::PKZVAR_T);
        replaceNaNWithZero(adata.T);
        pkzvar_ = AnalogMetric<float>(std::move(adata), frameRate, framesPerBlock);
    }

    if (rawMetrics.HasMetric(MetricFieldName::BPZVAR_A))
    {
        adata.A = rawMetrics.FloatMetric(MetricFieldName::BPZVAR_A);
        replaceNaNWithZero(adata.A);
        adata.C = rawMetrics.FloatMetric(MetricFieldName::BPZVAR_C);
        replaceNaNWithZero(adata.C);
        adata.G = rawMetrics.FloatMetric(MetricFieldName::BPZVAR_G);
        replaceNaNWithZero(adata.G);
        adata.T = rawMetrics.FloatMetric(MetricFieldName::BPZVAR_T);
        replaceNaNWithZero(adata.T);
        bpzvar_ = AnalogMetric<float>(std::move(adata), frameRate, framesPerBlock);
    }

    if (rawMetrics.HasMetric(MetricFieldName::PKMAX_A))
    {
        adata.A = rawMetrics.FloatMetric(MetricFieldName::PKMAX_A);
        adata.C = rawMetrics.FloatMetric(MetricFieldName::PKMAX_C);
        adata.G = rawMetrics.FloatMetric(MetricFieldName::PKMAX_G);
        adata.T = rawMetrics.FloatMetric(MetricFieldName::PKMAX_T);
        pkMax_ = AnalogMetric<float>(std::move(adata), frameRate, framesPerBlock);
    }

    adata.A = rawMetrics.FloatMetric(MetricFieldName::PKMID_A);
    adata.C = rawMetrics.FloatMetric(MetricFieldName::PKMID_C);
    adata.G = rawMetrics.FloatMetric(MetricFieldName::PKMID_G);
    adata.T = rawMetrics.FloatMetric(MetricFieldName::PKMID_T);
    pkMid_ = AnalogMetric<float>(std::move(adata), frameRate, framesPerBlock);

    AnalogMetricData<std::vector<uint32_t>> bdata;
    bdata.A = rawMetrics.UIntMetric(MetricFieldName::PKMID_FRAMES_A);
    bdata.C = rawMetrics.UIntMetric(MetricFieldName::PKMID_FRAMES_C);
    bdata.G = rawMetrics.UIntMetric(MetricFieldName::PKMID_FRAMES_G);
    bdata.T = rawMetrics.UIntMetric(MetricFieldName::PKMID_FRAMES_T);
    pkMidFrames_ = AnalogMetric<uint32_t>(std::move(bdata), frameRate, framesPerBlock);

    bdata.A = rawMetrics.UIntMetric(MetricFieldName::NUM_BASES_A);
    bdata.C = rawMetrics.UIntMetric(MetricFieldName::NUM_BASES_C);
    bdata.G = rawMetrics.UIntMetric(MetricFieldName::NUM_BASES_G);
    bdata.T = rawMetrics.UIntMetric(MetricFieldName::NUM_BASES_T);
    std::vector<uint32_t> basesSum(bdata.size());
    for (size_t i = 0; i < basesSum.size(); ++i)
    {
        basesSum[i] = bdata.A[i] + bdata.G[i] + bdata.C[i] +bdata.T[i];
    }
    numBases_ = AnalogMetric<uint32_t>(std::move(bdata), frameRate, framesPerBlock);
    numBasesAll_ = SingleMetric<uint32_t>(std::move(basesSum), frameRate, framesPerBlock);

    const float relampA = relAmps[baseMap.find('A')];
    const float relampC = relAmps[baseMap.find('C')];
    const float relampG = relAmps[baseMap.find('G')];
    const float relampT = relAmps[baseMap.find('T')];
    const float minamp = std::min({relampA, relampC, relampG, relampT});

    // These differ depending on if we're Spider or Sequel.  For now at least,
    // we're just going to cram spider metrics into sequel layout, which is a
    // bit weird, but insulates downstream file formats (like stats.h5) from
    // the change (for now)
    FilterMetricData<std::vector<float>> fdata;

    if (rawMetrics.HasMetric(MetricFieldName::BASELINE_GREEN_MEAN))
    {
        fdata.red = rawMetrics.FloatMetric(MetricFieldName::BASELINE_RED_SD);
        fdata.green = rawMetrics.FloatMetric(MetricFieldName::BASELINE_GREEN_SD);
        baselineSD_ = FilterMetric<float>(std::move(fdata), frameRate, framesPerBlock);

        for (size_t i = 0; i < numBasesAll_.size(); ++i)
        {
            // We want to do a little work to avoid nans/divide by zero. We are
            // assuming that relamps aren't 0.
            float stdev = baselineSD_.data().red[i];
            float totbases = static_cast<float>(numBases_.data().A[i] + numBases_.data().C[i]);
            if (stdev != 0 && totbases != 0)
            {
                fdata.red.push_back(
                    pkMid_.data().A[i] / stdev * numBases_.data().A[i] / totbases * minamp / relampA +
                    pkMid_.data().C[i] / stdev * numBases_.data().C[i] / totbases * minamp / relampC);
            }
            else
            {
                fdata.red.push_back(0.0);
            }

            stdev = baselineSD_.data().green[i];
            totbases = static_cast<float>(numBases_.data().G[i] + numBases_.data().T[i]);
            if (stdev != 0 && totbases != 0)
            {
                fdata.green.push_back(
                    pkMid_.data().G[i] / stdev * numBases_.data().G[i] / totbases * minamp / relampG +
                    pkMid_.data().T[i] / stdev * numBases_.data().T[i] / totbases * minamp / relampT);
            }
            else
            {
                fdata.green.push_back(0.0);
            }
        }
        channelMinSNR_ = FilterMetric<float>(std::move(fdata), frameRate, framesPerBlock);

        fdata.red = rawMetrics.FloatMetric(MetricFieldName::BASELINE_RED_MEAN);
        fdata.green = rawMetrics.FloatMetric(MetricFieldName::BASELINE_GREEN_MEAN);
        baselineMean_ = FilterMetric<float>(std::move(fdata), frameRate, framesPerBlock);

        fdata.red = rawMetrics.FloatMetric(MetricFieldName::ANGLE_RED);
        fdata.green = rawMetrics.FloatMetric(MetricFieldName::ANGLE_GREEN);
        angle_ = FilterMetric<float>(std::move(fdata), frameRate, framesPerBlock);
    } else {
        fdata.red = fdata.green = rawMetrics.FloatMetric(MetricFieldName::BASELINE_SD);
        baselineSD_ = FilterMetric<float>(std::move(fdata), frameRate, framesPerBlock);

        if (rawMetrics.HasMetric(MetricFieldName::ACTIVITY_LABEL))
        {
            activityLabels_ = SingleMetric<uint32_t>(rawMetrics.UIntMetric(MetricFieldName::ACTIVITY_LABEL),
                                                     frameRate, framesPerBlock);
        }

        for (size_t i = 0; i < numBasesAll_.size(); ++i)
        {
            float rstdev = baselineSD_.data().red[i];
            float gstdev = baselineSD_.data().green[i];
            float totbases = static_cast<float>(numBasesAll_.data()[i]);
            if (rstdev != 0 && gstdev != 0 && totbases != 0)
            {
                fdata.red.push_back(
                    pkMid_.data().A[i] / rstdev * numBases_.data().A[i] / totbases * minamp / relampA +
                    pkMid_.data().C[i] / rstdev * numBases_.data().C[i] / totbases * minamp / relampC +
                    pkMid_.data().G[i] / gstdev * numBases_.data().G[i] / totbases * minamp / relampG +
                    pkMid_.data().T[i] / gstdev * numBases_.data().T[i] / totbases * minamp / relampT);
            }
            else
            {
                fdata.red.push_back(0.0);
            }
        }
        fdata.green = fdata.red;
        channelMinSNR_ = FilterMetric<float>(std::move(fdata), frameRate, framesPerBlock);

        fdata.red = fdata.green = rawMetrics.FloatMetric(MetricFieldName::BASELINE_MEAN);
        baselineMean_ = FilterMetric<float>(std::move(fdata), frameRate, framesPerBlock);

        fdata.red = fdata.green = std::vector<float>(expectedSize, -1.0f);
        angle_ = FilterMetric<float>(std::move(fdata), frameRate, framesPerBlock);
    }

    // Compute some derived metrics (not explicitly in baz file)
    //==============================================================================================

    {
        auto GetAnalogBaseRate = [](const std::vector<uint32_t>& numBases, float blockTime)
        {
            std::vector<float> baseRate;
            baseRate.reserve(numBases.size());
            for (const auto& bases : numBases)
                // Scale to pulses per minute
                baseRate.push_back((60 * static_cast<float>(bases)) / blockTime);
            return baseRate;
        };

        const float blockTime = numBases_.BlockTime();

        adata.A = GetAnalogBaseRate(numBases_.data().A, blockTime),
        adata.C = GetAnalogBaseRate(numBases_.data().C, blockTime),
        adata.G = GetAnalogBaseRate(numBases_.data().G, blockTime),
        adata.T = GetAnalogBaseRate(numBases_.data().T, blockTime);
        basesVsTime_ = AnalogMetric<float>(std::move(adata), frameRate, framesPerBlock);
    }

    if (!pkMax_.empty())
    {
        std::vector<float> minSnr;
        std::vector<float> lowSnr;
        std::vector<float> pkMaxNorms;
        minSnr.reserve(baselineSD_.size());
        lowSnr.reserve(baselineSD_.size());
        pkMaxNorms.reserve(baselineSD_.size());
        for (size_t i = 0; i < baselineSD_.size(); i++)
        {
            // If we somehow have a baselineSD value of 0 (which is possible with
            // some baseline modes) we want to produce a not particularly
            // meaningful SNR value, e.g. 0
            float rSD = baselineSD_.data().red[i];
            float gSD = baselineSD_.data().green[i];
            float snrA = 0;
            float snrC = 0;
            float snrG = 0;
            float snrT = 0;
            float maxSnrA = 0;
            float maxSnrC = 0;
            float maxSnrG = 0;
            float maxSnrT = 0;
            // TODO this hard code red/green might cause problems when going to spider...
            if (rSD != 0 && !std::isnan(rSD))
            {
                snrA = pkMid_.data().A[i] / rSD;
                snrC = pkMid_.data().C[i] / rSD;
                maxSnrA = (pkMax_.data().A[i] - pkMid_.data().A[i]) / rSD;
                maxSnrC = (pkMax_.data().C[i] - pkMid_.data().C[i]) / rSD;
            }
            if (gSD != 0 && !std::isnan(gSD))
            {
                snrG = pkMid_.data().G[i] / gSD;
                snrT = pkMid_.data().T[i] / gSD;
                maxSnrG = (pkMax_.data().G[i] - pkMid_.data().G[i]) / gSD;
                maxSnrT = (pkMax_.data().T[i] - pkMid_.data().T[i]) / gSD;
            }
            minSnr.push_back(std::min({snrA, snrC, snrG, snrT}));

            if (channelMinSNR_.data().green[i] > 0 && channelMinSNR_.data().red[i] > 0)
            {
                lowSnr.push_back(std::min(channelMinSNR_.data().green[i], channelMinSNR_.data().red[i]));
            }
            else
            {
                lowSnr.push_back(std::max(channelMinSNR_.data().green[i], channelMinSNR_.data().red[i]));
            }

            pkMaxNorms.push_back(std::max({maxSnrA, maxSnrC, maxSnrG, maxSnrT}));
        }

        blockMinSNR_ = SingleMetric<float>(std::move(minSnr), frameRate, framesPerBlock);
        blockLowSNR_ = SingleMetric<float>(std::move(lowSnr), frameRate, framesPerBlock);
        maxPkMaxNorms_ = SingleMetric<float>(std::move(pkMaxNorms), frameRate, framesPerBlock);
    }

    std::vector<float> pkzNorms;
    std::vector<float> bpzNorms;
    pkzNorms.reserve(pkzvar_.size());
    bpzNorms.reserve(pkzvar_.size());
    for (size_t i = 0; i < pkzvar_.size(); i++)
    {
        // G is consistently low amplitude, so we'll exclude it on the
        // hypothesis that low amplitude bases are noisier.
        pkzNorms.push_back((pkzvar_.data().A[i] + pkzvar_.data().C[i] + pkzvar_.data().T[i]) / 3.0f);
        bpzNorms.push_back((bpzvar_.data().A[i] + bpzvar_.data().C[i] + bpzvar_.data().T[i]) / 3.0f);
    }

    pkzVarNorms_ = SingleMetric<float>(std::move(pkzNorms), frameRate, framesPerBlock);
    bpzVarNorms_ = SingleMetric<float>(std::move(bpzNorms), frameRate, framesPerBlock);
}

MetricRegion BlockLevelMetrics::GetFullRegion() const
{
    return MetricRegion(0, numPulsesAll_.size());
}

MetricRegion BlockLevelMetrics::GetMetricRegion(const RegionLabel& region) const
{
    // Short circuit, to handle the default empty range that other code produces
    // when no HQ is found.
    if (region.pulseBegin == 0 && region.pulseEnd == 0)
        return MetricRegion(0, 0);

    // Check if we've already done this particular computation.  Ideally this
    // will go away in the near future, when all the HQ metrics/regions/etc are
    // computed together and this function is only called some very small number
    // of times.
    if (cachedMap_.first.first == region.pulseBegin && cachedMap_.first.second == region.pulseEnd)
    {
        assert(mapSet_);
        return cachedMap_.second;
    }

    assert(!mapSet_);
    mapSet_ = true;
    cachedMap_.first.first = region.pulseBegin;
    cachedMap_.first.second = region.pulseEnd;

    assert(region.pulseBegin < region.pulseEnd);
    // If we're not in internal mode, the stored `pulse` indexes are really base indexes,
    // so we need to use the appropriate metric data
    const auto& eventData = internal_ ? numPulsesAll_.data() : numBasesAll_.data();

    auto FindMetricBlock = [](const std::vector<uint32_t>& data, size_t eventIdx) -> size_t
    {
        size_t accum = 0;
        // lastActive is for special handling where the end pulse isn't actually
        // contained in the metrics.  If this is the case the HQ region ends
        // not at the end of the metric data, but at the last metric block to
        // have any pulse activity
        int lastActive = -1;
        for (size_t i = 0; i < data.size(); ++i)
        {
            if (data[i] == 0) continue;
            lastActive = i;

            if (accum <= eventIdx && accum + data[i] > eventIdx)
                return i;
            accum += data[i];
        }
        assert(lastActive >= -1);
        return lastActive + 1;
    };

    size_t startBlock = FindMetricBlock(eventData, region.pulseBegin);
    size_t endBlock = FindMetricBlock(eventData, region.pulseEnd);

    MetricRegion ret(startBlock, endBlock);
    cachedMap_.second = ret;
    return ret;
}

}}
