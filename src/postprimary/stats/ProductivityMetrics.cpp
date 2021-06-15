// Copyright (c) 2014-2018, Pacific Biosciences of California, Inc.
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
#include <vector>

#include <pacbio/logging/Logger.h>

#include "ZmwMetrics.h"
#include "ProductivityMetrics.h"


using namespace PacBio::Primary;

namespace PacBio {
namespace Primary {
namespace Postprimary {

static constexpr float sticksPerMin = 30;
static constexpr int polymeraseLengthCut = 50;
static constexpr double accuracyCut = 0.8;

ProductivityInfo ProductivityMetrics::ComputeProductivityInfo(
        const size_t zmwNum,
        const RegionLabel& hqRegion,
        const BlockLevelMetrics& metrics) const
{
    return ComputeProductivityInfo(hqRegion, metrics,
                                   IsSequencingZmw(zmwNum));
}

ProductivityInfo ProductivityMetrics::ComputeProductivityInfo(
        const RegionLabel& hqRegion,
        const BlockLevelMetrics& metrics,
        bool isSequencing) const
{
    ProductivityInfo ret;
    ret.isSequencing = isSequencing;

    if (!ret.isSequencing)
    {
        ret.readAccuracy = 0.0f;
        ret.productivity = ProductivityClass::UNDEFINED;
        ret.loading = LoadingClass::UNDEFINED;
        ret.readType = ReadTypeClass::UNDEFINED;
    }
    else
    {
        // HQRegion SNR
        auto region = metrics.GetMetricRegion(hqRegion);
        auto signalMetrics = SignalMetrics(region, metrics);
        ret.snr = signalMetrics.Snr();

        // Compute rq
        double minSnr = std::numeric_limits<float>::quiet_NaN();
        const auto& snr = *ret.snr;
        minSnr = std::min(snr.A,
                          std::min(snr.C,
                                   std::min(snr.G,
                                            snr.T)));
        ret.readAccuracy = minSnr >= minSnrCut_ ? 0.8f : 0.0f;

        // Productivity
        ret.productivity = Productivity(hqRegion, metrics, ret.readAccuracy);
        ret.readType = ReadType(hqRegion, metrics);
        ret.loading = Loading(ret.readType);
        if (ret.productivity == ProductivityClass::OTHER)
        {
            if (hqRegion.Length() > 0)
            {
                if (!std::isnan(minSnr))
                {
                    // Had an HQ-region but was filtered out.
                    if (minSnr < minSnrCut_) ret.failedSNRCut = true;
                    if (minSnr < 0) ret.failedDME = true;
                }
            }
        }
    }
    return ret;
}

bool ProductivityMetrics::IsSequencingZmw(size_t zmwNum)
{
    static bool warnOnce = [](){PBLOG_WARN << "Marking all ZMWs as sequencing, Kestrel does not yet have a replacement for Sequel layout"; return true;}();
    (void)warnOnce;
    return true;
}

bool ProductivityMetrics::IsP1(const RegionLabel& hqRegion, float accuracy)
{
    const bool passHQCut = hqRegion.Length() > polymeraseLengthCut;
    if (!passHQCut) return false;

    const bool passRQCut = accuracy >= accuracyCut;
    if (!passRQCut) return false;

    return true;
}

ProductivityClass ProductivityMetrics::Productivity(const RegionLabel& hqRegion, const BlockLevelMetrics& metrics, float accuracy) const
{
    if (IsP1(hqRegion, accuracy))
        return ProductivityClass::PRODUCTIVE;

    if (IsEmpty(metrics.BasesVsTime()))
        return ProductivityClass::EMPTY;

    return ProductivityClass::OTHER;
}

LoadingClass ProductivityMetrics::Loading(ReadTypeClass readType) const
{
    switch(readType) {
        // Empty
        case ReadTypeClass::EMPTY:
            return LoadingClass::EMPTY;

        // Single
        case ReadTypeClass::FULLHQREAD0:
        case ReadTypeClass::FULLHQREAD1:
        case ReadTypeClass::PARTIALHQREAD1:
        case ReadTypeClass::PARTIALHQREAD2:
            return LoadingClass::SINGLE;

        // Multi
        case ReadTypeClass::PARTIALHQREAD0:
            return LoadingClass::MULTI;

        // Indeterminate
        case ReadTypeClass::INDETERMINATE:
        default:
            return LoadingClass::INDETERMINATE;
    }
}

ReadTypeClass ProductivityMetrics::ReadType(const RegionLabel& hqRegion, const BlockLevelMetrics& metrics) const
{
    // Compute per-analog channel pulse rates.
    const auto& baseRates = metrics.BasesVsTime();

    // EMPTY
    if (IsEmpty(baseRates))
        return ReadTypeClass::EMPTY;

    // INDETERMINATE
    if (hqRegion.Length() <= polymeraseLengthCut)
        return ReadTypeClass::INDETERMINATE;

    const auto& metricRegion = metrics.GetMetricRegion(hqRegion);

    const auto& baseRatesBeforeHQ = baseRates.GetRegionBefore(metricRegion);
    const auto& baseRatesAfterHQ =  baseRates.GetRegionAfter(metricRegion);

    bool sequencingBeforeHQRegion = (baseRatesBeforeHQ.size() > 0) &&
        IsSequencingInRegion(baseRatesBeforeHQ);
    bool sequencingAfterHQRegion = (baseRatesAfterHQ.size() > 0) &&
        IsSequencingInRegion(baseRatesAfterHQ);

    // Determine read type using start of HQ-region and what occurs
    // before and after HQ-region.

    const int mFMBSingleLoadMaxDelay = 1;

    // ================================================================
    // Early-starter: determined by start of HQ-region.
    // Currently set to the HQ-region occurring in the first MF block.
    if (baseRatesBeforeHQ.size() < mFMBSingleLoadMaxDelay)
    {
        if (!sequencingAfterHQRegion || (baseRatesAfterHQ.size() == 0))
        {
            // Normal (early-starter) HQ-region with no sequencing after HQ-region.
            return ReadTypeClass::FULLHQREAD0;
        }

        // Early-starter HQ-region but with sequencing after HQ-region.
        return ReadTypeClass::PARTIALHQREAD1;
    }

    // ================================================================
    // Late-starter: expected (usually) to be multi-load
    //
    // spec oddness: (hqRegionMFBegin == 0) falls into the above branch
    // with mFMBSingleLoadMaxDelay > 0. Therefore the first part of the
    // following condition will generally be false unless something drastic
    // changes. We can update the spec and remove this safely.
    if ((baseRatesBeforeHQ.size() == 0) || sequencingBeforeHQRegion)
    {
        // Late-starter that is sequencing before the HQ-region
        return ReadTypeClass::PARTIALHQREAD0;
    }

    // Clean this up, pulsesAfter should know it's own size
    if (!sequencingAfterHQRegion || (baseRatesAfterHQ.size() == 0))
    {
        // Late-starter but full HQ read.
        return ReadTypeClass::FULLHQREAD1;
    }

    // Late-starter HQ-region but with sequencing after HQ-region.
    return ReadTypeClass::PARTIALHQREAD2;

}

bool ProductivityMetrics::IsEmpty(const AnalogMetric<float>& baseRates) const
{
    // Convert minEmptyTime in minutes to a number of blocks
    // We want a minEmptyTime of zero to translate to checking the whole
    // read, instead of checking the first 0 minutes
    size_t nblocks = baseRates.data().size();
    float minutes_per_block = baseRates.BlockTime() / 60.0f;
    if (minEmptyTime_ > 0)
    {
        // We don't want to check beyond the length of the movie:
        nblocks = std::min(static_cast<size_t>(ceil(minEmptyTime_/minutes_per_block)), nblocks);
    }

    // emptyOutlierTime_ of 0 translates to throw no blocks out
    // Translate the emptyOutlierTime_ to number of blocks to throw out.
    size_t nthBlock = 0;
    if (emptyOutlierTime_ > 0)
    {
        nthBlock = static_cast<size_t>(round(emptyOutlierTime_/minutes_per_block));
        nthBlock = std::min(nthBlock, nblocks-1);
    }

    return !IsSequencingInRegion(baseRates.GetRegion(0, nblocks), nthBlock);
}

bool ProductivityMetrics::IsSequencingInRegion(const AnalogMetric<float>::View_T& baseRatesByAnalog,
                                 size_t nOutlier)
{
    if (baseRatesByAnalog.size() == 0) return false;

    // mfEnd Needs to be exclusive! By default the baseToBlock maps are
    // inclusive!
    const double channelPpmEmptyCut = 1.5 * (sticksPerMin / 4);

    if (nOutlier >= baseRatesByAnalog.size())
    {
        PBLOG_ERROR << "nOutlier too large "
        << ": size=" << baseRatesByAnalog.size()
        << ", nOutlier=" << nOutlier;
        return false;
    }

    std::vector<double> minRates;
    minRates.reserve(baseRatesByAnalog.size());
    for (size_t i = 0; i < baseRatesByAnalog.size(); ++i)
    {
        minRates.push_back(std::min(std::min(baseRatesByAnalog.A[i],
                                             baseRatesByAnalog.C[i]),
                                    std::min(baseRatesByAnalog.G[i],
                                             baseRatesByAnalog.T[i])));
    }

    auto it = minRates.begin() + (minRates.size() - nOutlier - 1);
    std::nth_element(minRates.begin(), it, minRates.end());

    if (it != minRates.end())
    {
        return (*it >= channelPpmEmptyCut);
    }

    return false;
}

}}}
