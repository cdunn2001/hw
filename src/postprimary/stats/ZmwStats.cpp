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


#include <postprimary/insertfinder/InsertState.h>

#include "ZmwStats.h"

namespace PacBio {
namespace Primary {
namespace Postprimary {

void ZmwStats::FillPerZmwStats(const Platform& platform,
                               const RegionLabel& hqRegion,
                               const ZmwMetrics& zmwMetrics,
                               const EventData& events,
                               const BlockLevelMetrics& blockMetrics,
                               const bool control,
                               const bool addDiagnostics,
                               PacBio::Primary::ZmwStats& zmwStats)
{
    // Enums to avoid typing magic numbers for indices
    enum Analog
    {
        A, C, G, T
    };

    static bool warnOnce = [](){PBLOG_WARN << "Hardcoding platform to SequelII and setting GREEN = 0, RED = 0"; return true;}();
    (void)warnOnce;
    enum Color
    {
        GREEN = 0, RED = 0
    };

    // Unconstrained auto isn't ideal here, but the dest parameter
    // is a boost type with a longer name, and the second parameter
    // really is somewhat generic.  Unfortunately can't specify
    // something like AnalogMetricData<T> to partially constrain it.
    auto FillAnalog = [&](auto& dest, const auto& analog)
    {
        assert(dest.size() == 4);
        dest[Analog::A] = analog.A;
        dest[Analog::C] = analog.C;
        dest[Analog::G] = analog.G;
        dest[Analog::T] = analog.T;
    };
    auto FillFilter = [&](auto& dest, const auto& filter)
    {
        static bool warnOnce = [](){PBLOG_WARN << "Hardcoding platform to SequelII for FillFilter"; return true;}();
        (void)warnOnce;
        dest[Color::GREEN] = filter.green;
    };

    const auto& excludedPulseMetrics = zmwMetrics.ZmwExcludedPulseMetrics();
    const auto& readMetrics = zmwMetrics.ZmwReadMetrics();
    const auto& zmwSignalMetrics = zmwMetrics.ZmwSignalMetrics();

    // ===============================================================================
    // Metrics that are always used

    zmwStats.Productivity = static_cast<uint8_t>(zmwMetrics.ZmwProdMetrics().productivity);
    zmwStats.Loading = static_cast<uint8_t>(zmwMetrics.ZmwProdMetrics().loading);

    zmwStats.HoleNumber = events.ZmwNumber();
    static bool warned = [](){PBLOG_WARN << "ZmwStats not setting hole type or holexy"; return true; }();
    (void)warned;
    //Primary::UnitCell uc(events.ZmwNumber());
    zmwStats.HoleXY[0] = 0;//uc.x;
    zmwStats.HoleXY[1] = 0;//uc.y;
    zmwStats.HoleType = 0;//chipLayout.UnitCellTypeId(static_cast<uint16_t>(uc.x), static_cast<uint16_t>(uc.y));
    zmwStats.UnitFeature = readMetrics.UnitFeatures();

    zmwStats.ReadType = static_cast<uint8_t>(zmwMetrics.ZmwProdMetrics().readType);
    zmwStats.ReadScore = zmwMetrics.ZmwProdMetrics().readAccuracy;
    zmwStats.ReadLength = static_cast<uint32_t>(readMetrics.ReadLength());

    zmwStats.NumBases = events.NumBases();

    FillAnalog(zmwStats.SnrMean, zmwSignalMetrics.Snr());

    FillFilter(zmwStats.BaselineLevel, zmwSignalMetrics.Baseline());
    FillFilter(zmwStats.BaselineStd, zmwSignalMetrics.BaselineSD());

    zmwStats.InsertionCounts[static_cast<int>(InsertState::BASE)] =
        excludedPulseMetrics.InsertCounts().base;
    zmwStats.InsertionCounts[static_cast<int>(InsertState::EX_SHORT_PULSE)] =
        excludedPulseMetrics.InsertCounts().exShortPulse;
    zmwStats.InsertionCounts[static_cast<int>(InsertState::BURST_PULSE)] =
        excludedPulseMetrics.InsertCounts().burstPulse;
    zmwStats.InsertionCounts[static_cast<int>(InsertState::PAUSE_PULSE)] =
        excludedPulseMetrics.InsertCounts().pausePulse;

    zmwStats.InsertionLengths[static_cast<int>(InsertState::BASE)] =
        excludedPulseMetrics.InsertLengths().base;
    zmwStats.InsertionLengths[static_cast<int>(InsertState::EX_SHORT_PULSE)] =
        excludedPulseMetrics.InsertLengths().exShortPulse;
    zmwStats.InsertionLengths[static_cast<int>(InsertState::BURST_PULSE)] =
        excludedPulseMetrics.InsertLengths().burstPulse;
    zmwStats.InsertionLengths[static_cast<int>(InsertState::PAUSE_PULSE)] =
        excludedPulseMetrics.InsertLengths().pausePulse;

    zmwStats.LoopOnly = !zmwMetrics.ZmwAdapterMetrics().hasStem;

    zmwStats.IsControl = static_cast<uint8_t>(control);
    if (zmwMetrics.ZmwControlMetrics().isControl)
    {
        zmwStats.ControlReadLength = zmwMetrics.ZmwControlMetrics().controlReadLength;
        zmwStats.ControlReadAccuracy = zmwMetrics.ZmwControlMetrics().controlReadScore;
    }

    // ===============================================================================
    // Statistics that require an HQ region
    if (hqRegion.Length() > 0)
    {
        const auto& hqSignalMetrics = *zmwMetrics.ZmwHqSignalMetrics();

        // HQ-region metrics.
        zmwStats.HQRegionStart = hqRegion.begin;
        zmwStats.HQRegionEnd = hqRegion.end;
        zmwStats.HQRegionStartTime = blockMetrics.StartTime(hqRegion);
        zmwStats.HQRegionEndTime = blockMetrics.StopTime(hqRegion);

        FillAnalog(zmwStats.HQRegionSnrMean, hqSignalMetrics.Snr());
        FillAnalog(zmwStats.HQPkmid, hqSignalMetrics.PkMid());
        
        FillFilter(zmwStats.HQChannelMinSnr, hqSignalMetrics.MinSnr());
        FillFilter(zmwStats.HQBaselineLevel, hqSignalMetrics.Baseline());
        FillFilter(zmwStats.HQBaselineStd, hqSignalMetrics.BaselineSD());

        const auto& baseMetrics = *zmwMetrics.ZmwBaseMetrics();
        const auto& insertMetrics = *zmwMetrics.ZmwSubreadMetrics();
        const auto& pulseMetrics = *zmwMetrics.ZmwPulseMetrics();

        zmwStats.HQRegionScore = zmwMetrics.ZmwProdMetrics().readAccuracy;

        zmwStats.BaseRate      = baseMetrics.Rate();
        zmwStats.BaseWidth     = baseMetrics.Width();
        zmwStats.BaseIpd       = baseMetrics.Ipd();
        zmwStats.LocalBaseRate = baseMetrics.LocalRate();
        zmwStats.Pausiness     = baseMetrics.Pausiness();

        zmwStats.NumPulses  = pulseMetrics.TotalCount();
        zmwStats.PulseRate  = pulseMetrics.Rate();
        zmwStats.PulseWidth = pulseMetrics.Width();

        FillAnalog(zmwStats.DyeAngle, hqSignalMetrics.Angle());

        if (insertMetrics.MedianLength() > 0)
            zmwStats.MedianInsertLength = static_cast<uint32_t>(insertMetrics.MedianLength());
        if (insertMetrics.MaxSubreadLength() > 0)
            zmwStats.InsertReadLength = static_cast<uint32_t>(insertMetrics.MaxSubreadLength());

        const auto& count = baseMetrics.Counts();
        auto total_count = static_cast<float>(count.A + count.C + count.G + count.T);
        if (total_count > 0)
        {
            zmwStats.BaseFraction[Analog::A] = count.A / total_count;
            zmwStats.BaseFraction[Analog::C] = count.C / total_count;
            zmwStats.BaseFraction[Analog::G] = count.G / total_count;
            zmwStats.BaseFraction[Analog::T] = count.T / total_count;
        }
    }
}

}}}

