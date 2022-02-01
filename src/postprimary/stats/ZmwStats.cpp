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


#include <pacbio/primary/RTMetricsConfig.h>
#include <postprimary/insertfinder/InsertState.h>

#include "ZmwStats.h"

namespace PacBio {
namespace Primary {
namespace Postprimary {

void ZmwStats::FillPerZmwStats(const Platform& platform,
                               const FileHeader& fh,
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

    // ===============================================================================
    // Time series data

    // Extracts a time series for a given metric. The data may live
    // in the low, mid or high frequency data structures.  By default,
    // the metric is subsampled to the low frequency rate that is stored in
    // the zmw stats file. If specified, factorDown is used to downsample
    // to a rate other than low frequency.
    //
    // The expected result size is based on an error-prone estimate from
    // the baz header, so that estimate is provided (newSize) and no longer
    // guessed from the actual size (data.size())

    const auto ExtractTimeSeries = [&fh] (
        const auto& data,
        MetricFrequency frequency,
        boost::multi_array<float, 1>::array_view<1>::type toFill,
        float defVal = std::numeric_limits<float>::quiet_NaN(),
        bool removeZeros = true,
        float scale = 1.0,
        size_t factorDown = 0)
    {
        // Data starts at the existing BAZ metric frequency then by default
        // we scale it back down to LF for the sts.h5. If you want to
        // change that, you can manually specify how it is downsampled by
        // supplying a factorDown arg other than 0. e.g. factorDown = 1
        // preserves the BAZ metric frequency in the sts.h5.
        if (factorDown == 0)
        {
            if (frequency == MetricFrequency::LOW)
            {
                factorDown = 1;
            }
            else if (frequency == MetricFrequency::MEDIUM)
            {
                assert(fh.HFbyLFRatio() % fh.HFbyMFRatio() == 0);
                factorDown = fh.HFbyLFRatio() / fh.HFbyMFRatio();
            }
            else
            {
                assert(frequency == MetricFrequency::HIGH);
                factorDown = fh.HFbyLFRatio();
            }
        }

        // If the data does not evently fit into an integral number of
        // low frequency bins, we neglect the extra bits.
        size_t newSize = std::min(toFill.size(), data.size() / factorDown);

        for (size_t i = 0; i < newSize; ++i)
        {
            float tally = 0;
            size_t nonZero = 0;
            // We don't want to run past the end of the higher frequency
            // metric array
            size_t currentFactor = std::min(data.size() - (factorDown * i),
                                            factorDown);
            for (size_t j = 0; j < currentFactor; ++j)
            {
                if (removeZeros && data[currentFactor * i + j] != 0)
                {
                    tally += data[currentFactor * i + j];
                    nonZero++;
                }
                else
                {
                    tally += data[currentFactor * i + j];
                }
            }

            if (removeZeros && nonZero == 0)
            {
                toFill[i] = defVal;
            }
            else
            {
                if (removeZeros && nonZero > 0)
                    tally /= nonZero;
                else
                    tally /= currentFactor;

                tally *= scale;
                toFill[i] = tally;
            }
        }
    };

    typedef boost::multi_array_types::index_range range;

    if (addDiagnostics)
    {
        // Note: VsMFTraceAutoCorr is over the full trace, not just the HQ-region.
        ExtractTimeSeries(blockMetrics.TraceAutocorr().data(),
                          blockMetrics.TraceAutocorr().Frequency(),
                          zmwStats.VsMFTraceAutoCorr[boost::indices[range()]], 0, false, 1.0, 1);

        const auto& baselineSD = blockMetrics.BaselineSD();
        ExtractTimeSeries(baselineSD.data().green,
                          baselineSD.Frequency(),
                          zmwStats.VsMFBaselineStd[boost::indices[range()][Color::GREEN]],
                          -1, true, 1.0, 1);
        ExtractTimeSeries(baselineSD.data().red,
                          baselineSD.Frequency(),
                          zmwStats.VsMFBaselineStd[boost::indices[range()][Color::RED]],
                          -1, true, 1.0, 1);

        const auto& baselineMean = blockMetrics.BaselineMean();
        ExtractTimeSeries(baselineMean.data().green,
                          baselineMean.Frequency(),
                          zmwStats.VsMFBaselineLevel[boost::indices[range()][Color::GREEN]],
                          std::numeric_limits<float>::quiet_NaN(), true, 1.0, 1);
        ExtractTimeSeries(baselineMean.data().red,
                          baselineMean.Frequency(),
                          zmwStats.VsMFBaselineLevel[boost::indices[range()][Color::RED]],
                          std::numeric_limits<float>::quiet_NaN(), true, 1.0, 1);

        ExtractTimeSeries(blockMetrics.NumBasesAll().data(),
                          blockMetrics.NumBasesAll().Frequency(),
                          zmwStats.VsMFBaseRate[boost::indices[range()]],
                          0, false, fh.FrameRateHz() / fh.MFMetricFrames(), 1);

        ExtractTimeSeries(blockMetrics.NumPulsesAll().data(),
                          blockMetrics.NumPulsesAll().Frequency(),
                          zmwStats.VsMFPulseRate[boost::indices[range()]],
                          0, false, fh.FrameRateHz() / fh.MFMetricFrames(), 1);

        const auto& pkmid = blockMetrics.PkMid();
        const RTMetricsConfig rtConfig;
        for (size_t i = 0; i < zmwStats.nMF_; ++i)
        {
            zmwStats.VsMFPulseWidth[i] = static_cast<float>(
                    blockMetrics.PulseWidth().data()[i]
                    / (fh.FrameRateHz() * blockMetrics.NumPulsesAll().data()[i]));

            zmwStats.VsMFBaseWidth[i] = static_cast<float>(
                    blockMetrics.BaseWidth().data()[i]
                    / (fh.FrameRateHz() * blockMetrics.NumBasesAll().data()[i]));

            zmwStats.VsMFPkmid[i][Analog::A] = std::numeric_limits<float>::quiet_NaN();
            zmwStats.VsMFPkmid[i][Analog::C] = std::numeric_limits<float>::quiet_NaN();
            zmwStats.VsMFPkmid[i][Analog::G] = std::numeric_limits<float>::quiet_NaN();
            zmwStats.VsMFPkmid[i][Analog::T] = std::numeric_limits<float>::quiet_NaN();
            zmwStats.VsMFSnrMean[i][Analog::A] = std::numeric_limits<float>::quiet_NaN();
            zmwStats.VsMFSnrMean[i][Analog::C] = std::numeric_limits<float>::quiet_NaN();
            zmwStats.VsMFSnrMean[i][Analog::G] = std::numeric_limits<float>::quiet_NaN();
            zmwStats.VsMFSnrMean[i][Analog::T] = std::numeric_limits<float>::quiet_NaN();
            if (blockMetrics.PkMidFrames().data().A[i] > rtConfig.minBaselineFrames)
            {
                zmwStats.VsMFPkmid[i][Analog::A] = pkmid.data().A[i];
                if (zmwStats.VsMFBaselineStd[i][Color::RED] > 0)
                {
                    zmwStats.VsMFSnrMean[i][Analog::A] = zmwStats.VsMFPkmid[i][Analog::A]
                                                         / zmwStats.VsMFBaselineStd[i][Color::RED];
                }
            }
            if (blockMetrics.PkMidFrames().data().C[i] > rtConfig.minBaselineFrames)
            {
                zmwStats.VsMFPkmid[i][Analog::C] = pkmid.data().C[i];
                if (zmwStats.VsMFBaselineStd[i][Color::RED] > 0)
                {
                    zmwStats.VsMFSnrMean[i][Analog::C] = zmwStats.VsMFPkmid[i][Analog::C]
                                                         / zmwStats.VsMFBaselineStd[i][Color::RED];
                }
            }
            if (blockMetrics.PkMidFrames().data().G[i] > rtConfig.minBaselineFrames)
            {
                zmwStats.VsMFPkmid[i][Analog::G] = pkmid.data().G[i];
                if (zmwStats.VsMFBaselineStd[i][Color::GREEN] > 0)
                {
                    zmwStats.VsMFSnrMean[i][Analog::G] = zmwStats.VsMFPkmid[i][Analog::G]
                                                         / zmwStats.VsMFBaselineStd[i][Color::GREEN];
                }
            }
            if (blockMetrics.PkMidFrames().data().T[i] > rtConfig.minBaselineFrames)
            {
                zmwStats.VsMFPkmid[i][Analog::T] = pkmid.data().T[i];
                if (zmwStats.VsMFBaselineStd[i][Color::GREEN] > 0)
                {
                    zmwStats.VsMFSnrMean[i][Analog::T] = zmwStats.VsMFPkmid[i][Analog::T]
                                                         / zmwStats.VsMFBaselineStd[i][Color::GREEN];
                }
            }
        }

        ExtractTimeSeries(
                blockMetrics.NumFrames().data(),
                blockMetrics.NumFrames().Frequency(),
                zmwStats.VsMFNumFrames[boost::indices[range()]],
                -1, true, 1.0, 1);
    }
}

}}}

