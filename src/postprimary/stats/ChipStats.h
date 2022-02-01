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

#pragma once

#include <iostream>
#include <memory>
#include <map>
#include <string>

#include <pacbio/logging/Logger.h>
#include <bazio/MetricData.h>

#include "ZmwMetrics.h"

namespace PacBio {
namespace Primary {
namespace Postprimary {

using namespace PacBio::Primary;

struct ChipMetricsCollection
{
    using SingleM = std::vector<float>;
    using FilterM = FilterMetricData<std::vector<float>>;
    using AnalogM = AnalogMetricData<std::vector<float>>;

    // Per Analog metrics
    AnalogM pkmidHQ;
    AnalogM angleHQ;
    AnalogM snr;
    AnalogM snrHQ;
    AnalogM baseCount;
    AnalogM analogBaselineSD;
    AnalogM analogBaseline;

    // Per Filter metrics
    FilterM baselineSequencing;
    FilterM baselineScatteringMetrology;

    // Single value metrics
    SingleM pulseWidth;
    SingleM baseWidth;

    SingleM sequencingReadLength;
    SingleM sequencingInsertLength;
    SingleM sequencingUmy;
    SingleM polyLength;
    SingleM controlReadLength;
    SingleM readScore;
    SingleM controlReadScore;
    SingleM sequencingReadScore;
    SingleM sequencingInsertReadScore;
    SingleM pulseRate;
    SingleM baseRate;
    SingleM localBaseRate;
    SingleM baseIPD;
    SingleM sequencingHQRatio;
    SingleM isSequencing;
    SingleM failedSNRCut;
    SingleM failedDME;
    SingleM internal;

    SingleM sequencingMedianInsert;
    SingleM productivity;
    SingleM loading;
    SingleM readType;
    SingleM isRead;
    SingleM pausiness;

    ChipMetricsCollection(size_t numZmw)
    {
        controlReadLength.reserve(numZmw);
        controlReadScore.reserve(numZmw);
        productivity.reserve(numZmw);
        loading.reserve(numZmw);
        readType.reserve(numZmw);
        isRead.reserve(numZmw);
        readScore.reserve(numZmw);
        pulseRate.reserve(numZmw);
        pulseWidth.reserve(numZmw);
        baseRate.reserve(numZmw);
        baseWidth.reserve(numZmw);
        baseIPD.reserve(numZmw);
        localBaseRate.reserve(numZmw);
        polyLength.reserve(numZmw);
        isSequencing.reserve(numZmw);
        failedSNRCut.reserve(numZmw);
        failedDME.reserve(numZmw);
        internal.reserve(numZmw);
        sequencingReadLength.reserve(numZmw);
        sequencingReadScore.reserve(numZmw);
        sequencingInsertLength.reserve(numZmw);
        sequencingInsertReadScore.reserve(numZmw);
        sequencingMedianInsert.reserve(numZmw);
        sequencingHQRatio.reserve(numZmw);
        pausiness.reserve(numZmw);
        sequencingUmy.reserve(numZmw);

        ReserveAnalogs(angleHQ, numZmw);
        ReserveAnalogs(baseCount, numZmw);
        ReserveAnalogs(snr, numZmw);
        ReserveAnalogs(snrHQ, numZmw);
        ReserveAnalogs(pkmidHQ, numZmw);
        ReserveAnalogs(analogBaseline, numZmw);
        ReserveAnalogs(analogBaselineSD, numZmw);
        ReserveFilters(baselineSequencing, numZmw);
        ReserveFilters(baselineScatteringMetrology, numZmw);
    }

    void ReserveAnalogs(AnalogMetricData<std::vector<float>>& amd, size_t numZmw)
    {
        amd.A.reserve(numZmw);
        amd.C.reserve(numZmw);
        amd.G.reserve(numZmw);
        amd.T.reserve(numZmw);
    }
    void ReserveFilters(FilterMetricData<std::vector<float>>& fmd, size_t numZmw)
    {
        fmd.green.reserve(numZmw);
        fmd.red.reserve(numZmw);
    }

};


class ChipStats
{
public: // structors
    // Default constructor
    ChipStats(size_t numZmw) : data_(numZmw) {}
    // Move constructor
    ChipStats(ChipStats&&) = delete;
    // Copy constructor
    ChipStats(const ChipStats&) = delete;
    // Move assignment operator
    ChipStats& operator=(ChipStats&& rhs) noexcept = delete;
    // Copy assignment operator
    ChipStats& operator=(const ChipStats&) = delete;

public:
    void AddData(const ZmwMetrics& metrics)
    {
        data_.productivity.push_back(static_cast<uint32_t>(metrics.ZmwProdMetrics().productivity));
        data_.loading.push_back(static_cast<uint32_t>(metrics.ZmwProdMetrics().loading));
        data_.readType.push_back(static_cast<uint32_t>(metrics.ZmwProdMetrics().readType));

        const auto& readMetrics = metrics.ZmwReadMetrics();
        const auto& zmwSignalMetrics = metrics.ZmwSignalMetrics();
        const auto& prodMetrics = metrics.ZmwProdMetrics();

        if (!metrics.ZmwControlMetrics().isControl)
        {
            data_.internal.push_back(readMetrics.Internal());;
            data_.isRead.push_back(readMetrics.IsRead());
            data_.isSequencing.push_back(prodMetrics.isSequencing);;
            data_.failedSNRCut.push_back(prodMetrics.failedSNRCut);;
            data_.failedDME.push_back(prodMetrics.failedDME);;

            if (prodMetrics.isSequencing)
            {
                AddFilters(data_.baselineSequencing, zmwSignalMetrics.Baseline());
            }
            static bool warnOnce = [](){PBLOG_WARN << "Scaterring Metrology disabled in ChipStats, Kestrel does not yet have a replacement for Sequel layout"; return true;}();
            (void)warnOnce;
            //Primary::UnitCell uc(readMetrics.HoleNumber());
            //if (layout.IsScatteringMetrology(static_cast<uint16_t>(uc.x), static_cast<uint16_t>(uc.y)))
            //{
            //    AddFilters(data_.baselineScatteringMetrology,  zmwSignalMetrics.Baseline());
            //}

            if (prodMetrics.productivity == ProductivityClass::PRODUCTIVE)
            {
                const auto& pulseMetrics = *metrics.ZmwPulseMetrics();
                const auto& baseMetrics = *metrics.ZmwBaseMetrics();
                const auto& hqSignalMetrics = *metrics.ZmwHqSignalMetrics();
                const auto& insertMetrics = *metrics.ZmwSubreadMetrics();
                
                AddAnalogs(data_.baseCount, baseMetrics.Counts());
                data_.baseRate.push_back(baseMetrics.Rate());
                data_.baseWidth.push_back(baseMetrics.Width());
                data_.baseIPD.push_back(baseMetrics.Ipd());
                data_.localBaseRate.push_back(baseMetrics.LocalRate());

                AddAnalogs(data_.angleHQ, hqSignalMetrics.Angle());
                AddAnalogs(data_.analogBaseline, AnalogMetricData<float>(hqSignalMetrics.Baseline()));
                AddAnalogs(data_.analogBaselineSD, AnalogMetricData<float>(hqSignalMetrics.BaselineSD()));
                AddAnalogs(data_.snrHQ, hqSignalMetrics.Snr());
                AddAnalogs(data_.pkmidHQ, hqSignalMetrics.PkMid());
                AddAnalogs(data_.snr, zmwSignalMetrics.Snr());

                data_.sequencingHQRatio.push_back(baseMetrics.HQRatio());
                data_.pausiness.push_back(baseMetrics.Pausiness());

                data_.polyLength.push_back(readMetrics.PolyLength());
                data_.sequencingReadLength.push_back(readMetrics.ReadLength());

                data_.pulseRate.push_back(pulseMetrics.Rate());
                data_.pulseWidth.push_back(pulseMetrics.Width());

                data_.sequencingReadScore.push_back(prodMetrics.readAccuracy);

                if (insertMetrics.MaxSubreadLength() > 0)
                {
                    data_.sequencingInsertLength.push_back(insertMetrics.MaxSubreadLength());
                    data_.sequencingInsertReadScore.push_back(insertMetrics.ReadScore());
                }
                if (insertMetrics.MedianLength() > 0)
                    data_.sequencingMedianInsert.push_back(insertMetrics.MedianLength());

                if (insertMetrics.Umy() > 0)
                    data_.sequencingUmy.push_back(insertMetrics.Umy());
            }
        }
        else
        {
            data_.controlReadLength.push_back(metrics.ZmwControlMetrics().controlReadLength);
            data_.controlReadScore.push_back(metrics.ZmwControlMetrics().controlReadScore);
        }
    }

    const ChipMetricsCollection& Data() const
    { return data_; }

private: // data
    template <typename T>
    void AddAnalogs(AnalogMetricData<std::vector<float>>& data, AnalogMetricData<T> element)
    {
        data.A.push_back(element.A);
        data.C.push_back(element.C);
        data.G.push_back(element.G);
        data.T.push_back(element.T);
    }
    template <typename T>
    void AddFilters(FilterMetricData<std::vector<float>>& data, const FilterMetricData<T>& element)
    {
        data.red.push_back(element.red);
        data.green.push_back(element.green);
    }

    ChipMetricsCollection data_;
};

}}} // ::PacBio::Primary::Postprimary

