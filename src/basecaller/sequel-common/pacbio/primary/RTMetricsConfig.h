// Copyright (c) 2017, Pacific Biosciences of California, Inc.
//
// All rights reserved.
//
// THIS SOFTWARE CONSTITUTES AND EMBODIES PACIFIC BIOSCIENCES' CONFIDENTIAL
// AND PROPRIETARY INFORMATION.
//
// Disclosure, redistribution and use of this software is subject to the
// terms and conditions of the applicable written agreement(s) between you
// and Pacific Biosciences, where "you" refers to you or your company or
// organization, as applicable.  Any other disclosure, redistribution or
// use is prohibited.
//
// THIS SOFTWARE IS PROVIDED BY PACIFIC BIOSCIENCES AND ITS CONTRIBUTORS "AS
// IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
// THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL PACIFIC BIOSCIENCES OR ITS
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
// OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
// WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
// OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
// ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Description:
/// \brief declaration of the object used to configure the Real Time Metrics

// TODO remove this file?

#ifndef SEQUEL_RTMETRICSCONFIG_H
#define SEQUEL_RTMETRICSCONFIG_H

#include <pacbio/configuration/PBConfig.h>
#include <pacbio/primary/SequelDefinitions.h>

namespace PacBio {
namespace Primary {

class RTMetricsRegion : public Configuration::PBConfig<RTMetricsRegion>
{
    PB_CONFIG(RTMetricsRegion);

    PB_CONFIG_PARAM(std::string, name, "");
    PB_CONFIG_PARAM(uint32_t, xMin, 0);
    PB_CONFIG_PARAM(uint32_t, yMin, 0);
    PB_CONFIG_PARAM(uint32_t, xExtent, 0);
    PB_CONFIG_PARAM(uint32_t, yExtent, 0);
};

class RTMetricsRegionNew : public Configuration::PBConfig<RTMetricsRegionNew>
{
    PB_CONFIG(RTMetricsRegionNew);

    SMART_ENUM(zmwType, SEQUENCING, PORSEQUENCING,
               LASERSCATTER, LPTITRATION2P0X, LPTITRATION1P5X, LPTITRATION0P5X, LPTITRATION0P0X);
    SMART_ENUM(zmwMetric, Baseline, BaselineStd, Pkmid, Snr,
               PulseRate, PulseWidth, BaseRate, BaseWidth);
    PB_CONFIG_PARAM(std::vector<zmwType>, zmwTypesFilter, std::vector<zmwType>{});
    PB_CONFIG_PARAM(std::vector<zmwMetric>, zmwMetricsReported, std::vector<zmwMetric>{});
    PB_CONFIG_PARAM(std::string, name, "");
    PB_CONFIG_PARAM(uint32_t, xMin, 0);
    PB_CONFIG_PARAM(uint32_t, yMin, 0);
    PB_CONFIG_PARAM(uint32_t, xExtent, 0);
    PB_CONFIG_PARAM(uint32_t, yExtent, 0);
    PB_CONFIG_PARAM(uint32_t, samplingFactorStride, 1);
    PB_CONFIG_PARAM(uint32_t, minSampleSize, 1000);
public:
    static std::vector<zmwMetric> ReportKineticMetrics()
    {
        return {zmwMetric::PulseRate,
                zmwMetric::PulseWidth,
                zmwMetric::BaseRate,
                zmwMetric::BaseWidth};
    }
    static std::vector<zmwMetric> ReportSignalMetrics()
    {
        return {zmwMetric::Baseline,
                zmwMetric::BaselineStd,
                zmwMetric::Pkmid,
                zmwMetric::Snr};
    }
    static std::vector<zmwMetric> ReportAllMetrics()
    {
        auto kin = ReportKineticMetrics();
        auto sig = ReportSignalMetrics();
        kin.insert(kin.end(), sig.begin(), kin.end());
        return kin;
    }
};

class RTMetricsConfig : public Configuration::PBConfig<RTMetricsConfig>
{
public:

    PB_CONFIG(RTMetricsConfig);

    // The two modes below control the filtering for reporting
    // baseline metrics - baseline and baseline sigma vs.
    // signal metrics - pkmid and snr. signalConfigMode is only
    // active when the new format is enabled whereas the old
    // format controlled baseline, baseline sigma, pkmid, and snr
    // with just baselineConfigMode. The modes are:
    //
    // MODE0 - no filtering applied
    // MODE1 - # baseline frames > minBaselineFrames
    // MODE2 - # baseline frames > minBaselineFrames && DME success
    //
    SMART_ENUM(BaselineMode, MODE0, MODE1, MODE2);
    SMART_ENUM(SignalMode, MODE0, MODE1, MODE2);
    PB_CONFIG_PARAM(SignalMode, signalConfigMode, SignalMode::MODE2);
    PB_CONFIG_PARAM(BaselineMode, baselineConfigMode,
                    Configuration::DefaultFunc([](bool newFmt) -> BaselineMode
                                               { return newFmt ?
                                                       BaselineMode::MODE1 :
                                                       BaselineMode::MODE2;
                                               },
                                               {"newJsonFormat"}));

    PB_CONFIG_PARAM(uint32_t, minBaselineFrames, 100);
    PB_CONFIG_PARAM(uint32_t, minSampleSize, 1000);

    // These thresholds are used to determine if
    // a ZMW is "sequencing". They are in units
    // of seconds. The upper thresholds are only used
    // when the new format is enabled.
    PB_CONFIG_PARAM(float, minBaseRate, 0.25f);
    PB_CONFIG_PARAM(float, maxBaseRate, -1.0f);
    PB_CONFIG_PARAM(float, minBaseWidth, 0.0625f);
    PB_CONFIG_PARAM(float, maxBaseWidth, -1.0f);

    // DME confidence score threshold
    PB_CONFIG_PARAM(float, minConfidenceScore, 0.0f);

    // Sampling factor stride used for all regions when
    // the old format is enabled. The new format allows for
    // a sampling factor to be set per region.
    PB_CONFIG_PARAM(uint32_t, samplingFactorStride, 1);
    PB_CONFIG_PARAM(uint32_t, numMetricsSuperChunks, 3);

    // Flag for specifying the new format, this
    // also enables other settings.
    PB_CONFIG_PARAM(bool, newJsonFormat, true);
    PB_CONFIG_PARAM(uint32_t, maxQueueSize, 5);

    PB_CONFIG_PARAM(bool, useRealtimeActivityLabels, false);

    // We keep both the old and new way of specifying regions.
    // When we have completely transitioned over, we
    // can remove the old way of specifying regions.
    //
    // TODO: Remove "regions" when new format becomes the default.
    PB_CONFIG_PARAM(std::vector<RTMetricsRegion>, regions, std::vector<RTMetricsRegion>{});
    PB_CONFIG_PARAM(std::vector<RTMetricsRegionNew>, newRegions, std::vector<RTMetricsRegionNew>{});

    void SetSequelChipRegionDefaults()
    {
        ///////////////////////////////////////////////////////////////////////
        //
        // OLD WAY
        regions.resize(0);
        regions.resize(6);

        regions[0].name = "TopStrip";
        regions[0].xMin = 74;
        regions[0].yMin = 69;
        regions[0].xExtent = 150;
        regions[0].yExtent = 940;

        regions[1].name = "TopHigh";
        regions[1].xMin = 74;
        regions[1].yMin = 1009;
        regions[1].xExtent = 150;
        regions[1].yExtent = 10;

        regions[2].name = "MidStrip";
        regions[2].xMin = 529;
        regions[2].yMin = 69;
        regions[2].xExtent = 150;
        regions[2].yExtent = 940;

        regions[3].name = "MidHigh";
        regions[3].xMin = 529;
        regions[3].yMin = 1009;
        regions[3].xExtent = 150;
        regions[3].yExtent = 10;

        regions[4].name = "BotStrip";
        regions[4].xMin = 984;
        regions[4].yMin = 69;
        regions[4].xExtent = 150;
        regions[4].yExtent = 940;

        regions[5].name = "BotHigh";
        regions[5].xMin = 984;
        regions[5].yMin = 1009;
        regions[5].xExtent = 150;
        regions[5].yExtent = 10;

        ///////////////////////////////////////////////////////////////////////
        //
        // NEW WAY
        newRegions.resize(0);
        newRegions.resize(4);

        // Full Chip - PORSEQUENCING
        newRegions[0].name = "FullChip";
        newRegions[0].xMin = 64;
        newRegions[0].yMin = 64;
        newRegions[0].xExtent = 1080;
        newRegions[0].yExtent = 960;
        newRegions[0].zmwTypesFilter.push_back(RTMetricsRegionNew::zmwType::PORSEQUENCING);
        newRegions[0].ReportAllMetrics();

        // Top Strip - PORSEQUENCING
        newRegions[1].name = "TopStrip";
        newRegions[1].xMin = 74;
        newRegions[1].yMin = 69;
        newRegions[1].xExtent = 150;
        newRegions[1].yExtent = 940;
        newRegions[1].zmwTypesFilter.push_back(RTMetricsRegionNew::zmwType::PORSEQUENCING);
        newRegions[1].ReportSignalMetrics();

        // Mid Strip - PORSEQUENCING
        newRegions[2].name = "MidStrip";
        newRegions[2].xMin = 529;
        newRegions[2].yMin = 69;
        newRegions[2].xExtent = 150;
        newRegions[2].yExtent = 940;
        newRegions[2].zmwTypesFilter.push_back(RTMetricsRegionNew::zmwType::PORSEQUENCING);
        newRegions[2].ReportSignalMetrics();

        // Bot Strip - PORSEQUENCING
        newRegions[3].name = "BotStrip";
        newRegions[3].xMin = 984;
        newRegions[3].yMin = 69;
        newRegions[3].xExtent = 150;
        newRegions[3].yExtent = 940;
        newRegions[3].zmwTypesFilter.push_back(RTMetricsRegionNew::zmwType::PORSEQUENCING);
        newRegions[3].ReportSignalMetrics();
    }

    void SetSpiderChipRegionDefaults()
    {
        ///////////////////////////////////////////////////////////////////////
        //
        // OLD WAY
        regions.resize(0);
        regions.resize(6);

        // In order to keep the format compatible for ICS we
        // use the same region names and types for Sequel for now
        // but this should be updated specifically for Spider.
        regions[0].name = "TopStrip";
        regions[0].xMin = 75;
        regions[0].yMin = 75;
        regions[0].xExtent = 190;
        regions[0].yExtent = 2762;

        regions[1].name = "TopHigh";
        regions[1].xMin = 0;
        regions[1].yMin = 0;
        regions[1].xExtent = 0;
        regions[1].yExtent = 0;

        regions[2].name = "MidStrip";
        regions[2].xMin = 1283;
        regions[2].yMin = 75;
        regions[2].xExtent = 190;
        regions[2].yExtent = 2762;

        regions[3].name = "MidHigh";
        regions[3].xMin = 0;
        regions[3].yMin = 0;
        regions[3].xExtent = 0;
        regions[3].yExtent = 0;

        regions[4].name = "BotStrip";
        regions[4].xMin = 2490;
        regions[4].yMin = 75;
        regions[4].xExtent = 190;
        regions[4].yExtent = 2762;

        regions[5].name = "BotHigh";
        regions[5].xMin = 0;
        regions[5].yMin = 0;
        regions[5].xExtent = 0;
        regions[5].yExtent = 0;

        // Set sampling factor to 8 to get roughly the same number of ZMWs as Sequel.
        samplingFactorStride = 8;

        ///////////////////////////////////////////////////////////////////////
        //
        // NEW WAY
        newRegions.resize(0);
        newRegions.resize(11);

        // Full Chip - PORSEQUENCING
        newRegions[0].name = "FullChip";
        newRegions[0].xMin = 0;
        newRegions[0].yMin = 0;
        newRegions[0].xExtent = 2755;
        newRegions[0].yExtent = 2911;
        newRegions[0].zmwTypesFilter.push_back(RTMetricsRegionNew::zmwType::PORSEQUENCING);
        newRegions[0].samplingFactorStride = 8;
        newRegions[0].ReportAllMetrics();

        // Top Strip - PORSEQUENCING
        newRegions[1].name = "TopStrip";
        newRegions[1].xMin = 75;
        newRegions[1].yMin = 75;
        newRegions[1].xExtent = 190;
        newRegions[1].yExtent = 2762;
        newRegions[1].zmwTypesFilter.push_back(RTMetricsRegionNew::zmwType::PORSEQUENCING);
        newRegions[1].samplingFactorStride = 8;
        newRegions[1].ReportSignalMetrics();

        // Mid Strip - PORSEQUENCING
        newRegions[2].name = "MidStrip";
        newRegions[2].xMin = 1283;
        newRegions[2].yMin = 75;
        newRegions[2].xExtent = 190;
        newRegions[2].yExtent = 2762;
        newRegions[2].zmwTypesFilter.push_back(RTMetricsRegionNew::zmwType::PORSEQUENCING);
        newRegions[2].samplingFactorStride = 8;
        newRegions[2].ReportSignalMetrics();

        // Bot Strip - PORSEQUENCING
        newRegions[3].name = "BotStrip";
        newRegions[3].xMin = 2490;
        newRegions[3].yMin = 75;
        newRegions[3].xExtent = 190;
        newRegions[3].yExtent = 2762;
        newRegions[3].zmwTypesFilter.push_back(RTMetricsRegionNew::zmwType::PORSEQUENCING);
        newRegions[3].samplingFactorStride = 8;
        newRegions[3].ReportSignalMetrics();

        // Top Strip - LASERSCATTER
        newRegions[4].name = "TopStrip_LS";
        newRegions[4].xMin = 0;
        newRegions[4].yMin = 0;
        newRegions[4].xExtent = 5;
        newRegions[4].yExtent = 2911;
        newRegions[4].minSampleSize = 100;
        newRegions[4].zmwTypesFilter.push_back(RTMetricsRegionNew::zmwType::LASERSCATTER);
        newRegions[4].ReportSignalMetrics();

        // Mid Strip - LASERSCATTER
        newRegions[5].name = "MidStrip_LS";
        newRegions[5].xMin = 1375;
        newRegions[5].yMin = 0;
        newRegions[5].xExtent = 5;
        newRegions[5].yExtent = 2911;
        newRegions[5].minSampleSize = 100;
        newRegions[5].zmwTypesFilter.push_back(RTMetricsRegionNew::zmwType::LASERSCATTER);
        newRegions[5].ReportSignalMetrics();

        // Bot Strip - LASERSCATTER
        newRegions[6].name = "BotStrip_LS";
        newRegions[6].xMin = 2751;
        newRegions[6].yMin = 0;
        newRegions[6].xExtent = 5;
        newRegions[6].yExtent = 2911;
        newRegions[6].minSampleSize = 100;
        newRegions[6].zmwTypesFilter.push_back(RTMetricsRegionNew::zmwType::LASERSCATTER);
        newRegions[6].ReportSignalMetrics();

        // Full Chip - LPTITRATION2P0X
        newRegions[7].name = "FullChip_LP2P0X";
        newRegions[7].xMin = 0;
        newRegions[7].yMin = 0;
        newRegions[7].xExtent = 2755;
        newRegions[7].yExtent = 2911;
        newRegions[7].minSampleSize = 100;
        newRegions[7].zmwTypesFilter.push_back(RTMetricsRegionNew::zmwType::LPTITRATION2P0X);
        newRegions[7].ReportSignalMetrics();

        // Full Chip - LPTITRATION1P5X
        newRegions[8].name = "FullChip_LP1P5X";
        newRegions[8].xMin = 0;
        newRegions[8].yMin = 0;
        newRegions[8].xExtent = 2755;
        newRegions[8].yExtent = 2911;
        newRegions[8].minSampleSize = 100;
        newRegions[8].zmwTypesFilter.push_back(RTMetricsRegionNew::zmwType::LPTITRATION1P5X);
        newRegions[8].ReportSignalMetrics();

        // Full Chip - LPTITRATION0P5X
        newRegions[9].name = "FullChip_LP0P5X";
        newRegions[9].xMin = 0;
        newRegions[9].yMin = 0;
        newRegions[9].xExtent = 2755;
        newRegions[9].yExtent = 2911;
        newRegions[9].minSampleSize = 100;
        newRegions[9].zmwTypesFilter.push_back(RTMetricsRegionNew::zmwType::LPTITRATION0P5X);
        newRegions[9].ReportSignalMetrics();

        // Full Chip - LPTITRATION0P0X
        newRegions[10].name = "FullChip_LP0P0X";
        newRegions[10].xMin = 0;
        newRegions[10].yMin = 0;
        newRegions[10].xExtent = 2755;
        newRegions[10].yExtent = 2911;
        newRegions[10].minSampleSize = 100;
        newRegions[10].zmwTypesFilter.push_back(RTMetricsRegionNew::zmwType::LPTITRATION0P0X);
        newRegions[10].ReportSignalMetrics();
    }

    };

}}

#endif //SEQUEL_RTMETRICSCONFIG_H
