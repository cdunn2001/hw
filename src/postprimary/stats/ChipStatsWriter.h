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

#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "ChipStats.h"
#include "Histogram.h"

namespace PacBio {
namespace Primary {
namespace Postprimary {

class ChipStatsWriter
{
public: // static data
    static const int adapterDimerLimit = 10;
    static const int shortInsertLimit = 100;

    // Hard-code bin widths, bin starting location, and number of bins for metrics
    // that will be merge-able for report generation.

    // One set of parameters for all read length metrics.
    static constexpr double readLengthBinWidth = 500;
    static constexpr double readLengthMinBin = 0;
    static constexpr double readLengthNumBins = 200;
    static constexpr uint32_t numBasesPerSecond = 3;

    // Score-based metrics in the range of [0,1]
    static constexpr double scoreBinWidth = 0.02;
    static constexpr double scoreMinBin = 0;
    static constexpr double scoreNumBins = 50;

public: // structors
    ChipStatsWriter(const std::string& fileName, bool controlUsed) 
        : fileName_(fileName)
        , fileNameTmp_(fileName + ".tmp")
        , controlUsed_(controlUsed)
    {
        Init();
    };
    // Default constructor
    ChipStatsWriter() = delete;
    // Move constructor
    ChipStatsWriter(ChipStatsWriter&&) = delete;
    // Copy constructor
    ChipStatsWriter(const ChipStatsWriter&) = delete;
    // Move assignment operator
    ChipStatsWriter& operator=(ChipStatsWriter&& rhs) noexcept = delete;
    // Copy assignment operator
    ChipStatsWriter& operator=(const ChipStatsWriter&) = delete;

    ~ChipStatsWriter()
    {
        xml_.close();
        if (std::rename(fileNameTmp_.c_str(), fileName_.c_str()))
            PBLOG_ERROR << "Could not rename " << fileNameTmp_ << " to " << fileName_;
        PBLOG_INFO << "Renaming file " << fileNameTmp_ << " -> " << fileName_;
    }

public:
    void WriteChip(const ChipStats& stats,
                   const FileHeader& fh, const std::string& schemaVersion)
    {
        WriteXMLHeader(schemaVersion);

        // Determine the maximum read length to use for read-length based histograms.
        maxReadLengthBinned_ = fh.MovieTimeInHrs() * 3600 * numBasesPerSecond;

        //=====================================================================
        // Movie name and movie length

        xml_ << "<MovieName>" << fh.MovieName() << "</MovieName>";
        xml_ << "<MovieLength>" << fh.MovieTimeInHrs() * 60 << "</MovieLength>";

        // # Sequencing ZMWS
        numSequencingZmws_ = std::accumulate(
                stats.Data().isSequencing.begin(),
                stats.Data().isSequencing.end(), 0);

        xml_ << "<NumSequencingZmws>" << numSequencingZmws_ << "</NumSequencingZmws>";

        size_t numFailedSnrFilter = std::accumulate(
            stats.Data().failedSNRCut.begin(),
            stats.Data().failedSNRCut.end(), 0);

        xml_ << "<NumFailedSnrFilterZmws>" << numFailedSnrFilter << "</NumFailedSnrFilterZmws>";

        size_t numFailedDmeZmws = std::accumulate(
            stats.Data().failedDME.begin(),
            stats.Data().failedDME.end(), 0);

        xml_ << "<NumFailedDmeZmws>" << numFailedDmeZmws << "</NumFailedDmeZmws>";

        // Selectively display metrics.

        // Display all external-facing metrics.
        for (const auto& metric : externalMetrics)
        {
            if (metricsWriteMap.find(metric) != metricsWriteMap.end())
            {
                metricsWriteMap[metric](*this, stats);
            }
        }

        // Display internal metrics only in internal mode.
        if (std::accumulate(stats.Data().internal.begin(), stats.Data().internal.end(), 0) > 0)
        {
            for (const auto& metric : internalMetrics)
            {
                if (metricsWriteMap.find(metric) != metricsWriteMap.end())
                {
                    metricsWriteMap[metric](*this, stats);
                }
            }
        }

        WriteXMLFooter();

    }

    void WriteStatChan(const std::vector<float>& stat,
                       const std::string& xmlElement,
                       const std::string& channel,
                       const std::string& metricDescription)
    {
        if (!stat.empty())
        {
            xml_ << "<" << xmlElement
            << " Channel=\"" << channel << "\"" << ">"
            << Histogram(stat, metricDescription)
            << "</" << xmlElement << ">";
        }
    };

    void WriteEmptyStat(const std::string& xmlElement,
                        const std::string& metricDescription,
                        const double binWidth, const double numBins)
    {
        xml_ << "<" << xmlElement << ">"
             << Histogram(metricDescription, binWidth, numBins)
             << "</" << xmlElement << ">";
    }

    void WriteStat(const std::vector<float>& stat,
                   const std::string& xmlElement,
                   const std::string& metricDescription)
    {
        if (!stat.empty())
        {
            xml_ << "<" << xmlElement << ">"
            << Histogram(stat, metricDescription)
            << "</" << xmlElement << ">";
        }
    };

    void WriteStatFixedBinWidthVarBins(const std::vector<float>& stat,
                                       const std::string& xmlElement,
                                       const std::string& metricDescription,
                                       const double binWidth,
                                       const double maxBinValue,
                                       bool outputN50=false)
    {
        if (!stat.empty())
        {
            Histogram h(stat, metricDescription, binWidth, maxBinValue);
            h.outputN50 = outputN50;
            xml_ << "<" << xmlElement << ">"
                 << h
                 << "</" << xmlElement << ">";
        }
    }


    void WriteStatFixedBinWidthNumBins(const std::vector<float>& stat,
                                       const std::string& xmlElement,
                                       const std::string& metricDescription,
                                       const double binWidth,
                                       const double minBinValue,
                                       const double numBins)
    {
        if (!stat.empty())
        {
            Histogram h(stat, metricDescription, binWidth, minBinValue, numBins);
            xml_ << "<" << xmlElement << ">"
                 << h
                 << "</" << xmlElement << ">";
        }
    };

    std::ofstream& XMLStream() { return xml_; }

private: // data
    std::string fileName_;
    std::string fileNameTmp_;
    bool controlUsed_;
    std::ofstream xml_;
    uint64_t numSequencingZmws_ = 0;
    uint32_t maxReadLengthBinned_;

    std::vector<std::string> externalMetrics
        {
            "TOTAL_BASEFRACTION",
            "BASELINE",
            "BASELINE_SD",
            "SNR",
            "HQSNR",
            "HQPKMID",
            "PAUSINESS",
            "AD_SI",
            "IS_READS",
            "CONTROL",
            "PRODUCTIVITY",
            "LOADING",
            "READ_TYPE",
            "READ_SCORE",
            "PULSE_METRICS",
            "POLYMERASE_READ",
            "POLYMERASE_HQ_READ",
            "BASELINE_ZMWS",
            "ANGLES",
            "UMY"
        };

    std::vector<std::string> internalMetrics
        {

        };

    std::map<std::string, std::function<void(ChipStatsWriter& c, const ChipStats& stats)>> metricsWriteMap{
        {"AD_SI", [this](ChipStatsWriter& c, const ChipStats& stats) {
            float adapterDimerFraction = 0;
            float shortInsertFraction = 0;
            if (this->numSequencingZmws_ != 0)
            {
                int adapterDimers = 0;
                int shortInserts = 0;
                for (const auto &m : stats.Data().sequencingMedianInsert)
                {
                    if (m < adapterDimerLimit)
                        adapterDimers++;
                    else if (m < shortInsertLimit)
                        shortInserts++;
                }

                adapterDimerFraction = static_cast<float>(adapterDimers) / static_cast<float>(this->numSequencingZmws_);
                shortInsertFraction = static_cast<float>(shortInserts) / static_cast<float>(this->numSequencingZmws_);
            }

            c.XMLStream() << "<AdapterDimerFraction>" << adapterDimerFraction << "</AdapterDimerFraction>";
            c.XMLStream() << "<ShortInsertFraction>" << shortInsertFraction << "</ShortInsertFraction>";
        }},
        {"UMY", [this](ChipStatsWriter& c, const ChipStats& stats) {
            // TODO: These length metrics are currently being stored as float and are derived
            // from the subread metrics that are currently being stored as double. These should
            // all be converted to unsigned ints but the blast radius is too large to address this
            // here for SE-2480. For now we compute them as floats to preserve the precision.
            float umy = std::accumulate(stats.Data().sequencingUmy.begin(), stats.Data().sequencingUmy.end(), 0.f);
            c.XMLStream() << "<SequencingUmy>" << std::fixed << std::setprecision(0) << std::round(umy) << "</SequencingUmy>";
        }},
        {"ANGLES", [](ChipStatsWriter& c, const ChipStats& stats) {
            c.WriteStatChan(stats.Data().angleHQ.A, "DmeAngleEstDist",
                            "A", "DME Angle Estimation: Analog A");
            c.WriteStatChan(stats.Data().angleHQ.C, "DmeAngleEstDist",
                            "C", "DME Angle Estimation: Analog C");
            c.WriteStatChan(stats.Data().angleHQ.G, "DmeAngleEstDist",
                            "G", "DME Angle Estimation: Analog G");
            c.WriteStatChan(stats.Data().angleHQ.T, "DmeAngleEstDist",
                            "T", "DME Angle Estimation: Analog T");
        }},
        {"IS_READS", [](ChipStatsWriter& c, const ChipStats& stats) {
            float isReadsFraction = 0;
            isReadsFraction =  std::accumulate(stats.Data().isRead.begin(),
                                               stats.Data().isRead.end(), 0)
                               / static_cast<float>(stats.Data().isRead.size());

            c.XMLStream() << "<IsReadsFraction>" << isReadsFraction << "</IsReadsFraction>";
        }},
        {"CONTROL", [this](ChipStatsWriter& c, const ChipStats& stats) {
            if (c.controlUsed_)
            {
                if (stats.Data().controlReadLength.empty())
                {
                    c.WriteEmptyStat("ControlReadLenDist", "Control Read Length", readLengthBinWidth, readLengthNumBins);
                    c.WriteEmptyStat("ControlReadQualDist", "Control Read Quality", scoreBinWidth, scoreNumBins);
                }
                else
                {
                    c.WriteStatFixedBinWidthVarBins(stats.Data().controlReadLength, "ControlReadLenDist", "Control Read Length",
                                                    readLengthBinWidth, maxReadLengthBinned_);

                    c.WriteStatFixedBinWidthNumBins(stats.Data().controlReadScore, "ControlReadQualDist", "Control Read Quality",
                                                    scoreBinWidth, scoreMinBin, scoreNumBins);
                }
            }
        }},
        {"PRODUCTIVITY", [](ChipStatsWriter& c, const ChipStats& stats) {
            c.XMLStream() << "<ProdDist>"
            << DiscreteHistogram(stats.Data().productivity, {{"Empty", "Productive", "Other", "Undefined"}}, "Productivity")
            << "</ProdDist>";
        }},
        {"LOADING", [](ChipStatsWriter& c, const ChipStats& stats) {
            c.XMLStream() << "<LoadingDist>"
            << DiscreteHistogram(stats.Data().loading, {{"Empty", "Single", "Multi", "Indeterminate", "Undefined"}}, "Loading")
            << "</LoadingDist>";
        }},
        {"READ_TYPE", [](ChipStatsWriter& c, const ChipStats& stats) {
            c.XMLStream() << "<ReadTypeDist>"
            << DiscreteHistogram(stats.Data().readType,
                                 {{"Empty", "FullHqRead0", "FullHqRead1", "PartialHqRead0", "PartialHqRead1", "PartialHqRead2", "Indeterminate", "Undefined"}}, "ReadTypeDist")
            << "</ReadTypeDist>";
        }},
        {"TOTAL_BASEFRACTION", [](ChipStatsWriter& c, const ChipStats& stats) {
            if (!stats.Data().baseCount.A.empty())
            {
                double countA = 0;
                double countC = 0;
                double countG = 0;
                double countT = 0;
                double sum = 0;

                for (const auto& d : stats.Data().baseCount.A) countA += d;
                for (const auto& d : stats.Data().baseCount.C) countC += d;
                for (const auto& d : stats.Data().baseCount.G) countG += d;
                for (const auto& d : stats.Data().baseCount.T) countT += d;
                sum = countA + countC + countG + countT;
                if (sum > 0)
                {
                    countA /= sum;
                    countC /= sum;
                    countG /= sum;
                    countT /= sum;
                }

                c.XMLStream() << "<TotalBaseFractionPerChannel Channel=\"A\"><TotalBaseFractionValue>" << countA << "</TotalBaseFractionValue></TotalBaseFractionPerChannel>";
                c.XMLStream() << "<TotalBaseFractionPerChannel Channel=\"C\"><TotalBaseFractionValue>" << countC << "</TotalBaseFractionValue></TotalBaseFractionPerChannel>";
                c.XMLStream() << "<TotalBaseFractionPerChannel Channel=\"G\"><TotalBaseFractionValue>" << countG << "</TotalBaseFractionValue></TotalBaseFractionPerChannel>";
                c.XMLStream() << "<TotalBaseFractionPerChannel Channel=\"T\"><TotalBaseFractionValue>" << countT << "</TotalBaseFractionValue></TotalBaseFractionPerChannel>";
            }
        }},
        {"BASELINE", [](ChipStatsWriter& c, const ChipStats& stats) {
            c.WriteStatChan(stats.Data().analogBaseline.A,
                            "BaselineLevelDist", "A",
                            "Baseline Level: Channel A");
            c.WriteStatChan(stats.Data().analogBaseline.C,
                            "BaselineLevelDist", "C",
                            "Baseline Level: Channel C");
            c.WriteStatChan(stats.Data().analogBaseline.G,
                            "BaselineLevelDist", "G",
                            "Baseline Level: Channel G");
            c.WriteStatChan(stats.Data().analogBaseline.T,
                            "BaselineLevelDist", "T",
                            "Baseline Level: Channel T");
        }},
        {"BASELINE_SD", [](ChipStatsWriter& c, const ChipStats& stats) {
            c.WriteStatChan(stats.Data().analogBaselineSD.A,
                            "BaselineStdDist", "A", "Baseline Std: Channel A");
            c.WriteStatChan(stats.Data().analogBaselineSD.C,
                            "BaselineStdDist", "C", "Baseline Std: Channel C");
            c.WriteStatChan(stats.Data().analogBaselineSD.G,
                            "BaselineStdDist", "G", "Baseline Std: Channel G");
            c.WriteStatChan(stats.Data().analogBaselineSD.T,
                            "BaselineStdDist", "T", "Baseline Std: Channel T");
        }},
        {"READ_SCORE", [](ChipStatsWriter& c, const ChipStats& stats) {
            c.WriteStat(stats.Data().readScore, "MovieReadQualDist",
                        "Movie Read Quality");
        }},
        {"PULSE_METRICS", [](ChipStatsWriter& c, const ChipStats& stats) {
            c.WriteStat(stats.Data().pulseRate, "PulseRateDist", "Pulse Rate");
            c.WriteStat(stats.Data().pulseWidth, "PulseWidthDist", "Pulse Width");
            c.WriteStat(stats.Data().baseRate, "BaseRateDist", "Base Rate");
            c.WriteStat(stats.Data().baseWidth, "BaseWidthDist", "Base Width");
            c.WriteStat(stats.Data().baseIPD, "BaseIpdDist", "Base Ipd");
            c.WriteStat(stats.Data().localBaseRate, "LocalBaseRateDist",
                            "Local Base Rate");
        }},
        {"POLYMERASE_READ", [this](ChipStatsWriter& c, const ChipStats& stats) {
            c.WriteStatFixedBinWidthVarBins(
                stats.Data().polyLength, "NumUnfilteredBasecallsDist",
                "Number of Unfiltered Basecalls", readLengthBinWidth,
                maxReadLengthBinned_);
        }},
        {"POLYMERASE_HQ_READ", [this](ChipStatsWriter& c, const ChipStats& stats) {
            c.WriteStatFixedBinWidthVarBins(
                stats.Data().sequencingReadLength, "ReadLenDist",
                "Read Length", readLengthBinWidth, maxReadLengthBinned_, true);
            // ReadQualDist is for the HQ-region, MovieReadQualDist is for the
            // full ZMW. Currently these are the same.
            c.WriteStatFixedBinWidthNumBins(
                stats.Data().sequencingReadScore, "ReadQualDist",
                "Read Quality", scoreBinWidth, scoreMinBin, scoreNumBins);
            c.WriteStatFixedBinWidthNumBins(
                stats.Data().sequencingReadScore, "MovieReadQualDist",
                "Movie Read Quality", scoreBinWidth, scoreMinBin,
                scoreNumBins);

            c.WriteStatFixedBinWidthVarBins(
                stats.Data().sequencingInsertLength,
                "InsertReadLenDist", "Insert Read Length",
                readLengthBinWidth, maxReadLengthBinned_, true);
            c.WriteStatFixedBinWidthNumBins(
                stats.Data().sequencingInsertReadScore,
                "InsertReadQualDist", "Insert Read Quality", scoreBinWidth,
                scoreMinBin, scoreNumBins);
            c.WriteStatFixedBinWidthVarBins(
                stats.Data().sequencingMedianInsert, "MedianInsertDist",
                "Median Insert Length", readLengthBinWidth,
                maxReadLengthBinned_);
            c.WriteStat(
                stats.Data().sequencingHQRatio, "HqBaseFractionDist",
                "Fraction of Bases in HQ Region");
        }},
        {"SNR", [](ChipStatsWriter& c, const ChipStats& stats) {
            c.WriteStatChan(stats.Data().snr.A, "SnrDist",  "A",
                            "Signal to Noise Ratio: Channel A");
            c.WriteStatChan(stats.Data().snr.C, "SnrDist",  "C",
                            "Signal to Noise Ratio: Channel C");
            c.WriteStatChan(stats.Data().snr.G, "SnrDist",  "G",
                            "Signal to Noise Ratio: Channel G");
            c.WriteStatChan(stats.Data().snr.T, "SnrDist",  "T",
                            "Signal to Noise Ratio: Channel T");
        }},
        {"HQSNR", [](ChipStatsWriter& c, const ChipStats& stats) {
            c.WriteStatChan(stats.Data().snrHQ.A, "HqRegionSnrDist",
                            "A", "HQ Region Signal to Noise Ratio: Channel A");
            c.WriteStatChan(stats.Data().snrHQ.C, "HqRegionSnrDist",
                            "C", "HQ Region Signal to Noise Ratio: Channel C");
            c.WriteStatChan(stats.Data().snrHQ.G, "HqRegionSnrDist",
                            "G", "HQ Region Signal to Noise Ratio: Channel G");
            c.WriteStatChan(stats.Data().snrHQ.T, "HqRegionSnrDist",
                            "T", "HQ Region Signal to Noise Ratio: Channel T");
        }},
        {"HQPKMID", [](ChipStatsWriter& c, const ChipStats& stats) {
            c.WriteStatChan(stats.Data().pkmidHQ.A, "HqBasPkMidDist",
                            "A", "HQ Region PkMid: Channel A");
            c.WriteStatChan(stats.Data().pkmidHQ.C, "HqBasPkMidDist",
                            "C", "HQ Region PkMid: Channel C");
            c.WriteStatChan(stats.Data().pkmidHQ.G, "HqBasPkMidDist",
                            "G", "HQ Region PkMid: Channel G");
            c.WriteStatChan(stats.Data().pkmidHQ.T, "HqBasPkMidDist",
                            "T", "HQ Region PkMid: Channel T");
        }},
        {"PAUSINESS", [](ChipStatsWriter& c, const ChipStats& stats) {
            c.WriteStat(stats.Data().pausiness, "PausinessDist", "Pausiness");
        }},
        {"BASELINE_ZMWS", [](ChipStatsWriter& c, const ChipStats& stats) {
            c.WriteStatChan(stats.Data().baselineSequencing.red,
                            "BaselineLevelSequencingDist", "A",
                            "Baseline Level Sequencing Zmws: Channel A");
            c.WriteStatChan(stats.Data().baselineSequencing.red,
                            "BaselineLevelSequencingDist", "C",
                            "Baseline Level Sequencing Zmws: Channel C");
            c.WriteStatChan(stats.Data().baselineSequencing.green,
                            "BaselineLevelSequencingDist", "G",
                            "Baseline Level Sequencing Zmws: Channel G");
            c.WriteStatChan(stats.Data().baselineSequencing.green,
                            "BaselineLevelSequencingDist", "T",
                            "Baseline Level Sequencing Zmws: Channel T");

            c.WriteStatChan(stats.Data().baselineScatteringMetrology.red,
                            "BaselineLevelScatteringMetrologyDist", "A",
                            "Baseline Level ScatteringMetrology Zmws: Channel A");
            c.WriteStatChan(stats.Data().baselineScatteringMetrology.red,
                            "BaselineLevelScatteringMetrologyDist", "C",
                            "Baseline Level ScatteringMetrology Zmws: Channel C");
            c.WriteStatChan(stats.Data().baselineScatteringMetrology.green,
                            "BaselineLevelScatteringMetrologyDist", "G",
                            "Baseline Level ScatteringMetrology Zmws: Channel G");
            c.WriteStatChan(stats.Data().baselineScatteringMetrology.green,
                            "BaselineLevelScatteringMetrologyDist", "T",
                            "Baseline Level ScatteringMetrology Zmws: Channel T");;
        }}
    };

private:

    void Init()
    {
        xml_.open(fileNameTmp_.c_str());
        PBLOG_INFO << "Opening file " << fileNameTmp_;
    }

    void WriteXMLHeader(const std::string& schemaVersion)
    {
        xml_ << "<?xml version=\"1.0\" encoding=\"utf-8\"?>";
        xml_ << "<PipeStats xmlns=\"http://pacificbiosciences.com/PacBioPipelineStats.xsd\"";
        xml_ << " xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\"";
        xml_ << " xmlns:ns=\"http://pacificbiosciences.com/PacBioBaseDataModel.xsd\"";
        xml_ << " Version=\"" << schemaVersion << "\">";
    }

    void WriteXMLFooter()
    {
        xml_ << "</PipeStats>";
    }


};

}}}
