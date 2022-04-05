// Copyright (c) 2022, Pacific Biosciences of California, Inc.
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

#ifndef basecaller_dataTypes_configs_RealTimeMetricsConfig_H_
#define basecaller_dataTypes_configs_RealTimeMetricsConfig_H_

#include <pacbio/configuration/PBConfig.h>
#include <pacbio/datasource/ZmwFeatures.h>
#include <pacbio/text/String.h>

#include <common/MongoConstants.h>

namespace PacBio::Mongo::Data
{

struct RealTimeMetricsReport : public Configuration::PBConfig<RealTimeMetricsReport>
{
    PB_CONFIG(RealTimeMetricsReport);
    struct SummaryStats : public Configuration::PBConfig<SummaryStats>
    {
        PB_CONFIG(SummaryStats);

        PB_CONFIG_PARAM(uint32_t, sampleTotal, 0);
        PB_CONFIG_PARAM(uint32_t, sampleSize, 0);
        PB_CONFIG_PARAM(float, sampleMean, std::numeric_limits<float>::quiet_NaN());
        PB_CONFIG_PARAM(float, sampleMedian, std::numeric_limits<float>::quiet_NaN());
        PB_CONFIG_PARAM(float, sampleCV, std::numeric_limits<float>::quiet_NaN());

        std::string CSVOutput() const
        {
            return std::to_string(sampleMean) + "," +
                   std::to_string(sampleMedian) + "," +
                   std::to_string(sampleCV) + "," +
                   std::to_string(sampleSize) + "," +
                   std::to_string(sampleTotal);
        }
    };

    PB_CONFIG_PARAM(SummaryStats, baseRate, {});
    PB_CONFIG_PARAM(SummaryStats, baseWidth, {});
    PB_CONFIG_PARAM(SummaryStats, pulseRate, {});
    PB_CONFIG_PARAM(SummaryStats, pulseWidth, {});

    using analogsMetric = std::array<SummaryStats,numAnalogs>;
    PB_CONFIG_PARAM(analogsMetric, snr, {});
    PB_CONFIG_PARAM(analogsMetric, pkmid, {});
    PB_CONFIG_PARAM(SummaryStats, baseline, {});
    PB_CONFIG_PARAM(SummaryStats, baselineSd, {});

    PB_CONFIG_PARAM(std::string, name, "");
    PB_CONFIG_PARAM(uint64_t, startFrame, 0);
    PB_CONFIG_PARAM(uint32_t, numFrames, 0);
    PB_CONFIG_PARAM(uint64_t, beginFrameTimeStamp, 0);
    PB_CONFIG_PARAM(uint64_t, endFrameTimeStamp, 0);

    static std::string Header(const std::vector<std::string>& regionNames)
    {
        const auto& statStr = [](const std::string& v) {
            const std::array<std::string,5> sHeaders = {{"Mean", "Median", "CV", "Size", "Total"}};
            std::vector<std::string> elems;
            std::transform(sHeaders.begin(), sHeaders.end(), std::back_inserter(elems),
                           [&v](const auto& s) { return v + "_" + s; });
            return PacBio::Text::String::Join(elems.begin(), elems.end(), ',');
        };

        std::string header = "StartFrame,NumFrames,StartFrameTS,EndFrameTS,";
        const std::array<std::string,4> bases = {{"A", "C", "G", "T"}};
        for (const auto& name : regionNames)
        {
            header += statStr(name + "_BaseRate") + "," + statStr(name + "_BaseWidth");
            header += ",";
            header += statStr(name + "_PulseRate") + "," + statStr(name + "_PulseWidth");
            header += ",";
            for (size_t i = 0; i < numAnalogs; i++)
            {
                header += statStr(name + "_Snr" + bases[i]) + "," + statStr(name + "_Pkmid" + bases[i]) + ",";
            }
            header += statStr(name + "_Baseline") + "," + statStr(name + "_BaselineSd");
        }

        return header;
    }

    std::string CSVOutput() const
    {
        std::string output = baseRate.CSVOutput() + "," + baseWidth.CSVOutput();
        output += ",";
        output += pulseRate.CSVOutput() + "," + pulseWidth.CSVOutput();
        output += ",";
        for (size_t i = 0; i < numAnalogs; i++)
        {
            output += snr[i].CSVOutput() + "," + pkmid[i].CSVOutput() + ",";
        }
        output += baseline.CSVOutput() + "," +
                  baselineSd.CSVOutput();
        return output;
    }
};

struct RealTimeMetricsRegion : public Configuration::PBConfig<RealTimeMetricsRegion>
{
    PB_CONFIG(RealTimeMetricsRegion);

    PB_CONFIG_PARAM(std::vector<DataSource::ZmwFeatures>, features,
                    std::vector<DataSource::ZmwFeatures>{DataSource::ZmwFeatures::Sequencing});
    PB_CONFIG_PARAM(std::string, name, "");
    PB_CONFIG_PARAM(std::vector<std::vector<int>>, roi, std::vector<std::vector<int>>());
    PB_CONFIG_PARAM(uint32_t, minSampleSize, 1000);
};

class RealTimeMetricsConfig : public Configuration::PBConfig<RealTimeMetricsConfig>
{
    PB_CONFIG(RealTimeMetricsConfig);

    PB_CONFIG_PARAM(std::vector<RealTimeMetricsRegion>, regions, std::vector<RealTimeMetricsRegion>());
    PB_CONFIG_PARAM(std::string, csvOutputFile, "");
};

} // namespace PacBio::Mongo::Data

#endif // basecaller_dataTypes_configs_RealTimeMetricsConfig_H_
