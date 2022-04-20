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
#include <dataTypes/HQRFPhysicalStates.h>

namespace PacBio::Mongo::Data
{

SMART_ENUM(MetricNames, Baseline, BaselineStd, Pkmid, SNR, PulseRate, PulseWidth, BaseRate, BaseWidth);

struct SummaryStats : public Configuration::PBConfig<SummaryStats>
{
public:
    PB_CONFIG(SummaryStats);

    PB_CONFIG_PARAM(MetricNames, name, MetricNames::Baseline);
    PB_CONFIG_PARAM(std::vector<uint32_t>, sampleTotal, {});
    PB_CONFIG_PARAM(std::vector<uint32_t>, sampleSize, {});
    PB_CONFIG_PARAM(std::vector<float>, sampleMean, {});
    PB_CONFIG_PARAM(std::vector<float>, sampleMed, {});
    PB_CONFIG_PARAM(std::vector<float>, sampleCV, {});

    std::stringstream& GenerateCSVHeader(const std::string& statPrefix, std::stringstream& ss) const
    {
        assert(sampleTotal.size() == sampleSize.size());
        assert(sampleTotal.size() == sampleMean.size());
        assert(sampleTotal.size() == sampleMed.size());
        assert(sampleTotal.size() == sampleCV.size());

        assert(sampleTotal.size() == 1 || sampleTotal.size() == 4);

        std::vector<std::string> prefixVec = (sampleTotal.size() == 1) ?
            std::vector<std::string>{statPrefix+"_"} :
            std::vector<std::string>{statPrefix+"_A_",statPrefix+"_C_",statPrefix+"_G_",statPrefix+"_T_"};

        const auto json = Serialize();
        for (const auto& memberName : json.getMemberNames())
        {
            if (memberName == "name") continue;
            assert(json[memberName].size() == prefixVec.size());

            for (const auto& prefix : prefixVec)
            {
                ss << prefix + memberName + ",";
            }
        }
        return ss;
    }

    std::stringstream& GenerateCSVRow(std::stringstream& ss) const
    {
        assert(sampleTotal.size() == sampleSize.size());
        assert(sampleTotal.size() == sampleMean.size());
        assert(sampleTotal.size() == sampleMed.size());
        assert(sampleTotal.size() == sampleCV.size());

        assert(sampleTotal.size() == 1 || sampleTotal.size() == 4);

        const auto json = Serialize();
        for (const auto& memberName : json.getMemberNames())
        {
            assert(json[memberName].isArray());

            for (const auto& val: json[memberName])
            {
                ss << val << ",";
            }
        }
        return ss;
    }
};

struct GroupStats : public Configuration::PBConfig<GroupStats>
{
    PB_CONFIG(GroupStats);

    PB_CONFIG_PARAM(std::string, region, "");
    PB_CONFIG_PARAM(std::vector<SummaryStats>, metrics, {});

    std::stringstream& GenerateCSVHeader(std::stringstream& ss) const
    {
        for (const auto& metric : metrics)
        {
            metric.GenerateCSVHeader(region + "_" + metric.name.toString(), ss);
        }
        return ss;
    }

    std::stringstream& GenerateCSVRow(std::stringstream& ss) const
    {
        for (const auto& metric : metrics)
        {
            metric.GenerateCSVRow(ss);
        }
        return ss;
    }
};

struct MetricBlock : public Configuration::PBConfig<MetricBlock>
{
    PB_CONFIG(MetricBlock);

    PB_CONFIG_PARAM(std::vector<GroupStats>, groups, {});

    PB_CONFIG_PARAM(uint64_t, startFrame, 0);
    PB_CONFIG_PARAM(uint32_t, numFrames, 0);
    PB_CONFIG_PARAM(uint64_t, beginFrameTimeStamp, 0);
    PB_CONFIG_PARAM(uint64_t, endFrameTimeStamp, 0);

    std::string GenerateCSVHeader() const
    {
        std::stringstream  ss;
        ss << "StartFrame,NumFrames,StartFrameTS,EndFrameTS,";
        for (const auto& group : groups)
        {
            group.GenerateCSVHeader(ss);
        }
        // Change trailing comma to a newline
        ss.seekp(-1, ss.cur);
        ss << std::endl;
        return ss.str();
    }

    std::string GenerateCSVRow() const
    {
        std::stringstream  ss;
        ss << startFrame << "," << numFrames << "," << beginFrameTimeStamp << "," << endFrameTimeStamp << ",";
        for (const auto& group : groups)
        {
            group.GenerateCSVRow(ss);
        }
        // Change trailing comma to a newline
        ss.seekp(-1, ss.cur);
        ss << std::endl;
        return ss.str();
    }
};

struct MetricsChunk : public Configuration::PBConfig<MetricsChunk>
{
    PB_CONFIG(MetricsChunk);

    PB_CONFIG_PARAM(uint32_t, numMetricsBlocks, 1);
    PB_CONFIG_PARAM(std::vector<MetricBlock>, metricsBlocks, {});
};

struct RealTimeMetricsReport : public Configuration::PBConfig<RealTimeMetricsReport>
{
    PB_CONFIG(RealTimeMetricsReport);

    PB_CONFIG_PARAM(uint64_t, startFrameTimeStamp, 0);
    PB_CONFIG_PARAM(uint64_t, frameTimeStampDelta, 0);

    PB_CONFIG_OBJECT(MetricsChunk, metricsChunk);

    PB_CONFIG_PARAM(std::string, token, "");
};

struct RealTimeMetricsRegion : public Configuration::PBConfig<RealTimeMetricsRegion>
{
    PB_CONFIG(RealTimeMetricsRegion);

    PB_CONFIG_PARAM(std::vector<DataSource::ZmwFeatures>, featuresForFilter,
                    std::vector<DataSource::ZmwFeatures>{DataSource::ZmwFeatures::Sequencing});
    PB_CONFIG_PARAM(std::string, name, "");
    PB_CONFIG_PARAM(std::vector<std::vector<int>>, roi, std::vector<std::vector<int>>());
    PB_CONFIG_PARAM(std::vector<MetricNames>, metrics, {});
    PB_CONFIG_PARAM(uint32_t, minSampleSize, 1000);
};

class RealTimeMetricsConfig : public Configuration::PBConfig<RealTimeMetricsConfig>
{
    PB_CONFIG(RealTimeMetricsConfig);

    PB_CONFIG_PARAM(std::vector<RealTimeMetricsRegion>, regions, std::vector<RealTimeMetricsRegion>());
    // The csv format will be continusouly appended, to maintain a history of all RT Metrics produced
    PB_CONFIG_PARAM(std::string, csvOutputFile, "");
    // The json format will only include the most recent RT Metrics produced.  It will be updated
    // in an atomic fashion via a tmp file, to avoid "slicing" the data in the event of a concurrent
    // update and read
    PB_CONFIG_PARAM(std::string, jsonOutputFile, "");
    PB_CONFIG_PARAM(bool, useSingleActivityLabels, true);
};

} // namespace PacBio::Mongo::Data

#endif // basecaller_dataTypes_configs_RealTimeMetricsConfig_H_
