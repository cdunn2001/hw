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

#include <common/MongoConstants.h>

namespace PacBio::Mongo::Data
{

struct RealTimeMetricsRegion : public Configuration::PBConfig<RealTimeMetricsRegion>
{
    PB_CONFIG(RealTimeMetricsRegion);

    struct Report : public Configuration::PBConfig<Report>
    {
        PB_CONFIG(Report);
        struct SummaryStats : public Configuration::PBConfig<SummaryStats>
        {
            PB_CONFIG(SummaryStats);

            PB_CONFIG_PARAM(uint32_t, sampleTotal, 0);
            PB_CONFIG_PARAM(uint32_t, sampleSize, 0);
            PB_CONFIG_PARAM(float, sampleMean, -1);
            PB_CONFIG_PARAM(float, sampleMedian, -1);
            PB_CONFIG_PARAM(float, sampleCV, -1);
        };

        PB_CONFIG_PARAM(SummaryStats, baseRate, SummaryStats());
        PB_CONFIG_PARAM(SummaryStats, baseWidth, SummaryStats());
        PB_CONFIG_PARAM(SummaryStats, pulseRate, SummaryStats());
        PB_CONFIG_PARAM(SummaryStats, pulseWidth, SummaryStats());

        using analogsMetric = std::array<SummaryStats,numAnalogs>;
        PB_CONFIG_PARAM(analogsMetric, snr,
                        analogsMetric({SummaryStats(), SummaryStats(), SummaryStats(), SummaryStats()}));
        PB_CONFIG_PARAM(analogsMetric, pkmid,
                        analogsMetric({SummaryStats(), SummaryStats(), SummaryStats(), SummaryStats()}));
        PB_CONFIG_PARAM(SummaryStats, baseline, SummaryStats());
        PB_CONFIG_PARAM(SummaryStats, baselineSd, SummaryStats());

        PB_CONFIG_PARAM(std::string, name, "");
        PB_CONFIG_PARAM(uint64_t, startFrame, 0);
        PB_CONFIG_PARAM(uint32_t, numFrames, 0);
        PB_CONFIG_PARAM(uint64_t, startFrameTimeStampe, 0);
        PB_CONFIG_PARAM(uint64_t, endFrameTimeStamp, 0);
    };

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
};

} // namespace PacBio::Mongo::Data

#endif // basecaller_dataTypes_configs_RealTimeMetricsConfig_H_
