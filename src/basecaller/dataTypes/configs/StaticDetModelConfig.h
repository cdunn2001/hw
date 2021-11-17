// Copyright (c) 2019-2020, Pacific Biosciences of California, Inc.
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

#ifndef mongo_dataTypes_configs_StaticDetectionModel_h
#define mongo_dataTypes_configs_StaticDetectionModel_h

#include <common/MongoConstants.h>
#include "AnalysisConfig.h"

#include <pacbio/configuration/PBConfig.h>

namespace PacBio {
namespace Mongo {
namespace Data {

class StaticDetModelConfig : public Configuration::PBConfig<StaticDetModelConfig>
{
public:
    PB_CONFIG(StaticDetModelConfig);

    PB_CONFIG_PARAM(float, baselineMean, 0.0f);
    PB_CONFIG_PARAM(float, baselineVariance, 33.0f);

public:
    struct AnalogMode
    {
        float mean;
        float var;
    };

    auto SetupAnalogs(const Data::AnalysisConfig& analysisConfig) const
    {
        std::array<AnalogMode, numAnalogs> analogs;
        auto& movieInfo = analysisConfig.movieInfo;
        const auto refSignal = movieInfo.refSnr * std::sqrt(baselineVariance);
        for (size_t i = 0; i < analogs.size(); i++)
        {
            const auto mean = baselineMean + movieInfo.analogs[i].relAmplitude * refSignal;
            const auto var = baselineVariance + mean + std::pow(movieInfo.analogs[i].excessNoiseCV * mean, 2.f);

            analogs[i].mean = mean;
            analogs[i].var = var;
        }

        return analogs;
    }
};

}}} // PacBio::Mongo::Data

#endif // mongo_dataTypes_configs_StaticDetectionModel_h
