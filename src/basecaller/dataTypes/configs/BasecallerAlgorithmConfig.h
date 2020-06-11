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
//
//  Description:
/// \brief  Global configuration for the Primary realtime pipeline. These values
///         may be changed at run time.

#ifndef mongo_dataTypes_BasecallerAlgorithmConfig_H_
#define mongo_dataTypes_BasecallerAlgorithmConfig_H_

#include <string>

#include <pacbio/configuration/PBConfig.h>

#include <dataTypes/configs/BasecallerBaselinerConfig.h>
#include <dataTypes/configs/BasecallerTraceHistogramConfig.h>
#include <dataTypes/configs/BasecallerDmeConfig.h>
#include <dataTypes/configs/BasecallerFrameLabelerConfig.h>
#include <dataTypes/configs/BasecallerPulseAccumConfig.h>
#include <dataTypes/configs/BasecallerPulseToBaseConfig.h>
#include <dataTypes/configs/BasecallerMetricsConfig.h>
#include <dataTypes/configs/SimulatedFaults.h>
#include <dataTypes/configs/StaticDetModelConfig.h>

namespace PacBio {
namespace Mongo {
namespace Data {

class BasecallerAlgorithmConfig : public Configuration::PBConfig<BasecallerAlgorithmConfig>
{
public:
    PB_CONFIG(BasecallerAlgorithmConfig);

    PB_CONFIG_OBJECT(BasecallerBaselinerConfig, baselinerConfig);
    PB_CONFIG_OBJECT(BasecallerTraceHistogramConfig, traceHistogramConfig);
    PB_CONFIG_OBJECT(BasecallerDmeConfig, dmeConfig);
    PB_CONFIG_OBJECT(BasecallerFrameLabelerConfig, frameLabelerConfig);
    PB_CONFIG_OBJECT(BasecallerPulseAccumConfig, pulseAccumConfig);
    PB_CONFIG_OBJECT(BasecallerPulseToBaseConfig, PulseToBase);
    PB_CONFIG_OBJECT(BasecallerMetricsConfig, Metrics);
    PB_CONFIG_OBJECT(SimulatedFaults, simulatedFaults);

    PB_CONFIG_OBJECT(StaticDetModelConfig, staticDetModelConfig);
    PB_CONFIG_PARAM(bool, staticAnalysis, true);

    PB_CONFIG_PARAM(uint32_t, framesPerHFMetricBlock, 4096);
    PB_CONFIG_PARAM(bool, realtimeActivityLabels, true);

public:
    std::string CombinedMethodName() const
    {
        return baselinerConfig.Method.toString() + "_"
             + dmeConfig.Method.toString() + "_"
             + frameLabelerConfig.Method.toString() + "_"
             + PulseToBase.Method.toString() + "_"
             + Metrics.Method.toString();
    }
};

}}}

#endif //mongo_dataTypes_BasecallerAlgorithmConfig_H_
