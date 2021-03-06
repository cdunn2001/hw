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

#ifndef mongo_dataTypes_configs_BasecallerFrameLabelerConfig_H_
#define mongo_dataTypes_configs_BasecallerFrameLabelerConfig_H_

#include <pacbio/configuration/PBConfig.h>
#include <pacbio/utilities/SmartEnum.h>

#include <basecaller/traceAnalysis/ComputeDevices.h>

namespace PacBio {
namespace Mongo {
namespace Data {

class BasecallerSubframeConfig : public Configuration::PBConfig<BasecallerSubframeConfig>
{
    PB_CONFIG(BasecallerSubframeConfig);

    PB_CONFIG_PARAM(float, alpha, 1.0f);
    PB_CONFIG_PARAM(float, beta, 1.0f);
    PB_CONFIG_PARAM(float, gamma, 1.0f);
};

class BasecallerRoiConfig : public Configuration::PBConfig<BasecallerRoiConfig>
{
    PB_CONFIG(BasecallerRoiConfig);

    // Placeholder config, as there is talk of evaluating different
    // Finite Impulse Response filters for the roi computation
    // Default is basically the original Sequel configuration,
    // and NoOp disables the Roi filter
    SMART_ENUM(RoiFilterType, Default, NoOp);
    PB_CONFIG_PARAM(RoiFilterType, filterType, RoiFilterType::Default);

    PB_CONFIG_PARAM(float, upperThreshold, 7.0f);
    PB_CONFIG_PARAM(float, lowerThreshold, 2.0f);
};

class BasecallerFrameLabelerConfig : public Configuration::PBConfig<BasecallerFrameLabelerConfig>
{
public:
    PB_CONFIG(BasecallerFrameLabelerConfig);

    SMART_ENUM(MethodName, NoOp, Device, Host);
    PB_CONFIG_PARAM(MethodName, Method, Configuration::DefaultFunc(
                        [](Basecaller::ComputeDevices device) -> MethodName
                        {
                            return device == Basecaller::ComputeDevices::Host ?
                                MethodName::Host :
                                MethodName::Device;
                        },
                        {"analyzerHardware"}
    ));

    bool UsesGpu() const { return Method == MethodName::Device; }

    // When/if we get alternative implementations, change these to be
    // variants over the implementation specific configs
    PB_CONFIG_OBJECT(BasecallerSubframeConfig, viterbi);
    PB_CONFIG_OBJECT(BasecallerRoiConfig, roi);
};

}}}     // namespace PacBio::Mongo::Data


#endif //mongo_dataTypes_configs_BasecallerFrameLabelerConfig_H_
