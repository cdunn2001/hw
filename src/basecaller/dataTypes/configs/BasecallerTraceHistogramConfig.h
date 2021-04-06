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

#ifndef mongo_dataTypes_configs_BasecallerTraceHistogramConfig_H_
#define mongo_dataTypes_configs_BasecallerTraceHistogramConfig_H_

#include <pacbio/configuration/PBConfig.h>
#include <pacbio/utilities/SmartEnum.h>

#include <basecaller/traceAnalysis/ComputeDevices.h>

namespace PacBio {
namespace Mongo {
namespace Data {

class BasecallerTraceHistogramConfig : public Configuration::PBConfig<BasecallerTraceHistogramConfig>
{
public:
    PB_CONFIG(BasecallerTraceHistogramConfig);

    SMART_ENUM(MethodName, Host, Gpu);
    PB_CONFIG_PARAM(MethodName, Method, Configuration::DefaultFunc(
                        [](Basecaller::ComputeDevices device) -> MethodName
                        {
                            return device == Basecaller::ComputeDevices::Host ?
                                MethodName::Host:
                                MethodName::Gpu;
                        },
                        {"analyzerHardware"}
    ));

    // Bin size of data histogram is nominally defined as initial estimate
    // of baseline sigma multiplied by this coefficient.
    PB_CONFIG_PARAM(float, BinSizeCoeff, 0.25f);

    // Use fall-back baseline sigma when number of baseline frames is
    // less than this value.
    PB_CONFIG_PARAM(unsigned int, BaselineStatMinFrameCount, 50u);

    // Use this value as an estimate for baseline standard deviation when
    // we have insufficient data.
    PB_CONFIG_PARAM(float, FallBackBaselineSigma, 10.0f);
};

}}}     // namespace PacBio::Mongo::Data

// Define validation specialization.  Specializations must happen in the
// same namespace as the generic declaration.
namespace PacBio {
namespace Configuration {

using PacBio::Mongo::Data::BasecallerTraceHistogramConfig;

template <>
inline void ValidateConfig<BasecallerTraceHistogramConfig>(const BasecallerTraceHistogramConfig& config,
                                                           ValidationResults* results)
{
    if (config.BinSizeCoeff <= 0.0f)
    {
        results->AddError("BinSizeCoeff must be positive");
    }

    if (config.FallBackBaselineSigma <= 0.0f)
    {
        results->AddError("FallBackBaselineSigma must be positive.");
    }
}

}}


#endif //mongo_dataTypes_configs_BasecallerTraceHistogramConfig_H_

