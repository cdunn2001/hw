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

#ifndef mongo_dataTypes_configs_BasecallerBaselinerConfig_H_
#define mongo_dataTypes_configs_BasecallerBaselinerConfig_H_

#include <sstream>

#include <pacbio/configuration/PBConfig.h>
#include <pacbio/utilities/SmartEnum.h>
#include <basecaller/traceAnalysis/ComputeDevices.h>

namespace PacBio {
namespace Mongo {
namespace Data {

class BasecallerBaselinerConfig : public Configuration::PBConfig<BasecallerBaselinerConfig>
{
public:
    PB_CONFIG(BasecallerBaselinerConfig);

    SMART_ENUM(FilterTypes,
               MultiScaleLarge, MultiScaleMedium, MultiScaleSmall,
               TwoScaleLarge, TwoScaleMedium, TwoScaleSmall);
    PB_CONFIG_PARAM(FilterTypes, Filter, FilterTypes::TwoScaleMedium);

    SMART_ENUM(MethodName, HostMultiScale, DeviceMultiScale, NoOp);

    PB_CONFIG_PARAM(MethodName, Method, Configuration::DefaultFunc(
                        [](Basecaller::ComputeDevices device) -> MethodName
                        {
                            return device == Basecaller::ComputeDevices::Host ?
                                MethodName::HostMultiScale :
                                MethodName::DeviceMultiScale;
                        },
                        {"analyzerHardware"}
    ));
    
    // The "half-life" for the exponential moving average used to smooth
    // the lower-upper-gap-based estimate of baseline sigma.
    // SigmaEmaScaleStrides must be >= +0.
    // Used by the HostMultiScale implementation.
    // TODO: Use it in a similar way in DeviceMultiScale implementation.
    PB_CONFIG_PARAM(float, SigmaEmaScaleStrides, 512);

    bool UsesGpu() const { return Method == MethodName::DeviceMultiScale; }
};

}}}     // namespace PacBio::Mongo::Data


namespace PacBio::Configuration {

template <>
inline void ValidateConfig<Mongo::Data::BasecallerBaselinerConfig>(
        const Mongo::Data::BasecallerBaselinerConfig& config,
        ValidationResults* results)
{
    const float sess = config.SigmaEmaScaleStrides;
    if (std::isnan(sess) || std::signbit(sess))
    {
        std::ostringstream msg;
        msg << "Bad value.  SigmaEmaScaleStrides = " << sess
            << ".  May not be negative, -0, or NaN.";
        results->AddError(msg.str());
    }
}

}   // namespace PacBio::Configuration


#endif //mongo_dataTypes_configs_BasecallerBaselinerConfig_H_
