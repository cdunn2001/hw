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
    PB_CONFIG_PARAM(FilterTypes, Filter, FilterTypes::MultiScaleLarge);

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
    
    // Two parameters solely for tuning calibration coefficients.
    // Each adjusts its corresponding coefficient by a factor of exp2(x).
    PB_CONFIG_PARAM(float, MeanBiasAdjust, 0.0f);
    PB_CONFIG_PARAM(float, SigmaBiasAdjust, 0.0f);

    // The "half-life" for the unbiased exponential moving average used to
    // smooth the estimate of the baseline.
    // MeanEmaScaleStrides must be >= 0.
    // Internal use of single-precision floating-point sets the practical
    // limit that MeanEmaScaleStrides should not exceed about 1500.
    PB_CONFIG_PARAM(float, MeanEmaScaleStrides, 0.0f);

    // The "half-life" for the exponential moving average used to smooth
    // the lower-upper-gap-based estimate of baseline sigma.
    // SigmaEmaScaleStrides must be >= +0.
    // Used by the HostMultiScale implementation.
    // TODO: Use it in a similar way in DeviceMultiScale implementation.
    PB_CONFIG_PARAM(float, SigmaEmaScaleStrides, 512);

    // Baseline standard deviation divided by JumpSuppression defines a
    // tolerance for large increases in the baseline estimate.  Any jump that
    // exceeds this tolerance is ignored and the most recent (accepted)
    // estimate is retained.
    // Notice that this effect is asymmetric; it applies only to increases,
    // not to decreases.
    // Notice also that this suppression is applied before smoothing by the EMA
    // controlled by MeanEmaScaleStrides.
    PB_CONFIG_PARAM(float, JumpSuppression, 0.0f);

    bool UsesGpu() const { return Method == MethodName::DeviceMultiScale; }
};

}}}     // namespace PacBio::Mongo::Data


namespace PacBio::Configuration {

template <>
inline void ValidateConfig<Mongo::Data::BasecallerBaselinerConfig>(
        const Mongo::Data::BasecallerBaselinerConfig& config,
        ValidationResults* results)
{
    const float mba = config.MeanBiasAdjust;
    if (!std::isfinite(mba))
    {
        std::ostringstream msg;
        msg << "Bad value.  MeanBiasAdjust = " << mba
            << ".  Should be finite.";
        results->AddError(msg.str());
    }

    const float sba = config.SigmaBiasAdjust;
    if (!std::isfinite(sba))
    {
        std::ostringstream msg;
        msg << "Bad value.  SigmaBiasAdjust = " << sba
            << ".  Should be finite.";
        results->AddError(msg.str());
    }

    const float sess = config.SigmaEmaScaleStrides;
    if (std::isnan(sess) || std::signbit(sess))
    {
        std::ostringstream msg;
        msg << "Bad value.  SigmaEmaScaleStrides = " << sess
            << ".  May not be negative, -0, or NaN.";
        results->AddError(msg.str());
    }

    const float mess = config.MeanEmaScaleStrides;
    if (mess < 0)
    {
        std::ostringstream msg;
        msg << "Bad value.  MeanEmaScaleStrides = " << mess
            << ".  Should be greater or equal to zero.";
        results->AddError(msg.str());
    }
}

}   // namespace PacBio::Configuration


#endif //mongo_dataTypes_configs_BasecallerBaselinerConfig_H_
