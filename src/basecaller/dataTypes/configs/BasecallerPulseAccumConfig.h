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

#ifndef mongo_dataTypes_configs_BasecallerPulseAccumConfig_H_
#define mongo_dataTypes_configs_BasecallerPulseAccumConfig_H_

#include <pacbio/configuration/PBConfig.h>
#include <pacbio/utilities/SmartEnum.h>

#include <basecaller/traceAnalysis/ComputeDevices.h>

namespace PacBio {
namespace Mongo {
namespace Data {

class BasecallerPulseAccumConfig : public Configuration::PBConfig<BasecallerPulseAccumConfig>
{
public:
    PB_CONFIG(BasecallerPulseAccumConfig);

    SMART_ENUM(MethodName, NoOp, HostSimulatedPulses, HostPulses, GpuPulses)
    PB_CONFIG_PARAM(MethodName, Method, Configuration::DefaultFunc(
                        [](Basecaller::ComputeDevices device) -> MethodName
                        {
                            return device == Basecaller::ComputeDevices::Host ?
                                MethodName::HostPulses :
                                MethodName::GpuPulses;
                        },
                        {"analyzerHardware"}
    ));

    // Increasing this number will directly increase memory usage, even if
    // we don't saturate the allowed number of calls, so be conservative.
    // Hard-coded for now to correspond to 512-frame chunk, 100 fps, with
    // max pulse rate of 10 pulses per second.
    PB_CONFIG_PARAM(uint32_t, maxCallsPerZmw, 48);
};

}}}     // namespace PacBio::Mongo::Data


#endif //mongo_dataTypes_configs_BasecallerPulseAccumConfig_H_
