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

class SimulatedPulseConfig : public Configuration::PBConfig<SimulatedPulseConfig>
{
public:
    PB_CONFIG(SimulatedPulseConfig);

    // These vectors do NOT need to be the same length.  They will be looped through independently,
    // and in fact having their lengths be relatively prime is a way to get a longer period between
    // repeats in the overall pulse sequence
    PB_CONFIG_PARAM(std::vector<char>,  basecalls,   std::vector<char>({ 'A', 'C', 'G', 'T'}));
    PB_CONFIG_PARAM(std::vector<float>, meanSignals, std::vector<float>({ 20.0f, 10.0f, 16.0f, 8.0f}));
    PB_CONFIG_PARAM(std::vector<float>, midSignals,  std::vector<float>({ 21.0f, 11.0f, 17.0f, 9.0f}));
    PB_CONFIG_PARAM(std::vector<float>, maxSignals,  std::vector<float>({ 21.0f, 11.0f, 17.0f, 9.0f}));
    PB_CONFIG_PARAM(std::vector<float>, ipds, std::vector<float>({6}));
    PB_CONFIG_PARAM(std::vector<float>, pws, std::vector<float>({5}));

    // When these vectors are empty, the corresponding vector above is treated as a sequence which
    // each ZMW will walk through in order.  When these vectors have at least one value, then
    // this vector and it's corresponding vector above is treated as a sequence of mean/variance
    // pairs.  These mean/variances will be distributed amongst ZMW in a round-robin fashion.
    PB_CONFIG_PARAM(std::vector<float>, meanSignalsVars, std::vector<float>());
    PB_CONFIG_PARAM(std::vector<float>, midSignalsVars,  std::vector<float>());
    PB_CONFIG_PARAM(std::vector<float>, maxSignalsVars,  std::vector<float>());
    PB_CONFIG_PARAM(std::vector<float>, ipdVars, std::vector<float>());
    PB_CONFIG_PARAM(std::vector<float>, pwVars, std::vector<float>());

    PB_CONFIG_PARAM(uint64_t, seed, 0);
};

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

    // Only has an effect if we are configured to use HostSimulatedPulses
    PB_CONFIG_OBJECT(SimulatedPulseConfig, simConfig);

    // Valid range is [0, 1]
    PB_CONFIG_PARAM(double, XspAmpThresh, 0.60);

    // Must be >= 0
    PB_CONFIG_PARAM(float, XspWidthThresh, 3.5f);
};

}}}     // namespace PacBio::Mongo::Data


#endif //mongo_dataTypes_configs_BasecallerPulseAccumConfig_H_
