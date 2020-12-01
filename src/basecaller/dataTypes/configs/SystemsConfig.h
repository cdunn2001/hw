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

#ifndef mongo_dataTypes_configs_SystemsConfig_H_
#define mongo_dataTypes_configs_SystemsConfig_H_

#include <pacbio/configuration/PBConfig.h>

namespace PacBio {
namespace Mongo {
namespace Data {

class SystemsConfig : public Configuration::PBConfig<SystemsConfig>
{
public:
    PB_CONFIG(SystemsConfig);
    /// The number of host worker threads to use.  Most parallelism should be
    /// handled internal to the GPU, so this does not need to be large.
    /// A minimum of 3 will allow efficient overlap of upload/download/compute,
    /// but beyond that it shouldn't really be any higher than what is necessary
    /// for active host stages to keep up with the gpu

    // TODO add hooks so that we can switch between gpu and host centric defaults
    // without manually specifying a million parameters
    PB_CONFIG_PARAM(uint32_t, numWorkerThreads, 8);

    /// If true, the threads are bound to a particular set of cores for the
    /// Sequel Alpha machines when running on the host.
    PB_CONFIG_PARAM(bool, bindCores, false);

    /// The maximum amount of gpu memory dedicated to permanently resident
    /// algorithm state data.  Anything beyond this threshold will have to
    /// be shuttled to-from the GPU on demand
    PB_CONFIG_PARAM(size_t, maxPermGpuDataMB, std::numeric_limits<size_t>::max());
};

}}}     // namespace PacBio::Mongo::Data


#endif //mongo_dataTypes_configs_SystemsConfig_H_
