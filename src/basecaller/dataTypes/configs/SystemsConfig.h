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

#include <basecaller/traceAnalysis/ComputeDevices.h>

namespace PacBio {
namespace Mongo {
namespace Data {

class SystemsConfig : public Configuration::PBConfig<SystemsConfig>
{
public:
    PB_CONFIG(SystemsConfig);
    /// The number of host worker threads to use.  For the most part we only
    /// need enough to satisfy the basecallerConcurrencyRequest below, but if
    /// any host computation stages are active then the more worker threades
    /// allowed the more TBB will be able to leverage any parallelism.
    PB_CONFIG_PARAM(uint32_t, numWorkerThreads, 8);

    /// The number of threads allowed to be working in the basecaller at any given
    /// time.  This will also correspond to the number of cuda streams in play.
    /// A minimum of 3 will allow efficient overlap of upload/download/compute,
    /// and anything beyond that can be used to smooth over any scheduling gaps
    /// that arrise from irregular compute/IO.  Shouldn't really need that
    /// many additional streams, though dropping down to 1 is necessary when
    /// measuring a robust compute budget breakdown.
    PB_CONFIG_PARAM(uint32_t, basecallerConcurrency, 5);

    /// The maximum amount of gpu memory dedicated to permanently resident
    /// algorithm state data.  Anything beyond this threshold will have to
    /// be shuttled to-from the GPU on demand
    PB_CONFIG_PARAM(size_t, maxPermGpuDataMB, std::numeric_limits<uint32_t>::max());

    /// Specifies the expected compute resource to use for basecalling.  Not currently
    /// a binding configuration, but will control various secondary behavior like the
    /// defaults between host vs gpu filter implementations, and whether we try to measure
    /// PCIe utilization or not.
    PB_CONFIG_PARAM(Basecaller::ComputeDevices, analyzerHardware, Basecaller::ComputeDevices::V100);
};

}}}     // namespace PacBio::Mongo::Data


#endif //mongo_dataTypes_configs_SystemsConfig_H_
