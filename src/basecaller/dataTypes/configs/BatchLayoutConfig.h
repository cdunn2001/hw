#ifndef mongo_dataTypes_PrimaryConfig_H_
#define mongo_dataTypes_PrimaryConfig_H_

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

#include <pacbio/configuration/PBConfig.h>
#include <pacbio/utilities/CpuInfo.h>
#include <pacbio/utilities/Finally.h>

namespace PacBio {
namespace Mongo {
namespace Data {

class BatchLayoutConfig :  public Configuration::PBConfig<BatchLayoutConfig>
{
    PB_CONFIG(BatchLayoutConfig);

    // TODO: zmwsPerLane should be deprecated and eventually removed.
    // In many places we use the constexpr laneSize defined in MongoConstants.h.
    PB_CONFIG_PARAM(uint32_t, zmwsPerLane, 64);
    PB_CONFIG_PARAM(uint32_t, lanesPerPool, 4096);
    PB_CONFIG_PARAM(uint32_t, framesPerChunk, 128);
};

}}}     // namespace PacBio::Mongo::Data

namespace PacBio {
namespace Configuration {

template <>
inline void ValidateConfig<Mongo::Data::BatchLayoutConfig>(
        const Mongo::Data::BatchLayoutConfig& config,
        ValidationResults* results)
{
    if (config.zmwsPerLane != 64) results->AddError("zmwsPerLane must equal 64");
}

}}

#endif //mongo_dataTypes_PrimaryConfig_H_
