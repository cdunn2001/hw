//  Copyright (c) 2019, Pacific Biosciences of California, Inc.
//
//  All rights reserved.
//
//  Redistribution and use in source and binary forms, with or without
//  modification, are permitted provided that the following conditions are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of Pacific Biosciences nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.
//
//  NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
//  THIS LICENSE.  THIS SOFTWARE IS PROVIDED BY PACIFIC BIOSCIENCES AND ITS
//  CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
//  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
//  PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL PACIFIC BIOSCIENCES OR
//  ITS CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
//  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
//  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
//  BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
//  IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
//  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
//  POSSIBILITY OF SUCH DAMAGE.

#include "DeviceMultiScaleBaseliner.h"
#include <dataTypes/configs/BasecallerBaselinerConfig.h>
#include <dataTypes/configs/MovieConfig.h>

#include <prototypes/BaselineFilter/BaselineFilterKernels.cuh>

namespace PacBio {
namespace Mongo {
namespace Basecaller {

using namespace PacBio::Cuda::Memory;

constexpr short  DeviceMultiScaleBaseliner::initVal;

void DeviceMultiScaleBaseliner::Configure(const Data::BasecallerBaselinerConfig&,
                                          const Data::MovieConfig& movConfig)
{
    const auto hostExecution = false;
    InitFactory(hostExecution, movConfig.photoelectronSensitivity);
}

void DeviceMultiScaleBaseliner::Finalize() {}

std::pair<Data::TraceBatch<Data::BaselinedTraceElement>,
          Data::BaselinerMetrics>
DeviceMultiScaleBaseliner::FilterBaseline(const Data::TraceBatch<ElementTypeIn>& rawTrace)
{
    auto out = batchFactory_->NewBatch(rawTrace.GetMeta(), rawTrace.StorageDims());

    Data::BatchData<ElementTypeIn> work1(rawTrace.StorageDims(), SyncDirection::HostReadDeviceWrite, SOURCE_MARKER());
    Data::BatchData<ElementTypeIn> work2(rawTrace.StorageDims(), SyncDirection::HostReadDeviceWrite, SOURCE_MARKER());

    filter_->RunBaselineFilter(rawTrace, out.first, out.second.baselinerStats, work1, work2);

    Cuda::CudaSynchronizeDefaultStream();
    return out;
}

DeviceMultiScaleBaseliner::DeviceMultiScaleBaseliner(uint32_t poolId,
                                                        const BaselinerParams& params, uint32_t lanesPerPool,
                                                        StashableAllocRegistrar* registrar)
    : Baseliner(poolId)
    , startupLatency_(params.LatentSize())
{
    filter_ = std::make_unique<Filter>(
        params,
        Scale(),
        lanesPerPool,
        initVal,
        SOURCE_MARKER(),
        registrar);
}

DeviceMultiScaleBaseliner::~DeviceMultiScaleBaseliner() = default;

}}}      // namespace PacBio::Mongo::Basecaller
