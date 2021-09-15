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
#include <basecaller/traceAnalysis/BaselinerParams.h>
#include <dataTypes/configs/BasecallerBaselinerConfig.h>

#include <prototypes/BaselineFilter/BaselineFilterKernels.cuh>

namespace PacBio {
namespace Mongo {
namespace Basecaller {

using namespace PacBio::Cuda::Memory;

constexpr size_t DeviceMultiScaleBaseliner::width1;
constexpr size_t DeviceMultiScaleBaseliner::width2;
constexpr size_t DeviceMultiScaleBaseliner::stride1;
constexpr size_t DeviceMultiScaleBaseliner::stride2;
constexpr short  DeviceMultiScaleBaseliner::initVal;

void DeviceMultiScaleBaseliner::Configure(const Data::BasecallerBaselinerConfig&,
                                          const Data::MovieConfig&)
{
    const auto hostExecution = false;
    Baseliner::InitFactory(hostExecution);
}

void DeviceMultiScaleBaseliner::Finalize() {}

std::pair<Data::TraceBatch<Data::BaselinedTraceElement>,
          Data::BaselinerMetrics>
DeviceMultiScaleBaseliner::FilterBaseline_(const Data::TraceBatch<ElementTypeIn>& rawTrace)
{
    auto out = batchFactory_->NewBatch(rawTrace.GetMeta(), rawTrace.StorageDims());

    Data::BatchData<ElementTypeIn> work1(rawTrace.StorageDims(), SyncDirection::HostReadDeviceWrite, SOURCE_MARKER());
    Data::BatchData<ElementTypeIn> work2(rawTrace.StorageDims(), SyncDirection::HostReadDeviceWrite, SOURCE_MARKER());

    filter_->RunBaselineFilter(rawTrace, out.first, out.second.baselinerStats, work1, work2);

    Cuda::CudaSynchronizeDefaultStream();
    return out;
}

DeviceMultiScaleBaseliner::DeviceMultiScaleBaseliner(uint32_t poolId,
                                                     uint32_t lanesPerPool,
                                                     StashableAllocRegistrar* registrar)
    : Baseliner(poolId)
{
    filter_ = std::make_unique<Filter>(
        SOURCE_MARKER(),
        lanesPerPool,
        initVal,
        registrar);
}

DeviceMultiScaleBaseliner::~DeviceMultiScaleBaseliner() = default;

size_t DeviceMultiScaleBaseliner::StartupLatency() const
{
    static const auto params = FilterParamsLookup(Data::BasecallerBaselinerConfig::MethodName::DeviceMultiScale);

    // Just check to make sure no one changed the hard coded filter parameters
    // without updating this here.
    assert([&]() -> bool
    {
        const auto& strides = params.Strides();
        const auto& widths = params.Widths();

        bool valid = true;
        valid &= strides.size() == 2;
        valid &= widths.size() == 2;
        if (!valid) throw PBException("Unexpected number of baseline filters");

        valid &= widths[0] == width1;
        valid &= widths[1] == width2;
        valid &= strides[0] == stride1;
        valid &= strides[1] == stride2;

        if (!valid)
            throw PBException("Unexpected filter settings in DeviceMultiScaleBaseliner");

        return valid;
    }());

    return params.LatentSize();
}

}}}      // namespace PacBio::Mongo::Basecaller
