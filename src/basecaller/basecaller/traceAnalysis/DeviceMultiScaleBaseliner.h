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

#ifndef MONGO_BASECALLER_DEVICE_MULTISCALE_BASELINER_H
#define MONGO_BASECALLER_DEVICE_MULTISCALE_BASELINER_H

#include <stdint.h>

#include <common/cuda/memory/DeviceAllocationStash.h>

#include <dataTypes/BasicTypes.h>
#include <dataTypes/configs/ConfigForward.h>
#include <dataTypes/TraceBatch.h>
#include <basecaller/traceAnalysis/Baseliner.h>
#include <basecaller/traceAnalysis/BaselinerParams.h>

namespace PacBio {
namespace Cuda {

// Forward declaring this for now, but really it should eventually be cleaned up and pulled
// out of prototypes
class ComposedFilterBase;

}}

namespace PacBio {
namespace Mongo {
namespace Basecaller {

class DeviceMultiScaleBaseliner : public Baseliner
{
    static constexpr size_t lag = 4;

private:     // Static data
    // Used to initialze the morpholigical filters.  Really the first bunch of frames
    // out are garbage, but setting this value at least close to the expected baseline
    // means it will potentially not be horribly off base.
    static constexpr short initVal = 150;

    static float cSigmaBiasAdj_;
    static float cMeanBiasAdj_;
    static float meanEmaAlpha_;
    static float sigmaEmaAlpha_;
    static float jumpTolCoeff_;

public:     // Static functions
    /// Sets algorithm configuration and system calibration properties.
    /// Static because the config types keep a JSON representation and
    /// deserialize on each reference, but the values will be the same for
    /// each Baseliner instance for a given movie.
    /// \note Not thread safe. Do not call this function while threads are
    /// running analysis.
    static void Configure(const Data::BasecallerBaselinerConfig&,
                          const Data::AnalysisConfig&);

    static void Finalize();

public:
    using EncodingFormat = DataSource::PacketLayout::EncodingFormat;
    DeviceMultiScaleBaseliner(uint32_t poolId,
                              const BaselinerParams& params,
                              uint32_t lanesPerPool,
                              Cuda::Memory::StashableAllocRegistrar* registrar = nullptr);

    DeviceMultiScaleBaseliner(const DeviceMultiScaleBaseliner&) = delete;
    DeviceMultiScaleBaseliner(DeviceMultiScaleBaseliner&&) = default;
    ~DeviceMultiScaleBaseliner() override;

    static float SigmaEmaScaleStrides()
    { 
        return -1.0f / std::log2(sigmaEmaAlpha_);
    }

    static float MeanEmaScaleStrides()
    { 
        return -1.0f / std::log2(meanEmaAlpha_);
    }

    size_t StartupLatency() const override 
    { 
        // pick the longest estimated ema transient in strides
        const float maxScaleStrides = std::max(MeanEmaScaleStrides(), SigmaEmaScaleStrides());
        // todo: consider scaling the second term here for additional
        // tunability.
        return latency_ + maxScaleStrides * framesPerStride_;
    }

private:    // Customizable implementation
    std::pair<Data::TraceBatch<Data::BaselinedTraceElement>, Data::BaselinerMetrics>
    FilterBaseline(const Data::TraceBatchVariant& rawTrace) override;

    using Filter = Cuda::ComposedFilterBase;
    std::unique_ptr<Filter> filter_;
    size_t latency_;
    size_t framesPerStride_;
};

}}}     // namespace PacBio::Mongo::Basecaller

#endif //MONGO_BASECALLER_DEVICE_MULTISCALE_BASELINER_H
