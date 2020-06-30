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

#include <dataTypes/BasicTypes.h>
#include <dataTypes/configs/ConfigForward.h>
#include <dataTypes/TraceBatch.h>
#include <basecaller/traceAnalysis/Baseliner.h>

namespace PacBio {
namespace Cuda {

// Forward declaring this for now, but really it should eventually be cleaned up and pulled
// out of prototypes
template <size_t blockThreads, size_t width1, size_t width2, size_t stride1, size_t stride2, size_t lag>
class ComposedFilter;

}}

namespace PacBio {
namespace Mongo {
namespace Basecaller {

class DeviceMultiScaleBaseliner : public Baseliner
{
    // Hard coding to the sequel `TwoScaleMedium` for now, but really
    // this should be configurable
    static constexpr size_t width1 = 9;
    static constexpr size_t width2 = 31;
    static constexpr size_t stride1 = 2;
    static constexpr size_t stride2 = 8;
    static constexpr size_t lag = 4;

    // Used to initialze the morpholigical filters.  Really the first bunch of frames
    // out are garbage, but setting this value at least close to the expected baseline
    // means it will potentially not be horribly off base.
    static constexpr short initVal = 150;
public:     // Types
    using ElementTypeIn = Baseliner::ElementTypeIn;

public:     // Static functions
    /// Sets algorithm configuration and system calibration properties.
    /// Static because the config types keep a JSON representation and
    /// deserialize on each reference, but the values will be the same for
    /// each Baseliner instance for a given movie.
    /// \note Not thread safe. Do not call this function while threads are
    /// running analysis.
    static void Configure(const Data::BasecallerBaselinerConfig&,
                          const Data::MovieConfig&);


    static void Finalize();

public:
    DeviceMultiScaleBaseliner(uint32_t poolId, uint32_t lanesPerPool);

    ~DeviceMultiScaleBaseliner() override;

private:    // Customizable implementation
    std::pair<Data::TraceBatch<Data::BaselinedTraceElement>,
              Data::BaselinerMetrics>
    Process(const Data::TraceBatch<ElementTypeIn>& rawTrace) override;

    using Filter = Cuda::ComposedFilter<laneSize/2, width1, width2, stride1, stride2, lag>;
    std::unique_ptr<Filter> filter_;
};

}}}     // namespace PacBio::Mongo::Basecaller

#endif //MONGO_BASECALLER_DEVICE_MULTISCALE_BASELINER_H
