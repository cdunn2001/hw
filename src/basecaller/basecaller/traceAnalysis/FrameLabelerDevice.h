// Copyright (c) 2019, Pacific Biosciences of California, Inc.
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

#ifndef PACBIO_MONGO_BASECALLER_FRAME_LABALER_DEVICE_H
#define PACBIO_MONGO_BASECALLER_FRAME_LABALER_DEVICE_H

#include "FrameLabeler.h"

#include <dataTypes/configs/AnalysisConfig.h>
#include <dataTypes/configs/BasecallerFrameLabelerConfig.h>
#include <dataTypes/BasicTypes.h>

#include <common/cuda/memory/DeviceAllocationStash.h>

namespace PacBio {
namespace Mongo {
namespace Basecaller {

/// Frame Labeler that runs on the GPU.  Currently hard coded to use
/// the Subframe Gauss Caps approximation, but this could change
/// once the underlying implementation (contained in the prototypes
/// directory) is finally cleaned up
class FrameLabelerDevice : public FrameLabeler
{

public:     // Static functions
    /// Sets algorithm configuration and system calibration properties.
    /// Static because the config types keep a JSON representation and
    /// deserialize on each reference, but the values will be the same for
    /// each DeviceSGCFrameLabeler instance for a given movie.
    /// \note Not thread safe. Do not call this function while threads are
    /// running analysis.
    static void Configure(const Data::AnalysisConfig& analysisConfig,
                          const Data::BasecallerFrameLabelerConfig& labelerConfig);
    static void Finalize();

public:
    FrameLabelerDevice(uint32_t poolId,
                       uint32_t lanesPerPool,
                       Cuda::Memory::StashableAllocRegistrar* registrar = nullptr);
    ~FrameLabelerDevice() override;

    struct Impl; // Processing interface

private:    // Customizable implementation
    std::pair<Data::LabelsBatch, Data::FrameLabelerMetrics>
    Process(Data::TraceBatch<Data::BaselinedTraceElement> trace,
            const PoolModelParameters& models) override;

    std::unique_ptr<Impl> labeler_;
};

}}}     // namespace PacBio::Mongo::Basecaller

#endif //PACBIO_MONGO_BASECALLER_FRAME_LABALER_DEVICE_H
