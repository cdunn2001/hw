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

#ifndef PACBIO_MONGO_BASECALLER_DEVICE_SGC_FRAME_LABELER_H
#define PACBIO_MONGO_BASECALLER_DEVICE_SGC_FRAME_LABELER_H

#include "FrameLabeler.h"

namespace PacBio {
namespace Cuda {
// Forward declare of cuda type that cannot be directly included here
// TODO prototype needs more cleaning, and should probably be renamed
class FrameLabeler;

}}

namespace PacBio {
namespace Mongo {
namespace Basecaller {

class DeviceSGCFrameLabeler : public FrameLabeler
{

public:     // Static functions
    /// Sets algorithm configuration and system calibration properties.
    /// Static because the config types keep a JSON representation and
    /// deserialize on each reference, but the values will be the same for
    /// each DeviceSGCFrameLabeler instance for a given movie.
    /// \note Not thread safe. Do not call this function while threads are
    /// running analysis.
    static void Configure(const Data::MovieConfig& movieConfig, int lanesPerPool, int framesPerChunk);
    static void Finalize();

public:
    DeviceSGCFrameLabeler(uint32_t poolId);
    ~DeviceSGCFrameLabeler() override;

private:    // Customizable implementation
    Data::LabelsBatch Process(Data::CameraTraceBatch trace,
                              const PoolModelParameters& models) override;

    std::unique_ptr<Cuda::FrameLabeler> labeler_;
};

}}}     // namespace PacBio::Mongo::Basecaller

#endif //PACBIO_MONGO_BASECALLER_DEVICE_SGC_FRAME_LABELER_H
