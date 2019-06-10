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

#include "DeviceSGCFrameLabeler.h"

#include <prototypes/FrameLabeler/FrameLabelerKernels.cuh>

using namespace PacBio::Cuda;
using namespace PacBio::Cuda::Memory;
using namespace PacBio::Mongo::Data;

namespace PacBio {
namespace Mongo {
namespace Basecaller {

// static
void DeviceSGCFrameLabeler::Configure(int lanesPerPool, int framesPerChunk)
{
    const auto hostExecution = false;
    InitAllocationPools(hostExecution);

    // TODO need to plumb through configurations for this somehow
    std::array<Subframe::AnalogMeta, 4> meta{};
    {
        int frameRate = 100;
        meta[0].ipdSSRatio = 0;
        meta[1].ipdSSRatio = 0;
        meta[2].ipdSSRatio = 0;
        meta[3].ipdSSRatio = 0;

        meta[0].ipd = frameRate * .308;
        meta[1].ipd = frameRate * .234;
        meta[2].ipd = frameRate * .234;
        meta[3].ipd = frameRate * .188;

        meta[0].pw = frameRate * .232;
        meta[1].pw = frameRate * .185;
        meta[2].pw = frameRate * .181;
        meta[3].pw = frameRate * .214;

        meta[0].pwSSRatio = 3.2;
        meta[1].pwSSRatio = 3.2;
        meta[2].pwSSRatio = 3.2;
        meta[3].pwSSRatio = 3.2;
    }
    Cuda::FrameLabeler::Configure(meta, lanesPerPool, framesPerChunk);
}

void DeviceSGCFrameLabeler::Finalize()
{
    DestroyAllocationPools();
    Cuda::FrameLabeler::Finalize();
}

DeviceSGCFrameLabeler::DeviceSGCFrameLabeler(uint32_t poolId)
    : FrameLabeler(poolId)
    , labeler_(std::make_unique<Cuda::FrameLabeler>())
{}

DeviceSGCFrameLabeler::~DeviceSGCFrameLabeler() = default;

LabelsBatch
DeviceSGCFrameLabeler::Process(CameraTraceBatch trace,
                               const PoolModelParameters& models)
{
    auto ret = batchFactory_->NewBatch(std::move(trace));

    //labeler_->ProcessBatch(models, ret.TraceData(), ret);

    return ret;
}

}}}     // namespace PacBio::Mongo::Basecaller
