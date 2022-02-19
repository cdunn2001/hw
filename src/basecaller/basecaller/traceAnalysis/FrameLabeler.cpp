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

#include "FrameLabeler.h"

namespace PacBio {
namespace Mongo {
namespace Basecaller {

std::unique_ptr<Data::LabelsBatchFactory> FrameLabeler::batchFactory_;
Data::BasecallerRoiConfig::RoiFilterType FrameLabeler::roiType = Data::BasecallerRoiConfig::RoiFilterType::RawEnum::Default;


// static
void FrameLabeler::Configure()
{
    const auto hostExecution = true;
    InitFactory(hostExecution, 0);
}

void FrameLabeler::Finalize()
{}

void FrameLabeler::InitFactory(bool hostExecution, size_t latentFrames)
{
    using Cuda::Memory::SyncDirection;

    SyncDirection syncDir = hostExecution ? SyncDirection::HostWriteDeviceRead : SyncDirection::HostReadDeviceWrite;
    batchFactory_ = std::make_unique<Data::LabelsBatchFactory>(
            latentFrames,
            syncDir);
}

FrameLabeler::FrameLabeler(uint32_t poolId)
    : poolId_ (poolId)
{

}

std::pair<Data::LabelsBatch, Data::FrameLabelerMetrics>
FrameLabeler::Process(Data::TraceBatch<Data::BaselinedTraceElement> trace,
                      const PoolModelParameters&)
{
    auto ret = batchFactory_->NewBatch(std::move(trace));
    for (size_t laneIdx = 0; laneIdx < ret.first.LanesPerBatch(); laneIdx++)
    {
        std::memset(ret.first.GetBlockView(laneIdx).Data(), 0,
                    ret.first.GetBlockView(laneIdx).Size() * sizeof(Data::LabelsBatch::ElementType));
    }
    return ret;
}

}}}     // namespace PacBio::Mongo::Basecaller
