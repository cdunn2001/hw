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

#include "PulseAccumulator.h"

#include <dataTypes/BasecallerConfig.h>

namespace PacBio {
namespace Mongo {
namespace Basecaller {

std::unique_ptr<Data::PulseBatchFactory> PulseAccumulator::batchFactory_;

// static

void PulseAccumulator::Configure(size_t maxCallsPerZmw)
{
    InitAllocationPools(true, maxCallsPerZmw);
}

void PulseAccumulator::Finalize()
{
    DestroyAllocationPools();
}

void PulseAccumulator::InitAllocationPools(bool hostExecution, size_t maxCallsPerZmw)
{
    using Cuda::Memory::SyncDirection;

    Data::BatchDimensions dims;
    dims.framesPerBatch = Data::GetPrimaryConfig().framesPerChunk;
    dims.lanesPerBatch = Data::GetPrimaryConfig().lanesPerPool;
    dims.laneWidth = laneSize;

    SyncDirection syncDir = hostExecution ? SyncDirection::HostWriteDeviceRead : SyncDirection::HostReadDeviceWrite;
    batchFactory_ = std::make_unique<Data::PulseBatchFactory>(
            maxCallsPerZmw,
            dims,
            syncDir);
}

std::pair<Data::PulseBatch, Data::PulseDetectorMetrics>
PulseAccumulator::Process(Data::LabelsBatch labels)
{
    auto ret = batchFactory_->NewBatch(labels.Metadata());

    for (size_t laneIdx = 0; laneIdx < labels.LanesPerBatch(); ++laneIdx)
    {
        auto lanePulses = ret.first.Pulses().LaneView(laneIdx);
        lanePulses.Reset();
    }

    // Need to make sure any potential kernels populating `labels`
    // finish before we destroy the object. 
    Cuda::CudaSynchronizeDefaultStream();

    return ret;
}

void PulseAccumulator::DestroyAllocationPools()
{
    batchFactory_.release();
}

PulseAccumulator::PulseAccumulator(uint32_t poolId)
    : poolId_ (poolId)
{

}

}}}     // namespace PacBio::Mongo::Basecaller
