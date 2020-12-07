#ifndef mongo_dataTypes_PulseDetectionMetrics_H_
#define mongo_dataTypes_PulseDetectionMetrics_H_

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
//  Description:
//  Defines various small metrics containers for different stages of the
//  pipelines, which ferry things like the Viterbi score from
//  the frame labeler and the baseline stats from the Pulse Accumulator (not to
//  be mistaken for the baseliner stats emitted by the Baseliner)

#include <common/cuda/utility/CudaArray.h>
#include <common/StatAccumState.h>
#include "BaselinerStatAccumState.h"
#include "BatchData.h"

namespace PacBio {
namespace Mongo {
namespace Data {

struct BaselinerMetrics
{
    BaselinerMetrics(uint32_t lanesPerBatch,
                     Cuda::Memory::SyncDirection syncDir,
                     const Cuda::Memory::AllocationMarker& marker)
        : baselinerStats(lanesPerBatch, syncDir, marker)
    { }

    Cuda::Memory::UnifiedCudaArray<BaselinerStatAccumState> baselinerStats;
};

struct FrameLabelerMetrics
{
    FrameLabelerMetrics(const BatchDimensions& dims,
                        Cuda::Memory::SyncDirection syncDir,
                        const Cuda::Memory::AllocationMarker& marker)
        : viterbiScore(dims.lanesPerBatch, syncDir, marker)
    { }

    Cuda::Memory::UnifiedCudaArray<Cuda::Utility::CudaArray<float, laneSize>> viterbiScore;
};

struct PulseDetectorMetrics
{
    PulseDetectorMetrics(const BatchDimensions& dims,
                         Cuda::Memory::SyncDirection syncDir,
                         const Cuda::Memory::AllocationMarker& marker)
        : baselineStats(dims.lanesPerBatch, syncDir, marker)
    { }

    Cuda::Memory::UnifiedCudaArray<StatAccumState> baselineStats;
};

}}}     // namespace PacBio::Mongo::Data

#endif // mongo_dataTypes_PulseDetectionMetrics_H_
