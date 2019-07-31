#ifndef mongo_dataTypes_BasecallBatch_H_
#define mongo_dataTypes_BasecallBatch_H_

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
//  Defines class BasecallBatch.

#include <array>
#include <numeric>
#include <vector>
#include <pacbio/smrtdata/Basecall.h>

#include <common/cuda/utility/CudaArray.h>
#include <common/cuda/memory/UnifiedCudaArray.h>

#include "BasecallingMetrics.h"
#include "BatchMetadata.h"
#include "BatchData.h"
#include "BatchVectors.h"

namespace PacBio {
namespace Mongo {
namespace Data {


/// BasecallBatch represents the results of basecalling, a sequence of basecalls
/// for each ZMW. Each basecall includes metrics. Some instances will also
/// include pulse and trace metrics.
/// Memory is allocated only at construction. No reallocation during the life
/// of an instance.
class BasecallBatch
{
public:     // Types
    using Basecall = PacBio::SmrtData::Basecall;

public:     // Structors & assignment operators
    BasecallBatch(const size_t maxCallsPerZmwChunk,
                  const BatchDimensions& batchDims,
                  const BatchMetadata& batchMetadata,
                  Cuda::Memory::SyncDirection syncDir,
                  bool pinned,
                  std::shared_ptr<Cuda::Memory::DualAllocationPools> callsPool,
                  std::shared_ptr<Cuda::Memory::DualAllocationPools> lenPool);

    BasecallBatch(const BasecallBatch&) = delete;
    BasecallBatch(BasecallBatch&&) = default;

    BasecallBatch& operator=(const BasecallBatch&) = delete;
    BasecallBatch& operator=(BasecallBatch&&) = default;

    ~BasecallBatch() = default;

public:     // Functions
    const BatchMetadata& GetMeta() const
    { return metaData_; }

    const BatchDimensions& Dims() const
    { return dims_; }

    BatchVectors<Basecall>& Basecalls() { return basecalls_; }
    const BatchVectors<Basecall>& Basecalls() const { return basecalls_; }

    bool HasMetrics() const
    { return metrics_.get() != nullptr; }

    // Safety first: call HasMetrics before you dig
    //Cuda::Memory::UnifiedCudaArray<BasecallingMetrics<laneSize>>& Metrics()
    //{ return *(metrics_.get()); }

    // Safety first: call HasMetrics before you dig
    const Cuda::Memory::UnifiedCudaArray<BasecallingMetrics<laneSize>>& Metrics() const
    { return *(metrics_.get()); }

    void Metrics(std::unique_ptr<Cuda::Memory::UnifiedCudaArray<BasecallingMetrics<laneSize>>> metrics)
    {
        metrics_ = std::move(metrics);
    }

private:    // Data
    BatchDimensions dims_;
    BatchMetadata   metaData_;
    BatchVectors<Basecall> basecalls_;

    // Metrics per ZMW. Size is dims_.zmwsPerBatch() or nullptr.
    std::unique_ptr<Cuda::Memory::UnifiedCudaArray<BasecallingMetrics<laneSize>>> metrics_;
};

class BasecallBatchFactory
{
    using Pools = Cuda::Memory::DualAllocationPools;
    using Basecall = PacBio::SmrtData::Basecall;
public:
    BasecallBatchFactory(const size_t maxCallsPerZmw,
                         const BatchDimensions& batchDims,
                         Cuda::Memory::SyncDirection syncDir,
                         bool pinned)
        : maxCallsPerZmw_(maxCallsPerZmw)
        , batchDims_(batchDims)
        , syncDir_(syncDir)
        , pinned_(pinned)
        , callsPool_(std::make_shared<Pools>(maxCallsPerZmw*batchDims.ZmwsPerBatch()*sizeof(Basecall), pinned))
        , lenPool_(std::make_shared<Pools>(batchDims.ZmwsPerBatch()*sizeof(uint32_t), pinned))
    {}

    BasecallBatch NewBatch(const BatchMetadata& batchMetadata)
    {
        return BasecallBatch(
                maxCallsPerZmw_,
                batchDims_,
                batchMetadata,
                syncDir_,
                pinned_,
                callsPool_,
                lenPool_);
    }

private:
    size_t maxCallsPerZmw_;
    BatchDimensions batchDims_;
    Cuda::Memory::SyncDirection syncDir_;
    bool pinned_;

    std::shared_ptr<Cuda::Memory::DualAllocationPools> callsPool_;
    std::shared_ptr<Cuda::Memory::DualAllocationPools> lenPool_;
};

}}}     // namespace PacBio::Mongo::Data

#endif // mongo_dataTypes_BasecallBatch_H_
