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

#include "BatchMetadata.h"
#include "BatchData.h"
#include "BatchVectors.h"

namespace PacBio {
namespace Mongo {
namespace Data {

// A type stub for representing the sundry basecalling and trace metrics for a
// single ZMW over a single "metrics block" (i.e., metrics frame interval).
// Will be modeled, to some degree, after PacBio::Primary::BasecallingMetrics.
class BasecallingMetrics
{
    // TODO: Fill in the details of class BasecallingMetrics.
public:
    using Basecall = PacBio::SmrtData::Basecall;

    BasecallingMetrics() = default;

    BasecallingMetrics& Count(const Basecall& base);

public:
    const Cuda::Utility::CudaArray<uint8_t,4> NumBasesByAnalog() const
    { return numBasesByAnalog_; }

    const Cuda::Utility::CudaArray<uint8_t,4> NumPulsesByAnalog() const
    { return numPulsesByAnalog_; }

    uint16_t NumBases() const
    { return std::accumulate(numBasesByAnalog_.begin(), numBasesByAnalog_.end(), 0); }

    uint16_t NumPulses() const
    { return std::accumulate(numPulsesByAnalog_.begin(), numPulsesByAnalog_.end(), 0); }

public:
    Cuda::Utility::CudaArray<uint8_t,4>& NumBasesByAnalog()
    { return numBasesByAnalog_; }

    Cuda::Utility::CudaArray<uint8_t,4>& NumPulsesByAnalog()
    { return numPulsesByAnalog_; }

private:
    Cuda::Utility::CudaArray<uint8_t,4> numBasesByAnalog_;
    Cuda::Utility::CudaArray<uint8_t,4> numPulsesByAnalog_;
};


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
                  std::shared_ptr<Cuda::Memory::DualAllocationPools> lenPool,
                  std::shared_ptr<Cuda::Memory::DualAllocationPools> metricsPool);

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

    Cuda::Memory::UnifiedCudaArray<BasecallingMetrics>& Metrics() { return metrics_; }
    const Cuda::Memory::UnifiedCudaArray<BasecallingMetrics>& Metrics() const { return metrics_; }

private:    // Data
    BatchDimensions dims_;
    BatchMetadata   metaData_;
    BatchVectors<Basecall> basecalls_;

    // Metrics per ZMW. Size is dims_.zmwsPerBatch() or 0.
    Cuda::Memory::UnifiedCudaArray<BasecallingMetrics> metrics_;
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
        , metricsPool_(std::make_shared<Pools>(batchDims.ZmwsPerBatch()*sizeof(BasecallingMetrics), pinned))
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
                lenPool_,
                metricsPool_);
    }

private:
    size_t maxCallsPerZmw_;
    BatchDimensions batchDims_;
    Cuda::Memory::SyncDirection syncDir_;
    bool pinned_;

    std::shared_ptr<Cuda::Memory::DualAllocationPools> callsPool_;
    std::shared_ptr<Cuda::Memory::DualAllocationPools> lenPool_;
    std::shared_ptr<Cuda::Memory::DualAllocationPools> metricsPool_;
};

}}}     // namespace PacBio::Mongo::Data

#endif // mongo_dataTypes_BasecallBatch_H_
