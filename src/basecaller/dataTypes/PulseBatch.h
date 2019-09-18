#ifndef mongo_dataTypes_PulseBatch_H_
#define mongo_dataTypes_PulseBatch_H_

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
//  Defines class PulseBatch.

#include "BatchData.h"
#include "BatchMetadata.h"
#include "BatchMetrics.h"
#include "BatchVectors.h"
#include "Pulse.h"

namespace PacBio {
namespace Mongo {
namespace Data {

class PulseBatch
{

public:     // Structors & assignment operators
    PulseBatch(const size_t maxCallsPerZmwChunk,
               const BatchDimensions& batchDims,
               const BatchMetadata& batchMetadata,
               Cuda::Memory::SyncDirection syncDir,
               const Cuda::Memory::AllocationMarker& marker)
        : dims_ (batchDims)
        , metaData_(batchMetadata)
        , pulses_(batchDims.ZmwsPerBatch(),  maxCallsPerZmwChunk, syncDir, marker)
    {}

    PulseBatch(const PulseBatch&) = delete;
    PulseBatch(PulseBatch&&) = default;

    PulseBatch& operator=(const PulseBatch&) = delete;
    PulseBatch& operator=(PulseBatch&&) = default;

    ~PulseBatch() = default;

public:     // Functions
    const BatchMetadata& GetMeta() const
    { return metaData_; }

    const BatchDimensions& Dims() const
    { return dims_; }

    BatchVectors<Pulse>& Pulses() { return pulses_; }
    const BatchVectors<Pulse>& Pulses() const { return pulses_; }

private:    // Data
    BatchDimensions dims_;
    BatchMetadata   metaData_;
    BatchVectors<Pulse> pulses_;
};

class PulseBatchFactory
{
public:
    PulseBatchFactory(const size_t maxCallsPerZmw,
                      const BatchDimensions& batchDims,
                      Cuda::Memory::SyncDirection syncDir)
        : maxCallsPerZmw_(maxCallsPerZmw)
        , batchDims_(batchDims)
        , syncDir_(syncDir)
    {}

    std::pair<PulseBatch, PulseDetectorMetrics>
    NewBatch(const BatchMetadata& batchMetadata) const
    {
        return std::make_pair(
            PulseBatch(
                maxCallsPerZmw_, batchDims_, batchMetadata, syncDir_, SOURCE_MARKER()),
            PulseDetectorMetrics(batchDims_, syncDir_, SOURCE_MARKER()));

    }

private:
    size_t maxCallsPerZmw_;
    BatchDimensions batchDims_;
    Cuda::Memory::SyncDirection syncDir_;
};

}}}     // namespace PacBio::Mongo::Data

#endif // mongo_dataTypes_PulseBatch_H_
