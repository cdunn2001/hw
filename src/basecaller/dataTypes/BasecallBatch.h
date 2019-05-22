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

#include "BatchMetadata.h"

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

    BasecallingMetrics();

    BasecallingMetrics& PushBack(const Basecall& base);

public:
    const std::array<uint16_t,4> NumBasesByAnalog() const
    { return numBasesByAnalog_; }

    const std::array<uint16_t,4> NumPulsesByAnalog() const
    { return numPulsesByAnalog_; }

    uint32_t NumBases() const
    { return std::accumulate(numBasesByAnalog_.begin(), numBasesByAnalog_.end(), 0); }

    uint32_t NumPulses() const
    { return std::accumulate(numPulsesByAnalog_.begin(), numPulsesByAnalog_.end(), 0); }

public:
    std::array<uint16_t,4>& NumBasesByAnalog()
    { return numBasesByAnalog_; }

    std::array<uint16_t,4>& NumPulsesByAnalog()
    { return numPulsesByAnalog_; }

private:
    std::array<uint16_t,4> numBasesByAnalog_;
    std::array<uint16_t,4> numPulsesByAnalog_;
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
    BasecallBatch() = default;

    BasecallBatch(const size_t maxCallsPerZmwChunk,
                  const BatchDimensions& batchDims,
                  const BatchMetadata& batchMetadata);

    BasecallBatch(const BasecallBatch&) = delete;
    BasecallBatch(BasecallBatch&&) = default;

    BasecallBatch& operator=(const BasecallBatch&) = delete;
    BasecallBatch& operator=(BasecallBatch&&) = default;

    ~BasecallBatch() = default;

private:    // Types
    // TODO: Should we use Cuda::Memory::UnifiedCudaArray?
    template <typename T>
    using ArrayType = std::vector<T>;

public:     // Functions
    const BatchMetadata& GetMeta() const
    { return metaData_; }

    const BatchDimensions& Dims() const
    { return dims_; }

    const ArrayType<Basecall>& Basecalls() const
    { return basecalls_; }

    const ArrayType<uint32_t>& SeqLengths() const
    { return seqLengths_; }

    const ArrayType<PacBio::Mongo::Data::BasecallingMetrics>& Metrics() const
    { return metrics_; }

public:    // Functions
    // Offset into basecalls_ for the start of the read for the z-th ZMW of
    // the batch.
    size_t zmwOffset(size_t z) const
    {
        assert(z < dims_.zmwsPerBatch());
        return z * maxCallsPerZmwChunk_;
    }

    // Offset into basecalls_ to the start of the read for the first ZMW of
    // lane l.
    size_t laneOffset(size_t l) const
    {
        assert(l < dims_.lanesPerBatch);
        return l * dims_.laneWidth * maxCallsPerZmwChunk_;
    }

public:   // Modifying methods
    void PushBack(uint32_t z, Basecall bc);

private:    // Data
    BatchDimensions dims_;
    size_t maxCallsPerZmwChunk_;
    BatchMetadata   metaData_;
    ArrayType<uint32_t> seqLengths_;        // Length of each zmw-read segment.
    ArrayType<Basecall> basecalls_;         // Storage for zmw-read segments.

    // Metrics per ZMW. Size is dims_.zmwsPerBatch() or 0.
    ArrayType<PacBio::Mongo::Data::BasecallingMetrics> metrics_;
};

}}}     // namespace PacBio::Mongo::Data

#endif // mongo_dataTypes_BasecallBatch_H_
