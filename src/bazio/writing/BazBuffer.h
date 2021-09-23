// Copyright (c) 2021, Pacific Biosciences of California, Inc.
//
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted (subject to the limitations in the
// disclaimer below) provided that the following conditions are met:
//
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//
//  * Redistributions in binary form must reproduce the above
//    copyright notice, this list of conditions and the following
//    disclaimer in the documentation and/or other materials provided
//    with the distribution.
//
//  * Neither the name of Pacific Biosciences nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE
// GRANTED BY THIS LICENSE. THIS SOFTWARE IS PROVIDED BY PACIFIC
// BIOSCIENCES AND ITS CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
// OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL PACIFIC BIOSCIENCES OR ITS
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
// USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
// OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
// SUCH DAMAGE.

#ifndef PACBIO_BAZIO_WRITING_BAZ_BUFFER_H
#define PACBIO_BAZIO_WRITING_BAZ_BUFFER_H

#include <cassert>
#include <memory>

#include <bazio/MetricBlock.h>
#include <bazio/writing/GrowableArray.h>
#include <bazio/writing/MemoryBuffer.h>
#include <bazio/writing/PacketBufferManager.h>

namespace PacBio::BazIO {

class BazBuffer
{
    using TMetric = Primary::SpiderMetricBlock;

public:
    BazBuffer(uint32_t bufferId,
              MemoryBuffer<TMetric>&& metricsBuffer,
              std::vector<MemoryBufferView<TMetric>>&& metrics,
              std::unique_ptr<const PacketBufferManager> packets)
        : numZmw_(metrics.size())
        , bufferId_(bufferId)
        , metricsBuffer_(std::move(metricsBuffer))
        , metrics_(std::move(metrics))
        , packets_(std::move(packets))
    {
        assert(numZmw_ == packets_->NumZmw());
    }

    size_t NumZmw() const { return numZmw_; }
    uint32_t BufferId() const { return bufferId_; }

    struct Slice
    {
        const MemoryBufferView<TMetric>& metrics;
        PacketBufferManager::PacketSlice packets;
    };
    Slice GetSlice(size_t zmw) const { return Slice {metrics_[zmw], packets_->GetSlice(zmw)}; };

    ~BazBuffer() = default;

private:
    size_t numZmw_;
    uint32_t bufferId_;

    MemoryBuffer<TMetric> metricsBuffer_;
    std::vector<MemoryBufferView<TMetric>> metrics_;

    std::unique_ptr<const PacketBufferManager> packets_;
};

}  // namespace PacBio::BazIO

#endif  // PACBIO_BAZIO_WRITING_BAZ_BUFFER_H
