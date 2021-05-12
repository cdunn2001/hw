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

#include <bazio/MetricBlock.h>
#include <bazio/writing/MemoryBuffer.h>

namespace PacBio {
namespace BazIO {

/// Stores ZmwSlices and the overall number of events and metric blocks.
struct BazBuffer
{
    using TMetric = Primary::SpiderMetricBlock;
    struct PacketPiece
    {
        MemoryBufferView<uint8_t> data_;
        uint32_t endIdx = 0;
        PacketPiece* next = nullptr;
    };
    struct Slice
    {
        Slice(size_t idx, PacketPiece& p)
            : zmwIdx{idx}
            , piece{p}
        {}
        size_t zmwIdx;
        size_t packetsByteSize = 0;
        size_t numEvents = 0;
        PacketPiece& piece;
        MemoryBufferView<TMetric> metric;
    };
public:
    BazBuffer(size_t numZmw, size_t expectedPulseBufferSize,
              std::unique_ptr<Memory::IAllocator> allocator = std::make_unique<DataSource::MallocAllocator>())
        : numZmw_(numZmw)
        , expectedPulseBufferSize_(expectedPulseBufferSize)
        , allocator_(std::move(allocator))
          // Hard coded 1MiB for allocation buffer size is not tuned. Feel
          // free to re-evaluate.
        , packetBuffer_(1<<20, 1, *allocator_)
        , metricsBuffer_(1<<20, 1, *allocator_)
    {
        Reset();
    }

    template <typename Iterator, typename Serializer>
    void AddZmw(size_t zmw, Iterator begin, Iterator end, Serializer&& s);

    template <typename MetricsConverter>
    void AddMetrics(size_t zmw, MetricsConverter&& m, size_t numMetrics);

    ~BazBuffer() = default;

    const std::vector<Slice>& ZmwData() const
    {
        return slices_;
    }

    uint64_t NumEvents() const
    {
        return numEvents_;
    }

    void Reset()
    {
        packetBuffer_.Reset();
        metricsBuffer_.Reset();

        numEvents_ = 0;
        packets_ = std::deque<PacketPiece>(numZmw_);
        size_t counter = 0;
        for (auto& p : packets_)
        {
            p.data_ = packetBuffer_.Allocate(expectedPulseBufferSize_);
            slices_.emplace_back(counter, p);
            counter++;
        }
    }

private:
    uint64_t numEvents_ = 0;
    size_t numZmw_;
    size_t expectedPulseBufferSize_;
    std::unique_ptr<Memory::IAllocator> allocator_;
    MemoryBuffer<uint8_t> packetBuffer_;
    MemoryBuffer<TMetric> metricsBuffer_;

    // One entry per ZMW is ideal, but the PacketPiece
    // does have a way to point to another entry, if we
    // need to handle some overflow
    std::deque<PacketPiece> packets_;
    // One entry per ZMW is expected
    std::vector<Slice> slices_;
};

}} // namespace

#endif //PACBIO_BAZIO_WRITING_BAZ_BUFFER_H
