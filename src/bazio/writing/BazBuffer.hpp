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

#ifndef PACBIO_BAZIO_WRITING_BAZ_BUFFER_HPP
#define PACBIO_BAZIO_WRITING_BAZ_BUFFER_HPP

#include <bazio/writing/BazBuffer.h>

namespace PacBio {
namespace BazIO {

template <typename Iterator, typename Serializer>
void BazBuffer::AddZmw(size_t zmw, Iterator begin, Iterator end, Serializer&& s)
{
    assert(slices_.size() > zmw);

    auto& slice = slices_[zmw];
    auto* packets = &slices_[zmw].piece;
    while (packets->next != nullptr) packets = packets->next;

    // Need to make a copy of the serializer, since it is stateful,
    // and calling BytesRequired will alter that state.
    //
    // This is definitely awkward and error prone and could be
    // improved, though at this point I'm waiting until we see
    // how this ends up being ported to the GPU first.  In this
    // implementation calls to BytesRequired are interleaved
    // between calls to Serialize.  On the GPU however,
    // BytesRequired is likely to either be unecessary, because
    // a single thread is working alone in a buffer guaranteed to
    // be large enough, or all the calls to BytesRequired will
    // happen up front so multiple threads can share the same
    // buffer and can coordinate as to where they all start writing.
    // I don't see much point in optimizing this API until the
    // necessary usage patterns are settled.
    auto sCopy = s;

    for(auto itr = begin; itr < end; ++itr)
    {
        size_t numBytes = sCopy.BytesRequired(*itr);
        if (packets->endIdx + numBytes > packets->data_.size())
        {
            packets_.push_back(PacketPiece{});
            packets->next = &packets_.back();
            packets = packets->next;
            packets->data_ = packetBuffer_.Allocate(expectedPulseBufferSize_);
        }
        auto* in = & packets->data_[packets->endIdx];
        s.Serialize(*itr, in);
        packets->endIdx += numBytes;
        numEvents_++;
        slice.packetsByteSize += numBytes;
        slice.numEvents++;
    }
}

template <typename MetricsConverter>
void BazBuffer::AddMetrics(size_t zmw, MetricsConverter&& metricsConverter, size_t numMetrics)
{
    assert(slices_.size() > zmw);

    auto& m = slices_[zmw].metric;
    m = metricsBuffer_.Allocate(numMetrics);
    metricsConverter(m);
}

}}

#endif //PACBIO_BAZIO_WRITING_BAZ_BUFFER_HPP
