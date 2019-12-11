// Copyright (c) 2014-2016, Pacific Biosciences of California, Inc.
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

// Programmer: Armin Töpfer

#pragma once

#include <pacbio/primary/SmartMemory.h>
#include <pacbio/primary/MetricBlock.h>
#include <pacbio/primary/MemoryBuffer.h>

namespace PacBio {
namespace Primary {

/// Contains packet and metric byte streams and the respective sizes.
template <typename TMetric>
struct ZmwSlice
{
public:
    ZmwSlice(const Primary::MemoryBufferView<uint8_t>& packetsArg,
             const Primary::MemoryBufferView<TMetric>& hfmbsArg) :
        packets(packetsArg),
        hfmbs(hfmbsArg) {};

    // Necessary to have an std::vector that is grown to full size before populating
    ZmwSlice() = default;
    
    ZmwSlice(const ZmwSlice&) = default;
    ZmwSlice(ZmwSlice&& src) = default;
    ZmwSlice& operator=(const ZmwSlice&) = default;
    ZmwSlice& operator=(ZmwSlice&& src) = default;

    ~ZmwSlice() {}

public:
    // Note, the size parameter in this object may not agree with
    // packetsByteSize * numEvents.  numEvents is the true number of pulses/bases
    // in this packet
    Primary::MemoryBufferView<uint8_t> packets; // BasePacket or PulsePacket byte stream
    Primary::MemoryBufferView<TMetric> hfmbs; // HighFreqMetricBlock
    uint32_t packetsByteSize;
    uint32_t zmwIndex; // maximal PrimaryToBaz::maximalNumZmws
    uint16_t numEvents;

public:
    inline ZmwSlice& NumEvents(uint16_t num)
    { ZmwSlice::numEvents = num; return *this; }

    inline ZmwSlice& PacketsByteSize(uint16_t packetsSize) 
    { ZmwSlice::packetsByteSize = packetsSize; return *this; }

    inline ZmwSlice& ZmwIndex(uint32_t zmwIdx)
    { ZmwSlice::zmwIndex = zmwIdx; return *this; }

public:
    const Primary::MemoryBufferView<uint8_t>& Packets() const
    { return packets; }

    const Primary::MemoryBufferView<TMetric>& Hfmbs() const
    { return hfmbs; }

    uint16_t NumEvents() const
    { return numEvents; }

    uint16_t PacketsByteSize() const
    { return packetsByteSize; }

    uint32_t ZmwIndex() const
    { return zmwIndex; }
};

}}
