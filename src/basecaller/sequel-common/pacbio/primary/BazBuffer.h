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

#include <pacbio/primary/MemoryBuffer.h>
#include <pacbio/primary/ZmwSlice.h>

namespace PacBio {
namespace Primary {

/// Stores ZmwSlices and the overall number of events and metric blocks.
template <typename TMetric>
struct BazBuffer
{
public:
    BazBuffer(size_t numZmw, size_t numMetrics, size_t initialBytes, size_t finalBytes)
      : zmwData_(numZmw)
      , packetBuffer_(initialBytes, finalBytes)
      , metricsBuffer_(numZmw * numMetrics)
    {}
    BazBuffer(size_t numZmw, size_t numMetrics, size_t numBytes)
      : zmwData_(numZmw)
      , packetBuffer_(numBytes)
      , metricsBuffer_(numMetrics)
    {}

    // Move constructor
    BazBuffer(BazBuffer&&) = default;
    // Copy constructor
    BazBuffer(const BazBuffer&) = default;
    // Move assignment operator
    BazBuffer& operator=(BazBuffer&& rhs)  = default;
    // Copy assignment operator
    BazBuffer& operator=(const BazBuffer&) = default;

    ~BazBuffer() = default;

    size_t PacketDataSize() const { return packetBuffer_.Size(); }
    size_t MetricsDataSize() const { return metricsBuffer_.Size(); }

    void AddZmwSlice(const ZmwSlice<TMetric>& zmwSlice, uint8_t hFbyMFRatio, uint8_t hFbyLFRatio)
    {
        numEvents_ += zmwSlice.numEvents;
        numHFMBs_ += zmwSlice.hfmbs.size();
        if (hFbyMFRatio > 0)
            numMFMBs_ += zmwSlice.hfmbs.size() / hFbyMFRatio;
        if (hFbyLFRatio > 0)
            numLFMBs_ += zmwSlice.hfmbs.size() / hFbyLFRatio;
        if (zmwData_.size() <= zmwSlice.zmwIndex)
        {
            PBLOG_DEBUG << "Unexpected resize is required in BazBuffer...\n";
            zmwData_.resize(zmwSlice.zmwIndex+1);
        }
        zmwData_[zmwSlice.zmwIndex] = zmwSlice;
        zmwSeen_++;
    }

    Primary::MemoryBuffer<TMetric>& MetricsBuffer()
    {
        return metricsBuffer_;
    }

    Primary::MemoryBuffer<uint8_t>& PacketBuffer()
    {
        return packetBuffer_;
    }

    const std::vector<ZmwSlice<TMetric>>& ZmwData() const
    {
        return zmwData_;
    }

    uint64_t NumLFMBs() const
    {
        return numLFMBs_;
    }

    uint64_t NumMFMBs() const
    {
        return numMFMBs_;
    }

    uint64_t NumHFMBs() const
    {
        return numHFMBs_;
    }

    uint64_t NumEvents() const
    {
        return numEvents_;
    }

    bool SeenAllZmw() const
    {
        return zmwSeen_ == zmwData_.size();
    }

    size_t Size() const
    {
        return zmwSeen_;
    }

    void Reset()
    {
        numEvents_ = 0;
        numHFMBs_ = 0;
        numMFMBs_ = 0;
        numLFMBs_ = 0;
        zmwData_.resize(0);
        zmwSeen_ = 0;
        packetBuffer_.Reset();
        metricsBuffer_.Reset();
    }

private:
    uint64_t numEvents_ = 0;
    uint64_t numHFMBs_ = 0;
    uint64_t numMFMBs_ = 0;
    uint64_t numLFMBs_ = 0;
    std::vector<ZmwSlice<TMetric>> zmwData_;
    size_t zmwSeen_ = 0;
    Primary::MemoryBuffer<uint8_t> packetBuffer_;
    Primary::MemoryBuffer<TMetric> metricsBuffer_;
};

}} // namespace
