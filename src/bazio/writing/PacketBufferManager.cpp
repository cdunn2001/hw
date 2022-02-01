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

#include "PacketBufferManager.h"

namespace PacBio::BazIO {

PacketBufferManager::PacketBufferManager(PacketBufferManager& other)
    : numHQ_(other.numHQ_)
    , expectedPulseBytesPerZmw_(other.expectedPulseBytesPerZmw_)
    , maxLookback_(other.maxLookback_)
    , allocator_(other.allocator_)
    // Beware the unexpected move!
    , hqPackets_(std::move(other.hqPackets_))
    , lookbackData_(other.lookbackData_)
    , indexInfo_(other.indexInfo_)
    , newHQ_(other.newHQ_)
{}

PacketBufferManager::PacketBufferManager(size_t numZmw, size_t expectedPulseBytesPerZmw)
    : numHQ_(numZmw)
    , expectedPulseBytesPerZmw_(expectedPulseBytesPerZmw)
    , maxLookback_(0)
    , allocator_(std::make_shared<DataSource::MallocAllocator>())
    , hqPackets_(numZmw, zmwBatchingSize, expectedPulseBytesPerZmw, allocator_)
    , newHQ_(numZmw, false)
{
    indexInfo_.resize(numZmw);
    for (size_t zmw = 0; zmw < numZmw; ++zmw)
    {
        indexInfo_[zmw].inHQBuffer_ = true;
        indexInfo_[zmw].latentIndex_ = zmw;
        indexInfo_[zmw].recentIndex_ = zmw;
    }
}

PacketBufferManager::PacketBufferManager(size_t numZmw,
                                         size_t expectedPulseBufferSize,
                                         size_t maxLookback,
                                         std::shared_ptr<Memory::IAllocator> allocator)
    : numHQ_(0)
    , expectedPulseBytesPerZmw_(expectedPulseBufferSize)
    , maxLookback_(maxLookback)
    , allocator_(allocator)
    , hqPackets_(0, zmwBatchingSize, expectedPulseBufferSize, allocator_)
    , newHQ_(numZmw, false)
{
    assert(maxLookback > 0);
    lookbackData_.push_back(
        std::make_shared<PacketBuffer>(numZmw, zmwBatchingSize, expectedPulseBufferSize, allocator_));

    indexInfo_.resize(numZmw);
    for (size_t zmw = 0; zmw < numZmw; ++zmw)
    {
        indexInfo_[zmw].inHQBuffer_ = false;
        indexInfo_[zmw].latentIndex_ = zmw;
        indexInfo_[zmw].recentIndex_ = zmw;
    }
}

PacketBufferManager::PacketSlice PacketBufferManager::GetSlice(size_t zmw) const
{
    PacketSlice ret;
    bool alreadyHQ = indexInfo_[zmw].inHQBuffer_;
    if (!alreadyHQ && !newHQ_[zmw]) return ret;

    size_t lookbackIdx = 0;
    const PacketBuffer* buffer = alreadyHQ ? &hqPackets_ : lookbackData_[lookbackIdx].get();
    const uint8_t* ptr = buffer->Data(indexInfo_[zmw].latentIndex_);
    const auto* piece = buffer->Piece(indexInfo_[zmw].latentIndex_);
    while (true)
    {
        ret.pieces.emplace_back(PacketSlice::Piece {ptr, piece->packetsByteSize});
        ret.numEvents += piece->numEvents;
        ret.packetByteSize += piece->packetsByteSize;

        if (piece->nextIdx == 0 && !piece->isNextExternal)
            break;

        if (piece->isNextExternal)
        {
            assert(!alreadyHQ);
            lookbackIdx++;
            buffer = lookbackData_[lookbackIdx].get();
        }

        ptr = buffer->Data(piece->nextIdx);
        piece = buffer->Piece(piece->nextIdx);
    }
    return ret;
}

std::unique_ptr<const PacketBufferManager> PacketBufferManager::CreateCheckpoint()
{
    // Copies the contents of our current instance, with the exception of hqPackets_
    // that gets destructively moved.
    auto ret = std::unique_ptr<PacketBufferManager>(new PacketBufferManager(*this));

    hqPackets_ = PacketBuffer(numHQ_, zmwBatchingSize, expectedPulseBytesPerZmw_, allocator_);
    assert(indexInfo_.size() >= numHQ_);

    size_t activeIdx = 0;
    size_t lookbackIdx = 0;
    for (size_t zmw = 0; zmw < indexInfo_.size(); ++zmw)
    {
        if (indexInfo_[zmw].inHQBuffer_ || newHQ_[zmw])
        {
            indexInfo_[zmw].inHQBuffer_ = true;
            indexInfo_[zmw].latentIndex_ = activeIdx;
            indexInfo_[zmw].recentIndex_ = activeIdx;
            activeIdx++;
        }
        else
        {
            lookbackData_.back()->Link(indexInfo_[zmw].recentIndex_, lookbackIdx);

            indexInfo_[zmw].recentIndex_ = lookbackIdx;
            lookbackIdx++;

            if (lookbackData_.size() == maxLookback_)
            {
                // find the index of the first piece in the second
                // to oldest buffer, since the oldest is about to be
                // dropped on the floor
                uint32_t idx = indexInfo_[zmw].latentIndex_;
                idx = lookbackData_.front()->FindNextExternalIndex(idx);
                indexInfo_[zmw].latentIndex_ = idx;
            }
        }
    }

    if (maxLookback_ > 0)
    {
        lookbackData_.push_back(std::make_shared<PacketBuffer>(
            indexInfo_.size() - numHQ_, zmwBatchingSize, expectedPulseBytesPerZmw_, allocator_));
        if (lookbackData_.size() > maxLookback_)
            lookbackData_.erase(lookbackData_.begin());
    }

    newHQ_ = std::vector<bool>(indexInfo_.size(), false);
    return ret;
}

}  // namespace PacBio::BazIO
