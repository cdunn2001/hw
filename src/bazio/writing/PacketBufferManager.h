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

#ifndef PACBIO_PACKETIO_WRITING_PACKET_BUFFER_MANAGER_H
#define PACBIO_PACKETIO_WRITING_PACKET_BUFFER_MANAGER_H

#include <cstddef>

#include <bazio/writing/PacketBuffer.h>

namespace PacBio::BazIO {

/// Class that manages pulse data as they incrementally stream in.
/// You can gather as much data as you like, and when desired you can
/// call `NewCheckpoint` which will return a copy of the existing state
/// and the current instance will reset itself for further data aggregation.
///
/// The PacketBufferManager has the concept of "HQ" vs "preHQ" zmw, where with
/// the latter we wish to accrue data, but maybe not output it to disk until
/// later when we are reasonably certain the data belongs to an HQ region. Data
/// for "preHQ" zmw are stored separately, in a vector of buffers where each
/// buffer corresponds to the interval between previous calls to ProduceBazBuffer.
/// When the current instance resets itself after a call to `NewCheckpoint` it
/// will:
///  * Update bookkeeping to migrate "preHQ" data that has been marked as newly
///    HQ to the HQ buffer
///  * The HQ buffers will be reset to an empty state
///  * A new preHQ buffer will be created for incoming data
///  * If there are more historical preHQ buffers than a pre-configured maximum,
///    the oldest will be discarded.
class PacketBufferManager
{
private:
    // We'll reserve memory for this many ZMW at a time.  Too large, and any
    // partially filled allocations will potentially greatly impact our overal
    // memory consumption.  Too small and our allocations will be fragmented.
    // If you go to the extreme and set it as 1 then the space overhead of the
    // pointers will be a non-negligable overhead to the actual data payload
    // itself, since a pointer and a size is 16 bytes, and a single ZMW is
    // probably ~320 bytes.
    static constexpr size_t zmwBatchingSize = 4096;

private:
    // An odd "copy" constructor, where we want to copy the contents of most of the
    // other CircularPacketBuffer, with the exception of the hqPackets, which are
    // moved out and we take posession of that data.  The moved-from hqPackets_
    // will need to be re-initialized after this call, which is why this is
    // a private ctor only
    PacketBufferManager(PacketBufferManager& other);

public:
    /// Simplified constructor, predominantly meant for simulation/test execution where
    /// we don't care about HQ vs preHQ data.  All data is marked as HQ from the onset,
    /// and the lookback functionality is disabled.
    ///
    /// \param numZmw    The number of ZMW handled by this manager
    /// \param expectedPulseBytesPerZmw The number of bytes to reserve for each ZMW
    ///                                 initially.  If this allotment is fully
    ///                                 consumed by any particular ZMW, then another
    ///                                 allocation of the same size will be reserved for it.
    PacketBufferManager(size_t numZmw, size_t expectedPulseBytesPerZmw);

    /// Primary constructor, where all ZMW start off as "preHQ" until they are
    /// explicitly marked otherwise
    ///
    /// \param numZmw    The number of ZMW handled by this manager
    /// \param expectedPulseBytesPerZmw The number of bytes to reserve for each zmw
    ///                                 initially.  If this allotment is fully
    ///                                 consumed by any particular ZMW, then
    ///                                 another allocation of the same size will
    ///                                 be reserved for it.
    /// \param maxLookback The number of data buffers we retain before dropping
    ///                    data. Every call to `ProduceBazFile` will create a new
    ///                    data buffer
    /// \param allocator   An IAllocator instance to use for allocating memory
    PacketBufferManager(size_t numZmw,
                        size_t expectedPulseBufferSize,
                        size_t maxLookback,
                        std::shared_ptr<Memory::IAllocator> allocator);

    std::unique_ptr<const PacketBufferManager> CreateCheckpoint();

    /// \return the number of ZMW belonging to this manager
    size_t NumZmw() const { return indexInfo_.size(); }
    /// \return the number of ZMW that are marked as HQ
    size_t NumHQ() const { return numHQ_; }

    /// Struct that exposes all of the existing data for a given ZMW.
    /// This is created upon demand, internal bookkeeping is done
    /// more compactly
    struct PacketSlice
    {
        size_t packetByteSize = 0;
        size_t numEvents = 0;
        struct Piece
        {
            const uint8_t* data;
            size_t count;
        };
        std::vector<Piece> pieces;
    };
    /// \return Full data for a given ZMW.  If the ZMQ requested is not HQ,
    ///         either upon initialization or via a call to `MarkAsHQ`, then
    ///         the returned data will be empty
    PacketSlice GetSlice(size_t zmw) const;

    /// Adds ZMW packet data to the current buffers
    ///
    /// \param zmw   The zmw index of the data
    /// \param begin An iterator to the first pulse to be serialized
    /// \param end   An iterator to one-past-the-end for the pulses to be serialized
    /// \predicate   A boolean functor to determine if a given pulse should be
    ///              serialized (e.g. to skip pulses not marked as bases)
    /// \serializer  A serializer that can convert the pulse to a byte stream
    template <typename Iterator, typename Predicate, typename Serializer>
    void AddZmw(size_t zmw, Iterator begin, Iterator end, Predicate&& predicate, Serializer&& serializer)
    {
        auto& buffer = indexInfo_[zmw].inHQBuffer_ ? hqPackets_ : *lookbackData_.back();
        auto result = buffer.AddZmw(indexInfo_[zmw].recentIndex_,
                                    begin,
                                    end,
                                    std::forward<Predicate>(predicate),
                                    std::forward<Serializer>(serializer));
        indexInfo_[zmw].recentIndex_ = result.nextIndex;
    }

    /// \return if a ZMW is considered HQ or not
    bool IsHQ(size_t zmw) const { return indexInfo_[zmw].inHQBuffer_ || newHQ_[zmw]; }

    /// Marks a ZMW as being HQ
    void MarkAsHQ(size_t zmw)
    {
        // Not 100% sure if this should be an error, but it's
        // at least strange to mark a ZMW as HQ when it's
        // already marked as such.
        if(indexInfo_[zmw].inHQBuffer_) return;
        if(newHQ_[zmw]) return;
        numHQ_++;
        newHQ_[zmw] = true;
    }

private:
    // struct to help keep track of where the data for a given ZMW resides.
    struct IndexInfo
    {
        // Does data live in `hqPackets` or `lookbackData_`?
        bool inHQBuffer_;
        // The PacketBuffer index for the start of data, either
        // in the hqPackets buffer, or the *first* buffer in
        // loockbackData_
        uint32_t latentIndex_;
        // The PacketBuffer index for the last (most recent)
        // snippet of data.  It resides either  in the hqPackets
        // buffer, or the *first* buffer in loockbackData_
        uint32_t recentIndex_;
    };

    size_t numHQ_;
    size_t expectedPulseBytesPerZmw_;
    size_t maxLookback_;

    std::shared_ptr<Memory::IAllocator> allocator_;

    PacketBuffer hqPackets_;
    std::vector<std::shared_ptr<PacketBuffer>> lookbackData_;
    std::vector<IndexInfo> indexInfo_;
    std::vector<bool> newHQ_;
};

}  // namespace PacBio::BazIO

#endif  // PACBIO_PACKETIO_WRITING_PACKET_BUFFER_MANAGER_H
