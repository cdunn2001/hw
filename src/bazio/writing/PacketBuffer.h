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

#ifndef PACBIO_PACKETIO_WRITING_PACKET_BUFFER_H
#define PACBIO_PACKETIO_WRITING_PACKET_BUFFER_H

#include <cstdint>

#include <pacbio/memory/IAllocator.h>
#include <pacbio/datasource/MallocAllocator.h>
#include <pacbio/PBException.h>

#include <bazio/writing/GrowableArray.h>

namespace PacBio {
namespace BazIO {


/// Class used to store pulse data as it is streamed in for a collection of
/// ZMW.  Data is intended to be added incrementally, and the buffer will
/// expand as required.  Be aware that data for a given ZMW is *not*
/// necessarily stored contiguously, but the parameters used for construction
/// can be tuned to help promote as much contiguous storage as possible.
///
/// Note: This class uses some fuggly and somewhat brittle bookkeeping to
///       keep track of ad ad-hoc linked list.  This is because the intended
///       use case involves literally managing hundres of millions of tiny
///       (e.g. 300 byte) pieces of memory.  Early and naive forms of bookkeeping
///       were more robust, but ended up with something like a 20% memory overhead.
///       The current bookkeeping overhead is closer to 5% overhead.
class PacketBuffer
{
public:
    // Bookkeeping information to keep track of ZMW data as it potentially
    // grows beyond a single contiguoius snippet of memory.  This is admittedly
    // brittle, and focused more on space savings than ease of use.  External
    // code should preferably be interacting instead with PacketBufferManager
    // and the Slice interface it exposes.
    struct PieceInfo
    {
        PieceInfo()
            : nextIdx(0)
            , isNextExternal(false)
        {}
        uint16_t packetsByteSize = 0;
        uint16_t numEvents = 0;
        // Sets up an ad-hoc linked list while saving a few
        // bytes.  nextIdx is the next index in the chain.
        // If it's 0 *and* `isNextExternal` is false, then
        // it is the end of the chain.
        uint32_t nextIdx : 31;
        // Indicates if the next piece is in this same buffer
        // or an external buffer
        bool isNextExternal : 1;
    };
    static_assert(sizeof(PieceInfo) == 8);

public:
    /// Constructs a buffer for a collection of ZMW, composed of fixed-size
    /// memory snippets.  If a given ZMW uses more than a single snippet,
    /// more memory will be reserved for it, and the data will be stitched
    /// together via an sort of ad-hoc linked list.
    ///
    /// \param numZmw The number of ZMW int his buffer
    /// \param batchSize The number of ZMW to group together into a single
    ///                  allocation.  If it's too large, then a partially
    ///                  filled allocation may noticeably impact overal
    ///                  memory usage.  If it's too small then we lose the
    ///                  benefits of streaming over large contiguous allocations.
    ///                  If it's trivially small, and snippetLen is also small,
    ///                  then the overhead of the pointers involved may also
    ///                  impact the overal memory consumption
    /// \param snippetLen The number of bytes reserved at a time for each ZMW.
    ///                   Ideally this is chosen to match the expected total
    ///                   payload size, and the use of additional snippets are
    ///                   for relatively rare and overly noisy ZMW.
    /// \param alloc  An IAllocator instance for provisioning memory
    PacketBuffer(size_t numZmw,
                 size_t batchSize,
                 size_t snippetLen,
                 std::shared_ptr<Memory::IAllocator> alloc)
        : packetsData_(alloc, batchSize, snippetLen)
        , pieceData_(alloc, batchSize)
    {
        packetsData_.GrowToSize(numZmw);
        pieceData_.GrowToSize(numZmw);
    }

    struct AddResult
    {
        // Number of events added (e.g. possibly excluding pulses)
        size_t numEvents;
        // The index for the next bit of data insertion.  It may not
        // be the same as the input index, in case we filled the
        // current buffer and provisioned a new one
        size_t nextIndex;
    };
    /// Adds some ZMW data to a given buffer.
    /// \return The number of events actuall added, as well as the next
    ///         index to use, as we may have completely filled the
    ///         previous index
    ///
    /// \param index The index of data to fill
    /// \param begin A begin iterator to the incoming pulse data
    /// \param end   An end iterator to the incoming pulse data
    /// \param predicate A functor predicate, which can be used to filter
    ///                  out incoming pulse data
    /// \param serializer A Serializer type that can convert pulses to binary
    ///                   data
    template <typename Iterator, typename Predicate, typename Serializer>
    AddResult AddZmw(uint32_t index,
                     Iterator begin,
                     Iterator end,
                     Predicate&& predicate,
                     Serializer&& serializer)
    {
        assert(index < packetsData_.Size());
        assert(index < pieceData_.Size());

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
        auto sCopy = serializer;
        size_t numEvents = 0;

        auto currIndex = index;
        auto* currPiece = pieceData_[index];
        uint8_t* data = packetsData_[index];

        assert(currPiece->nextIdx == 0);
        assert(!currPiece->isNextExternal);

        for(auto itr = begin; itr < end; ++itr)
        {
            if (!predicate(*itr)) continue;

            size_t numBytes = sCopy.BytesRequired(*itr);
            if (currPiece->packetsByteSize + numBytes > packetsData_.ElementLen())
            {
                currIndex = packetsData_.Size();
                currPiece->nextIdx = currIndex;
                currPiece = pieceData_.AppendOne();
                data = packetsData_.AppendOne();

                assert(currPiece->packetsByteSize == 0);
                assert(currPiece->numEvents == 0);
                assert(currPiece->nextIdx == 0);
                assert(currPiece->isNextExternal == false);
            }
            auto* in = data + currPiece->packetsByteSize;
            serializer.Serialize(*itr, in);
            currPiece->packetsByteSize += numBytes;
            currPiece->numEvents++;
            numEvents++;
        }
        return {numEvents, currIndex};
    }


    /// Creates an implicit and external link between one index
    /// in this buffer with an index in a separate buffer.
    /// This is used to continue the ad-hoc linked-list
    /// data structure for a given ZMW across multiple PacketBuffer
    /// instances
    ///
    /// It is brittle, but it's currently the responsibility of
    /// external code to keep track of exactly what that *other*
    /// buffer is
    ///
    /// \param thisIdx The index in *this* buffer
    /// \param nextIdx The index in the *other* buffer
    void Link(uint32_t thisIdx, uint32_t nextIdx)
    {
        assert(pieceData_[thisIdx]->nextIdx == 0);
        assert(pieceData_[thisIdx]->isNextExternal == false);
        pieceData_[thisIdx]->nextIdx = nextIdx;
        pieceData_[thisIdx]->isNextExternal = true;
    }

    /// Accepts an index and does a forward traversal of the
    /// ad-hoc linked list until it finds an index belonging
    /// to the next PacketBuffer.  If there is no "next PacketBuffer"
    /// (i.e. you call this on the most recent PacketBuffer) then
    /// it will throw.
    uint32_t FindNextExternalIndex(uint32_t idx) const
    {
        const PieceInfo* piece = pieceData_[idx];
        while (!piece->isNextExternal)
        {
            if(piece->nextIdx == 0)
                throw PBException("Broken/Corrupted bookkeeping in PacketBuffer");
            piece = pieceData_[piece->nextIdx];
        }
        return piece->nextIdx;
    }

    const uint8_t* Data(size_t idx) const
    {
        return packetsData_[idx];
    }
    const PieceInfo* Piece(size_t idx) const
    {
        return pieceData_[idx];
    }

private:
    GrowableArray<uint8_t> packetsData_;
    GrowableArray<PieceInfo> pieceData_;
};

}}

#endif //PACBIO_PACKETIO_WRITING_PACKET_BUFFER_H
