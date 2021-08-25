// Copyright (c) 2021, Pacific Biosciences of California, Inc.
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

#include <gtest/gtest.h>

#include <bazio/writing/PacketBuffer.h>

#include <pacbio/datasource/MallocAllocator.h>

using namespace PacBio::BazIO;
using namespace PacBio::DataSource;

// Constructs an empty packet buffer.  Doesn't fill it with data,
// but we do try to check that the batchSize is respected and we get
// exactly one allocation
TEST(PacketBuffer, Construct)
{
    auto alloc = std::make_shared<MallocAllocator>();

    uint32_t numZmw = 10;
    uint32_t batchSize = 10;
    uint32_t snippetLen = 20;
    PacketBuffer buf(numZmw, batchSize, snippetLen, alloc);

    // Things are constructed such that everything should be in the same
    // allocation
    EXPECT_EQ(buf.Data(0) + (numZmw-1) * snippetLen, buf.Data(numZmw-1));
    EXPECT_EQ(buf.Piece(0) + (numZmw-1), buf.Piece(numZmw-1));
}

// Just serializes integers to a byte stream, in the fashion
// expected by the AddZmw function
struct DummySerializer
{
    size_t BytesRequired(int) { return 4; }
    uint8_t*  Serialize(int val, uint8_t* ptr)
    {
        memcpy(ptr, &val, sizeof(int));
        return ptr+sizeof(int);
    }
};

TEST(PacketBuffer, AddData)
{
    auto alloc = std::make_shared<MallocAllocator>();

    uint32_t numZmw = 10;
    uint32_t batchSize = 10;
    uint32_t snippetLen = 20;
    PacketBuffer buf(numZmw, batchSize, snippetLen, alloc);

    // We're going to write this data to ZMW 0 and 5.
    std::vector<int> data{1,2,3,4};

    auto [numEvents1, nextIdx1 ] = buf.AddZmw(0, data.begin(), data.end(),
                                              [](auto){return true;},
                                              DummySerializer{});
    EXPECT_EQ(numEvents1, 4);
    EXPECT_EQ(nextIdx1, 0);

    auto [numEvents2, nextIdx2 ] = buf.AddZmw(5, data.begin(), data.end(),
                                              [](auto){return true;},
                                              DummySerializer{});
    EXPECT_EQ(numEvents2, 4);
    EXPECT_EQ(nextIdx2, 5);

    // Make sure zmw 0 and 5 have the expected data,
    // and that the rest are still empty
    for (size_t i = 0; i < numZmw; ++i)
    {
        auto* piece = buf.Piece(i);
        if (i == 0 || i == 5)
        {
            EXPECT_EQ(piece->numEvents, 4);
            EXPECT_EQ(piece->packetsByteSize, 16);
            EXPECT_EQ(piece->nextIdx, 0);
            EXPECT_FALSE(piece->isNextExternal);

            const auto* data = reinterpret_cast<const int*>(buf.Data(i));
            EXPECT_EQ(data[0], 1);
            EXPECT_EQ(data[1], 2);
            EXPECT_EQ(data[2], 3);
            EXPECT_EQ(data[3], 4);
        } else {
            EXPECT_EQ(piece->numEvents, 0);
            EXPECT_EQ(piece->packetsByteSize, 0);
            EXPECT_EQ(piece->nextIdx, 0);
            EXPECT_FALSE(piece->isNextExternal);
        }
    }
}

// Make sure that the filtering predicate works as expected, so
// we'll do the same test as previously, but this time cutting out
// over other value
TEST(PacketBuffer, FilterData)
{
    auto alloc = std::make_shared<MallocAllocator>();

    uint32_t numZmw = 10;
    uint32_t batchSize = 10;
    uint32_t snippetLen = 20;
    PacketBuffer buf(numZmw, batchSize, snippetLen, alloc);

    std::vector<int> data{1,2,3,4};

    auto [numEvents1, nextIdx1 ] = buf.AddZmw(0, data.begin(), data.end(),
                                              [](auto val){return val % 2 == 0;},
                                              DummySerializer{});
    EXPECT_EQ(numEvents1, 2);
    EXPECT_EQ(nextIdx1, 0);

    auto [numEvents2, nextIdx2 ] = buf.AddZmw(5, data.begin(), data.end(),
                                              [](auto val){return val % 2 == 0;},
                                              DummySerializer{});
    EXPECT_EQ(numEvents2, 2);
    EXPECT_EQ(nextIdx2, 5);

    for (size_t i = 0; i < numZmw; ++i)
    {
        auto* piece = buf.Piece(i);
        if (i == 0 || i == 5)
        {
            // With every other value removed, only 2 of the 4 entries
            // made it into teh buffer
            EXPECT_EQ(piece->numEvents, 2);
            EXPECT_EQ(piece->packetsByteSize, 8);
            EXPECT_EQ(piece->nextIdx, 0);
            EXPECT_FALSE(piece->isNextExternal);

            const auto* data = reinterpret_cast<const int*>(buf.Data(i));
            EXPECT_EQ(data[0], 2);
            EXPECT_EQ(data[1], 4);
        } else {
            EXPECT_EQ(piece->numEvents, 0);
            EXPECT_EQ(piece->packetsByteSize, 0);
            EXPECT_EQ(piece->nextIdx, 0);
            EXPECT_FALSE(piece->isNextExternal);
        }
    }
}

// Make sure we can handle the situation where all the data for a ZMW
// does not fit in a single snippet.  We'll do the same as AddData test
// again, this time with two extra values so the last value has to
// go into a new buffer
TEST(PacketBuffer, OverflowData)
{
    auto alloc = std::make_shared<MallocAllocator>();

    uint32_t numZmw = 10;
    uint32_t batchSize = 10;
    uint32_t snippetLen = 20;
    PacketBuffer buf(numZmw, batchSize, snippetLen, alloc);

    // Only 1-5 will fit in the initial buffer
    std::vector<int> data{1,2,3,4,5,6};

    auto [numEvents1, nextIdx1 ] = buf.AddZmw(0, data.begin(), data.end(),
                                              [](auto){return true;},
                                              DummySerializer{});
    EXPECT_EQ(numEvents1, 6);
    // nextIdx gets set to a new entry at the end, because we overflowed
    EXPECT_EQ(nextIdx1, 10);

    auto [numEvents2, nextIdx2 ] = buf.AddZmw(5, data.begin(), data.end(),
                                              [](auto){return true;},
                                              DummySerializer{});
    EXPECT_EQ(numEvents2, 6);
    // nextIdx gets set one past the new entry we created for zmw 0.  Only
    // zmw that overflow get a new entry, we don't reserve overflow for all
    // zmw when a subset overflow.
    EXPECT_EQ(nextIdx2, 11);

    for (size_t i = 0; i < numZmw+2; ++i)
    {
        auto* piece = buf.Piece(i);
        if (i == 0 || i == 5)
        {
            EXPECT_EQ(piece->numEvents, 5);
            EXPECT_EQ(piece->packetsByteSize, 20);
            // This time our `nextIdx` will be different from our zmwIdx,
            // since we had an overflow.
            EXPECT_EQ(piece->nextIdx, i == 0 ? 10 : 11);
            EXPECT_FALSE(piece->isNextExternal);

            const auto* data = reinterpret_cast<const int*>(buf.Data(i));
            EXPECT_EQ(data[0], 1);
            EXPECT_EQ(data[1], 2);
            EXPECT_EQ(data[2], 3);
            EXPECT_EQ(data[3], 4);
            EXPECT_EQ(data[4], 5);
        } else if (i >= numZmw ) {

            // Check the overflow to make sure there is the one expected
            // element
            EXPECT_EQ(piece->numEvents, 1);
            EXPECT_EQ(piece->packetsByteSize, 4);
            EXPECT_EQ(piece->nextIdx, 0);
            EXPECT_FALSE(piece->isNextExternal);

            const auto* data = reinterpret_cast<const int*>(buf.Data(i));
            EXPECT_EQ(data[0], 6);
        } else {
            EXPECT_EQ(piece->numEvents, 0);
            EXPECT_EQ(piece->packetsByteSize, 0);
            EXPECT_EQ(piece->nextIdx, 0);
            EXPECT_FALSE(piece->isNextExternal);
        }
    }
}
