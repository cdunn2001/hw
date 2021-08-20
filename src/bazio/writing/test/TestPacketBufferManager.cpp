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

#include <bazio/writing/PacketBufferManager.h>

#include <pacbio/datasource/MallocAllocator.h>

using namespace PacBio::BazIO;
using namespace PacBio::DataSource;

// Constructs an empty packet buffer.  Doesn't fill it with data,
// but we do try to check that the batchSize is respected and we get
// exactly one allocation
TEST(PacketBufferManager, ConstructAllHQ)
{
    uint32_t numZmw = 10;
    uint32_t bytesPerZmw = 10;
    PacketBufferManager manager(numZmw, bytesPerZmw);

    EXPECT_EQ(manager.NumHQ(), numZmw);
    EXPECT_EQ(manager.NumZmw(), numZmw);

    for (size_t i = 0; i < numZmw; ++i)
    {
        EXPECT_TRUE(manager.IsHQ(i));

        auto slice = manager.GetSlice(i);
        EXPECT_EQ(slice.numEvents, 0);
        EXPECT_EQ(slice.packetByteSize, 0);
        ASSERT_EQ(slice.pieces.size(), 1);
        EXPECT_EQ(slice.pieces[0].count, 0);
    }
}

TEST(PacketBufferManager, ConstructNoHQ)
{
    uint32_t numZmw = 10;
    uint32_t bytesPerZmw = 10;
    uint32_t maxLookback = 2;
    PacketBufferManager manager(numZmw, bytesPerZmw, maxLookback,
                                std::make_shared<MallocAllocator>());

    EXPECT_EQ(manager.NumHQ(), 0);
    EXPECT_EQ(manager.NumZmw(), numZmw);

    for (size_t i = 0; i < numZmw; ++i)
    {
        EXPECT_FALSE(manager.IsHQ(i));

        auto slice = manager.GetSlice(i);
        EXPECT_EQ(slice.numEvents, 0);
        EXPECT_EQ(slice.packetByteSize, 0);
        ASSERT_EQ(slice.pieces.size(), 0);
    }
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

TEST(PacketBuffer, AddDataFullHQ)
{
    uint32_t numZmw = 10;
    uint32_t bytesPerZmw = 10;
    PacketBufferManager manager(numZmw, bytesPerZmw);

    // We're going to write this data to ZMW 0 and 5.
    // 4 ints will not fit in the 10 bytes allocated,
    // so we should overflow
    std::vector<int> data{1,2,3,4};

    manager.AddZmw(0, data.begin(), data.end(),
                   [](auto){return true;},
                   DummySerializer{});

    manager.AddZmw(5, data.begin(), data.end(),
                   [](auto){return true;},
                   DummySerializer{});


    // Make sure zmw 0 and 5 have the expected data,
    // and that the rest are still empty
    for (size_t i = 0; i < numZmw; ++i)
    {
        auto slice = manager.GetSlice(i);
        if (i == 0 || i == 5)
        {
            EXPECT_EQ(slice.numEvents, 4);
            EXPECT_EQ(slice.packetByteSize, 16);
            ASSERT_EQ(slice.pieces.size(), 2);
            EXPECT_EQ(slice.pieces[0].count, 8);
            EXPECT_EQ(slice.pieces[1].count, 8);

            const auto* data = reinterpret_cast<const int*>(slice.pieces[0].data);
            EXPECT_EQ(data[0], 1);
            EXPECT_EQ(data[1], 2);
            data = reinterpret_cast<const int*>(slice.pieces[1].data);
            EXPECT_EQ(data[0], 3);
            EXPECT_EQ(data[1], 4);
        } else {
            EXPECT_EQ(slice.numEvents, 0);
            EXPECT_EQ(slice.packetByteSize, 0);
            ASSERT_EQ(slice.pieces.size(), 1);
            EXPECT_EQ(slice.pieces[0].count, 0);
        }
    }
}

TEST(PacketBuffer, AddDataMarkHQ)
{
    uint32_t numZmw = 10;
    uint32_t bytesPerZmw = 10;
    uint32_t maxLookback = 1;
    PacketBufferManager manager(numZmw, bytesPerZmw, maxLookback,
                                std::make_shared<MallocAllocator>());

    // We're going to write this data to ZMW 0 and 5.
    // 4 ints will not fit in the 10 bytes allocated,
    // so we should overflow
    std::vector<int> data{1,2,3,4};

    manager.AddZmw(0, data.begin(), data.end(),
                   [](auto){return true;},
                   DummySerializer{});

    manager.AddZmw(5, data.begin(), data.end(),
                   [](auto){return true;},
                   DummySerializer{});

    // Data not marked as HQ is not returned in a slice
    EXPECT_EQ(manager.NumHQ(), 0);
    for (size_t i = 0; i < numZmw; ++i)
    {
        auto slice = manager.GetSlice(i);
        EXPECT_EQ(slice.numEvents, 0);
        EXPECT_EQ(slice.packetByteSize, 0);
        ASSERT_EQ(slice.pieces.size(), 0);
    }

    manager.MarkAsHQ(0);
    manager.MarkAsHQ(5);
    EXPECT_EQ(manager.NumHQ(), 2);

    // Make sure zmw 0 and 5 have the expected data,
    // and that the rest are still empty
    for (size_t i = 0; i < numZmw; ++i)
    {
        auto slice = manager.GetSlice(i);
        if (i == 0 || i == 5)
        {
            EXPECT_EQ(slice.numEvents, 4);
            EXPECT_EQ(slice.packetByteSize, 16);
            ASSERT_EQ(slice.pieces.size(), 2);
            EXPECT_EQ(slice.pieces[0].count, 8);
            EXPECT_EQ(slice.pieces[1].count, 8);

            const auto* data = reinterpret_cast<const int*>(slice.pieces[0].data);
            EXPECT_EQ(data[0], 1);
            EXPECT_EQ(data[1], 2);
            data = reinterpret_cast<const int*>(slice.pieces[1].data);
            EXPECT_EQ(data[0], 3);
            EXPECT_EQ(data[1], 4);
        } else {
            EXPECT_EQ(slice.numEvents, 0);
            EXPECT_EQ(slice.packetByteSize, 0);
            ASSERT_EQ(slice.pieces.size(), 0);
        }
    }
}

TEST(PacketBuffer, LookbackBuffers)
{
    uint32_t numZmw = 10;
    uint32_t bytesPerZmw = 10;
    uint32_t maxLookback = 3;
    PacketBufferManager manager(numZmw, bytesPerZmw, maxLookback,
                                std::make_shared<MallocAllocator>());

    // We're going to write this data to ZMW 0,1 and 2
    // 4 ints will not fit in the 10 bytes allocated,
    // so we should overflow.  We're going to repeat and mark
    // the ZMW as hq one by one.  We shouldn't lose any data,
    // even if the ZMW marked as HQ later won't output it until
    // later
    std::vector<int> data{1,2,3,4};
    manager.AddZmw(0, data.begin(), data.end(),
                   [](auto){return true;},
                   DummySerializer{});
    manager.AddZmw(1, data.begin(), data.end(),
                   [](auto){return true;},
                   DummySerializer{});
    manager.AddZmw(2, data.begin(), data.end(),
                   [](auto){return true;},
                   DummySerializer{});

    EXPECT_EQ(manager.NumHQ(), 0);
    manager.MarkAsHQ(0);

    auto managerCheck1 = manager.CreateCheckpoint();
    EXPECT_EQ(managerCheck1->NumHQ(), 1);
    EXPECT_TRUE(managerCheck1->IsHQ(0));
    auto slice = managerCheck1->GetSlice(0);
    EXPECT_EQ(slice.numEvents, 4);
    EXPECT_EQ(slice.pieces.size(), 2);
    // previous tests have checked the data payload in similar cases,
    // so this time I won't bother

    // Add the data again.  ZMW 0 is effectively starting fresh,
    // but 1 and 2 still have their previous data.
    manager.AddZmw(0, data.begin(), data.end(),
                   [](auto){return true;},
                   DummySerializer{});
    manager.AddZmw(1, data.begin(), data.end(),
                   [](auto){return true;},
                   DummySerializer{});
    manager.AddZmw(2, data.begin(), data.end(),
                   [](auto){return true;},
                   DummySerializer{});

    manager.MarkAsHQ(1);
    EXPECT_EQ(manager.NumHQ(), 2);
    auto managerCheck2 = manager.CreateCheckpoint();
    EXPECT_EQ(managerCheck2->NumHQ(), 2);
    EXPECT_TRUE(managerCheck2->IsHQ(0));
    EXPECT_TRUE(managerCheck2->IsHQ(1));
    // ZMW 0 had a fresh start, so it only has a single copy of the above
    // 4 element vector
    slice = managerCheck2->GetSlice(0);
    EXPECT_EQ(slice.numEvents, 4);
    EXPECT_EQ(slice.pieces.size(), 2);
    // ZMW 1 has accrued data, so it should have two coppies.  This time
    // I'm not validating the data payload because we'll do that the next
    // iteration for zmw 3
    slice = managerCheck2->GetSlice(1);
    EXPECT_EQ(slice.numEvents, 8);
    EXPECT_EQ(slice.pieces.size(), 4);

    // One final round.  ZMW 0 and 1 are starting fresh,
    // but ZMW 2 has three coppies now.
    manager.AddZmw(0, data.begin(), data.end(),
                   [](auto){return true;},
                   DummySerializer{});
    manager.AddZmw(1, data.begin(), data.end(),
                   [](auto){return true;},
                   DummySerializer{});
    manager.AddZmw(2, data.begin(), data.end(),
                   [](auto){return true;},
                   DummySerializer{});

    manager.MarkAsHQ(2);
    EXPECT_EQ(manager.NumHQ(), 3);
    auto managerCheck3 = manager.CreateCheckpoint();
    EXPECT_EQ(managerCheck3->NumHQ(), 3);
    EXPECT_TRUE(managerCheck3->IsHQ(0));
    EXPECT_TRUE(managerCheck3->IsHQ(1));
    EXPECT_TRUE(managerCheck3->IsHQ(2));
    // ZMW 0 had a fresh start, so it only has a single copy of the above
    // 4 element vector
    slice = managerCheck3->GetSlice(0);
    EXPECT_EQ(slice.numEvents, 4);
    EXPECT_EQ(slice.pieces.size(), 2);
    // Same story now for ZMW 1
    slice = managerCheck3->GetSlice(1);
    EXPECT_EQ(slice.numEvents, 4);
    EXPECT_EQ(slice.pieces.size(), 2);
    // ZMW 2 has accrued data, so it should have three coppies.  This time
    // we'll validate the full payload
    slice = managerCheck3->GetSlice(2);
    EXPECT_EQ(slice.numEvents, 12);
    EXPECT_EQ(slice.pieces.size(), 6);
    int val = 1;
    for (const auto& piece : slice.pieces)
    {
        EXPECT_EQ(piece.count, 8);
        const auto* data = reinterpret_cast<const int*>(piece.data);
        for (size_t i = 0; i < 2; ++i)
        {
            EXPECT_EQ(val, data[i]);
            val++;
            if (val > 4) val = 1;
        }
    }
}

// Data older than our max lookback gets dropped on the floow
TEST(PacketBuffer, DropData)
{
    uint32_t numZmw = 10;
    uint32_t bytesPerZmw = 10;
    uint32_t maxLookback = 3;
    PacketBufferManager manager(numZmw, bytesPerZmw, maxLookback,
                                std::make_shared<MallocAllocator>());

    std::vector<int> data{1,2,3,4};

    std::unique_ptr<const PacketBufferManager> checkpoint;
    for (size_t i = 0; i < 10; ++i)
    {
        manager.AddZmw(0, data.begin(), data.end(),
                       [](auto){return true;},
                       DummySerializer{});
        if (i == 9) manager.MarkAsHQ(0);
        checkpoint = manager.CreateCheckpoint();
    }

    // We did 10 checkpoints but we're configured to only store 3,
    // so we only expect three coppies of the vector
    const auto slice = checkpoint->GetSlice(0);
    EXPECT_EQ(slice.numEvents, 12);
    EXPECT_EQ(slice.packetByteSize, 12*sizeof(int));
    EXPECT_EQ(slice.pieces.size(), 6);
}
