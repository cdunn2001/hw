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

#include <bazio/writing/GrowableArray.h>

#include <pacbio/memory/IAllocator.h>

using namespace PacBio::BazIO;
using namespace PacBio::Memory;

namespace {

// Test Allocator, just to get some visibility into what allocations
// actually happen under the hood
struct TestAllocator : public IAllocator
{
    virtual SmartAllocation GetAllocation(size_t count)
    {
        return SmartAllocation(count,
                               [this](size_t c)
                               {
                                   ++totalAllocs;
                                   ++currAllocs;
                                   totalBytes += c;
                                   currBytes += c;
                                   auto ret = malloc(c);
                                   memset(ret, 0, c);
                                   return ret;
                               },
                               [this, count](void* ptr)
                               {
                                   currBytes -= count;
                                   --currAllocs;
                                   free(ptr);
                               });
    }
    virtual bool SupportsAllFlags(uint32_t flags) const
    {
        return flags == 0;
    }
    virtual std::string Name() const {
        return "Test allocator for GrowableArray";
    }

    uint32_t totalAllocs = 0;
    int32_t currAllocs = 0;
    uint32_t totalBytes = 0;
    int32_t currBytes = 0;
};

}  // namespace

// Currently these are only empty on construction
TEST(GrowableArray, Construct)
{
    auto alloc = std::make_shared<TestAllocator>();

    uint32_t batchSize = 10;
    uint32_t allocationLength = 10;
    GrowableArray<int> arr(alloc, batchSize, allocationLength);
    EXPECT_EQ(alloc->totalAllocs , 0);
    EXPECT_EQ(alloc->totalBytes, 0);

    EXPECT_EQ(arr.Size(), 0);
    EXPECT_EQ(arr.ElementLen(), allocationLength);
}

TEST(GrowableArray, Grow)
{
    auto alloc = std::make_shared<TestAllocator>();

    uint32_t batchSize = 10;
    uint32_t allocationLength = 10;
    GrowableArray<int> arr(alloc, batchSize, allocationLength);

    EXPECT_EQ(alloc->totalAllocs, 0);
    EXPECT_EQ(alloc->totalBytes, 0);
    EXPECT_EQ(arr.Size(), 0);
    EXPECT_EQ(arr.ElementLen(), allocationLength);

    // Should require two allocations, but not actually add
    // any elements
    arr.Reserve(19);
    EXPECT_EQ(arr.Size(), 0);
    EXPECT_EQ(alloc->totalAllocs, 2);
    EXPECT_EQ(alloc->totalBytes, allocationLength * batchSize * sizeof(int) * 2);

    // Should consume only part of our capacity, no new allocations are expected
    arr.GrowToSize(15);
    EXPECT_EQ(arr.Size(), 15);
    EXPECT_EQ(alloc->totalAllocs, 2);
    EXPECT_EQ(alloc->totalBytes, allocationLength * batchSize * sizeof(int) * 2);

    // Should exceed capacity and require one more allocation
    arr.GrowToSize(25);
    EXPECT_EQ(arr.Size(), 25);
    EXPECT_EQ(alloc->totalAllocs, 3);
    EXPECT_EQ(alloc->totalBytes, allocationLength * batchSize * sizeof(int) * 3);

    // Grow by one to max out the current allocation.  Not expecting to spill over
    arr.AppendOne();
    arr.AppendOne();
    arr.AppendOne();
    arr.AppendOne();
    arr.AppendOne();
    EXPECT_EQ(arr.Size(), 30);
    EXPECT_EQ(alloc->totalAllocs, 3);
    EXPECT_EQ(alloc->totalBytes, allocationLength * batchSize * sizeof(int) * 3);

    // A single new element should require a full new allocation
    arr.AppendOne();
    EXPECT_EQ(arr.Size(), 31);
    EXPECT_EQ(alloc->totalAllocs, 4);
    EXPECT_EQ(alloc->totalBytes, allocationLength * batchSize * sizeof(int) * 4);
}

TEST(GrowableArray, StablePointers)
{
    auto alloc = std::make_shared<TestAllocator>();

    uint32_t batchSize = 10;
    uint32_t allocationLength = 10;
    GrowableArray<int> arr(alloc, batchSize, allocationLength);
    arr.GrowToSize(batchSize);

    auto ptr1 = arr[0];

    // Growing should never invalidate pointers.  The data should
    // never be coppied under the hood
    arr.GrowToSize(10 *batchSize);
    EXPECT_EQ(ptr1, arr[0]);
    EXPECT_EQ(alloc->totalAllocs, alloc->currAllocs);
}

TEST(GrowableArray, SetValues)
{
    auto alloc = std::make_shared<TestAllocator>();

    uint32_t batchSize = 10;
    uint32_t allocationLength = 10;
    GrowableArray<int> arr(alloc, batchSize, allocationLength);
    arr.GrowToSize(2);

    // Elements within the same allocation are contiguous.  This
    // isn't something to rely on too much in production code,
    // we just want to at least make sure we are actually respecting
    // the `batchSize` parameter
    auto* ptr1 = arr[0];
    auto* ptr2 = arr[1];
    EXPECT_EQ(ptr1+allocationLength, ptr2);

    for (size_t i = 0; i < 20; ++i)
    {
        ptr1[i] = i;
    };

    EXPECT_EQ(arr[0][0], 0);
    EXPECT_EQ(arr[0][9], 9);
    EXPECT_EQ(arr[1][0], 10);
    EXPECT_EQ(arr[1][9], 19);

    EXPECT_EQ(arr[1][10], 0);
}

namespace {

struct TestClass
{
    int foo = 12;
    int bar = 99;
};

}

TEST(GrowableArray, ClassTypes)
{
    auto alloc = std::make_shared<TestAllocator>();

    uint32_t batchSize = 10;
    uint32_t allocationLength = 10;
    GrowableArray<TestClass> arr(alloc, batchSize, allocationLength);
    arr.AppendOne();
    // We added one index, so all instances in that (0-9) should be
    // fully constructed objects
    EXPECT_EQ(arr[0][0].foo, 12);
    EXPECT_EQ(arr[0][0].bar, 99);
    EXPECT_EQ(arr[0][9].foo, 12);
    EXPECT_EQ(arr[0][9].bar, 99);
    // If we "walk off the end", then things shouldn't be constructed.
    // The test allocator above is what set these locations to 0
    EXPECT_EQ(arr[0][10].foo, 0);
    EXPECT_EQ(arr[0][10].bar, 0);
}
