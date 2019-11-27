// Copyright (c) 2018, Pacific Biosciences of California, Inc.
//
// All rights reserved.
//
// THIS SOFTWARE CONSTITUTES AND EMBODIES PACIFIC BIOSCIENCES' CONFIDENTIAL
// AND PROPRIETARY ERRORRMATION.
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

#include <pacbio/primary/MemoryBuffer.h>
#include <pacbio/logging/Logger.h>

#include <memory>

using namespace PacBio::Primary;
using namespace PacBio;

namespace {

class Dummy
{
public:
    Dummy(int a, int b, int c, std::unique_ptr<int> d)
      : one(a), two(b), three(c), four(std::move(d)) {}
    Dummy() = default;
    Dummy(const Dummy&) = default;
    Dummy(Dummy&&) = default;
    Dummy& operator =(const Dummy&) = default;
    Dummy& operator =(Dummy&&) = default;

    int one, two, three;
    std::unique_ptr<int> four; //just tossing in someting non-copyable
};

// Helper class to keep track of system allocations/deallocations
// Implements the concept of an `Allocator` as defined by the standard for
// use in containers.
template <typename T>
class Allocator
{
public:
    using value_type = T;
    T * allocate(size_t n)
    {
        allocatedCount += n;
        //PBLOG_ERROR << "Allocate " << n;
       return reinterpret_cast<T*>(malloc(n * sizeof(T)));
    }

    void deallocate(T* ptr, size_t n)
    {
        //PBLOG_ERROR << "Deallocate " << n;
        deallocatedCount += n;
        free(ptr);
    }
    size_t max_size() const
    {
        return std::numeric_limits<size_t>::max() / sizeof(value_type);
    }
    static size_t allocatedCount;
    static size_t deallocatedCount;
};

template <typename T>
size_t Allocator<T>::allocatedCount = 0;
template <typename T>
size_t Allocator<T>::deallocatedCount = 0;

}

template <typename T = Dummy>
std::unique_ptr<MemoryBuffer<T, Allocator>>
CreateBuffer(size_t initialCapacity, size_t estimatedMaximum)
{
    std::unique_ptr<MemoryBuffer<T, Allocator>> ret;
    ret.reset(new MemoryBuffer<T, Allocator>(initialCapacity, estimatedMaximum));
    return ret;
}

TEST(MemoryBuffer, CreateDestroy)
{
    auto initialDeallocations = Allocator<Dummy>::deallocatedCount;
    auto initialAllocations = Allocator<Dummy>::allocatedCount;

    auto buffer = CreateBuffer(100, 1000);
    ASSERT_EQ(0, buffer->Size());
    ASSERT_EQ(100, buffer->Capacity());
    ASSERT_EQ(0, Allocator<Dummy>::deallocatedCount - initialDeallocations);
    ASSERT_EQ(100, Allocator<Dummy>::allocatedCount - initialAllocations);

    buffer.reset();
    ASSERT_EQ(100, Allocator<Dummy>::deallocatedCount - initialDeallocations);
    ASSERT_EQ(100, Allocator<Dummy>::allocatedCount - initialAllocations);
}

TEST(MemoryBuffer, NoGrowth)
{
    auto initialDeallocations = Allocator<Dummy>::deallocatedCount;
    auto initialAllocations = Allocator<Dummy>::allocatedCount;

    auto buffer = CreateBuffer(100, 1000);
    ASSERT_EQ(0, buffer->Size());
    ASSERT_EQ(100, buffer->Capacity());
    ASSERT_EQ(0, Allocator<Dummy>::deallocatedCount - initialDeallocations);
    ASSERT_EQ(100, Allocator<Dummy>::allocatedCount - initialAllocations);

    (void) buffer->Allocate(10);
    (void) buffer->Allocate(11);
    (void) buffer->Allocate(5);
    (void) buffer->Allocate(8);
    (void) buffer->Allocate(20);

    // Only thing that should change from above is the number of entries
    ASSERT_EQ(54, buffer->Size());
    ASSERT_EQ(100, buffer->Capacity());
    ASSERT_EQ(0, Allocator<Dummy>::deallocatedCount - initialDeallocations);
    ASSERT_EQ(100, Allocator<Dummy>::allocatedCount - initialAllocations);

    buffer.reset();
    ASSERT_EQ(Allocator<Dummy>::allocatedCount, Allocator<Dummy>::deallocatedCount);
}

TEST(MemoryBuffer, ZeroAlloc)
{
    auto initialDeallocations = Allocator<Dummy>::deallocatedCount;
    auto initialAllocations = Allocator<Dummy>::allocatedCount;

    auto buffer = CreateBuffer(0, 1000);
    ASSERT_EQ(0, buffer->Size());
    ASSERT_EQ(0, buffer->Capacity());
    ASSERT_EQ(0, Allocator<Dummy>::deallocatedCount - initialDeallocations);
    ASSERT_EQ(0, Allocator<Dummy>::allocatedCount - initialAllocations);

    auto view = buffer->Allocate(0);
    ASSERT_EQ(0, view.size());
    ASSERT_EQ(0, buffer->Size());
    ASSERT_EQ(0, buffer->Capacity());
    ASSERT_EQ(0, Allocator<Dummy>::deallocatedCount - initialDeallocations);
    ASSERT_EQ(0, Allocator<Dummy>::allocatedCount - initialAllocations);
}

TEST(MemoryBuffer, SingleGrowthExact)
{
    auto initialDeallocations = Allocator<Dummy>::deallocatedCount;
    auto initialAllocations = Allocator<Dummy>::allocatedCount;

    auto buffer = CreateBuffer(100, 1000);
    ASSERT_EQ(0, buffer->Size());
    ASSERT_EQ(100, buffer->Capacity());
    ASSERT_EQ(0, Allocator<Dummy>::deallocatedCount - initialDeallocations);
    ASSERT_EQ(100, Allocator<Dummy>::allocatedCount - initialAllocations);

    // Should exactly saturate initial allocation
    (void) buffer->Allocate(10);
    (void) buffer->Allocate(10);
    (void) buffer->Allocate(10);
    (void) buffer->Allocate(10);
    (void) buffer->Allocate(10);
    (void) buffer->Allocate(10);
    (void) buffer->Allocate(10);
    (void) buffer->Allocate(10);
    (void) buffer->Allocate(10);
    (void) buffer->Allocate(10);
    ASSERT_EQ(100, buffer->Size());
    ASSERT_EQ(100, buffer->Capacity());
    ASSERT_EQ(0, Allocator<Dummy>::deallocatedCount - initialDeallocations);
    ASSERT_EQ(100, Allocator<Dummy>::allocatedCount - initialAllocations);

    (void) buffer->Allocate(10);
    ASSERT_EQ(110, buffer->Size());
    // Capacity must have been increased
    ASSERT_LT(100, buffer->Capacity());
    // Nothing must be deallocated (else we've invalidated a live pointer)
    ASSERT_EQ(0, Allocator<Dummy>::deallocatedCount - initialDeallocations);
    // But there must be a new allocation for the new capacity
    ASSERT_LT(100, Allocator<Dummy>::allocatedCount - initialAllocations);

    buffer.reset();
    ASSERT_EQ(Allocator<Dummy>::allocatedCount, Allocator<Dummy>::deallocatedCount);
}

TEST(MemoryBuffer, SingleGrowthInexact)
{
    auto initialDeallocations = Allocator<Dummy>::deallocatedCount;
    auto initialAllocations = Allocator<Dummy>::allocatedCount;

    auto buffer = CreateBuffer(100, 1000);
    ASSERT_EQ(0, buffer->Size());
    ASSERT_EQ(100, buffer->Capacity());
    ASSERT_EQ(0, Allocator<Dummy>::deallocatedCount - initialDeallocations);
    ASSERT_EQ(100, Allocator<Dummy>::allocatedCount - initialAllocations);

    // Should be barely shy of saturating existing storage
    (void) buffer->Allocate(11);
    (void) buffer->Allocate(11);
    (void) buffer->Allocate(11);
    (void) buffer->Allocate(11);
    (void) buffer->Allocate(11);
    (void) buffer->Allocate(11);
    (void) buffer->Allocate(11);
    (void) buffer->Allocate(11);
    (void) buffer->Allocate(11);
    ASSERT_EQ(99, buffer->Size());
    ASSERT_EQ(100, buffer->Capacity());
    ASSERT_EQ(0, Allocator<Dummy>::deallocatedCount - initialDeallocations);
    ASSERT_EQ(100, Allocator<Dummy>::allocatedCount - initialAllocations);

    (void) buffer->Allocate(11);
    ASSERT_EQ(110, buffer->Size());
    // Capacity must have been increased
    ASSERT_LT(100, buffer->Capacity());
    // Nothing must be deallocated (else we've invalidated a live pointer)
    ASSERT_EQ(0, Allocator<Dummy>::deallocatedCount - initialDeallocations);
    // But there must be a new allocation for the new capacity
    ASSERT_LT(100, Allocator<Dummy>::allocatedCount - initialAllocations);


    // There must also be minor fragmentation now
    ASSERT_LT(0, buffer->FragmentationFraction());

    buffer.reset();
    ASSERT_EQ(Allocator<Dummy>::allocatedCount, Allocator<Dummy>::deallocatedCount);
}

TEST(MemoryBuffer, LargeAllocation)
{
    auto initialDeallocations = Allocator<Dummy>::deallocatedCount;
    auto initialAllocations = Allocator<Dummy>::allocatedCount;

    auto buffer = CreateBuffer(100, 1000);
    buffer->Allocate(10000);
    ASSERT_EQ(10000, buffer->Size());
    ASSERT_EQ(10100, buffer->Capacity());
    ASSERT_EQ(0, Allocator<Dummy>::deallocatedCount - initialDeallocations);
    ASSERT_EQ(10100, Allocator<Dummy>::allocatedCount - initialAllocations);

    // The initial 100 elements end up being unused.
    ASSERT_FLOAT_EQ(0.01f, buffer->FragmentationFraction());

    buffer.reset();
    ASSERT_EQ(Allocator<Dummy>::allocatedCount, Allocator<Dummy>::deallocatedCount);
}

TEST(MemoryBuffer, ManySmallAllocations)
{
    auto initialDeallocations = Allocator<Dummy>::deallocatedCount;

    auto buffer = CreateBuffer(1000, 1000000);

    auto lastAllocated = Allocator<Dummy>::allocatedCount;
    size_t linearRate = 0;
    // Should be (fast) linear growth until we hit the soft cap
    for (size_t i = 0; i < 100000; ++i)
    {
        (void) buffer->Allocate(10);
        if (lastAllocated != Allocator<Dummy>::allocatedCount)
        {
            auto delta = Allocator<Dummy>::allocatedCount - lastAllocated;
            if (linearRate != 0) ASSERT_EQ(linearRate, delta);
            else linearRate = delta;
            lastAllocated = Allocator<Dummy>::allocatedCount;
        }
    }

    // Next allocation is allowed to be weird as we transition to new behavior.
    // Burn through values until we skip it
    while (Allocator<Dummy>::allocatedCount == lastAllocated) buffer->Allocate(10);

    lastAllocated = Allocator<Dummy>::allocatedCount;

    // Now growth should be a (slow) exponential
    size_t remainingAllocs = 5;
    float growthRatio = 0.0f;
    while (remainingAllocs > 0)
    {
        (void) buffer->Allocate(10);
        if (lastAllocated != Allocator<Dummy>::allocatedCount)
        {
            float delta = static_cast<float>(Allocator<Dummy>::allocatedCount - lastAllocated);
            auto prevSize = buffer->Size() - 10;
            if (growthRatio != 0) ASSERT_FLOAT_EQ(growthRatio, delta / prevSize);
            else growthRatio = delta / prevSize;
            lastAllocated = Allocator<Dummy>::allocatedCount;

            remainingAllocs--;
        }
    }

    // Should still be no deallocations, else we've invalidated pointers
    ASSERT_EQ(0, Allocator<Dummy>::deallocatedCount - initialDeallocations);

    buffer.reset();
    ASSERT_EQ(Allocator<Dummy>::allocatedCount, Allocator<Dummy>::deallocatedCount);
}

TEST(MemoryBuffer, CallAllocate)
{
    auto buffer = CreateBuffer(1000, 1000000);

    auto first = buffer->Allocate(12);
    auto second = buffer->Allocate(12);

    ASSERT_EQ(12, first.size());
    ASSERT_EQ(12, second.size());

    // We are under the initial capacity.  One shouldn't strictly rely on this
    // in production code, but the two memory segments should be contiguous
    ASSERT_EQ(&first[0]+12, &second[0]);
}

TEST(MemoryBuffer, CallCopy)
{
    auto buffer = CreateBuffer<int>(100, 500);

    std::vector<int> input1 {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    auto output1 = buffer->Copy(input1.data(), 10);

    std::vector<int> input2 {30, 31, 32, 33, 34, 35, 36, 37, 38, 39};
    auto output2 = buffer->Copy(input2.data(), 10);

    ASSERT_EQ(10, output1.size());
    ASSERT_EQ(10, output2.size());
    for (size_t i = 0; i < 10; ++i)
    {
        ASSERT_EQ(input1[i], output1[i]);
        ASSERT_EQ(input2[i], output2[i]);
    }
}

TEST(MemoryBuffer, CallCreate)
{
    auto buffer = CreateBuffer(1000, 1000000);

    std::unique_ptr<int> tmp{new int{12}};
    // Call the forwarding constructor with a unique_ptr, to make sure the
    // forwarding works properly
    auto out = buffer->Create(9,6,3, std::move(tmp));
    ASSERT_EQ(1, out.size());
    ASSERT_EQ(9, out[0].one);
    ASSERT_EQ(6, out[0].two);
    ASSERT_EQ(3, out[0].three);
    ASSERT_EQ(12, *out[0].four);
}

TEST(MemoryBuffer, Reset)
{
    auto initialAllocations = Allocator<Dummy>::allocatedCount;
    auto initialDeallocations = Allocator<Dummy>::deallocatedCount;

    auto buffer = CreateBuffer(0, 10000);
    ASSERT_EQ(0, Allocator<Dummy>::allocatedCount - initialAllocations);

    buffer->Allocate(10000);
    ASSERT_EQ(10000, Allocator<Dummy>::allocatedCount - initialAllocations);

    // Should not actually release any memory
    buffer->Reset();
    ASSERT_EQ(Allocator<Dummy>::deallocatedCount, initialDeallocations);
    ASSERT_EQ(buffer->Size(), 0);
    ASSERT_EQ(buffer->Capacity(), 10000);

    // There should be space for all these without new reservations as well
    for (size_t i = 0; i < 10; ++i)
    {
        buffer->Allocate(1000);
        ASSERT_EQ(10000, Allocator<Dummy>::allocatedCount - initialAllocations);
    }
}
