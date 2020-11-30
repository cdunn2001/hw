// Copyright (c) 2020, Pacific Biosciences of California, Inc.
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

#include <numeric>

#include <gtest/gtest.h>

#include <common/cuda/memory/DeviceAllocationStash.h>
#include <common/cuda/memory/UnifiedCudaArray.h>
#include <common/cuda/memory/AllocationViews.cuh>
#include <common/cuda/streams/LaunchManager.cuh>

using namespace PacBio::Cuda::Memory;
using namespace PacBio::Cuda;

// Checks the inital empty state.  We don't initially grab any memory
// on either the host or the device.  This is because:
// * If it's an allocation that will always reside on the GPU,
//   no need to waste host resources we won't use
// * If it's an allocation that will be stashed, we don't want to
//   grab any GPU memory until the proper pool is activated, oterwise
//   we may blow out GPU memory
TEST(StashableDeviceAllocation, Empty)
{
    StashableDeviceAllocation alloc(100, SOURCE_MARKER(),
                                    std::make_unique<SingleStreamMonitor>());

    EXPECT_FALSE(alloc.hasDeviceAlloc());
    EXPECT_FALSE(alloc.hasHostAlloc());
    EXPECT_EQ(alloc.size(), 100);
    EXPECT_EQ(alloc.state(), StashableDeviceAllocation::NO_ALLOC);
}

// Stashing an Empty allocation should effectively do nothing.
TEST(StashableDeviceAllocation, HostFirst)
{
    StashableDeviceAllocation alloc(100, SOURCE_MARKER(),
                                    std::make_unique<SingleStreamMonitor>());

    alloc.Stash();

    EXPECT_FALSE(alloc.hasDeviceAlloc());
    EXPECT_FALSE(alloc.hasHostAlloc());
    EXPECT_EQ(alloc.size(), 100);
    EXPECT_EQ(alloc.state(), StashableDeviceAllocation::NO_ALLOC);
}

// Make sure that we don't consume any host resources unless we know
// we're going to need them.
TEST(StashableDeviceAllocation, DeviceFirst)
{
    StashableDeviceAllocation alloc(100, SOURCE_MARKER(),
                                    std::make_unique<SingleStreamMonitor>());
    alloc.Retrieve();

    EXPECT_TRUE(alloc.hasDeviceAlloc());
    EXPECT_FALSE(alloc.hasHostAlloc());
    EXPECT_EQ(alloc.size(), 100);
    EXPECT_EQ(alloc.state(), StashableDeviceAllocation::DEVICE);
}

// Make sure that after "stashing" an allocation, we have a host allocation
// but the GPU memory has been released for re-use elsewhere.
TEST(StashableDeviceAllocation, Stashed)
{
    StashableDeviceAllocation alloc(100, SOURCE_MARKER(),
                                    std::make_unique<SingleStreamMonitor>());
    alloc.Retrieve();
    alloc.Stash();

    EXPECT_FALSE(alloc.hasDeviceAlloc());
    EXPECT_TRUE(alloc.hasHostAlloc());
    EXPECT_EQ(alloc.size(), 100);
    EXPECT_EQ(alloc.state(), StashableDeviceAllocation::HOST);
}

// Helper kernel to set up some real data inside a StashableDeviceAllocation
__global__ void RoundTripInit(DeviceView<int> data)
{
    for (int i = 0; i < data.Size(); ++i)
    {
        data[i] = i;
    }
}

// Helper kernel to copy data from a StashableDeviceAllocation
// to a UnifiedCudaArray, so it can be validated on the host
__global__ void RoundTripExtract(DeviceView<int> stashed, DeviceView<int> ret)
{
    assert(stashed.Size() == ret.Size());
    for (int i = 0; i < stashed.Size(); ++i)
    {
        ret[i] = stashed[i];
    }
}

// Make sure that if we stash an allocation, it comes back with all its
// data intact
TEST(StashableDeviceAllocation, RoundTrip)
{
    // Helper class, to get access to a typed version of the underlying
    // data so we can actualy run a cuda kernel using it.
    struct TestAlloc : StashableDeviceAllocation
    {
        using StashableDeviceAllocation::StashableDeviceAllocation;

        DeviceView<int> GetDeviceHandle()
        {
            return StashableDeviceAllocation::GetDeviceHandle<int>(DataKey());
        }
    };

    TestAlloc alloc(100, SOURCE_MARKER(),
                    std::make_unique<SingleStreamMonitor>());
    PBLauncher(RoundTripInit, 1,1)(alloc.GetDeviceHandle());
    CudaSynchronizeDefaultStream();

    // We've used the data in a kernel, it had better be present...!
    EXPECT_TRUE(alloc.hasDeviceAlloc());
    EXPECT_FALSE(alloc.hasHostAlloc());
    EXPECT_EQ(alloc.size(), 100);
    EXPECT_EQ(alloc.state(), StashableDeviceAllocation::DEVICE);

    // We've stashed it, so now there should only be a host allocation
    alloc.Stash();
    EXPECT_FALSE(alloc.hasDeviceAlloc());
    EXPECT_TRUE(alloc.hasHostAlloc());
    EXPECT_EQ(alloc.size(), 100);
    EXPECT_EQ(alloc.state(), StashableDeviceAllocation::HOST);

    // It's now been activated back on the GPU, so make sure the
    // GPU allocation is present.
    // Note: Currently there still will be a host allocation, but
    //       that's a technical detail that we may decide to
    //       change. Not testing it here, because the API is
    //       currently agnostic to that choice.
    alloc.Retrieve();
    EXPECT_TRUE(alloc.hasDeviceAlloc());
    EXPECT_EQ(alloc.size(), 100);
    EXPECT_EQ(alloc.state(), StashableDeviceAllocation::DEVICE);

    // Run a kernel to make sure the data remainded intact
    UnifiedCudaArray<int> result(25, SyncDirection::HostReadDeviceWrite, SOURCE_MARKER());
    PBLauncher(RoundTripExtract, 1,1)(alloc.GetDeviceHandle(), result);

    auto view = result.GetHostView();
    for (size_t i = 0; i < result.Size(); ++i)
    {
        EXPECT_EQ(view[i], i);
    }
}

// Creates a pool of allocations with `count` entries.  Entries will start at
// 1 MB in size, with each subsequent entry being 1MB larger
std::vector<std::shared_ptr<StashableDeviceAllocation>> CreateAllocPool(size_t count)
{

    std::vector<std::shared_ptr<StashableDeviceAllocation>> allocPool;
    for (size_t i = 0; i < count; ++i)
    {
        allocPool.push_back(std::make_shared<StashableDeviceAllocation>(
                (i+1)<<20, SOURCE_MARKER(),
                std::make_unique<SingleStreamMonitor>()));
    }
    return allocPool;
}


// Checks the behavior of the stash when the limits are set such
// that *all* memory can live on the GPU permanently.
TEST(DeviceAllocationStash, NoStashing)
{
    DeviceAllocationStash stash;

    auto allocPool1 = CreateAllocPool(10);
    auto allocPool2 = CreateAllocPool(10);

    for (auto& alloc : allocPool1) stash.Register(1, alloc);
    for (auto& alloc : allocPool2) stash.Register(2, alloc);

    stash.PartitionData(std::numeric_limits<size_t>::max());

    stash.RetrievePool(1);
    stash.RetrievePool(2);

    stash.StashPool(1);
    stash.StashPool(2);

    for (size_t i = 0; i < allocPool1.size(); ++i)
    {
        EXPECT_TRUE(allocPool1[i]->hasDeviceAlloc()) << i;
        EXPECT_TRUE(allocPool2[i]->hasDeviceAlloc()) << i;
    }
}

// Checks the behavior of the stash when the limits are set such
// that *no* memory can live on the GPU permanently.
TEST(DeviceAllocationStash, AllStashing)
{
    DeviceAllocationStash stash;

    auto allocPool1 = CreateAllocPool(10);
    auto allocPool2 = CreateAllocPool(10);

    for (auto& alloc : allocPool1) stash.Register(1, alloc);
    for (auto& alloc : allocPool2) stash.Register(2, alloc);

    stash.PartitionData(0);

    stash.RetrievePool(1);
    stash.RetrievePool(2);

    stash.StashPool(1);
    stash.StashPool(2);

    for (size_t i = 0; i < allocPool1.size(); ++i)
    {
        EXPECT_FALSE(allocPool1[i]->hasDeviceAlloc()) << i;
        EXPECT_FALSE(allocPool2[i]->hasDeviceAlloc()) << i;
    }
}

// Checks the behavior of the stash when the limits are set such
// that *some* memory can live on the GPU permanently.
TEST(DeviceAllocationStash, PartialStashing)
{
    DeviceAllocationStash stash;

    auto allocPool1 = CreateAllocPool(10);
    auto allocPool2 = CreateAllocPool(10);

    for (auto& alloc : allocPool1) stash.Register(1, alloc);
    for (auto& alloc : allocPool2) stash.Register(2, alloc);

    const size_t maxGpuMB = 40;
    stash.PartitionData(maxGpuMB);

    stash.RetrievePool(1);
    stash.RetrievePool(2);

    stash.StashPool(1);
    stash.StashPool(2);

    auto tallyMemUsage = [](auto allocPool)
    {
        return std::accumulate(allocPool.begin(), allocPool.end(), 0ull,
                               [](size_t count, std::shared_ptr<StashableDeviceAllocation> alloc)
                               {
                                   if (alloc->hasDeviceAlloc())
                                       return count + alloc->size();
                                   else
                                       return count;
                               });
    };
    auto fullTally = tallyMemUsage(allocPool1) + tallyMemUsage(allocPool2);

    // There is no guarantee we'll have fully filled out the maximum
    // for permanent GPU memory, but make sure we use at least a
    // large and reasonable portion of it.
    EXPECT_LE(fullTally, maxGpuMB<<20);
    EXPECT_GT(fullTally, (maxGpuMB * 3 / 4)<<20);
}

// Make sure that the pool based upload/download really do behave
// independantly with respect to each other.
TEST(DeviceAllocationStash, IndependantPools)
{
    DeviceAllocationStash stash;

    auto allocPool1 = CreateAllocPool(10);
    auto allocPool2 = CreateAllocPool(10);

    for (auto& alloc : allocPool1) stash.Register(1, alloc);
    for (auto& alloc : allocPool2) stash.Register(2, alloc);

    stash.PartitionData(0);

    stash.RetrievePool(1);
    for (auto& alloc : allocPool1) EXPECT_TRUE(alloc->hasDeviceAlloc());
    for (auto& alloc : allocPool2) EXPECT_FALSE(alloc->hasDeviceAlloc());

    stash.RetrievePool(2);
    for (auto& alloc : allocPool1) EXPECT_TRUE(alloc->hasDeviceAlloc());
    for (auto& alloc : allocPool2) EXPECT_TRUE(alloc->hasDeviceAlloc());

    stash.StashPool(1);
    for (auto& alloc : allocPool1) EXPECT_FALSE(alloc->hasDeviceAlloc());
    for (auto& alloc : allocPool2) EXPECT_TRUE(alloc->hasDeviceAlloc());

    stash.StashPool(2);
    for (auto& alloc : allocPool1) EXPECT_FALSE(alloc->hasDeviceAlloc());
    for (auto& alloc : allocPool2) EXPECT_FALSE(alloc->hasDeviceAlloc());
}
