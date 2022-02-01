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

#ifndef PACBIO_CUDA_MEMORY_DEVICE_ALLOCATION_STASH_H
#define PACBIO_CUDA_MEMORY_DEVICE_ALLOCATION_STASH_H

#include <memory>
#include <map>
#include <unordered_set>
#include <vector>

#include <common/cuda/memory/StashableDeviceAllocation.h>

namespace PacBio {
namespace Cuda {
namespace Memory {

// This class maintains a collection of weak pointers to StashableDeviceAllocations,
// with the intent that it will coordinate application wide data transfers to
// potentially lessen device memory requirements, at the cost of additional
// PCIe traffic.
class DeviceAllocationStash
{
    using Allocation = std::weak_ptr<StashableDeviceAllocation>;
    using PoolMap = std::map<uint32_t, std::vector<Allocation>>;
public:
    DeviceAllocationStash(const DeviceAllocationStash&) = delete;
    DeviceAllocationStash(DeviceAllocationStash&&) = delete;
    DeviceAllocationStash& operator=(const DeviceAllocationStash&) = delete;
    DeviceAllocationStash& operator=(DeviceAllocationStash&&) = delete;

    DeviceAllocationStash() = default;

    /// Goes through all currently registered data and segregates them into
    /// two groups:
    /// 1. Data that will always reside on the GPU
    /// 2. Data that will be uploaded just in time for use and moved back
    ///    to the host afterwards.
    /// maxResidentMB controls how large the first group is.
    ///
    /// \param maxResidentMB maximum space to use on the GPU for permanent
    ///        algorithm state data storage
    void PartitionData(size_t maxResidentMB);

    /// Registers a new StashableDeviceAllocation with this stash.
    /// Until PartitionData is subsequently called, the new allocation
    /// will be stashed on the host rather than permanently resident
    /// on the GPU.
    ///
    /// \param poolId the pool associated with the provided allocation
    /// \param alloc a weak_ptr to a StashableDeviceAllocation
    void Register(uint32_t poolId, Allocation alloc);

    /// Walks through all registered StashableDeviceAllocations associated
    /// with a given poolID, and makes sure that they are moved to the GPU
    /// and ready to be used.
    ///
    /// \param poolId the pool for which we want to upload all stashed data
    /// \return total number of bytes transfered
    size_t RetrievePool(uint32_t poolId);

    /// Walks through all registered StashableDeviceAllocations associated
    /// with a given poolID, and if they are marked for storage on the host,
    /// copy them down and free up the GPU memory.
    ///
    /// \param poolId the pool for which we can stash data back on the host
    /// \return total number of bytes transfered
    size_t StashPool(uint32_t poolId);

private:
    // Using std::greater as the sort predicate so that loops over the data
    // are in descending order, which should hopefully make it easier to
    // select a subset of allocations that add up as close as possible to
    // a specified memory allotment.
    std::map<size_t, PoolMap, std::greater<size_t>> mobileData_;
    std::map<size_t, PoolMap, std::greater<size_t>> stationaryData_;
    std::unordered_set<uint32_t> poolsSeen_;

    // Can only alter the maps if you hold this lock
    std::mutex mutateMutex_;
    // Less restrictive locks, that allow either upload or
    // download of data.  There can be a simultaneous upload
    // and download.
    std::mutex uploadMutex_;
    std::mutex downloadMutex_;
};

// Helper class used to facilitate registering a StashableDeviceAllocation
// with a DeviceAllocationStash.  Mostly here just to wrap the poolId, since
// lower level code that actually owns the allocations likely doesn't know
// what pool it's associated with.
class StashableAllocRegistrar
{
public:
    using Allocation = std::weak_ptr<StashableDeviceAllocation>;

    StashableAllocRegistrar(uint32_t poolId, DeviceAllocationStash& manager)
        : manager_(manager)
        , poolId_(poolId)
    {}

    StashableAllocRegistrar(const StashableAllocRegistrar&) = delete;
    StashableAllocRegistrar(StashableAllocRegistrar&&) = delete;
    StashableAllocRegistrar& operator=(const StashableAllocRegistrar&) = delete;
    StashableAllocRegistrar& operator=(StashableAllocRegistrar&&) = delete;

    void Record(Allocation alloc)
    {
        manager_.Register(poolId_, alloc);
    }

private:
    DeviceAllocationStash& manager_;
    uint32_t poolId_;

};

}}}


#endif //PACBIO_CUDA_MEMORY_DEVICE_ALLOCATION_STASH_H
