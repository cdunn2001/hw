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

#ifndef PACBIO_CUDA_MEMORY_STASHABLE_DEVICE_ALLOCATION_H
#define PACBIO_CUDA_MEMORY_STASHABLE_DEVICE_ALLOCATION_H

#include <common/cuda/memory/AllocationViews.h>
#include <common/cuda/memory/ManagedAllocations.h>
#include <common/cuda/memory/SmartDeviceAllocation.h>
#include <common/cuda/streams/StreamMonitors.h>

#include <pacbio/memory/SmartAllocation.h>

namespace PacBio {
namespace Cuda {
namespace Memory {

// This class represents a device allocation that may potentially
// be put into "cold storage" on the host, freeing up GPU memory
// at the cost of PCIe bandwidth.  This is necessary for cases
// where we cannot fit all of the algorithm state data permanently
// on the GPU.
//
// Note: This class does have some major similaries with the
//       implementation details of UnifiedCudaArray.  Ultimately
//       though this serves a somewhat different need, so I
//       decided to keep this implementation entirely separate.
class StashableDeviceAllocation : public detail::DataManager
{
public:
    enum State
    {
        NO_ALLOC = 0,
        HOST,
        DEVICE
    };

    StashableDeviceAllocation(size_t size,
                              const AllocationMarker& marker,
                              std::unique_ptr<StreamMonitorBase> monitor)
        : monitor_(std::move(monitor))
        , mutex_(std::make_unique<std::mutex>())
        , size_(size)
        , marker_(marker)
        , state_(NO_ALLOC)
    {}

    StashableDeviceAllocation(const StashableDeviceAllocation&) = delete;
    StashableDeviceAllocation(StashableDeviceAllocation&& o)
        : device_(std::move(o.device_))
        , host_(std::move(o.host_))
        , monitor_(std::move(o.monitor_))
        , mutex_(std::move(o.mutex_))
        , size_(std::exchange(o.size_, 0))
        , marker_(std::exchange(o.marker_, AllocationMarker{""}))
        , state_(std::exchange(o.state_, NO_ALLOC))
    {}

    StashableDeviceAllocation& operator=(const StashableDeviceAllocation&) = delete;
    StashableDeviceAllocation& operator=(StashableDeviceAllocation&& o)
    {
        device_  = std::move(o.device_);
        host_    = std::move(o.host_);
        monitor_ = std::move(o.monitor_);
        mutex_   = std::move(o.mutex_);
        size_    = std::exchange(o.size_, 0);
        marker_  = std::exchange(o.marker_, AllocationMarker{""});
        state_   = std::exchange(o.state_, NO_ALLOC);

        return *this;
    }

    ~StashableDeviceAllocation()
    {
        if (host_)
            IMongoCachedAllocator::ReturnHostAllocation(std::move(host_));
        if (device_)
            IMongoCachedAllocator::ReturnDeviceAllocation(std::move(device_));
    }

    // Makes sure the data currently resides on the GPU.  This is cheap
    // to call if the data is already on the GPU
    void Retrieve()
    {
        if (size_ == 0) return;
        assert(mutex_);
        std::lock_guard<std::mutex> lm(*mutex_);

        if (!device_)
            device_ = GetGlobalAllocator().GetDeviceAllocation(size_, marker_);

        if (state_ == HOST)
        {
            assert(host_);
            assert(state_);
            CudaRawCopyDevice(device_.get<void>(DataKey()), host_.get<void>(), size_);
            CudaSynchronizeDefaultStream();
        }
        state_ = DEVICE;
    }

    // Copy data to the host and free up the GPU memory.  This is cheap
    // to call if data is already on the host.
    void Stash()
    {
        // Extra short-circuit, because it's not going to make sense to
        // go straight from NO_ALLOC to HOST.  The memory is only visable
        // and usable on the gpu, so that means if we allowed the copy we'd
        // just be shuffling around uninitialized bits
        if (size_ == 0 || state_ == NO_ALLOC) return;

        assert(mutex_);
        std::lock_guard<std::mutex> lm(*mutex_);

        if (!host_)
            host_ = GetGlobalAllocator().GetAllocation(size_, marker_);

        if (state_ == DEVICE)
        {
            monitor_->Reset();
            assert(host_);
            assert(device_);
            CudaRawCopyHost(host_.get<void>(),device_.get<void>(DataKey()),  size_);
            CudaSynchronizeDefaultStream();
            IMongoCachedAllocator::ReturnDeviceAllocation(std::move(device_));
        }
        state_ = HOST;
    }

    template <typename T>
    DeviceHandle<T> GetDeviceHandle(const KernelLaunchInfo& info)
    {
        assert(size_ % sizeof(T) == 0);
        monitor_->Update(info);

        return GetDeviceHandle<T>(DataKey());
    }

    template <typename T>
    DeviceHandle<T> GetDeviceHandle(detail::DataManagerKey)
    {
        assert(size_ % sizeof(T) == 0);

        Retrieve();
        return DeviceHandle<T>(device_.get<T>(DataKey()), size_/sizeof(T), DataKey());
    }

    bool empty() const { return !device_ && !host_; }
    bool hasHostAlloc() const { return host_; }
    bool hasDeviceAlloc() const { return device_; }
    size_t size() const { return size_; }
    State state() const { return state_; }

private:
    SmartDeviceAllocation device_;
    PacBio::Memory::SmartAllocation host_;
    std::unique_ptr<StreamMonitorBase> monitor_;
    std::unique_ptr<std::mutex> mutex_;

    size_t size_;
    Memory::AllocationMarker marker_;
    State state_;
};

}}}


#endif //PACBIO_CUDA_MEMORY_STASHABLE_DEVICE_ALLOCATION_H
