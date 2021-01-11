// Copyright (c) 2019, Pacific Biosciences of California, Inc.
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

#ifndef PACBIO_CUDA_MEMORY_UNIFIED_CUDA_ARRAY_H
#define PACBIO_CUDA_MEMORY_UNIFIED_CUDA_ARRAY_H

#include <cassert>
#include <memory>

#include <pacbio/memory/SmartAllocation.h>

#include <common/cuda/PBCudaSimd.h>
#include <common/cuda/streams/KernelLaunchInfo.h>
#include <common/cuda/streams/StreamMonitors.h>

#include "AllocationViews.h"
#include "DataManagerKey.h"
#include "ManagedAllocations.h"
#include "SmartDeviceAllocation.h"

namespace PacBio {
namespace Cuda {
namespace Memory {

// Controls automatic data synchronization direction.
enum class SyncDirection
{
    HostWriteDeviceRead,  // Write on host read on device
    HostReadDeviceWrite,    // Write on device, read on host
    Symmetric  // Read/write either direction
};

template <typename T> struct gpu_type { using type = T; };
template <> struct gpu_type<int16_t> { using type = PBShort2; };
template <> struct gpu_type<PBHalf> { using type = PBHalf2; };

// TODO handle pitched allocations for multidimensional data?
template <typename T, bool allow_expensive_types = false>
class UnifiedCudaArray : private detail::DataManager
{

public:
    using HostType = T;
    using GpuType = typename gpu_type<T>::type;
    static constexpr size_t size_ratio = sizeof(GpuType) / sizeof(HostType);
    static_assert(std::is_trivially_copyable<HostType>::value, "Host type must be trivially copyable to support CudaMemcpy transfer");

    // Constructor for creating UCA from a pre-existing host allocation.  Predominantly
    // designed to allow a DeviceOnlyArray to copy out it's contents into something that
    // can be downloaded to the host, and thus requires a DataManagerKey to invoke.  Not
    // meant for 1-off random usage.
    UnifiedCudaArray(PacBio::Cuda::Memory::SmartDeviceAllocation alloc,
                     size_t count,
                     SyncDirection dir,
                     const AllocationMarker& marker,
                     detail::DataManagerKey)
        : activeOnHost_(false)
        , hostData_{GetGlobalAllocator().GetAllocation(count*sizeof(HostType), marker)}
        , gpuData_{std::move(alloc)}
        , syncDir_(dir)
        , marker_(marker)
        , checker_(std::make_unique<SingleStreamMonitor>())
        , transferMutex_(std::make_unique<std::mutex>())
    {
        if (dir == SyncDirection::HostWriteDeviceRead)
            throw PBException("Invalid SyncDirection for UnifiedCudaArray with manual gpu allocation");
        if (count != gpuData_.size() / sizeof(HostType))
            throw PBException("Incorrectly sized host allocation");
        if (count % (sizeof(GpuType) / sizeof(HostType)) != 0)
        {
            // If we're doing something special like using int16_t on the host and
            // PBShort2 on the gpu, we need to make sure things tile evenly, else the
            // last gpu element will walk off the array
            throw PBException("Invalid array length.");
        }
    }

    UnifiedCudaArray(PacBio::Memory::SmartAllocation alloc,
                     size_t count,
                     SyncDirection dir,
                     const AllocationMarker& marker)
        : activeOnHost_(true)
        , hostData_{std::move(alloc)}
        , gpuData_{}
        , syncDir_(dir)
        , marker_(marker)
        , checker_(std::make_unique<SingleStreamMonitor>())
        , transferMutex_(std::make_unique<std::mutex>())
    {
        // This is likely paranoia on my part, and this static assert can be relaxed if someone finds a need.
        // This is here to make it impossible to hand in data that is technically uninitialized.  Fundamental
        // types have trivial default construction, so their lifetime has effectively already begun before
        // this function is called.
        static_assert(std::is_trivially_default_constructible<HostType>::value,
                      "Cannot create UnifiedCudaArray from a manual host allocation when the type is not trivially default constructible");
        if (dir == SyncDirection::HostReadDeviceWrite)
            throw PBException("Invalid SyncDirection for UnifiedCudaArray with manual host allocation");
        if (count != hostData_.size() / sizeof(HostType))
            throw PBException("Incorrectly sized host allocation");
        if (count % (sizeof(GpuType) / sizeof(HostType)) != 0)
        {
            // If we're doing something special like using int16_t on the host and
            // PBShort2 on the gpu, we need to make sure things tile evenly, else the
            // last gpu element will walk off the array
            throw PBException("Invalid array length.");
        }
    }

    UnifiedCudaArray(size_t count,
                     SyncDirection dir,
                     const AllocationMarker& marker)
        : activeOnHost_(true)
        , hostData_{GetGlobalAllocator().GetAllocation(count*sizeof(HostType), marker)}
        , gpuData_{}
        , syncDir_(dir)
        , marker_(marker)
        , checker_(std::make_unique<SingleStreamMonitor>())
        , transferMutex_(std::make_unique<std::mutex>())
    {
        // Preferrably we'd ensure GpuType is trivially copyable as well, but for whatever reason
        // the half2 type is not implemented as such, which casues problems.  Best we can do is
        // try and ensure binary compatability on our end as much as we can.
        static_assert(sizeof(HostType) == sizeof(GpuType) || 2*sizeof(HostType) == sizeof(GpuType), "");

        static_assert(std::is_trivially_default_constructible<HostType>::value || allow_expensive_types,
                      "Must explicitly opt-in to use non-trivial construction types by setting the allow_expensive_types template parameter");
        static_assert(std::is_trivially_destructible<HostType>::value || allow_expensive_types,
                      "Must explicitly opt-in to use non-trivial destruction types by setting the allow_expensive_types template parameter");

        if (count % (sizeof(GpuType) / sizeof(HostType)) != 0)
        {
            // If we're doing something special like using int16_t on the host and
            // PBShort2 on the gpu, we need to make sure things tile evenly, else the
            // last gpu element will walk off the array
            throw PBException("Invalid array length.");
        }

        // default construct all elements on host.  Gpu memory is initialized via memcpy only
        // In theory this loop should be optimized away, but see notes on dtor as to why
        // I'm manually disabling it.
        if (!std::is_trivially_default_constructible<HostType>::value)
        {
            auto* ptr = hostData_.get<HostType>();
            for (size_t i = 0; i < count; ++i)
            {
                new(ptr+i) HostType;
            }
        }

        if (!std::is_trivially_default_constructible<HostType>::value && dir == SyncDirection::HostReadDeviceWrite)
        {
            CopyToDevice();
            CudaSynchronizeDefaultStream();
        }
    }

    UnifiedCudaArray(const UnifiedCudaArray&) = delete;
    UnifiedCudaArray(UnifiedCudaArray&& other) = default;
    UnifiedCudaArray& operator=(const UnifiedCudaArray&) = delete;
    UnifiedCudaArray& operator=(UnifiedCudaArray&& other) = default;

    ~UnifiedCudaArray()
    {
        // If we are trivially destructible, we can short-circuit any final
        // download from the gpu, because desconstruction won't actually
        // do anything observable anyway.
        if (std::is_trivially_destructible<HostType>::value)
        {
            activeOnHost_ = true;
        }
        DeactivateGpuMem();

        // Need to formally call destructors on host data.
        // Device side memory is just a bitwise memcpy mirror,
        // so no destructor invocations necessary on device
        //
        // Semi-interesting note: One would think the below loop
        // would completely optimize away for trivially destructible
        // types, and indeed it did for a long time.  Until an
        // unrelated change somehow caused it to not.  At one point
        // I profiled it and the application was spending a ton of
        // time executing an empty loop.  In looking at the assembly
        // it was clearly doing nothing but counting from 0 to count,
        // checking each time if the loop should terminate.  After
        // that experience I took matters into my own hand and
        // manually disabled this loop if it should be a noop.
        if (!std::is_trivially_destructible<HostType>::value)
        {
            auto * ptr = hostData_.get<HostType>();
            const size_t count = hostData_.size() / sizeof(HostType);
            for (size_t i = 0; i < count; ++i)
            {
                ptr[i].~HostType();
            }
        }

        //recycle our allocation
        IMongoCachedAllocator::ReturnHostAllocation(std::move(hostData_));
    }

    size_t Size() const { return hostData_.size() / sizeof(HostType); }
    bool ActiveOnHost() const { return activeOnHost_; }

    // Retires the gpu memory, so it can be used again elsewhere
    //
    // This is a blocking operation, and will not complete until all kernels run by this thread have
    // completed (to make sure no kernel that could know the gpu side address is still running). It
    // will also cause a data download if the data is device side and the synchronization scheme is
    // HostWriteDeviceRead.  This situation is not strictly an error, but does indicate perhaps
    // the synchronization scheme was set incorrectly.
    void DeactivateGpuMem()
    {
        if (!gpuData_) return;

        if (!activeOnHost_)
        {
            GetHostView();
        }

        IMongoCachedAllocator::ReturnDeviceAllocation(std::move(gpuData_));
    }

    // Calling these functions may cause
    // memory to be synchronized to the host side
    // (no performance penalty if it already is),
    // and will only remain valid until a view of
    // the device side is requested.
    HostView<HostType> GetHostView()
    {
        if (!activeOnHost_)
        {
            // Make sure any associated GPU kernels have completed
            checker_->Reset();

            CopyImpl(true, false);

        }

        return HostView<HostType>(hostData_.get<HostType>(), Size(), DataKey());
    }
    HostView<const HostType> GetHostView() const
    {
        if (!activeOnHost_)
        {
            // If necesary, make sure any associated GPU kernels have completed.
            // Since we provide immutable access, this is really only necessary if we're
            // not set to upload only.  That means there won't be future downloads to
            // alter the data we are seeing, and since we have retrieving a const view,
            // we cannot corrupt any pending uploads to the GPU.
            if (syncDir_ != SyncDirection::HostWriteDeviceRead)
                checker_->Reset();

            CopyImpl(true, false);

        }

        return HostView<const HostType>(hostData_.get<HostType>(), Size(), DataKey());
    }

    DeviceHandle<GpuType> GetDeviceHandle(const KernelLaunchInfo& info)
    {
        checker_->Update(info);
        if (activeOnHost_) CopyImpl(false, false);
        return DeviceHandle<GpuType>(gpuData_.get<GpuType>(DataKey()), Size()/size_ratio, DataKey());
    }
    DeviceHandle<const GpuType> GetDeviceHandle(const KernelLaunchInfo& info) const
    {
        checker_->Update(info);
        if (activeOnHost_) CopyImpl(false, false);
        return DeviceHandle<const GpuType>(gpuData_.get<GpuType>(DataKey()), Size()/size_ratio, DataKey());
    }

    void CopyToDevice() const
    {
        CopyImpl(false, true);
    }
    void CopyToHost() const
    {
        CopyImpl(true, true);
    }

private:

    // Used for just-in-time gpu allocations, to minimize our memory utilization
    // on the device
    void ActivateGpuMem() const
    {
        if (gpuData_) return;

        gpuData_ = GetGlobalAllocator().GetDeviceAllocation(hostData_.size(), marker_);
    }

    void CopyImpl(bool toHost, bool manual) const
    {
        // This is primarily to guard against a parallel host filter following
        // a GPU filter.  If multiple threads are requesting the host data, we
        // want to make sure only a single download is triggered.
        std::lock_guard<std::mutex> lg(*transferMutex_);
        if (toHost)
        {
            if (activeOnHost_) return; // Another thread performed the download.

            activeOnHost_ = true;
            if (manual || syncDir_ != SyncDirection::HostWriteDeviceRead)
            {
                // TODO this first synchronization is almost certainly to handle a quirk
                //      specific to the jetson xavier agx system.  On that system data
                //      transfers were always satisfied in the order they were requested,
                //      instead of whenever there is available bandwidth and all previous
                //      dependancies were met.  This meant that sending a transfer request
                //      too early could result in very poor scheduling, and we did better
                //      manually waiting before hand.
                //
                //      This should not be necessary on normal (non-integrated) nvidia
                //      systems, but I still need to do a quick experiment to prove that
                CudaSynchronizeDefaultStream();
                CudaCopyDeviceToHost(hostData_.get<HostType>(), gpuData_.get<HostType>(DataKey()), Size());
                CudaSynchronizeDefaultStream();
            }
        } else {
            if (!activeOnHost_) return; // Another thread performed the upload

            ActivateGpuMem();
            activeOnHost_ = false;
            if (manual || syncDir_ != SyncDirection::HostReadDeviceWrite)
                CudaCopyHostToDevice(gpuData_.get<HostType>(DataKey()), hostData_.get<HostType>(), Size());
        }
    }

    // NOTE: `const` in c++ normally implies bitwise const, not necessarily logical const
    //       In this case, this class is designed to represent data that has a dual life
    //       between host and gpu storage.  Data synchronization between the two does not
    //       logically alter the data, but it does violate bitwise const.  This class views
    //       logical constness as the more important semantics, thus the liberal useage of
    //       `mutable`.  A const version of this class will still be capable of copying the
    //       data to-from the GPU, but will only provide const accessors, so that the actual
    //       logical payload being represented cannot be altered.
    mutable bool activeOnHost_;
    mutable PacBio::Memory::SmartAllocation hostData_;
    mutable SmartDeviceAllocation gpuData_;
    SyncDirection syncDir_;
    AllocationMarker marker_;
    std::unique_ptr<SingleStreamMonitor> checker_;
    std::unique_ptr<std::mutex> transferMutex_;
};

// Define overloads for this function, so that we can track kernel invocations, and
// so that we can be converted to our gpu specific representation
template <typename T, bool allow_expensive_types = false>
auto KernelArgConvert(UnifiedCudaArray<T, allow_expensive_types>& obj,
                      const KernelLaunchInfo& info)
{
    return obj.GetDeviceHandle(info);
}

template <typename T, bool allow_expensive_types = false>
auto KernelArgConvert(const UnifiedCudaArray<T, allow_expensive_types>& obj,
                      const KernelLaunchInfo& info)
{
    return obj.GetDeviceHandle(info);
}

}}} //::Pacbio::Cuda::Memory

#endif //PACBIO_CUDA_MEMORY_UNIFIED_CUDA_ARRAY_H
