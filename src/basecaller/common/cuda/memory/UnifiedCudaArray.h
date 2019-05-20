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

#include "AllocationViews.h"
#include "DataManagerKey.h"
#include "GpuAllocationPool.h"
#include "SmartDeviceAllocation.h"
#include "SmartHostAllocation.h"

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

// TODO handle pitched allocations for multidimensional data?
template <typename T>
class UnifiedCudaArray : private detail::DataManager {
public:
    UnifiedCudaArray(size_t count, SyncDirection dir, std::shared_ptr<GpuAllocationPool<T>> pool = nullptr)
        : activeOnHost_(true)
        , hostData_(count, true)
        , gpuData_(pool ? 0 : count) // Only allocate now if we're not pooling gpu allocations
        , syncDir_(dir)
        , pool_(pool)
    {
        // default construct all elements on host.  Gpu memory is initialized via memcpy only
        auto* ptr = hostData_.get(DataKey());
        for (size_t i = 0; i < count; ++i)
        {
            new(ptr+i) T();
        }

        if (pool && pool->AllocSize() != count)
        {
            throw PBException("Inconsistent gpu allocation pool and allocation size");
        }
    }

    UnifiedCudaArray(const UnifiedCudaArray&) = delete;
    UnifiedCudaArray(UnifiedCudaArray&& other) = default;
    UnifiedCudaArray& operator=(const UnifiedCudaArray&) = delete;
    UnifiedCudaArray& operator=(UnifiedCudaArray&& other) = default;

    ~UnifiedCudaArray()
    {
        // Potentially hand this back to the memory pool (if active)
        // instead of actually freeing the memory.  Will triger
        // synchronization if necessary
        DeactivateGpuMem();

        // Need to formally call destructors on host data.
        // Device side memory is just a bitwise memcpy mirror,
        // so no destructor invocations necessary on device
        auto* ptr = hostData_.get(DataKey());
        for (size_t i = 0; i < hostData_.size(); ++i)
        {
            ptr[i].~T();
        }
    }

    size_t Size() const { return hostData_.size(); }
    bool ActiveOnHost() const { return activeOnHost_; }

    // Marks the gpu memory as "inactive".  If the gpu side is using pooled allocations, this means it can
    // now be stolen and used to back a different UnifiedCudaArray.
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
        CudaSynchronizeDefaultStream();
        activeOnHost_ = true;

        if (auto pool = pool_.lock()) pool->PushAlloc(std::move(gpuData_));
    }

    // Calling these functions may cause
    // memory to be synchronized to the host side
    // (no performance penalty if it already is),
    // and will only remain valid until a view of
    // the device side is requested.
    HostView<T> GetHostView(size_t idx, size_t len)
    {
        if (!activeOnHost_) CopyImpl(true, false);
        assert(idx + len <= hostData_.size());
        return HostView<T>(hostData_.get(DataKey()), idx, len, DataKey());
    }
    HostView<T> GetHostView() { return GetHostView(0, hostData_.size()); }
    operator HostView<T>() { return GetHostView(); }

    DeviceHandle<T> GetDeviceHandle(size_t idx, size_t len)
    {
        if (activeOnHost_) CopyImpl(false, false);
        assert(idx + len <= hostData_.size());
        return DeviceHandle<T>(gpuData_.get(DataKey()), idx, len, DataKey());
    }
    DeviceHandle<T> GetDeviceHandle() { return GetDeviceHandle(0, hostData_.size()); }
    operator DeviceHandle<T>() { return GetDeviceHandle(); }

    void CopyToDevice()
    {
        CopyImpl(false, true);
    }
    void CopyToHost()
    {
        CopyImpl(true, true);
    }

private:
    // Makes sure that if we are using pooled gpu allocations, we actually
    // have an active allocation to use.  If we don't, try and check one
    // out of the memory pool.  If the memory pool is empty (or we just
    // don't have one) then allocate a new swath of memory.
    void ActivateGpuMem()
    {
        if (gpuData_) return;

        if (auto pool = pool_.lock())
        {
            gpuData_ = pool->PopAlloc(hostData_.size());
        }
        if (!gpuData_)
            gpuData_ = SmartDeviceAllocation<T>(hostData_.size());
    }

    void CopyImpl(bool toHost, bool manual)
    {
        if (toHost)
        {
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
                CudaCopyHost(hostData_.get(DataKey()), gpuData_.get(DataKey()), hostData_.size());
                CudaSynchronizeDefaultStream();
            }
        } else {
            ActivateGpuMem();
            activeOnHost_ = false;
            if (manual || syncDir_ != SyncDirection::HostReadDeviceWrite)
                CudaCopyDevice(gpuData_.get(DataKey()), hostData_.get(DataKey()), hostData_.size());
        }
    }

    bool activeOnHost_;
    SmartHostAllocation<T> hostData_;
    SmartDeviceAllocation<T> gpuData_;
    SyncDirection syncDir_;
    std::weak_ptr<GpuAllocationPool<T>> pool_;
};

}}} //::Pacbio::Cuda::Memory

#endif //PACBIO_CUDA_MEMORY_UNIFIED_CUDA_ARRAY_H
