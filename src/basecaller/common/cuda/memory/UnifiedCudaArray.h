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
#include "AllocationPool.h"
#include "SmartDeviceAllocation.h"
#include "SmartHostAllocation.h"

#include <vector_types.h>

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
template <> struct gpu_type<int16_t> { using type = short2; };

// TODO handle pitched allocations for multidimensional data?
template <typename T1>
class UnifiedCudaArray : private detail::DataManager
{
    using T2 = typename gpu_type<T1>::type;
    static constexpr size_t size_ratio = sizeof(T2) / sizeof(T1);
public:
    UnifiedCudaArray(size_t count,
                     SyncDirection dir,
                     bool pinnedHost = true,
                     std::shared_ptr<DualAllocationPools> pools = nullptr)
        : activeOnHost_(true)
        , hostData_{}
        , gpuData_{}
        , syncDir_(dir)
        , pools_(pools)
    {
        static_assert(sizeof(T2) % sizeof(T1) == 0, "Gpu type not even multiple of host types");
        if (count % (sizeof(T2) / sizeof(T1)) != 0)
        {
            // If we're doing something special like using int16_t on the host and
            // short2 on the gpu, we need to make sure things tile evenly, else the
            // last gpu element will walk off the array
            throw PBException("Invalid array length.");
        }
        // get host allocation from pool if we can. Gpu allocations are always deffered
        // until necessary
        if (pools)
        {
            hostData_ = pools->hostPool.PopAlloc(count*sizeof(T1));
        } else {
            hostData_ = SmartHostAllocation(count*sizeof(T1), pinnedHost);
        }

        if (pools && pools->gpuPool.AllocSize() != count * sizeof(T1))
        {
            throw PBException("Inconsistent gpu allocation pool and allocation size");
        }

        // default construct all elements on host.  Gpu memory is initialized via memcpy only
        auto* ptr = hostData_.get<T1>(DataKey());
        for (size_t i = 0; i < count; ++i)
        {
            new(ptr+i) T1();
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
        auto * ptr = hostData_.get<T1>(DataKey());
        for (size_t i = 0; i < hostData_.size(); ++i)
        {
            ptr[i].~T1();
        }

        // If there is a pool for host data, recycle our allocation
        if (auto pools = pools_.lock()) pools->hostPool.PushAlloc(std::move(hostData_));
    }

    size_t Size() const { return hostData_.size() / sizeof(T1); }
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

        if (auto pools = pools_.lock()) pools->gpuPool.PushAlloc(std::move(gpuData_));
    }

    // Calling these functions may cause
    // memory to be synchronized to the host side
    // (no performance penalty if it already is),
    // and will only remain valid until a view of
    // the device side is requested.
    HostView<T1> GetHostView(size_t idx, size_t len)
    {
        if (!activeOnHost_) CopyImpl(true, false);
        assert(idx + len <= Size());
        return HostView<T1>(hostData_.get<T1>(DataKey()), idx, len, DataKey());
    }
    HostView<T1> GetHostView() { return GetHostView(0, Size()); }
    operator HostView<T1>() { return GetHostView(); }

    DeviceHandle<T2> GetDeviceHandle(size_t idx, size_t len)
    {
        if (activeOnHost_) CopyImpl(false, false);
        assert(idx + len <= Size());
        return DeviceHandle<T2>(gpuData_.get<T2>(DataKey()), idx, len/size_ratio, DataKey());
    }
    DeviceHandle<T2> GetDeviceHandle() { return GetDeviceHandle(0, Size()); }
    operator DeviceHandle<T2>() { return GetDeviceHandle(); }

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

        if (auto pools = pools_.lock())
        {
            gpuData_ = pools->gpuPool.PopAlloc(hostData_.size());
        }
        if (!gpuData_)
            gpuData_ = SmartDeviceAllocation(hostData_.size());
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
                CudaCopyHost(hostData_.get<T1>(DataKey()), gpuData_.get<T1>(DataKey()), Size());
                CudaSynchronizeDefaultStream();
            }
        } else {
            ActivateGpuMem();
            activeOnHost_ = false;
            if (manual || syncDir_ != SyncDirection::HostReadDeviceWrite)
                CudaCopyDevice(gpuData_.get<T1>(DataKey()), hostData_.get<T1>(DataKey()), Size());
        }
    }

    bool activeOnHost_;
    SmartHostAllocation hostData_;
    SmartDeviceAllocation gpuData_;
    SyncDirection syncDir_;
    std::weak_ptr<DualAllocationPools> pools_;
};

}}} //::Pacbio::Cuda::Memory

#endif //PACBIO_CUDA_MEMORY_UNIFIED_CUDA_ARRAY_H
