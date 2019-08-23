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

#ifndef PACBIO_CUDA_MEMORY_DEVICE_ONLY_ARRAY_CUH_
#define PACBIO_CUDA_MEMORY_DEVICE_ONLY_ARRAY_CUH_

#include <cassert>
#include <cmath>
#include <utility>

#include <pacbio/PBException.h>

#include <common/cuda/memory/AllocationViews.cuh>
#include <common/cuda/memory/SmartDeviceAllocation.h>
#include <common/cuda/PBCudaRuntime.h>
#include <common/cuda/streams/KernelLaunchInfo.h>
#include <common/cuda/streams/StreamMonitors.h>

namespace PacBio {
namespace Cuda {
namespace Memory {

namespace detail {
// Helper kernels to actually run the constructor and destructor on the gpu
template <typename T, typename... Args>
__global__ void InitFilters(T* data, size_t count, Args... args)
{
    size_t batchSize = blockDim.x * gridDim.x;
    size_t myIdx = threadIdx.x + blockDim.x * blockIdx.x;
    for (size_t i = 0; i < count; i+= batchSize)
    {
        if (i + myIdx < count)
        {
            auto in_ptr = data + i + myIdx;
            auto result_ptr = new(in_ptr) T(args...);
        }
    }
}
template <typename T>
__global__ void DestroyFilters(T* data, size_t count)
{
    size_t batchSize = blockDim.x * gridDim.x;
    size_t myIdx = threadIdx.x + blockDim.x * blockIdx.x;
    for (size_t i = 0; i < count; i+= batchSize)
    {
        if (i + myIdx < count)
        {
            data[i+myIdx].~T();
        }
    }
}

} // detail

// RAII host class that controls the lifetime of a device only object.  Useful if the
// object is designed to only be constructible on the device, or if you wish to
// avoid copying large objects and construct them directly on the gpu.  The entire
// array of objects must be constructible with the same arguments.
template <typename T>
class DeviceOnlyArray : private detail::DataManager
{
    // Try to find the best way to allocate one thread per entry in the array, somewhat
    // taking in to account both register requirements as well as warp size
    static constexpr size_t maxThreadsPerBlock = 1024;
    static std::pair<size_t, size_t> ComputeBlocksThreads(size_t arrayLen, const void* func)
    {
        auto requiredRegisters = RequiredRegisterCount(func);
        auto regsPerBlock = AvailableRegistersPerBlock();

        auto threadsPerBlock = regsPerBlock / requiredRegisters;
        if (threadsPerBlock == 0) throw PBException("Cannot invoke constructor on device, insufficient registers\n");

        threadsPerBlock = std::min(threadsPerBlock, std::min(arrayLen, maxThreadsPerBlock));

        // If we need more than one block, try to efficiently map to warps for better occupancy
        if (threadsPerBlock > 32 && threadsPerBlock < arrayLen)
        {
            size_t numWarps = threadsPerBlock / 32;
            numWarps = static_cast<size_t>(std::pow(2, std::floor(std::log(numWarps)/std::log(2))));
            threadsPerBlock = 32*numWarps;
        }

        auto numBlocks = (arrayLen + threadsPerBlock-1)/threadsPerBlock;
        return std::make_pair(numBlocks, threadsPerBlock);
    }
public:
    template <typename... Args>
    DeviceOnlyArray(size_t count, Args&&... args)
        : data_(count*sizeof(T))
        , count_(count)
    {
        // Even if we are an array of const types, we need to be non-const during construction
        using U = typename std::remove_const<T>::type;
        auto launchParams = ComputeBlocksThreads(count, (void*)&detail::InitFilters<U, Args...>);
        detail::InitFilters<<<launchParams.first, launchParams.second>>>(
                data_.get<U>(DataKey()),
                count_,
                std::forward<Args>(args)...);
        CudaSynchronizeDefaultStream();
    }

    DeviceOnlyArray() = delete;
    DeviceOnlyArray(const DeviceOnlyArray&) = delete;
    DeviceOnlyArray(DeviceOnlyArray&& other) = default;
    DeviceOnlyArray& operator=(const DeviceOnlyArray&) = delete;
    DeviceOnlyArray& operator=(DeviceOnlyArray&& other) = default;

    DeviceView<T> GetDeviceView(const KernelLaunchInfo& info)
    {
        checker_.Update(info);
        return DeviceHandle<T>(data_.get<T>(DataKey()), count_, DataKey());
    }
    DeviceView<const T> GetDeviceView(const KernelLaunchInfo& info) const
    {
        checker_.Update(info);
        return DeviceHandle<T>(data_.get<T>(DataKey()), count_, DataKey());
    }

    ~DeviceOnlyArray()
    {
        if (data_)
        {
            // Even if we are an array of const types, we need to be non-const during destruction
            using U = std::remove_const_t<T>;
            auto launchParams = ComputeBlocksThreads(count_, (void*)&detail::DestroyFilters<U>);
            detail::DestroyFilters<<<launchParams.first, launchParams.second>>>(
                    data_.get<U>(DataKey()),
                    count_);
            CudaSynchronizeDefaultStream();
        }
    }

private:

    SmartDeviceAllocation data_;
    size_t count_;

    // It's only safe for us to allow concurrent access among different streams
    // if we are storing a const type, and know that no stream is capable
    // of mutating the data.
    using CheckerType = typename std::conditional<std::is_const<T>::value,
                                                  MultiStreamMonitor,
                                                  SingleStreamMonitor>::type;
    CheckerType checker_;
};

template <typename T>
constexpr size_t DeviceOnlyArray<T>::maxThreadsPerBlock;

// Define overloads for this function, so that we can track kernel invocations, and
// so that we can be converted to our gpu specific representation
template <typename T>
DeviceView<T> KernelArgConvert(DeviceOnlyArray<T>& obj, const KernelLaunchInfo& info)
{
    return obj.GetDeviceView(info);
}
template <typename T>
DeviceView<T> KernelArgConvert(const DeviceOnlyArray<T>& obj, const KernelLaunchInfo& info)
{
    return obj.GetDeviceView(info);
}

}}}

#endif // PACBIO_CUDA_MEMORY_DEVICE_ONLY_ARRAY_CUH_
