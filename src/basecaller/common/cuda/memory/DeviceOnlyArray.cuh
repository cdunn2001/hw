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

#include <utility>

#include <common/cuda/PBCudaRuntime.h>
#include <common/cuda/memory/AllocationViews.cuh>
#include <common/cuda/memory/SmartDeviceAllocation.h>

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
    // Not really choosing these because they are efficient or well suited to the
    // problem, just choosing something that works, and takes advantage of some
    // ammount of parallelism if there are a lot of elements in the array
    static constexpr size_t cudaBlocks = 1;
    static constexpr size_t cudaThreadsPerBlock = 1024;
public:
    template <typename... Args>
    DeviceOnlyArray(size_t count, Args&&... args)
        : data_(count)
    {
        detail::InitFilters<<<cudaBlocks, cudaThreadsPerBlock>>>(
                data_.get(DataKey()),
                data_.size(),
                std::forward<Args>(args)...);
    }

    DeviceOnlyArray() = delete;
    DeviceOnlyArray(const DeviceOnlyArray&) = delete;
    DeviceOnlyArray(DeviceOnlyArray&& other) = default;
    DeviceOnlyArray& operator=(const DeviceOnlyArray&) = delete;
    DeviceOnlyArray& operator=(DeviceOnlyArray&& other) = default;

    DeviceView<T> GetDeviceView(size_t idx, size_t len)
    {
        return DeviceHandle<T>(data_.get(DataKey()), idx, len, DataKey());
    }
    DeviceView<T> GetDeviceView()
    {
        return GetDeviceView(0, data_.size());
    }

    ~DeviceOnlyArray()
    {
        detail::DestroyFilters<<<cudaBlocks, cudaThreadsPerBlock>>>(
                data_.get(DataKey()),
                data_.size());
    }

private:

    SmartDeviceAllocation<T> data_;
};

}}}

#endif // PACBIO_CUDA_MEMORY_DEVICE_ONLY_ARRAY_CUH_
