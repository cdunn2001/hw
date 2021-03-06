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

#ifndef PACBIO_CUDA_MEMORY_DEVICE_ONLY_OBJECT_CUH_
#define PACBIO_CUDA_MEMORY_DEVICE_ONLY_OBJECT_CUH_

#include <common/cuda/memory/DeviceOnlyArray.cuh>

namespace PacBio {
namespace Cuda {
namespace Memory {

// Non-owning smart pointer for use on the GPU to access a gpu-only instance.
template <typename T>
class DevicePtr : private detail::DataManager
{
public:
    __host__ __device__ DevicePtr(DeviceView<T> data, detail::DataManagerKey)
        : data_(data)
    {}

    __device__ T* operator->() { return data_.Data(); }
    __device__ const T* operator->() const { return data_.Data(); }
    __device__ T& operator*() { return *data_.Data();}
    __device__ const T& operator*() const { return *data_.Data();}

    // Implicit conversion to DevicePtr<const T> needs to be disabled if we're already
    // templated on a const T
    template <typename U = T, std::enable_if_t<!std::is_same<const U, U>::value, int> = 0>
    __host__ __device__ operator DevicePtr<const T>()
    {
        return DevicePtr<const T>(data_, DataKey());
    }
private:
    DeviceView<T> data_;
};

// Owning host-side smart pointer for an obect residing exclusively
// in GPU memory.  The host will handle lifetime of the object,
// but the object itself can only be interacted with in a cuda kernel
// via a DevicePtr
template <typename T>
class DeviceOnlyObj : private detail::DataManager
{
public:
    template <typename... Args>
    DeviceOnlyObj(const AllocationMarker& marker, Args&&... args)
        : data_(marker, 1, std::forward<Args>(args)...)
    {}

    DevicePtr<T> GetDevicePtr(const KernelLaunchInfo& info)
    {
        return DevicePtr<T>(data_.GetDeviceView(info), DataKey());
    }
    DevicePtr<const T> GetDevicePtr(const KernelLaunchInfo& info) const
    {
        return DevicePtr<T>(data_.GetDeviceView(info), DataKey());
    }

private:
    Memory::DeviceOnlyArray<T> data_;
};

// Define overloads for this function, so that we can track kernel invocations, and
// so that we can be converted to our gpu specific representation
template <typename T>
DevicePtr<T> KernelArgConvert(DeviceOnlyObj<T>& obj, const KernelLaunchInfo& info)
{
    return obj.GetDevicePtr(info);
}
template <typename T>
DevicePtr<T> KernelArgConvert(const DeviceOnlyObj<T>& obj, const KernelLaunchInfo& info)
{
    return obj.GetDevicePtr(info);
}


}}}

#endif // PACBIO_CUDA_MEMORY_DEVICE_ONLY_OBJECT_CUH_
