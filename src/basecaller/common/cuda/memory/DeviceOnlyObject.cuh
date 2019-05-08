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
class DevicePtr
{
public:
    DevicePtr(T* data, detail::DataManagerKey)
        : data_(data)
    {}

    __device__ T* operator->() { return data_; }
    __device__ const T* operator->() const { return data_; }
private:
    T* data_;
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
    DeviceOnlyObj(Args&&... args)
        : data_(1, std::forward<Args>(args)...)
    {}

    DevicePtr<T> GetDevicePtr()
    {
        return DevicePtr<T>(data_.GetDeviceView().Data(DataKey()), DataKey());
    }

private:
    Memory::DeviceOnlyArray<T> data_;
};


}}}

#endif // PACBIO_CUDA_MEMORY_DEVICE_ONLY_OBJECT_CUH_
