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

#ifndef PACBIO_CUDA_MEMORY_SMART_DEVICE_ALLOCATION_H
#define PACBIO_CUDA_MEMORY_SMART_DEVICE_ALLOCATION_H

#include <memory>

#include <common/cuda/PBCudaRuntime.h>

#include "DataManagerKey.h"

namespace PacBio {
namespace Cuda {
namespace Memory {

// RAII managed allocation on the GPU device.  This handles
// raw allocations *ONLY*.  No constructors or destructors will
// be called.  Initialization of gpu data could be done
// either by construction on the host and copying up to the
// gpu, or actually invoking cuda kernels to call constructors
// directly on the gpu.  By handling only allocations, this
// allows a unified representation of gpu memory, while letting
// individual implementations choose their own initialization
// scheme.
class SmartDeviceAllocation
{
public:
    SmartDeviceAllocation(size_t size = 0)
        : data_(size ? CudaRawMalloc(size) : nullptr)
        , size_(size)
    {}

    SmartDeviceAllocation(const SmartDeviceAllocation&) = delete;
    SmartDeviceAllocation(SmartDeviceAllocation&& other)
        : data_(std::move(other.data_))
        , size_(other.size_)
    {
        other.size_ = 0;
    }

    SmartDeviceAllocation& operator=(const SmartDeviceAllocation&) = delete;
    SmartDeviceAllocation& operator=(SmartDeviceAllocation&& other)
    {
        data_ = std::move(other.data_);
        size_ = other.size_;
        other.size_ = 0;
        return *this;
    }

    ~SmartDeviceAllocation() = default;

    template <typename T>
    T* get(detail::DataManagerKey) { return static_cast<T*>(data_.get()); }
    template <typename T>
    const T* get(detail::DataManagerKey) const { return static_cast<const T*>(data_.get()); }
    size_t size() const { return size_; }
    operator bool() const { return static_cast<bool>(data_); }

private:
    struct Deleter
    {
        void operator()(void* ptr) { CudaFree(ptr); }
    };
    std::unique_ptr<void, Deleter> data_;
    size_t size_;
};


}}}

#endif //PACBIO_CUDA_MEMORY_SMART_DEVICE_ALLOCATION_H
