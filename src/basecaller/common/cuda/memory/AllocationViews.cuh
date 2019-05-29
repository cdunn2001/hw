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

// Extensions to UnifiedCudaArray available only in cuda compilation units.
// In particular provide array access to device data when in device code

#ifndef PACBIO_CUDA_MEMORY_ALLOCATION_VIEW_CUH_
#define PACBIO_CUDA_MEMORY_ALLOCATION_VIEW_CUH_

#include <common/cuda/memory/AllocationViews.h>

namespace PacBio {
namespace Cuda {
namespace Memory {

template <typename T>
class DeviceView : public DeviceHandle<T>
{
    using Parent = DeviceHandle<T>;
    using Parent::data_;
    using Parent::len_;
public:
    __device__ __host__ DeviceView(const DeviceHandle<T>& handle) : DeviceHandle<T>(handle) {}

    __device__ T& operator[](size_t idx) { return data_[idx]; }
    __device__ const T& operator[](size_t idx) const { return data_[idx]; }

    __device__ T* Data() { return data_; }
    __device__ size_t Size() const { return len_; }
};


}}}

#endif /* PACBIO_CUDA_MEMORY_ALLOCATION_VIEW_CUH_ */
