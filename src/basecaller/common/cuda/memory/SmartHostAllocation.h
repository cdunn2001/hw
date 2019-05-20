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

#ifndef PACBIO_CUDA_MEMORY_SMART_HOST_ALLOCATION_H
#define PACBIO_CUDA_MEMORY_SMART_HOST_ALLOCATION_H

#include <memory>

#include <common/cuda/PBCudaRuntime.h>

#include "DataManagerKey.h"

namespace PacBio {
namespace Cuda {
namespace Memory {


// RAII managed host allocation that is compatible with
// efficient gpu data transfers.
template <typename T>
class SmartHostAllocation
{
private:
    // Stateful deleter, to keep track of if we have to talk to
    // cuda runtime or not for deallocating memory
    struct Deleter
    {
        Deleter(bool pinned = true)
            : pinned_(pinned)
        {}
        void operator()(T* ptr)
        {
            if (pinned_) CudaFreeHost(ptr);
            else free(ptr);
        }
    private:
        bool pinned_;
    };
    using Storage = std::unique_ptr<T[], Deleter>;
    static Storage AllocateHelper(size_t count, bool pinned)
    {
        if (count == 0) return Storage(nullptr);
        else
        {
            return Storage(pinned ? CudaMallocHost<T>(count) : static_cast<T*>(malloc(count*sizeof(T))),
                           Deleter(pinned));
        }
    }
public:
    SmartHostAllocation(size_t count, bool pinned = true)
        : data_(AllocateHelper(count, pinned))
        , count_(count)
    {}

    SmartHostAllocation(const SmartHostAllocation&) = delete;
    SmartHostAllocation(SmartHostAllocation&& other)
        : data_(std::move(other.data_))
        , count_(other.count_)
    {
        other.count_ = 0;
    }

    SmartHostAllocation& operator=(const SmartHostAllocation&) = delete;
    SmartHostAllocation& operator=(SmartHostAllocation&& other)
    {
        data_ = std::move(other.data_);
        count_ = other.count_;
        other.count_ = 0;
        return *this;
    }

    ~SmartHostAllocation(){}

    T* get(detail::DataManagerKey) { return data_.get(); }
    const T* get(detail::DataManagerKey) const { return data_.get(); }
    size_t size() const { return count_; }
    operator bool() const { return static_cast<bool>(data_); }

private:
    std::unique_ptr<T[], Deleter> data_;
    size_t count_;
};


}}}

#endif // PACBIO_CUDA_MEMORY_SMART_HOST_ALLOCATION_H
