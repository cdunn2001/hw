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

#include <atomic>
#include <cassert>
#include <memory>
#include <mutex>

#include <common/cuda/PBCudaRuntime.h>

#include "DataManagerKey.h"

namespace PacBio {
namespace Cuda {
namespace Memory {


// RAII managed host allocation that is compatible with
// efficient gpu data transfers.
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
        void operator()(void* ptr)
        {
            if (pinned_) CudaFreeHost(ptr);
            else free(ptr);
        }
    private:
        bool pinned_;
    };
    using Storage = std::unique_ptr<void, Deleter>;
    static Storage AllocateHelper(size_t size, bool pinned)
    {
        if (size == 0) return Storage(nullptr, Deleter(pinned));
        else
        {
            return Storage(pinned ? CudaRawMallocHost(size) : malloc(size),
                           Deleter(pinned));
        }
    }
public:
    SmartHostAllocation(size_t size = 0, bool pinned = true, size_t hash = 0)
        : data_(AllocateHelper(size, pinned))
        , size_(size)
        , hash_(hash)
    {
        if (size_ > 0)
        {
            std::lock_guard<std::mutex> lm(m_);
            bytesAllocated_ += size;
            peakBytesAllocated_ = std::max(peakBytesAllocated_.load(), bytesAllocated_.load());
        }
    }

    SmartHostAllocation(const SmartHostAllocation&) = delete;
    SmartHostAllocation(SmartHostAllocation&& other)
        : data_(std::move(other.data_))
        , size_(other.size_)
        , hash_(other.hash_)
    {
        other.size_ = 0;
    }

    SmartHostAllocation& operator=(const SmartHostAllocation&) = delete;
    SmartHostAllocation& operator=(SmartHostAllocation&& other)
    {
        data_ = std::move(other.data_);
        size_ = other.size_;
        hash_ = other.hash_;
        other.size_ = 0;
        return *this;
    }

    ~SmartHostAllocation()
    {
        if (size_ != 0)
        {
            std::lock_guard<std::mutex> lm(m_);
            assert(bytesAllocated_ >= size_);
            bytesAllocated_ -= size_;
        }
    }

    size_t Hash() const { return hash_; }

    template <typename T>
    T* get(detail::DataManagerKey) { return static_cast<T*>(data_.get()); }
    template <typename T>
    const T* get(detail::DataManagerKey) const { return static_cast<const T*>(data_.get()); }
    size_t size() const { return size_; }
    operator bool() const { return static_cast<bool>(data_); }

    static size_t CurrentAllocatedBytes()
    {
        return bytesAllocated_;
    }

    static size_t PeekAllocatedBytes()
    {
        return peakBytesAllocated_;
    }

private:
    Storage data_;
    size_t size_;
    size_t hash_;

    static std::atomic<size_t> bytesAllocated_;
    static std::atomic<size_t> peakBytesAllocated_;
    static std::mutex m_;
};


}}}

#endif // PACBIO_CUDA_MEMORY_SMART_HOST_ALLOCATION_H
