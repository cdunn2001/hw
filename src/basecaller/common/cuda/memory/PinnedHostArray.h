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

#ifndef PACBIO_CUDA_MEMORY_PINNED_HOST_ARRAY_H
#define PACBIO_CUDA_MEMORY_PINNED_HOST_ARRAY_H

#include <memory>

#include <common/cuda/PBCudaRuntime.h>

namespace PacBio {
namespace Cuda {
namespace Memory {

// RAII managed host array of data that is compatible with
// efficient gpu data transfers.  This class will explicitly
// ensure all constructors and destructors are correctly
// called.
template <typename T>
class PinnedHostArray
{
public:
    template <typename... Args>
    PinnedHostArray(size_t count = 0, Args&&... args)
        : data_(count ? CudaMallocHost<T>(count) : nullptr)
        , count_(count)
    {
        for (size_t i = 0; i < count_; ++i)
        {
            new (data_.get()+i) T(args...);
        }
    }

    PinnedHostArray(const PinnedHostArray&) = delete;
    PinnedHostArray(PinnedHostArray&& other)
        : data_(std::move(other.data_))
        , count_(other.count_)
    {
        other.count_ = 0;
    }

    PinnedHostArray& operator=(const PinnedHostArray&) = delete;
    PinnedHostArray& operator=(PinnedHostArray&& other)
    {
        data_ = std::move(other.data_);
        count_ = other.count_;
        other.count_ = 0;
        return *this;
    }

    ~PinnedHostArray()
    {
        for (size_t i = 0; i < count_; ++i)
        {
            data_[i].~T();
        }
    }

    T* get() { return data_.get(); }
    size_t size() const { return count_; }
    operator bool() const { return static_cast<bool>(data_); }

private:
    struct Deleter
    {
        void operator()(T* ptr) { CudaFreeHost(ptr); }
    };
    std::unique_ptr<T[], Deleter> data_;
    size_t count_;
};


}}}

#endif // PACBIO_CUDA_MEMORY_PINNED_HOST_ARRAY_H
