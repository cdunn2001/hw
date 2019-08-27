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
//

#ifndef PACBIO_MONGO_DATA_BATCH_VECTORS_CUH
#define PACBIO_MONGO_DATA_BATCH_VECTORS_CUH

#include "BatchVectors.h"
#include <common/cuda/memory/AllocationViews.cuh>

namespace PacBio {
namespace Mongo {
namespace Data {

template <typename T,
          typename Len_t = typename std::conditional<
              std::is_const<T>::value,
              const uint32_t,
              uint32_t>::type
          >
class VectorView : private Cuda::Memory::detail::DataManager
{
public:
    __device__ VectorView(T* data,
               Len_t * len,
               uint32_t maxLen,
               Cuda::Memory::detail::DataManagerKey)
        : data_(Cuda::Memory::DeviceHandle<T>(data, maxLen, DataKey()))
        , len_(len)
    {}

    template <typename U = T, std::enable_if_t<!std::is_const<U>::value, int> = 0>
    __device__ operator VectorView<const T>()
    {
        return VectorView<const T>(data_.Data(), len_, data_.Size());
    }

    __device__ void Reset()
    {
        *len_ = 0;
    }

    __device__ const T& operator[](size_t idx) const
    {
        assert(idx < data_.Size());
        return data_[idx];
    }
    __device__ T& operator[](size_t idx)
    {
        return const_cast<T&>(const_cast<const VectorView&>(*this)[idx]);
    }

    template <typename U, std::enable_if_t<std::is_assignable<T, U>::value, int> = 0>
    __device__ void push_back(U&& val)
    {
        assert(*len_ < data_.Size());
        data_[*len_] = std::forward<U>(val);
        (*len_)++;
    }

    // Creates a default initialized entry at the back of the vector
    // Note that this implementation is limited to trivially_default_constructible
    // types, so this is in fact a noop.
    __device__ void emplace_back_default()
    {
        assert(*len_ < data_.Size());
        (*len_)++;
    }

    __device__ T& back()
    {
        assert(*len_ <= data_.Size());
        assert(*len_ > 0);
        return data_[*len_-1];
    }

private:
    Cuda::Memory::DeviceView<T> data_;
    Len_t* len_;
};

template <typename T>
class GpuBatchVectors : private Cuda::Memory::detail::DataManager
{
public:
    GpuBatchVectors(BatchVectors<T>& vecs, const Cuda::KernelLaunchInfo& info)
        : maxLen_(vecs.MaxLen())
        , data_(vecs.Data({}).GetDeviceHandle(info))
        , lens_(vecs.Lens({}).GetDeviceHandle(info))
    {}

    template <typename U = T, std::enable_if_t<std::is_const<U>::value, int> = 0>
    GpuBatchVectors(const BatchVectors<typename std::remove_const<T>::type>& vecs,
                    const Cuda::KernelLaunchInfo& info)
        : maxLen_(vecs.MaxLen())
        , data_(vecs.Data({}).GetDeviceHandle(info))
        , lens_(vecs.Lens({}).GetDeviceHandle(info))
    {}

    __device__ VectorView<T> GetVector(int zmw)
    {
        assert(zmw < lens_.Size());
        auto offset = zmw*maxLen_;
        return VectorView<T>(data_.Data() + offset,
                             lens_.Data() + zmw,
                             maxLen_,
                             DataKey());
    }
    __device__ VectorView<const T> GetVector(int zmw) const
    {
        assert(zmw < lens_.Size());
        auto offset = zmw*maxLen_;
        return VectorView<const T>(data_.Data() + offset,
                                   lens_.Data() + zmw,
                                    maxLen_,
                                    DataKey());
    }

private:
    uint32_t maxLen_;

    using idx_t = typename std::conditional<std::is_const<T>::value, const uint32_t, uint32_t>::type;
    Cuda::Memory::DeviceView<T> data_;
    Cuda::Memory::DeviceView<idx_t> lens_;
};

template <typename T>
auto KernelArgConvert(BatchVectors<T>& vecs, const Cuda::KernelLaunchInfo& info)
{
    return GpuBatchVectors<T>(vecs, info);
}

template <typename T>
auto KernelArgConvert(const BatchVectors<T>& vecs, const Cuda::KernelLaunchInfo& info)
{
    return GpuBatchVectors<const T>(vecs, info);
}

}}}

#endif // PACBIO_MONGO_DATA_BATCH_VECTORS_CUH
