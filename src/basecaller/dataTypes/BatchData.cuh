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

#ifndef PACBIO_MONGO_DATA_TRACE_BATCH_CUH
#define PACBIO_MONGO_DATA_TRACE_BATCH_CUH

#include <common/cuda/memory/AllocationViews.cuh>
#include <common/cuda/memory/UnifiedCudaArray.h>
#include <common/cuda/KernelManager.h>

#include <dataTypes/TraceBatch.h>

namespace PacBio {
namespace Mongo {
namespace Data {

// Class for efficient access of an individual zmw on the gpu.  It
// is intended that the different threads in a cuda block will create
// views for adjacent zmw in a pacbio data block.  This way the interleaved
// nature of the data along with the lockstep nature of warp execution
// means we should have optimal memory access patterns.
template <typename T>
class StridedBlockView
{
public:
    // Helper iterator to allow range-based for loops.
    template <typename U>
    class StrideIterator
    {
    public:
        __device__ StrideIterator(T* ptr, size_t stride)
            : ptr_(ptr)
            , stride_(stride)
        {}

        __device__ void operator++()
        {
            ptr_ += stride_;
        }

        __device__ bool operator==(const StrideIterator& other) const
        {
            return ptr_ == other.ptr_;
        }

        __device__ bool operator!=(const StrideIterator& other) const
        {
            return ptr_ != other.ptr_;
        }

        __device__ U& operator*() { return *ptr_; }
        __device__ const U& operator*() const { return *ptr_; }

    private:
        T* ptr_;
        size_t stride_;
    };

    __device__ StridedBlockView(T* start, T* end, size_t laneWidth, Cuda::Memory::detail::DataManagerKey key)
        : start_(start)
        , end_(end)
        , laneWidth_(laneWidth)
    {}

    __device__ StrideIterator<T> begin() { return StrideIterator<T>(start_, laneWidth_); }
    __device__ StrideIterator<T> end()   { return StrideIterator<T>(end_,   laneWidth_); }
    __device__ StrideIterator<const T> begin() const { return StrideIterator<const T>(start_, laneWidth_); }
    __device__ StrideIterator<const T> end()   const { return StrideIterator<const T>(end_,   laneWidth_); }
    __device__ int size() const { return (end_ - start_) / laneWidth_; }
    __device__ T& operator[](unsigned int idx) { return start_[laneWidth_*idx]; }
    __device__ const T& operator[](unsigned int idx) const { return start_[laneWidth_*idx]; }
private:
    T* start_;
    T* end_;
    size_t laneWidth_;
};

template <typename T>
class GpuBatchData : public GpuBatchDataHandle<T>, private Cuda::Memory::detail::DataManager
{
    using GpuBatchDataHandle<T>::data_;
    using GpuBatchDataHandle<T>::availableFrames_;
    using GpuBatchDataHandle<T>::dims_;
public:
    GpuBatchData(const GpuBatchDataHandle<T>& handle)
        : GpuBatchDataHandle<T>(handle)
    {}
    template <typename U = T, std::enable_if_t<std::is_const<U>::value, int> = 0>
    GpuBatchData(const GpuBatchDataHandle<typename std::remove_const<T>::type>& handle)
        : GpuBatchDataHandle<T>(handle.Dimensions(), handle.NumFrames(), handle.Data(DataKey()), DataKey())
    {}

    __device__ StridedBlockView<T> ZmwData(size_t laneIdx, size_t zmwIdx)
    {
        return ZmwDataImpl<T>(laneIdx, zmwIdx);
    }
    __device__ StridedBlockView<const T> ZmwData(size_t laneIdx, size_t zmwIdx) const
    {
        return ZmwDataImpl<const T>(laneIdx, zmwIdx);
    }

    __device__ uint32_t NumFrames() const { return availableFrames_; }

private:
    template <typename U>
    __device__ StridedBlockView<U> ZmwDataImpl(size_t laneIdx, size_t zmwIdx) const
    {
        Cuda::Memory::DeviceView<T> view(data_);
        auto startIdx = laneIdx * dims_.laneWidth * dims_.framesPerBatch + zmwIdx;
        auto endIdx = startIdx + availableFrames_ * dims_.laneWidth;
        return StridedBlockView<U>(view.Data() + startIdx,
                                   view.Data() + endIdx,
                                   dims_.laneWidth,
                                   DataKey());
    }
};

}}}

#endif // PACBIO_MONGO_DATA_TRACE_BATCH_CUH
