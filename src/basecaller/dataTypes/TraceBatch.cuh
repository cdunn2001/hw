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

        __device__ T& operator*() { return *ptr_; }
        __device__ const T& operator*() const { return *ptr_; }

    private:
        T* ptr_;
        size_t stride_;
    };

    __device__ StridedBlockView(T* start, T* end, size_t laneWidth, Cuda::Memory::detail::DataManagerKey key)
        : start_(start)
        , end_(end)
        , laneWidth_(laneWidth)
    {}

    __device__ StrideIterator begin() { return StrideIterator(start_, laneWidth_); }
    __device__ StrideIterator end()   { return StrideIterator(end_,   laneWidth_); }
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
    using GpuBatchDataHandle<T>::dims_;
public:
    GpuBatchData(const GpuBatchDataHandle<T>& handle)
        : GpuBatchDataHandle<T>(handle)
    {}
    template <typename U>
    GpuBatchData(BatchData<U>& data)
        : GpuBatchDataHandle<T>(data.Dimensions(),
                                data.GetRawData(DataKey()).GetDeviceHandle(),
                                DataKey())
    {
        // We support using things like int16_t on the host but short2 on
        // the device.  To enable that, we may need to tweak our apparent
        // lane width
        if (sizeof(U) != sizeof(T))
        {
            static_assert(sizeof(T) % sizeof(U) == 0, "Invalid types");
            this->dims_.laneWidth /= 2;
        }
    }

    __device__ const BatchDimensions& Dims() const { return dims_; }

    __device__ StridedBlockView<T> ZmwData(size_t laneIdx, size_t zmwIdx)
    {
        Cuda::Memory::DeviceView<T> view(data_);
        auto startIdx = laneIdx * dims_.laneWidth * dims_.framesPerBatch + zmwIdx;
        auto endIdx = startIdx + dims_.framesPerBatch * dims_.laneWidth;
        return StridedBlockView<T>(view.Data() + startIdx,
                                   view.Data() + endIdx,
                                   dims_.laneWidth,
                                   DataKey());
    }
};

}}}

#endif // PACBIO_MONGO_DATA_TRACE_BATCH_CUH
