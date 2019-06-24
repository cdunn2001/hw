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

#ifndef PACBIO_MONGO_DATA_TRACE_BATCH_H
#define PACBIO_MONGO_DATA_TRACE_BATCH_H

#include <cstdint>

#include <common/cuda/memory/DataManagerKey.h>
#include <common/cuda/memory/UnifiedCudaArray.h>
#include <common/MongoConstants.h>

#include "BatchMetadata.h"

namespace PacBio {
namespace Mongo {

template <typename T, unsigned int N>
class ConstLaneArrayRef;

template <typename T, unsigned int N>
class LaneArrayRef;

template <typename T, unsigned int N>
class LaneArray;

namespace Data {

// Non-owning host-side representation of a gpu batch.  Does not
// grant access to the data and is meant primarily as a shim class
// helping segregate vanilla c++ code from cuda code.  Actual
// device functions can access the data by including TraceBatch.cuh.
// which defines the GpuBatchData class and can interact with the
// data while running on the device.
template <typename T>
class GpuBatchDataHandle
{
    using DataManagerKey = Cuda::Memory::detail::DataManagerKey;
public:
    GpuBatchDataHandle(const BatchDimensions& dims,
                       Cuda::Memory::DeviceHandle<T> data,
                       DataManagerKey key)
        : dims_(dims)
        , data_(data)
    {}

    const BatchDimensions& Dimensions() const { return dims_; }
    const Cuda::Memory::DeviceHandle<T>& Data(DataManagerKey key) { return data_; }

protected:
    BatchDimensions dims_;
    Cuda::Memory::DeviceHandle<T> data_;
};

// Non-owning host-side view of an individual block contained within a batch.
// Facilitates both 1D raw access, as well as 2D (frame, zmw) access.
template <typename T>
class BlockView
{
    using DataManagerKey = Cuda::Memory::detail::DataManagerKey;
public:
    class ConstLaneIterator;

    class LaneIterator
    {
    public:
        using DiffType = std::ptrdiff_t;
    public:
        static DiffType distance(const LaneIterator& first, const LaneIterator& last)
        {
            return (last.curFrame_ - first.curFrame_);
        }

        friend ConstLaneIterator;
    public:
        LaneIterator(T* ptr, size_t curFrame, size_t laneWidth, size_t numFrames)
            : ptr_(ptr)
            , curFrame_(curFrame)
            , laneWidth_(laneWidth)
            , numFrames_(numFrames)
        { }

        bool operator==(const LaneIterator& other) const
        {
            return curFrame_ == other.curFrame_;
        }

        bool operator!=(const LaneIterator& other) const
        {
            return curFrame_ != other.curFrame_;
        }

        LaneIterator& operator++()
        {
            curFrame_ = std::min(curFrame_ + 1, numFrames_);
            return *this;
        }

        LaneIterator operator++(int)
        {
            LaneIterator tmp = *this;
            ++*this;
            return tmp;
        }

        LaneIterator& operator+=(size_t v)
        {
            curFrame_ = std::min(curFrame_ + v, numFrames_);
            return *this;
        }

        LaneArrayRef<T, laneSize> operator*()
        {
            return LaneArrayRef<T, laneSize>(ptr_ + (curFrame_ * laneWidth_));
        }

        ConstLaneArrayRef<T, laneSize> operator*() const
        {
            return ConstLaneArrayRef<T, laneSize>(ptr_ + (curFrame_ * laneWidth_));
        }

        LaneArrayRef<T, laneSize> operator[](size_t idx)
        {
            assert(curFrame_ + idx < numFrames_);
            return LaneArrayRef<T, laneSize>(ptr_ + ((curFrame_ + idx) * laneWidth_));
        }

        ConstLaneArrayRef<T, laneSize> operator[](size_t idx) const
        {
            assert(curFrame_ + idx < numFrames_);
            return ConstLaneArrayRef<T, laneSize>(ptr_ + ((curFrame_ + idx) * laneWidth_));
        }
    private:
        T* ptr_;
        size_t curFrame_;
        size_t laneWidth_;
        size_t numFrames_;
    };

    class ConstLaneIterator
    {
    public:
        using DiffType = std::ptrdiff_t;
        using ValueType = LaneArray<T, laneSize>;
    public:
        static DiffType distance(const ConstLaneIterator& first, const ConstLaneIterator& last)
        {
            return (last.curFrame_ - first.curFrame_);
        }
    public:
        ConstLaneIterator(T* ptr, size_t curFrame, size_t laneWidth, size_t numFrames)
            : ptr_(ptr)
            , curFrame_(curFrame)
            , laneWidth_(laneWidth)
            , numFrames_(numFrames)
        { }

        ConstLaneIterator(LaneIterator& iter)
            : ptr_(iter.ptr_)
            , curFrame_(iter.curFrame_)
            , laneWidth_(iter.laneWidth_)
            , numFrames_(iter.numFrames_)
        { }

        bool operator==(const ConstLaneIterator& other) const
        {
            return curFrame_ == other.curFrame_;
        }

        bool operator!=(const ConstLaneIterator& other) const
        {
            return curFrame_ != other.curFrame_;
        }

        ConstLaneIterator& operator++()
        {
            curFrame_ = std::min(curFrame_ + 1, numFrames_);
            return *this;
        }

        ConstLaneIterator operator++(int)
        {
            ConstLaneIterator tmp = *this;
            ++*this;
            return tmp;
        }

        ConstLaneIterator& operator+=(size_t v)
        {
            curFrame_ = std::min(curFrame_ + v, numFrames_);
            return *this;
        }

        ConstLaneArrayRef<T, laneSize> operator*()
        {
            return ConstLaneArrayRef<T, laneSize>(ptr_ + (curFrame_ * laneWidth_));
        }

        ConstLaneArrayRef<T, laneSize> operator*() const
        {
            return ConstLaneArrayRef<T, laneSize>(ptr_ + (curFrame_ * laneWidth_));
        }

        ConstLaneArrayRef<T, laneSize> operator[](size_t idx)
        {
            assert(curFrame_ + idx < numFrames_);
            return ConstLaneArrayRef<T, laneSize>(ptr_ + ((curFrame_ + idx) * laneWidth_));
        }

        ConstLaneArrayRef<T, laneSize> operator[](size_t idx) const
        {
            assert(curFrame_ + idx < numFrames_);
            return ConstLaneArrayRef<T, laneSize>(ptr_ + ((curFrame_ + idx) * laneWidth_));
        }
    private:
        T* ptr_;
        size_t curFrame_;
        size_t laneWidth_;
        size_t numFrames_;
    };

public:
    using Iterator = LaneIterator;
    using ConstIterator = ConstLaneIterator;

public:

    Iterator Begin()
    { return LaneIterator(data_, 0, laneWidth_, numFrames_); }

    Iterator End()
    { return LaneIterator(data_, numFrames_, laneWidth_, numFrames_); }

    ConstIterator CBegin() const
    { return ConstLaneIterator(data_, 0, laneWidth_, numFrames_); }

    ConstIterator CEnd() const
    { return ConstLaneIterator(data_ , numFrames_, laneWidth_, numFrames_); }

public:
    BlockView(T* data, size_t laneWidth, size_t numFrames, DataManagerKey key)
        : data_(data)
        , laneWidth_(laneWidth)
        , numFrames_(numFrames)
    {}

    size_t LaneWidth() const { return laneWidth_; }
    size_t NumFrames() const { return numFrames_; }
    size_t Size() const { return laneWidth_ * numFrames_; }

    T* Data() { return data_; }
    const T* Data() const { return data_; }

    T& operator[](size_t idx) { return data_[idx]; }
    const T& operator[](size_t idx) const { return data_[idx]; }

    T& operator()(size_t frame, size_t zmw) { return data_[frame*laneWidth_ + zmw]; }
    const T& operator()(size_t frame, size_t zmw) const { return data_[frame*laneWidth_ + zmw]; }
private:
    T* data_;
    size_t laneWidth_;
    size_t numFrames_;
};

// BatchData defines a 3D layout of data designed for efficient usage on the GPU.
// Data is conceptually laid out as [lane][frame][zmw], where this is standard C
// convention and the rightmost index is contiguous in memory.  Dimensions are
// configurable, but in general will be chosen to accommodate efficiency on the GPU.
// In particular, the lane width will be chosen to correspond to the size of gpu
// cuda blocks, and the number of lanes will be chosen to have sufficiently large
// kernel executions such that any launch overhead is negligable.  The zmw for
// a given frame are interleaved to match efficient gpu memory access patterns.
//
// A BatchData class can freely migrate to and from the GPU, though there are
// some caveates.  Migration is automatic, accessing a `BlockView` will make
// sure the data is on the host (if not already there), and accessing GetGpuHandle
// will make sure the data is on the gpu (again if not already there).  As with
// UnifiedCudaArray, the syncDirection parameter controls the semantics of that
// migration.  Data upload to the gpu is asynchronous and download to the host
// is synchronous, though both of those rely on this oject being owned by the
// same thread for the whole duration.  In order to move this object to another
// thread you should explicitly cause a synchronization first.
//
// To avoid unecessary utilization of the relatively scarce gpu memory, this
// class can be constructed with a GpuAllocationPool.  This is important as
// the host will need at least an entire chip's worth of batches to place
// data as it streams in, but only a small handful will be processed on the gpu
// at one time.  By calling `DeactivateGpuMem` whenever gpu processing on a
// batch is finished, the underlying gpu allocation can be placed back in
// the memory pool for another batch to check out once it becomes active.
template <typename T>
class BatchData : private Cuda::Memory::detail::DataManager
{
public:
    BatchData(const BatchDimensions& dims,
              Cuda::Memory::SyncDirection syncDirection,
              std::shared_ptr<Cuda::Memory::DualAllocationPools> pool,
              bool pinnedHost = true)
        : dims_(dims)
        , data_(dims.laneWidth * dims.framesPerBatch * dims.lanesPerBatch,
                syncDirection, pinnedHost, pool)
    {}

    BatchData(const BatchData&) = delete;
    BatchData(BatchData&&) = default;
    BatchData& operator=(const BatchData&) = delete;
    BatchData& operator=(BatchData&&) = default;

    ~BatchData() = default;

    // Can be null, if there are no pools in use
    std::shared_ptr<Cuda::Memory::DualAllocationPools> GetAllocationPools() const
    {
        return data_.GetAllocationPools();
    }

    size_t LaneWidth()     const { return dims_.laneWidth; }
    size_t numFrames()     const { return dims_.framesPerBatch; }
    size_t LanesPerBatch() const { return dims_.lanesPerBatch; }

    const BatchDimensions& Dimensions() const { return dims_; }

    Cuda::Memory::UnifiedCudaArray<T>& GetRawData(Cuda::Memory::detail::DataManagerKey) { return data_; }
    const Cuda::Memory::UnifiedCudaArray<T>& GetRawData(Cuda::Memory::detail::DataManagerKey) const { return data_; }

    void DeactivateGpuMem() { data_.DeactivateGpuMem(); }
    void CopyToDevice() { data_.CopyToDevice(); }

    BlockView<T> GetBlockView(size_t laneIdx)
    {
        auto view = data_.GetHostView();
        return BlockView<T>(view.Data() + laneIdx * dims_.framesPerBatch * dims_.laneWidth,
                            dims_.laneWidth,
                            dims_.framesPerBatch,
                            DataKey());
    }
    BlockView<const T> GetBlockView(size_t laneIdx) const
    {
        auto view = data_.GetHostView();
        return BlockView<const T>(view.Data() + laneIdx * dims_.framesPerBatch * dims_.laneWidth,
                            dims_.laneWidth,
                            dims_.framesPerBatch,
                            DataKey());
    }
private:
    BatchDimensions dims_;
    Cuda::Memory::UnifiedCudaArray<T> data_;
};

// A TraceBatch is just a wrapper around a BatchData, with some associated metadata
// that identifies the batch within the larger acquisition.
template <typename T>
class TraceBatch : public BatchData<T>
{
public:
    TraceBatch(const BatchMetadata& meta,
               const BatchDimensions& dims,
               Cuda::Memory::SyncDirection syncDirection,
               std::shared_ptr<Cuda::Memory::DualAllocationPools> pool,
               bool pinnedHost = true)
        : BatchData<T>(dims, syncDirection, pool, pinnedHost)
        , meta_(meta)
    {}

    TraceBatch(const BatchMetadata& meta,
               TraceBatch<T>&& batch)
        : BatchData<T>(std::move(batch))
        , meta_(meta)
    {}

    void SetMeta(const BatchMetadata& meta)
    {
        meta_ = meta;
    }
    const BatchMetadata& GetMeta() const { return meta_; }

    TraceBatch(const TraceBatch&) = delete;
    TraceBatch(TraceBatch&&) = default;
    TraceBatch& operator=(const TraceBatch&) = delete;
    TraceBatch& operator=(TraceBatch&&) = default;

    ~TraceBatch() = default;

    const BatchMetadata& Metadata() const { return meta_; }
private:
    BatchMetadata meta_;
};

}}}

#endif //PACBIO_MONGO_DATA_TRACE_BATCH_H
