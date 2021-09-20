#ifndef mongo_dataTypes_BatchData_H_
#define mongo_dataTypes_BatchData_H_

// Copyright (c) 2019-2021, Pacific Biosciences of California, Inc.
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
//  Description:
//  Defines classes BatchData, BatchDimensions, and BlockView.

#include <pacbio/datasource/SensorPacket.h>

#include <common/cuda/memory/DataManagerKey.h>
#include <common/cuda/memory/UnifiedCudaArray.h>
#include <common/LaneArray_fwd.h>
#include <common/MongoConstants.h>

namespace PacBio {
namespace Mongo {
namespace Data {

class BatchDimensions
{
public:     // Functions
    BatchDimensions() = default;
    BatchDimensions(const BatchDimensions&) = default;
    BatchDimensions(BatchDimensions&&) = default;
    BatchDimensions& operator=(const BatchDimensions&) = default;
    BatchDimensions& operator=(BatchDimensions&&) = default;

    uint32_t ZmwsPerBatch() const
    {
        // TODO: Strictly speaking, there's an overflow risk here. Pretty sure,
        // however, that we won't have more than four billion ZMWs per batch in
        // the forseeable future.
        return laneWidth * lanesPerBatch;
    }

    uint32_t LaneStride() const
    { return framesPerBatch * laneWidth; }

public:
    // Notice that the declaration order matches the BatchData layout order
    // in the sense of C-style indexing.
    uint32_t lanesPerBatch;
    uint32_t framesPerBatch;

    // TODO: In many places we assume at compile time that laneWidth = laneSize. Eliminate the laneWidth field.
    uint32_t laneWidth = laneSize;
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
            return *this+=1;
        }

        LaneIterator operator++(int)
        {
            LaneIterator tmp = *this;
            ++*this;
            return tmp;
        }

        LaneIterator& operator+=(int32_t v)
        {
            curFrame_ = curFrame_ + v;
            return *this;
        }

        LaneIterator operator+(int32_t v)
        {
            auto ret = *this;
            return ret+=v;
        }

        LaneIterator& operator--()
        {
            return *this-=1;
        }

        LaneIterator operator--(int)
        {
            LaneIterator tmp = *this;
            --*this;
            return tmp;
        }

        LaneIterator& operator-=(int32_t v)
        {
            curFrame_ = curFrame_ - v;
            return *this;
        }
        LaneIterator operator-(int32_t v)
        {
            auto ret = *this;
            return ret-=v;
        }

        LaneArray<T, laneSize> Extract() const
        {
            if (curFrame_ >= numFrames_) throw PBException("Out of bounds: Past End");
            if (curFrame_ < 0) throw PBException("Out of bounds: Before Start");
            return LaneArray<T, laneSize>(MemoryRange<T, laneSize>{ptr_ + (curFrame_ * laneWidth_)});
        }

        void Store(const LaneArray<T, laneSize>& lane)
        {
            if (curFrame_ >= numFrames_) throw PBException("Out of bounds: Past End");
            if (curFrame_ < 0) throw PBException("Out of bounds: Before Start");
            memcpy(ptr_ + (curFrame_ * laneWidth_), &lane, sizeof(lane));
        }

    private:
        T* ptr_;
        int32_t curFrame_;
        int32_t laneWidth_;
        int32_t numFrames_;
    };

    class ConstLaneIterator
    {
    public:
        using DiffType = std::ptrdiff_t;
        using ValueType = LaneArray<std::remove_const_t<T>, laneSize>;
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
            return *this+=1;
        }

        ConstLaneIterator operator++(int)
        {
            ConstLaneIterator tmp = *this;
            ++*this;
            return tmp;
        }

        ConstLaneIterator& operator+=(int32_t v)
        {
            curFrame_ = curFrame_ + v;
            if (curFrame_ > numFrames_) throw PBException("Out of bounds LaneIterator increment");
            return *this;
        }

        ConstLaneIterator operator+(int32_t v)
        {
            auto ret = *this;
            return ret+=v;
        }

        ConstLaneIterator& operator--()
        {
            return *this-=1;
        }

        ConstLaneIterator operator--(int)
        {
            ConstLaneIterator tmp = *this;
            --*this;
            return tmp;
        }

        ConstLaneIterator& operator-=(int32_t v)
        {
            if (curFrame_ < v) throw PBException("Out of bounds LaneIterator decrement");
            curFrame_ = curFrame_ - v;
            return *this;
        }

        ConstLaneIterator operator-(int32_t v)
        {
            auto ret = *this;
            return ret-=v;
        }

        ValueType Extract() const
        {
            if (curFrame_ >= numFrames_) throw PBException("Out of bounds: Past End");
            if (curFrame_ < 0) throw PBException("Out of bounds: Before Start");
            return ValueType(MemoryRange<std::remove_const_t<T>, laneSize>{ptr_ + (curFrame_ * laneWidth_)});
        }

    private:
        T* ptr_;
        int32_t curFrame_;
        int32_t laneWidth_;
        int32_t numFrames_;
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
    BlockView(T* data, size_t laneWidth, size_t numFrames, DataManagerKey)
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
                       uint32_t availableFrames,
                       Cuda::Memory::DeviceHandle<T> data,
                       DataManagerKey)
        : dims_(dims)
        , availableFrames_(availableFrames)
        , data_(data)
    {}

    const BatchDimensions& Dimensions() const { return dims_; }
    const Cuda::Memory::DeviceHandle<T>& Data(DataManagerKey) const { return data_; }
    uint32_t NumFrames() const { return availableFrames_; }

protected:
    BatchDimensions dims_;
    uint32_t availableFrames_;
    Cuda::Memory::DeviceHandle<T> data_;
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
template <typename T>
class BatchData : private Cuda::Memory::detail::DataManager
{
    using GpuType = typename Cuda::Memory::UnifiedCudaArray<T>::GpuType;
    using HostType = typename Cuda::Memory::UnifiedCudaArray<T>::HostType;


    // Helper validation function, to make sure if we are constructing using
    // data from a SensorPacket, that packet's dimensions are consistent
    const BatchDimensions& ValidateDims(const BatchDimensions& dims,
                                        const DataSource::PacketLayout& layout)
    {
        if (layout.Encoding() != DataSource::PacketLayout::INT16)
            throw PBException("Cannot create batch from 12 bit SensorPacket");
        if (layout.Type() == DataSource::PacketLayout::FRAME_LAYOUT)
            throw PBException("Cannot create batch from SensorPacket with frame data");

        if (dims.framesPerBatch != layout.NumFrames())
            throw PBException("PacketLayout had " + std::to_string(layout.NumFrames()) +
                              " frames, but " + std::to_string(dims.framesPerBatch) + " was expected");
        if (dims.laneWidth != layout.BlockWidth())
            throw PBException("PacketLayout had " + std::to_string(layout.BlockWidth()) +
                              " lane width, but " + std::to_string(dims.laneWidth) + " was expected");
        if (dims.lanesPerBatch != layout.NumBlocks())
            throw PBException("PacketLayout had " + std::to_string(layout.NumBlocks()) +
                              " num blocks, but " + std::to_string(dims.lanesPerBatch) + " was expected");

        return dims;
    }
public:
    BatchData(DataSource::SensorPacket packet,
              const BatchDimensions& dims,
              Cuda::Memory::SyncDirection syncDirection,
              const Cuda::Memory::AllocationMarker& marker)
        : dims_(ValidateDims(dims, packet.Layout()))
        , availableFrames_(dims.framesPerBatch)
        , data_(std::move(packet).RelinquishAllocation(), dims.laneWidth * dims.framesPerBatch * dims.lanesPerBatch,
                syncDirection, marker)
    {}

    BatchData(const BatchDimensions& dims,
              Cuda::Memory::SyncDirection syncDirection,
              const Cuda::Memory::AllocationMarker& marker)
        : dims_(dims)
        , availableFrames_(dims.framesPerBatch)
        , data_(dims.laneWidth * dims.framesPerBatch * dims.lanesPerBatch,
                syncDirection, marker)
    {}

    BatchData(const BatchData&) = delete;
    BatchData(BatchData&&) = default;
    BatchData& operator=(const BatchData&) = delete;
    BatchData& operator=(BatchData&&) = default;

    ~BatchData() = default;

    size_t LaneWidth()     const { return dims_.laneWidth; }
    size_t NumFrames()     const { return availableFrames_; }
    size_t LanesPerBatch() const { return dims_.lanesPerBatch; }

    void SetFrameLimit(uint32_t frames)
    {
        if (frames > dims_.framesPerBatch) throw PBException("New BatchData frame limit exceeds underlying storage");
        availableFrames_ = frames;
    }

    const BatchDimensions& StorageDims() const { return dims_; }

    GpuBatchDataHandle<GpuType> GetDeviceHandle(const Cuda::KernelLaunchInfo& info)
    {
        auto gpuDims = dims_;
        gpuDims.laneWidth /= (sizeof(GpuType) / sizeof(HostType));
        return GpuBatchDataHandle<GpuType>(gpuDims, NumFrames(), data_.GetDeviceHandle(info), DataKey());
    }
    GpuBatchDataHandle<const GpuType> GetDeviceHandle(const Cuda::KernelLaunchInfo& info) const
    {
        auto gpuDims = dims_;
        gpuDims.laneWidth /= (sizeof(GpuType) / sizeof(HostType));
        return GpuBatchDataHandle<const GpuType>(gpuDims, NumFrames(), data_.GetDeviceHandle(info), DataKey());
    }

    // Note: For this class (and UnifiedCudaArray) `const` implies that the contents of the data stays the same,
    //       but not necessarily the location.  `mutable` has been applied in a few select places such that
    //       we can copy data to (or from) the GPU even for a const object (On the gpu you'll only be able to
    //       get a const view of the data, but we still have to technically modify things in order to get
    //       the payload up there in the first place).
    // Relinquishes the device allocation, after downloading the data if necessary.
    // Returns the number of bytes actually downloaded
    size_t DeactivateGpuMem() const { return data_.DeactivateGpuMem(); }
    // Manually uploads the data to the GPU if not already present.  Returns
    // the number of bytes actually uploaded
    size_t CopyToDevice() const { return data_.CopyToDevice(); }

    BlockView<T> GetBlockView(size_t laneIdx)
    {
        auto view = data_.GetHostView();
        return BlockView<T>(view.Data() + laneIdx * dims_.framesPerBatch * dims_.laneWidth,
                            dims_.laneWidth,
                            availableFrames_,
                            DataKey());
    }
    BlockView<const T> GetBlockView(size_t laneIdx) const
    {
        auto view = data_.GetHostView();
        return BlockView<const T>(view.Data() + laneIdx * dims_.framesPerBatch * dims_.laneWidth,
                            dims_.laneWidth,
                            availableFrames_,
                            DataKey());
    }

private:
    BatchDimensions dims_;
    uint32_t availableFrames_; // Can have fewer frames than specified in dims_;
    Cuda::Memory::UnifiedCudaArray<T> data_;
};

// Define overloads for this function, so that we can track kernel invocations, and
// so that we can be converted to our gpu specific representation
template <typename T>
auto KernelArgConvert(BatchData<T>& obj, const Cuda::KernelLaunchInfo& info)
{
    return obj.GetDeviceHandle(info);
}
template <typename T>
auto KernelArgConvert(const BatchData<T>& obj, const Cuda::KernelLaunchInfo& info)
{
    return obj.GetDeviceHandle(info);
}

}}}     // namespace PacBio::Mongo::Data

#endif // mongo_dataTypes_BatchData_H_
