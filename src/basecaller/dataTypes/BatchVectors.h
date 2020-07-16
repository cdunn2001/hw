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
// Defines a set of vectors, one for each zmw in a bach.  These vectors are
// quasi-dynamic, in that they start size 0 and can grow, but as the whole
// batch shares an allocation, there is a max size per vector.  Simply
// creating this type will consume the same memory as if it were entirely full,
// so this is only really suitable for cases where each zmw is expected to be
// (roughly) the same size and things have conservative bounds.
//
// File: Defines a set of classes used to give a consistent and synchronized
//       view of a fixed allocation size vector implementation, with one
//       vector per zmw in a pool

#ifndef PACBIO_MONGO_DATA_BATCH_VECTORS_H
#define PACBIO_MONGO_DATA_BATCH_VECTORS_H

#include <cstddef>

#include <common/cuda/memory/UnifiedCudaArray.h>
#include <common/cuda/CudaFunctionDecorators.h>
#include <common/MongoConstants.h>

namespace PacBio {
namespace Mongo {
namespace Data {

// Forwards, necessary for setting up PassKey permissions below
template <typename T>
class BatchVectors;

template <typename T>
class GpuBatchVectors;

// Host side view of the vectors for a given Lane.  If you are doing the
// first write operations after initial construction, you must call the
// Reset function to make things initialized.
template <typename T,
          typename Len_t = typename std::conditional<
              std::is_const<T>::value,
              const uint32_t,
              uint32_t>::type
          >
class LaneVectorView : private Cuda::Memory::detail::DataManager
{
public:
    LaneVectorView(T* data,
                   Len_t * lens,
                   Len_t * overflows,
                   uint32_t maxLen,
                   Cuda::Memory::detail::PassKey<BatchVectors<typename std::remove_const<T>::type>>)
        : maxLen_(maxLen)
        , data_(data, maxLen*laneSize, DataKey())
        , lens_(lens, laneSize, DataKey())
        , overflows_(overflows, laneSize, DataKey())
    {}

    template <typename U = T, std::enable_if_t<!std::is_const<U>::value, int> = 0>
    operator LaneVectorView<const T>()
    {
        return LaneVectorView<const T>(data_.Data(), lens_.Data(), overflows_.Data(), maxLen_);
    }

    // Sets all lengths back to zero.
    void Reset()
    {
        std::fill(lens_.Data(), lens_.Data() + lens_.Size(), 0);
        std::fill(overflows_.Data(), overflows_.Data() + overflows_.Size(), 0);
    }

    const T& operator()(size_t zmw, size_t idx) const
    {
        assert(zmw < laneSize);
        assert(idx < lens_[zmw]);
        return data_[zmw*maxLen_ + idx];
    }
    T& operator()(size_t zmw, size_t idx)
    {
        return const_cast<T&>(const_cast<const LaneVectorView&>(*this)(zmw, idx));
    }

    template <typename U, std::enable_if_t<std::is_assignable<T, U>::value, int> = 0>
    bool push_back(size_t zmw, U&& val)
    {
        assert(zmw < laneSize);
        assert(lens_[zmw] <= maxLen_);
        if (lens_[zmw] < maxLen_)
        {
            data_[zmw*maxLen_ + lens_[zmw]] = std::forward<U>(val);
            lens_[zmw]++;
            return true;
        } else
        {
            overflows_[zmw]++;
            return false;
        }
    }

    T* ZmwData(int zmw) { return data_.Data() + zmw * maxLen_; }
    const T* ZmwData(int zmw) const { return data_.Data() + zmw * maxLen_; }

    uint32_t size(int zmw) const { return lens_[zmw]; }
    uint32_t overflows(int zmw) const { return overflows_[zmw]; }

private:
    uint32_t maxLen_;

    Cuda::Memory::HostView<T> data_;
    Cuda::Memory::HostView<Len_t> lens_;
    Cuda::Memory::HostView<Len_t> overflows_;
};

// This class manages a set of vectors, with one vector for each zmw
// in a pool.  The vector has a fixed maximum capacity, and the class
// will always allocate enough memory for each zmw to saturate this capacity.
// It has the same semantics as classes like BatchData, where this is an
// opaque owning class, and you request host/gpu 'views' that
// automatically synchronize the data as necessary.
//
// As a special note, to avoid unecessary data transfers this class only
// supports trivially default constructible types, and upon construction
// will not initialize the per-zmw `length` information.  The first writes
// to a newly constructed `BatchData` need to be preceeded by a call to the
// `Reset` functions provided by the host/gpu view classes, to set all
// the lengths to zero.
template <typename T>
class BatchVectors
{
    template <typename U>
    using UnifiedCudaArray = Cuda::Memory::UnifiedCudaArray<U>;
    template <typename U>
    using PassKey= Cuda::Memory::detail::PassKey<U>;
public:
    BatchVectors(uint32_t zmwPerBatch,
                 uint32_t maxLen,
                 Cuda::Memory::SyncDirection syncDir,
                 const Cuda::Memory::AllocationMarker& marker)
        : zmwPerBatch_(zmwPerBatch)
        , maxLen_(maxLen)
        , data_(zmwPerBatch * maxLen, syncDir, marker)
        , lens_(zmwPerBatch, syncDir, marker)
        , overflows_(zmwPerBatch, syncDir, marker)
    {}

    LaneVectorView<T> LaneView(uint32_t laneId)
    {
        auto laneOffset = laneId * laneSize;
        assert(laneOffset < zmwPerBatch_);
        auto dView = data_.GetHostView();
        auto lView = lens_.GetHostView();
        auto oView = overflows_.GetHostView();
        return LaneVectorView<T>(dView.Data() + laneOffset*maxLen_,
                           lView.Data() + laneOffset,
                           oView.Data() + laneOffset,
                           maxLen_,
                           {});
    }
    LaneVectorView<const T> LaneView(uint32_t laneId) const
    {
        auto laneOffset = laneId * laneSize;
        assert(laneOffset < zmwPerBatch_);
        auto dView = data_.GetHostView();
        auto lView = lens_.GetHostView();
        auto oView = overflows_.GetHostView();
        return LaneVectorView<const T>(dView.Data() + laneOffset*maxLen_,
                           lView.Data() + laneOffset,
                           oView.Data() + laneOffset,
                           maxLen_,
                           {});
    }

    uint32_t MaxLen() const { return maxLen_; }

    // Semi-private functions.  `GpuBatchVectors` uses these to construct themselves,
    // but no one else should pay attention to these.
    UnifiedCudaArray<T>& Data(PassKey<GpuBatchVectors<T>>) { return data_; }
    const UnifiedCudaArray<T>& Data(PassKey<GpuBatchVectors<const T>>) const { return data_; }

    UnifiedCudaArray<uint32_t>& Lens(PassKey<GpuBatchVectors<T>>) { return lens_; }
    const UnifiedCudaArray<uint32_t>& Lens(PassKey<GpuBatchVectors<const T>>) const { return lens_; }

    UnifiedCudaArray<uint32_t>& Overflows(PassKey<GpuBatchVectors<T>>) { return overflows_; }
    const UnifiedCudaArray<uint32_t>& Overflows(PassKey<GpuBatchVectors<const T>>) const { return overflows_; }

    void DeactivateGpuMem()
    {
        data_.DeactivateGpuMem();
        lens_.DeactivateGpuMem();
    }

private:
    uint32_t zmwPerBatch_;
    uint32_t maxLen_;

    Cuda::Memory::UnifiedCudaArray<T> data_;
    Cuda::Memory::UnifiedCudaArray<uint32_t> lens_;
    Cuda::Memory::UnifiedCudaArray<uint32_t> overflows_;
};

}}} //::PacBio::Mongo::Data

#endif // PACBIO_MONGO_DATA_BATCH_VECTORS_H
