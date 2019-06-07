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

#include "BatchData.h"
#include "BatchMetadata.h"

namespace PacBio {
namespace Mongo {
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
