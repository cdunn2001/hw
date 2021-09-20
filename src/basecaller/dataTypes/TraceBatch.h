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

#ifndef PACBIO_MONGO_DATA_TRACE_BATCH_H
#define PACBIO_MONGO_DATA_TRACE_BATCH_H

#include <cstdint>
#include <variant>

#include <pacbio/datasource/SensorPacket.h>

#include "BatchData.h"
#include "BatchMetadata.h"

namespace PacBio {
namespace Mongo {
namespace Data {

// A TraceBatch is just a wrapper around a BatchData, with some associated metadata
// that identifies the batch within the larger acquisition.
template <typename T>
class TraceBatch : public BatchData<T>
{
public:
    TraceBatch(DataSource::SensorPacket packet,
               const BatchMetadata& meta,
               const BatchDimensions& dims,
               Cuda::Memory::SyncDirection syncDirection,
               const Cuda::Memory::AllocationMarker& marker)
        : BatchData<T>(std::move(packet), dims, syncDirection, marker)
        , meta_(meta)
    {}

    TraceBatch(const BatchMetadata& meta,
               const BatchDimensions& dims,
               Cuda::Memory::SyncDirection syncDirection,
               const Cuda::Memory::AllocationMarker& marker)
        : BatchData<T>(dims, syncDirection, marker)
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

// Variant that can hold the supported types for raw trace inputs
// into the compute graph
using TraceVariant = std::variant<TraceBatch<int16_t>, TraceBatch<uint8_t>>;

// Define overloads for this function, so that we can track kernel invocations, and
// so that we can be converted to our gpu specific representation
template <typename T>
auto KernelArgConvert(TraceBatch<T>& obj, const Cuda::KernelLaunchInfo& info)
{
    return obj.GetDeviceHandle(info);
}
template <typename T>
auto KernelArgConvert(const TraceBatch<T>& obj, const Cuda::KernelLaunchInfo& info)
{
    return obj.GetDeviceHandle(info);
}

}}}

#endif //PACBIO_MONGO_DATA_TRACE_BATCH_H
