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

#ifndef PACBIO_MONGO_DATA_LABELS_BATCH_H_
#define PACBIO_MONGO_DATA_LABELS_BATCH_H_

#include "TraceBatch.h"

#include <common/cuda/memory/UnifiedCudaArray.h>
#include <common/MongoConstants.h>

#include "BatchMetrics.h"

namespace PacBio {
namespace Mongo {
namespace Data {

using Label_t = int16_t;

/// Baseline-subtracted trace data with statistics
class LabelsBatch : public TraceBatch<Label_t>
{
    static BatchDimensions LatentDimensions(const BatchDimensions& traceDims, size_t latentFrames)
    {
        BatchDimensions ret = traceDims;
        ret.framesPerBatch = latentFrames;
        return ret;
    }
public:     // Types
    using ElementType = Label_t;

public:     // Structors and assignment
    LabelsBatch(const BatchMetadata& meta,
                const BatchDimensions& dims,
                TraceBatch<ElementType> trace,
                size_t latentFrames,
                Cuda::Memory::SyncDirection syncDirection,
                const Cuda::Memory::AllocationMarker& marker)
        : TraceBatch<ElementType>(meta, dims, syncDirection, marker)
        , curTrace_(std::move(trace))
        , latTrace_(LatentDimensions(dims, latentFrames), syncDirection, marker)
    { }

    LabelsBatch(const LabelsBatch&) = delete;
    LabelsBatch(LabelsBatch&&) = default;

    LabelsBatch& operator=(const LabelsBatch&) = delete;
    LabelsBatch& operator=(LabelsBatch&&) = default;

    const BatchData& TraceData() const { return curTrace_; }
    BatchData& TraceData() { return curTrace_; }

    const BatchData& LatentTrace() const { return latTrace_; }
    BatchData& LatentTrace() { return latTrace_; }

private:    // Data
    // Full trace input to label filter, but the last few frames are held back for
    // viterbi stitching, so this class will prevent access to those
    TraceBatch<ElementType> curTrace_;

    // Latent camera trace data held over by frame labeling from the previous block
    BatchData latTrace_;
};

// Factory class, to simplify the construction of LabelsBatch instances.
// This class will handle the small collection of constructor arguments that
// need to change depending on the pipeline configuration, but otherwise are
// generally constant between different batches
class LabelsBatchFactory
{
public:
    LabelsBatchFactory(uint32_t latentFrames,
                       Cuda::Memory::SyncDirection syncDirection)
        : latentFrames_(latentFrames)
        , syncDirection_(syncDirection)
    {}

    std::pair<LabelsBatch, FrameLabelerMetrics>
    NewBatch(TraceBatch<LabelsBatch::ElementType> trace)
    {
        auto traceMeta = trace.Metadata();
        auto dims = trace.StorageDims();

        // Adjust metadata to account for latent frames.
        BatchMetadata labelsMeta(traceMeta.PoolId(),
                                 std::max(static_cast<size_t>(traceMeta.FirstFrame()), latentFrames_) - latentFrames_,
                                 std::max(static_cast<size_t>(traceMeta.LastFrame()), latentFrames_) - latentFrames_,
                                 traceMeta.FirstZmw());

        return std::make_pair(
            LabelsBatch(
                labelsMeta, dims, std::move(trace), latentFrames_, syncDirection_, SOURCE_MARKER()),
            FrameLabelerMetrics(dims, syncDirection_, SOURCE_MARKER()));
    }

    LabelsBatch NewLabels(TraceBatch<LabelsBatch::ElementType> trace)
    {
        auto traceMeta = trace.Metadata();
        auto dims = trace.StorageDims();

        // Adjust metadata to account for latent frames.
        BatchMetadata labelsMeta(traceMeta.PoolId(),
                                 std::max(static_cast<size_t>(traceMeta.FirstFrame()), latentFrames_) - latentFrames_,
                                 std::max(static_cast<size_t>(traceMeta.LastFrame()), latentFrames_) - latentFrames_,
                                 traceMeta.FirstZmw());

        return LabelsBatch(labelsMeta, dims, std::move(trace), latentFrames_, syncDirection_, SOURCE_MARKER());
    }

    FrameLabelerMetrics NewMetrics(const Data::BatchDimensions& dims)
    {
        return FrameLabelerMetrics(dims, syncDirection_, SOURCE_MARKER());
    }

private:
    size_t latentFrames_;
    Cuda::Memory::SyncDirection syncDirection_;
};

// Define overloads for this function, so that we can track kernel invocations, and
// so that we can be converted to our gpu specific representation
inline auto KernelArgConvert(LabelsBatch& obj, const Cuda::KernelLaunchInfo& info)
{
    return obj.GetDeviceHandle(info);
}
inline auto KernelArgConvert(const LabelsBatch& obj, const Cuda::KernelLaunchInfo& info)
{
    return obj.GetDeviceHandle(info);
}

}}}     // namespace PacBio::Mongo::Data

#endif // PACBIO_MONGO_DATA_LABELS_BATCH_H_
