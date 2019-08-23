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

#include "PulseDetectionMetrics.h"

namespace PacBio {
namespace Mongo {
namespace Data {

using Label_t = int16_t;

/// Baseline-subtracted trace data with statistics
class LabelsBatch : public TraceBatch<Label_t>
{
    static BatchDimensions LatentDimensions(const BatchDimensions& traceDims, size_t latentFrames)
    {
        BatchDimensions ret{traceDims};
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
                bool pinned,
                Cuda::Memory::SyncDirection syncDirection,
                std::shared_ptr<Cuda::Memory::DualAllocationPools> tracePool,
                std::shared_ptr<Cuda::Memory::DualAllocationPools> latPool,
                std::shared_ptr<Cuda::Memory::DualAllocationPools> metricsPool)
        : TraceBatch<ElementType>(meta, dims, syncDirection, tracePool, pinned)
        , curTrace_(std::move(trace))
        , latTrace_(LatentDimensions(dims, latentFrames), syncDirection, latPool, pinned)
        , pdMetrics_(dims.lanesPerBatch, syncDirection, pinned, metricsPool)
    { }

    LabelsBatch(const LabelsBatch&) = delete;
    LabelsBatch(LabelsBatch&&) = default;

    LabelsBatch& operator=(const LabelsBatch&) = delete;
    LabelsBatch& operator=(LabelsBatch&&) = default;

    const BatchData& TraceData() const { return curTrace_; }
    BatchData& TraceData() { return curTrace_; }

    const BatchData& LatentTrace() const { return latTrace_; }
    BatchData& LatentTrace() { return latTrace_; }

    Cuda::Memory::UnifiedCudaArray<PulseDetectionMetrics>& PdMetrics()
    { return pdMetrics_; }

    Cuda::Memory::UnifiedCudaArray<PulseDetectionMetrics> TakePdMetrics()
    { return std::move(pdMetrics_); }

private:    // Data
    // Full trace input to label filter, but the last few frames are held back for
    // viterbi stitching, so this class will prevent access to those
    TraceBatch<ElementType> curTrace_;

    // Latent camera trace data held over by frame labeling from the previous block
    BatchData latTrace_;

    // Viterbi score from FrameLabeler and baseline stats from PulseAccumulator
    Cuda::Memory::UnifiedCudaArray<PulseDetectionMetrics> pdMetrics_;
};

// Factory class, to simplify the construction of LabelsBatch instances.
// This class will handle the small collection of constructor arguments that
// need to change depending on the pipeline configuration, but otherwise are
// generally constant between different batches
class LabelsBatchFactory
{
public:
    LabelsBatchFactory(size_t framesPerChunk,
                       size_t lanesPerPool,
                       size_t latentFrames,
                       Cuda::Memory::SyncDirection syncDirection,
                       bool pinned = true)
        : latentFrames_(latentFrames)
        , syncDirection_(syncDirection)
        , pinned_(pinned)
        , tracePool_(std::make_shared<Cuda::Memory::DualAllocationPools>(framesPerChunk * lanesPerPool * laneSize * sizeof(int16_t), pinned))
        , latPool_(std::make_shared<Cuda::Memory::DualAllocationPools>(latentFrames_* lanesPerPool * laneSize * sizeof(int16_t), pinned))
        , metricsPool_(std::make_shared<Cuda::Memory::DualAllocationPools>(lanesPerPool * sizeof(PulseDetectionMetrics), pinned))
    {}

    LabelsBatch NewBatch(TraceBatch<LabelsBatch::ElementType> trace)
    {
        auto meta = trace.Metadata();
        auto dims = trace.StorageDims();
        return LabelsBatch(meta, dims, std::move(trace),
                           latentFrames_, pinned_,
                           syncDirection_, tracePool_, latPool_, metricsPool_);
    }

private:
    size_t latentFrames_;
    Cuda::Memory::SyncDirection syncDirection_;
    bool pinned_;
    std::shared_ptr<Cuda::Memory::DualAllocationPools> tracePool_;
    std::shared_ptr<Cuda::Memory::DualAllocationPools> latPool_;
    std::shared_ptr<Cuda::Memory::DualAllocationPools> metricsPool_;
};

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
