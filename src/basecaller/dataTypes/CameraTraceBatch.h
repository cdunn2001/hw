#ifndef mongo_dataTypes_CameraTraceBatch_H_
#define mongo_dataTypes_CameraTraceBatch_H_

#include "TraceBatch.h"

#include <common/cuda/memory/UnifiedCudaArray.h>
#include <common/MongoConstants.h>

#include "BaselinerStatAccumState.h"

namespace PacBio {
namespace Mongo {
namespace Data {

// Factory class, to simplify the construction of
// baselineSubtractedTraces+Stats pair instances.
// This class will handle the small collection of constructor arguments that
// need to change depending on the pipeline configuration, but otherwise are
// generally constant between different batches
class CameraBatchFactory
{
public:
    using ElementType = BaselinedTraceElement;

    CameraBatchFactory(size_t framesPerChunk,
                       size_t lanesPerPool,
                       Cuda::Memory::SyncDirection syncDirection,
                       bool pinned = true)
        : syncDirection_(syncDirection)
        , pinned_(pinned)
        , tracePool_(std::make_shared<Cuda::Memory::DualAllocationPools>(framesPerChunk * lanesPerPool * laneSize * sizeof(int16_t), pinned))
        , statsPool_(std::make_shared<Cuda::Memory::DualAllocationPools>(lanesPerPool* sizeof(BaselinerStatAccumState), pinned))
    {
        dims_.laneWidth = laneSize;
        dims_.framesPerBatch = framesPerChunk;
        dims_.lanesPerBatch = lanesPerPool;
    }

    std::pair<TraceBatch<ElementType>, Cuda::Memory::UnifiedCudaArray<BaselinerStatAccumState>> NewBatch(const BatchMetadata& meta) const
    {
        return std::make_pair(TraceBatch<ElementType>(meta, dims_, syncDirection_, tracePool_, pinned_),
                              Cuda::Memory::UnifiedCudaArray<BaselinerStatAccumState>(
                                  dims_.lanesPerBatch, syncDirection_, pinned_, statsPool_));
    }

private:
    Cuda::Memory::SyncDirection syncDirection_;
    bool pinned_;
    BatchDimensions dims_;
    std::shared_ptr<Cuda::Memory::DualAllocationPools> tracePool_;
    std::shared_ptr<Cuda::Memory::DualAllocationPools> statsPool_;
};

}}}     // namespace PacBio::Mongo::Data

#endif // mongo_dataTypes_CameraTraceBatch_H_
