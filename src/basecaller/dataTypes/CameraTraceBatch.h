#ifndef mongo_dataTypes_CameraTraceBatch_H_
#define mongo_dataTypes_CameraTraceBatch_H_

#include "TraceBatch.h"

#include <common/cuda/memory/UnifiedCudaArray.h>
#include <common/MongoConstants.h>

#include "BaselinerStatAccumState.h"

namespace PacBio {
namespace Mongo {
namespace Data {

// This class will handle the small collection of constructor arguments that
// need to change depending on the pipeline configuration, but otherwise are
// generally constant between different batches
class CameraBatchFactory
{
public:
    using ElementType = BaselinedTraceElement;

    CameraBatchFactory(size_t framesPerChunk,
                       size_t lanesPerPool,
                       Cuda::Memory::SyncDirection syncDirection)
        : syncDirection_(syncDirection)
    {
        dims_.laneWidth = laneSize;
        dims_.framesPerBatch = framesPerChunk;
        dims_.lanesPerBatch = lanesPerPool;
    }

    std::pair<TraceBatch<ElementType>, Cuda::Memory::UnifiedCudaArray<BaselinerStatAccumState>> NewBatch(const BatchMetadata& meta) const
    {
        const auto& marker = SOURCE_MARKER();
        return std::make_pair(TraceBatch<ElementType>(meta, dims_, syncDirection_, marker),
                              Cuda::Memory::UnifiedCudaArray<BaselinerStatAccumState>(
                                  dims_.lanesPerBatch, syncDirection_, marker));
    }

private:
    Cuda::Memory::SyncDirection syncDirection_;
    BatchDimensions dims_;
};

}}}     // namespace PacBio::Mongo::Data

#endif // mongo_dataTypes_CameraTraceBatch_H_
