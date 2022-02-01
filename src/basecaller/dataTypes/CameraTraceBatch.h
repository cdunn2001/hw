#ifndef mongo_dataTypes_CameraTraceBatch_H_
#define mongo_dataTypes_CameraTraceBatch_H_

#include "TraceBatch.h"

#include <common/cuda/memory/UnifiedCudaArray.h>
#include <common/MongoConstants.h>

#include "BaselinerStatAccumState.h"
#include "BatchMetrics.h"

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

    CameraBatchFactory(Cuda::Memory::SyncDirection syncDirection)
        : syncDirection_(syncDirection)
    {}

    std::pair<TraceBatch<ElementType>, BaselinerMetrics>
    NewBatch(const BatchMetadata& meta, const BatchDimensions dims) const
    {
        const auto& marker = SOURCE_MARKER();
        return std::make_pair(TraceBatch<ElementType>(meta, dims, syncDirection_, marker),
                              BaselinerMetrics(dims.lanesPerBatch, syncDirection_, marker));
    }

private:
    Cuda::Memory::SyncDirection syncDirection_;
};

}}}     // namespace PacBio::Mongo::Data

#endif // mongo_dataTypes_CameraTraceBatch_H_
