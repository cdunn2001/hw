#ifndef mongo_dataTypes_CameraTraceBatch_H_
#define mongo_dataTypes_CameraTraceBatch_H_

#include "TraceBatch.h"

#include <common/cuda/memory/UnifiedCudaArray.h>
#include <common/MongoConstants.h>

#include "BaselinerStatAccumState.h"
#include "BasicTypes.h"

namespace PacBio {
namespace Mongo {
namespace Data {


/// Baseline-subtracted trace data with statistics
class CameraTraceBatch : public TraceBatch<BaselinedTraceElement>
{
public:     // Types
    using ElementType = BaselinedTraceElement;

public:     // Structors and assignment
    CameraTraceBatch(const BatchMetadata& meta,
                     const BatchDimensions& dims,
                     bool pinned,
                     Cuda::Memory::SyncDirection syncDirection,
                     const Cuda::Memory::AllocationMarker& marker)
        : TraceBatch<ElementType>(meta, dims, syncDirection, marker, pinned)
        , stats_ (dims.lanesPerBatch, syncDirection, marker, pinned)
    { }

    CameraTraceBatch(const CameraTraceBatch&) = delete;
    CameraTraceBatch(CameraTraceBatch&&) = default;

    CameraTraceBatch& operator=(const CameraTraceBatch&) = delete;
    CameraTraceBatch& operator=(CameraTraceBatch&&) = default;

public:     // Access to statistics
    const Cuda::Memory::UnifiedCudaArray<BaselinerStatAccumState>& Stats() const
    { return stats_; }

    Cuda::Memory::UnifiedCudaArray<BaselinerStatAccumState>& Stats()
    { return stats_; }

private:    // Data
    // Statistics for each ZMW in the batch, one element per lane.
    // TODO: Use half-precision for floating-point members of BaselinerStatAccumulator.
    Cuda::Memory::UnifiedCudaArray<BaselinerStatAccumState> stats_;
};

// Factory class, to simplify the construction of CameraTraceBatch instances.
// This class will handle the small collection of constructor arguments that
// need to change depending on the pipeline configuration, but otherwise are
// generally constant between different batches
class CameraBatchFactory
{
public:
    CameraBatchFactory(size_t framesPerChunk,
                       size_t lanesPerPool,
                       Cuda::Memory::SyncDirection syncDirection,
                       bool pinned = true)
        : syncDirection_(syncDirection)
        , pinned_(pinned)
    {
        dims_.laneWidth = laneSize;
        dims_.framesPerBatch = framesPerChunk;
        dims_.lanesPerBatch = lanesPerPool;
    }

    CameraTraceBatch NewBatch(const BatchMetadata& meta) const
    {
        return CameraTraceBatch(meta, dims_, pinned_, syncDirection_, SOURCE_MARKER());
    }

private:
    Cuda::Memory::SyncDirection syncDirection_;
    bool pinned_;
    BatchDimensions dims_;
};

// Define overloads for this function, so that we can track kernel invocations, and
// so that we can be converted to our gpu specific representation
inline auto KernelArgConvert(CameraTraceBatch& obj, const Cuda::KernelLaunchInfo& info)
{
    return obj.GetDeviceHandle(info);
}
inline auto KernelArgConvert(const CameraTraceBatch& obj, const Cuda::KernelLaunchInfo& info)
{
    return obj.GetDeviceHandle(info);
}

}}}     // namespace PacBio::Mongo::Data

#endif // mongo_dataTypes_CameraTraceBatch_H_
