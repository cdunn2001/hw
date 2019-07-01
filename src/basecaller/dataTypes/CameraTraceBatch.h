#ifndef mongo_dataTypes_CameraTraceBatch_H_
#define mongo_dataTypes_CameraTraceBatch_H_

#include "TraceBatch.h"

#include <common/cuda/memory/UnifiedCudaArray.h>
#include <common/MongoConstants.h>

#include "BasicTypes.h"
#include "BaselineStats.h"

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
                     std::shared_ptr<Cuda::Memory::DualAllocationPools> tracePool,
                     std::shared_ptr<Cuda::Memory::DualAllocationPools> statsPool)
        : TraceBatch<ElementType>(meta, dims, syncDirection, tracePool, pinned)
        , stats_ (dims.lanesPerBatch, syncDirection, pinned, statsPool)
    { }

    CameraTraceBatch(const CameraTraceBatch&) = delete;
    CameraTraceBatch(CameraTraceBatch&&) = default;

    CameraTraceBatch& operator=(const CameraTraceBatch&) = delete;
    CameraTraceBatch& operator=(CameraTraceBatch&&) = default;

public:     // Access to statistics
    const Cuda::Memory::UnifiedCudaArray<BaselineStats<laneSize>>& Stats() const
    { return stats_; }

    Cuda::Memory::UnifiedCudaArray<BaselineStats<laneSize>>& Stats()
    { return stats_; }

private:    // Data
    // Statistics for each ZMW in the batch, one element per lane.
    // TODO: Use half-precision for floating-point members of BaselinerStatAccumulator.
    Cuda::Memory::UnifiedCudaArray<BaselineStats<laneSize>> stats_;
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
        , tracePool_(std::make_shared<Cuda::Memory::DualAllocationPools>(framesPerChunk * lanesPerPool * laneSize * sizeof(int16_t), pinned))
        , statsPool_(std::make_shared<Cuda::Memory::DualAllocationPools>(lanesPerPool* sizeof(BaselineStats<laneSize>), pinned))
    {}

    CameraTraceBatch NewBatch(const BatchMetadata& meta,
                              const BatchDimensions& dims)
    {
        return CameraTraceBatch(meta, dims, pinned_, syncDirection_, tracePool_, statsPool_);
    }

private:
    Cuda::Memory::SyncDirection syncDirection_;
    bool pinned_;
    std::shared_ptr<Cuda::Memory::DualAllocationPools> tracePool_;
    std::shared_ptr<Cuda::Memory::DualAllocationPools> statsPool_;
};

}}}     // namespace PacBio::Mongo::Data

#endif // mongo_dataTypes_CameraTraceBatch_H_
