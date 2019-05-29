#ifndef mongo_dataTypes_CameraTraceBatch_H_
#define mongo_dataTypes_CameraTraceBatch_H_

#include "TraceBatch.h"

#include <common/cuda/memory/UnifiedCudaArray.h>
#include <common/MongoConstants.h>
#include <common/NumericUtil.h>
#include <common/StatAccumulator.h>

#include "BaselinerStatAccumulator.h"

namespace PacBio {
namespace Mongo {
namespace Data {

using BaselinedTraceElement = int16_t;

/// Baseline-subtracted trace data with statistics
class CameraTraceBatch : public TraceBatch<BaselinedTraceElement>
{
public:     // Types
    using ElementType = BaselinedTraceElement;

public:     // Structors and assignment
    CameraTraceBatch(const BatchMetadata& meta,
                     const BatchDimensions& dims,
                     Cuda::Memory::SyncDirection syncDirection,
                     std::shared_ptr<Cuda::Memory::DualAllocationPools> pool = nullptr)
        : TraceBatch<ElementType>(meta, dims, syncDirection, pool)
        , stats_ (dims.zmwsPerBatch(), syncDirection /* TODO: , pool for BaselineStats? */)
    { }

    // Move constructor variant that accepts an rvalue reference to TraceBatch.
    CameraTraceBatch(TraceBatch&& t)
        : TraceBatch<ElementType>(std::move(t))
        , stats_ (Dimensions().zmwsPerBatch(), Cuda::Memory::SyncDirection::Symmetric /* TODO: , pool for BaselineStats? */)
    { }

    CameraTraceBatch(const CameraTraceBatch&) = delete;
    CameraTraceBatch(CameraTraceBatch&&) = default;

    CameraTraceBatch& operator=(const CameraTraceBatch&) = delete;
    CameraTraceBatch& operator=(CameraTraceBatch&&) = default;

public:     // Access to statistics
    // TODO: How do we support const access?
//    const BaselinerStatAccumulator<ElementType>& Stats(unsigned int lane) const
//    { return stats_.GetHostView()[lane]; }

    BaselinerStatAccumulator<ElementType>& Stats(unsigned int lane)
    { return stats_.GetHostView()[lane]; }

private:    // Data
    // Statistics for each ZMW in the batch.
    Cuda::Memory::UnifiedCudaArray<BaselinerStatAccumulator<ElementType>> stats_;
};

}}}     // namespace PacBio::Mongo::Data

#endif // mongo_dataTypes_CameraTraceBatch_H_
