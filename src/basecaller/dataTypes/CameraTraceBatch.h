#ifndef mongo_dataTypes_CameraTraceBatch_H_
#define mongo_dataTypes_CameraTraceBatch_H_

#include "TraceBatch.h"

#include <common/cuda/memory/UnifiedCudaArray.h>
#include <common/NumericUtil.h>
#include <common/StatAccumulator.h>

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

    CameraTraceBatch(const CameraTraceBatch&) = delete;
    CameraTraceBatch(CameraTraceBatch&&) = default;

    CameraTraceBatch& operator=(const CameraTraceBatch&) = delete;
    CameraTraceBatch& operator=(CameraTraceBatch&&) = default;

    // TODO: Move constructor variant that accepts an rvalue reference to TraceBatch?

private:    // Types
    // Statistics computed by the baseliner.
    struct BaselinerStats
    {
        // Autocorrelation coefficient of baseline subtracted trace.
        // TODO: Use AutocorrAccumulator from Sequel.
        float autocorrelation;

        // Minimum and maximum of baseline-subtraced trace.
        ElementType traceMin;
        ElementType traceMax;

        // First three moments of raw trace baseline.
        StatAccumulator<float> stats;

        // Sum of baseline-subtracted baseline frames.
        ElementType residualSum;
    };

private:    // Data
    // Statistics for each ZMW in the batch.
    Cuda::Memory::UnifiedCudaArray<BaselinerStats> stats_;
};

}}}     // namespace PacBio::Mongo::Data

#endif // mongo_dataTypes_CameraTraceBatch_H_
