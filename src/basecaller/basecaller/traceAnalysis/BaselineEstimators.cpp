
#include "BaselineEstimators.h"

#include <dataTypes/BasicTypes.h>
#include <dataTypes/BaselinerStatAccumulator.h>

namespace PacBio {
namespace Mongo {
namespace Basecaller {

Data::CameraTraceBatch NoOpBaseliner::Process(Data::TraceBatch <ElementTypeIn> rawTrace)
{
    auto out = batchFactory_->NewBatch(rawTrace.GetMeta(), rawTrace.Dimensions());

    for (size_t laneIdx = 0; laneIdx < rawTrace.LanesPerBatch(); ++laneIdx)
    {
        auto frameData = rawTrace.GetBlockView(laneIdx);
        auto baselinerStats = Data::BaselinerStatAccumulator<Data::BaselinedTraceElement>{};

        for (size_t frame = 0; frame < frameData.NumFrames(); ++frame)
        {
            ElementTypeIn* f = frameData.Data() + (frame * frameData.LaneWidth());
            LaneArray rawData;
            std::memcpy(rawData.Data(), f, sizeof(ElementTypeIn) * frameData.LaneWidth());
            Mask isBaseline { false };
            baselinerStats.AddSample(rawData, rawData, isBaseline);
        }

        // TODO: Convert BaslinerStatAccumulator to BaselineStats.
    }

    return std::move(out);
}

MultiScaleBaseliner::MultiScaleBaseliner(uint32_t poolId, float scaler,
                                         const PacBio::Mongo::Basecaller::BaselinerParams& config)
    : Baseliner(poolId, scaler)
    , msLowerOpen_(config.Strides(), config.Widths())
    , msUpperOpen_(config.Strides(), config.Widths())
    , stride_(config.AggregateStride())
    , cSigmaBias_{config.SigmaBias()}
    , cMeanBias_{config.MeanBias()}
{ }

Data::CameraTraceBatch MultiScaleBaseliner::Process(Data::TraceBatch <ElementTypeIn> rawTrace)
{
    auto out = batchFactory_->NewBatch(rawTrace.GetMeta(), rawTrace.Dimensions());
    auto pools = rawTrace.GetAllocationPools();

    for (size_t laneIdx = 0; laneIdx < rawTrace.LanesPerBatch(); ++laneIdx)
    {
        auto frameData = rawTrace.GetBlockView(laneIdx);
        auto baselinerStats = Data::BaselinerStatAccumulator<Data::BaselinedTraceElement>{};

        // TODO: Run filter and perform baseline subtraction.

        // TODO: Convert BaslinerStatAccumulator to BaselineStats.
    }

    return std::move(out);
}

}}}      // namespace PacBio::Mongo::Basecaller
