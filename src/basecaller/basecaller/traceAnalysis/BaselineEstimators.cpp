
#include "BaselineEstimators.h"

#include <dataTypes/BaselinerStatAccumulator.h>

namespace PacBio {
namespace Mongo {
namespace Basecaller {


Data::CameraTraceBatch
NoOpBaseliner::Process(Data::TraceBatch <ElementTypeIn> rawTrace)
{
    Data::CameraTraceBatch ctb(std::move(rawTrace));

    for (size_t laneIdx = 0; laneIdx < ctb.LanesPerBatch(); ++laneIdx)
    {
        auto frameData = ctb.GetBlockView(laneIdx);
        auto& baselineStats = ctb.Stats(laneIdx);

        for (size_t frame = 0; frame < frameData.NumFrames(); ++frame)
        {
            ElementTypeIn* f = frameData.Data() + (frame * frameData.LaneWidth());
            LaneArray rawData;
            std::memcpy(rawData.Data(), f, frameData.LaneWidth());
            Mask isBaseline { false };
            baselineStats.AddSample(rawData, rawData, isBaseline);
        }
    }

    return ctb;
}

}}}      // namespace PacBio::Mongo::Basecaller
