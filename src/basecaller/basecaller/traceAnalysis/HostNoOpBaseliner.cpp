
#include "HostNoOpBaseliner.h"

#include <dataTypes/BasicTypes.h>
#include <dataTypes/BaselinerStatAccumulator.h>

namespace PacBio {
namespace Mongo {
namespace Basecaller {

void HostNoOpBaseliner::Configure(const Data::BasecallerBaselinerConfig &baselinerConfig,
                                          const Data::MovieConfig &movConfig)
{
    const auto hostExecution = false;
    Baseliner::InitAllocationPools(hostExecution);
}

void HostNoOpBaseliner::Finalize()
{
    Baseliner::DestroyAllocationPools();
}

Data::CameraTraceBatch HostNoOpBaseliner::Process(Data::TraceBatch <ElementTypeIn> rawTrace)
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

        out.Stats(laneIdx) = baselinerStats.ToBaselineStats();
    }

    return std::move(out);
}

HostNoOpBaseliner::~HostNoOpBaseliner() = default;

}}}      // namespace PacBio::Mongo::Basecaller
