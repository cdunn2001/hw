
#include "HostNoOpBaseliner.h"

#include <dataTypes/BasicTypes.h>
#include <dataTypes/BaselinerStatAccumulator.h>

namespace PacBio {
namespace Mongo {
namespace Basecaller {

void HostNoOpBaseliner::Configure(const Data::BasecallerBaselinerConfig &baselinerConfig,
                                  const Data::MovieConfig &movConfig)
{
    const auto hostExecution = true;
    Baseliner::InitAllocationPools(hostExecution);
}

void HostNoOpBaseliner::Finalize()
{
    Baseliner::DestroyAllocationPools();
}

Data::CameraTraceBatch HostNoOpBaseliner::Process(Data::TraceBatch <ElementTypeIn> rawTrace)
{
    auto out = batchFactory_->NewBatch(rawTrace.GetMeta());

    for (size_t laneIdx = 0; laneIdx < rawTrace.LanesPerBatch(); ++laneIdx)
    {
        auto traceData = rawTrace.GetBlockView(laneIdx);
        auto cameraTraceData = out.GetBlockView(laneIdx);
        auto baselinerStats = Data::BaselinerStatAccumulator<Data::BaselinedTraceElement>{};
        auto statsView = out.Stats().GetHostView();
        for (size_t frame = 0; frame < traceData.NumFrames(); ++frame)
        {
            ElementTypeIn* src = traceData.Data() + (frame * traceData.LaneWidth());
            ElementTypeOut* dest = cameraTraceData.Data() + (frame * cameraTraceData.LaneWidth());
            std::memcpy(dest, src, sizeof(ElementTypeIn) * traceData.LaneWidth());
            LaneArray rawTrace(dest, dest + cameraTraceData.LaneWidth());
            Mask isBaseline { false };
            baselinerStats.AddSample(rawTrace, rawTrace, isBaseline);
        }

        statsView[laneIdx] = baselinerStats.ToBaselineStats();
    }

    return out;
}

HostNoOpBaseliner::~HostNoOpBaseliner() = default;

}}}      // namespace PacBio::Mongo::Basecaller
