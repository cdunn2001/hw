
#include "HostNoOpBaseliner.h"

#include <dataTypes/BasicTypes.h>
#include <dataTypes/BaselinerStatAccumulator.h>

namespace PacBio {
namespace Mongo {
namespace Basecaller {

void HostNoOpBaseliner::Configure(const Data::BasecallerBaselinerConfig&,
                                  const Data::MovieConfig&)
{
    const auto hostExecution = true;
    Baseliner::InitAllocationPools(hostExecution);
}

void HostNoOpBaseliner::Finalize()
{
    Baseliner::DestroyAllocationPools();
}

std::pair<Data::TraceBatch<Data::BaselinedTraceElement>,
          Data::BaselinerMetrics>
HostNoOpBaseliner::Process(Data::TraceBatch<ElementTypeIn> rawTrace)
{
    auto out = batchFactory_->NewBatch(rawTrace.GetMeta());

    for (size_t laneIdx = 0; laneIdx < rawTrace.LanesPerBatch(); ++laneIdx)
    {
        auto traceData = rawTrace.GetBlockView(laneIdx);
        auto cameraTraceData = out.first.GetBlockView(laneIdx);
        auto baselinerStats = Data::BaselinerStatAccumulator<Data::BaselinedTraceElement>{};
        auto statsView = out.second.baselinerStats.GetHostView();
        for (size_t frame = 0; frame < traceData.NumFrames(); ++frame)
        {
            ElementTypeIn* src = traceData.Data() + (frame * traceData.LaneWidth());
            ElementTypeOut* dest = cameraTraceData.Data() + (frame * cameraTraceData.LaneWidth());
            std::memcpy(dest, src, sizeof(ElementTypeIn) * traceData.LaneWidth());
            LaneArray data(dest, dest + cameraTraceData.LaneWidth());
            Mask isBaseline { false };
            baselinerStats.AddSample(data, data, isBaseline);
        }

        statsView[laneIdx] = baselinerStats.GetState();
    }

    return out;
}

HostNoOpBaseliner::~HostNoOpBaseliner() = default;

}}}      // namespace PacBio::Mongo::Basecaller
