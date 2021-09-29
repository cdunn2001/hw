
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
    Baseliner::InitFactory(hostExecution, 1.0f);
}

void HostNoOpBaseliner::Finalize() {}

std::pair<Data::TraceBatch<Data::BaselinedTraceElement>,
          Data::BaselinerMetrics>
HostNoOpBaseliner::FilterBaseline(const Data::TraceBatchVariant& batch)
{
    const auto& rawTrace = [&]() ->decltype(auto)
    {
        try
        {
            return std::get<Mongo::Data::TraceBatch<int16_t>>(batch);
        } catch (const std::exception&)
        {
            throw PBException("Basecaller currently only supports int16_t trace data");
        }
    }();

    auto out = batchFactory_->NewBatch(rawTrace.GetMeta(), rawTrace.StorageDims());

    for (size_t laneIdx = 0; laneIdx < rawTrace.LanesPerBatch(); ++laneIdx)
    {
        auto traceData = rawTrace.GetBlockView(laneIdx);
        auto cameraTraceData = out.first.GetBlockView(laneIdx);
        if (cameraTraceData.NumFrames() < traceData.NumFrames())
        {
            throw PBException("Destination frame buffer is smaller than input buffer");
        }
        auto baselinerStats = Data::BaselinerStatAccumulator<Data::BaselinedTraceElement>{};
        auto statsView = out.second.baselinerStats.GetHostView();
        auto outItr = cameraTraceData.Begin();
        for (auto inItr = traceData.CBegin(); inItr != traceData.CEnd(); inItr++, outItr++)
        {
            auto copy = inItr.Extract();
            outItr.Store(copy);
            Mask isBaseline { false };
            baselinerStats.AddSample(copy, copy, isBaseline);
        }

        statsView[laneIdx] = baselinerStats.GetState();
    }

    return out;
}

HostNoOpBaseliner::~HostNoOpBaseliner() = default;

}}}      // namespace PacBio::Mongo::Basecaller
