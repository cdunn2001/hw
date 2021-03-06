
#include "HostNoOpBaseliner.h"

#include <dataTypes/BasicTypes.h>
#include <dataTypes/BaselinerStatAccumulator.h>

namespace PacBio {
namespace Mongo {
namespace Basecaller {

void HostNoOpBaseliner::Configure(const Data::BasecallerBaselinerConfig&,
                                  const Data::AnalysisConfig& analysisConfig)
{
    const auto hostExecution = true;
    Baseliner::InitFactory(hostExecution, analysisConfig);
}

void HostNoOpBaseliner::Finalize() {}

std::pair<Data::TraceBatch<Data::BaselinedTraceElement>,
          Data::BaselinerMetrics>
HostNoOpBaseliner::FilterBaseline(const Data::TraceBatchVariant& batch)
{
    return std::visit([&](const auto& rawTrace)
    {
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
            FloatArray background {0.0f};
            for (auto inItr = traceData.CBegin(); inItr != traceData.CEnd(); inItr++, outItr++)
            {
                auto copy = (inItr.Extract() - pedestal_) * movieScaler_;
                outItr.Store(copy);
                Mask isBaseline { false };
                baselinerStats.AddSample(copy, copy, isBaseline);
            }

            baselinerStats.AddSampleBackground(background);

            statsView[laneIdx] = baselinerStats.GetState();
        }
        
        const auto& tracemd = out.first.Metadata();
        out.second.frameInterval = {tracemd.FirstFrame(), tracemd.LastFrame()};

        return out;
    }, batch.Data());
}

HostNoOpBaseliner::~HostNoOpBaseliner() = default;

}}}      // namespace PacBio::Mongo::Basecaller
