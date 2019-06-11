#include "BaselinerStatAccumulator.h"

namespace PacBio {
namespace Mongo {
namespace Data {

template <typename T>
void BaselinerStatAccumulator<T>::AddSample(const LaneArray& x, const LaneArray& y, const Mask& isBaseline)
{
    const auto fy = y.AsFloat();

    // Add frame to complete trace statistics.
    baselineSubtractedStats_.AddSample(fy);
    traceMin = min(fy, traceMin);
    traceMax = max(fy, traceMax);

    // Add frame to baseline statistics if so flagged.
    baselineStats_.AddSample(fy, isBaseline);
    rawBaselineSum_ += Blend(isBaseline, x, {0});
}

template <typename T>
const BaselineStats<laneSize> BaselinerStatAccumulator<T>::ToBaselineStats() const
{
    return Data::BaselineStats<laneSize>{}
        .TraceMin(TraceMin().AsCudaArray())
        .TraceMax(TraceMax().AsCudaArray())
        .RawBaselineSum(RawBaselineSum().AsCudaArray())
        .BaselineMoments(BaselineFramesStats().Count().AsCudaArray(),
                         BaselineFramesStats().Mean().AsCudaArray(),
                         BaselineFramesStats().Variance().AsCudaArray())
        .AutocorrMoments(BaselineSubtractedStats().M1First().AsCudaArray(),
                         BaselineSubtractedStats().M1Last().AsCudaArray(),
                         BaselineSubtractedStats().M2().AsCudaArray());
}

//
// Explicit Instantiations
//

template class BaselinerStatAccumulator<short>;

}}}     // namespace PacBio::Mongo::Data
