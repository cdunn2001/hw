#include "BaselinerStatAccumulator.h"

namespace PacBio {
namespace Mongo {
namespace Data {

template <typename T>
void BaselinerStatAccumulator<T>::AddSample(const LaneArray& rawTrace, const LaneArray& baselineSubtracted, const Mask& isBaseline)
{
    const auto& bs = baselineSubtracted.AsFloat();

    // Add frame to complete trace statistics.
    baselineSubtractedStats_.AddSample(bs);
    traceMin = min(bs, traceMin);
    traceMax = max(bs, traceMax);

    // Add frame to baseline statistics if so flagged.
    baselineStats_.AddSample(bs, isBaseline);
    rawBaselineSum_ += Blend(isBaseline, rawTrace, {0});
}

template <typename T>
const BaselineStats<laneSize> BaselinerStatAccumulator<T>::ToBaselineStats() const
{
    auto baselineStats = Data::BaselineStats<laneSize>{};

    std::copy(BaselineSubtractedStats().M1First().begin(), BaselineSubtractedStats().M1First().end(),
              baselineStats.lagM1First_.data());
    std::copy(BaselineSubtractedStats().M1Last().begin(), BaselineSubtractedStats().M1Last().end(),
              baselineStats.lagM1Last_.data());
    std::copy(BaselineSubtractedStats().M2().begin(), BaselineSubtractedStats().M2().end(),
              baselineStats.lagM2_.data());

    std::copy(TraceMin().begin(), TraceMin().end(), baselineStats.traceMin_.data());
    std::copy(TraceMax().begin(), TraceMax().end(), baselineStats.traceMax_.data());

    std::copy(BaselineFramesStats().Count().begin(), BaselineFramesStats().Count().end(),
              baselineStats.m0_.data());
    std::copy(BaselineFramesStats().M1().begin(), BaselineFramesStats().M1().end(),
              baselineStats.m1_.data());
    std::copy(BaselineFramesStats().M2().begin(), BaselineFramesStats().M2().end(),
              baselineStats.m2_.data());

    std::copy(RawBaselineSum().begin(), RawBaselineSum().end(), baselineStats.rawBaselineSum_.data());

    return baselineStats;
}

//
// Explicit Instantiations
//

template class BaselinerStatAccumulator<short>;

}}}     // namespace PacBio::Mongo::Data
