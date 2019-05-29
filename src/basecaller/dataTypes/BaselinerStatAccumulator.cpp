#include "BaselinerStatAccumulator.h"

namespace PacBio {
namespace Mongo {
namespace Data {

template <typename T>
void
BaselinerStatAccumulator<T>::AddSample(const LaneArray& x, const LaneArray& y, const Mask& isBaseline)
{
    const auto fy = FloatArray(y);

    // Add frame to complete trace statistics.
    baselineSubtractedStats_.AddSample(fy);
    traceMin = min(fy, traceMin);
    traceMax = max(fy, traceMax);

    // Add frame to baseline statistics if so flagged.
    baselineStats_.AddSample(fy, isBaseline);
    rawBaselineSum_ += Blend(isBaseline, x, 0);
}

}}}     // namespace PacBio::Mongo::Data
