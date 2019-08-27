#ifndef mongo_dataTypes_BaselinerStatAccumulator_H_
#define mongo_dataTypes_BaselinerStatAccumulator_H_

#include <common/AutocorrAccumulator.h>
#include <common/LaneArray.h>
#include <common/MongoConstants.h>

#include "BaselinerStatAccumState.h"

namespace PacBio {
namespace Mongo {
namespace Data {

// TODO: Add declaration decorators to enable use on CUDA device.
// TODO: Add another template parameter to control the precision used for floating-point members.

/// Statistics computed by the baseliner for one lane of ZMWs.
/// \tparam T The elemental type of the raw trace data.
template <typename T>
class BaselinerStatAccumulator
{
public:     // Types
    using LaneArray = PacBio::Mongo::LaneArray<T>;
    using FloatArray = PacBio::Mongo::LaneArray<float>;
    using Mask = PacBio::Mongo::LaneMask<>;

public:     // Structors
    BaselinerStatAccumulator() = default;

    BaselinerStatAccumulator(const BaselinerStatAccumState& state)
        : baselineSubtractedStats_ {state.fullAutocorrState}
        , traceMin {state.traceMin}
        , traceMax {state.traceMax}
        , baselineStats_ {state.baselineStats}
        , rawBaselineSum_ {state.rawBaselineSum}
    { }

public:     // Mutating functions
    /// Add a lane-frame to the statistics.
    /// \a x contains raw trace values.
    /// \a y contains baseline-subtracted trace values.
    /// Only add to baseline statistics where \a isBaseline is true.
    void AddSample(const LaneArray& rawTrace,
                   const LaneArray& baselineSubtracted,
                   const Mask& isBaseline);

public:     // Const functions
    BaselinerStatAccumState GetState() const
    {
        return BaselinerStatAccumState
        {
            baselineSubtractedStats_.GetState(),
            {traceMin.cbegin(), traceMin.cend()},
            {traceMax.cbegin(), traceMax.cend()},
            baselineStats_.GetState(),
            {rawBaselineSum_.cbegin(), rawBaselineSum_.cend()}
        };
    }

    const AutocorrAccumulator<FloatArray>& BaselineSubtractedStats() const
    { return baselineSubtractedStats_; }

    const StatAccumulator<FloatArray>& BaselineFramesStats() const
    { return baselineStats_; }

    const LaneArray& TraceMin() const
    { return traceMin; }

    const LaneArray& TraceMax() const
    { return traceMax; }

    const LaneArray& RawBaselineSum() const
    { return rawBaselineSum_; }

public:     // Non-const functions
    BaselinerStatAccumulator& Merge(const BaselinerStatAccumulator& other);

private:
    // Statistics of trace after baseline estimate has been subtracted.
    AutocorrAccumulator<FloatArray> baselineSubtractedStats_;
    LaneArray traceMin;
    LaneArray traceMax;

    // Statistics of baseline frames after baseline estimate has
    // been subtracted.
    StatAccumulator<FloatArray> baselineStats_;

    // Sum of baseline frames _prior_ to baseline subtraction.
    LaneArray rawBaselineSum_;
};

}}}     // namespace PacBio::Mongo::Data

#endif  // mongo_dataTypes_BaselinerStatAccumulator_H_
