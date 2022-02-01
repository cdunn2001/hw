#ifndef mongo_dataTypes_BaselinerStatAccumulator_H_
#define mongo_dataTypes_BaselinerStatAccumulator_H_

#include <common/AutocorrAccumulator.h>
#include <common/LaneArray.h>
#include <common/MongoConstants.h>

#include "BaselinerStatAccumState.h"

namespace PacBio {
namespace Mongo {
namespace Data {

// TODO: Don't assume that the element types of the raw and baseline-subtracted
// traces are the same.
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
        : baselineSubtractedCorr_ {state.fullAutocorrState}
        , traceMin {state.traceMin}
        , traceMax {state.traceMax}
        , baselineSubtractedStats_ {state.baselineStats}
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
            baselineSubtractedCorr_.GetState(),
            traceMin,
            traceMax,
            baselineSubtractedStats_.GetState(),
            rawBaselineSum_,
        };
    }

    const AutocorrAccumulator<FloatArray>& BaselineSubtractedStats() const
    { return baselineSubtractedCorr_; }

    const StatAccumulator<FloatArray>& BaselineFramesStats() const
    { return baselineSubtractedStats_; }

    const LaneArray& TraceMin() const
    { return traceMin; }

    const LaneArray& TraceMax() const
    { return traceMax; }

    const FloatArray& RawBaselineSum() const
    { return rawBaselineSum_; }

public:     // Non-const functions
    BaselinerStatAccumulator& Merge(const BaselinerStatAccumulator& other);

private:
    // Trace statistics after baseline estimate has been subtracted
    AutocorrAccumulator<FloatArray> baselineSubtractedCorr_;

    LaneArray traceMin{std::numeric_limits<T>::max()};
    LaneArray traceMax{std::numeric_limits<T>::lowest()};

    // Baseline frames statistics after baseline estimate has been subtracted
    StatAccumulator<FloatArray> baselineSubtractedStats_;

    // Sum of baseline frames _prior_ to baseline subtraction
    FloatArray rawBaselineSum_{0};
};

}}}     // namespace PacBio::Mongo::Data

#endif  // mongo_dataTypes_BaselinerStatAccumulator_H_
