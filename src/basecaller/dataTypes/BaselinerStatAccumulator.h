#ifndef mongo_dataTypes_BaselinerStatAccumulator_H_
#define mongo_dataTypes_BaselinerStatAccumulator_H_

#include <common/AutocorrAccumulator.h>
#include <common/LaneArray.h>
#include <common/MongoConstants.h>

namespace PacBio {
namespace Mongo {
namespace Data {

// Statistics computed by the baseliner for one lane of ZMWs.
template <typename T>
class BaselinerStatAccumulator
{
public:     // Types
    using LaneArray = PacBio::Mongo::LaneArray<T, laneSize>;
    using FloatArray = PacBio::Mongo::LaneArray<float, laneSize>;
    using Mask = PacBio::Mongo::LaneMask<laneSize>;

public:     // Mutating functions
    /// Add a lane-frame to the statistics.
    /// \a x contains raw trace values.
    /// \a y contains baseline-subtracted trace values.
    /// Only add to baseline statistics where \a isBaseline is true.
    void AddSample(const LaneArray& x,
                   const LaneArray& y,
                   const Mask& isBaseline);

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
