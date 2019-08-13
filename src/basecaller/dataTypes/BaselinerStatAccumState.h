#ifndef mongo_dataTypes_BaselinerStatAccumStat_H_
#define mongo_dataTypes_BaselinerStatAccumStat_H_

#include <common/AutocorrAccumState.h>
#include <common/cuda/utility/CudaArray.h>
#include "BasicTypes.h"

namespace PacBio {
namespace Mongo {
namespace Data {

/// A CUDA-friendly POD struct that represents the state of a BaselinerStatAccumState.
/// The ability to add more data samples is not preserved by this
/// representation.
struct BaselinerStatAccumState
{
    using StatElement = AutocorrAccumState::FloatArray::value_type;
    using IntArray = Cuda::Utility::CudaArray<RawTraceElement, laneSize>;

    // Statistics of all frames after baseline estimate has been subtracted.
    AutocorrAccumState fullAutocorrState;
    IntArray traceMin;
    IntArray traceMax;

    // Statistics of the frames crudely classified as baseline after baseline
    // estimate has been subtracted.
    StatAccumState baselineStats;

    // Sum of raw baseline frames (i.e., prior to baseline subtraction).
    IntArray rawBaselineSum;

    StatAccumState::FloatArray NumBaselineFrames() const
    {
        return baselineStats.moment0;
    }

    StatAccumState::FloatArray TotalFrames() const
    {
        return fullAutocorrState.basicStats.moment0;
    }
};

}}}     // namesapce PacBio::Mongo::Data

#endif  // mongo_dataTypes_BaselinerStatAccumStat_H_
