#ifndef FRAME_LABELER_H
#define FRAME_LABELER_H

#include <cstddef>

#include <pacbio/datasource/AnalogMode.h>

#include <common/DataGenerators/PicketFenceGenerator.h>
#include <common/DataGenerators/SignalGenerator.h>

#include <dataTypes/LaneDetectionModel.h>

namespace PacBio {
namespace Cuda {

void run(const Data::DataManagerParams& dataParams,
         const Data::PicketFenceParams& picketParams,
         const Data::TraceFileParams& traceParams,
         const std::array<DataSource::AnalogMode,4>& meta,
         const Data::LaneModelParameters<PBHalf, Mongo::laneSize>& referenceModel,
         size_t simulKernels);

}}

#endif // FRAME_LABELER_H
