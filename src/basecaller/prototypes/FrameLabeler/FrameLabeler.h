#ifndef FRAME_LABELER_H
#define FRAME_LABELER_H

#include <cstddef>

#include <common/DataGenerators/PicketFenceGenerator.h>
#include <common/DataGenerators/SignalGenerator.h>

namespace PacBio {
namespace Cuda {

void run(const Data::DataManagerParams& dataParams,
         const Data::PicketFenceParams& picketParams,
         const Data::TraceFileParams& traceParams,
         size_t simulKernels);

}}

#endif // FRAME_LABELER_H
