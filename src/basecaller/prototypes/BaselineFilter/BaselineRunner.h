#ifndef BASELINE_FILTER_BASELINE_H
#define BASELINE_FILTER_BASELINE_H

#include <common/DataGenerators/PicketFenceGenerator.h>
#include <common/DataGenerators/SignalGenerator.h>
#include <common/ZmwDataManager.h>
#include <pacbio/utilities/SmartEnum.h>

SMART_ENUM(BaselineFilterMode, GlobalMax, SharedMax, LocalMax, GlobalFull, SharedFull, SharedFullCompressed, MultipleFull);

namespace PacBio {
namespace Cuda {

void run(const Data::DataManagerParams& dataParams,
         const Data::PicketFenceParams& picketParams,
         const Data::TraceFileParams& traceParams,
         size_t simulKernels,
         BaselineFilterMode mode);

}}

#endif // BASELINE_FILTER_BASELINE_H
