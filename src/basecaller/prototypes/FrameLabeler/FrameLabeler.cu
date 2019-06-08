#include "FrameLabeler.h"

#include <common/cuda/PBCudaSimd.cuh>
#include <common/cuda/memory/DeviceOnlyObject.cuh>
#include <common/cuda/utility/CudaArray.h>
#include <common/KernelThreadPool.h>

#include <dataTypes/TraceBatch.cuh>

#include "SubframeScorer.cuh"
#include "FrameLabelerKernels.cuh"

using namespace PacBio::Cuda::Memory;
using namespace PacBio::Cuda::Data;
using namespace PacBio::Cuda::Utility;
using namespace PacBio::Cuda::Subframe;
using namespace PacBio::Mongo::Data;

namespace PacBio {
namespace Cuda {


std::unique_ptr<GeneratorBase<int16_t>> MakeDataGenerator(const Data::DataManagerParams& dataParams,
                                                          const Data::PicketFenceParams& picketParams,
                                                          const Data::TraceFileParams& traceParams)
{
    return traceParams.traceFileName.empty()
        ? std::unique_ptr<GeneratorBase<int16_t>>(new PicketFenceGenerator(dataParams, picketParams))
        : std::unique_ptr<GeneratorBase<int16_t>>(new SignalGenerator(dataParams, traceParams));

}


void run(const Data::DataManagerParams& dataParams,
         const Data::PicketFenceParams& picketParams,
         const Data::TraceFileParams& traceParams,
         const std::array<Subframe::AnalogMeta, 4>& meta,
         const Subframe::AnalogMeta& baselineMeta,
         size_t simulKernels)
{
    static constexpr size_t gpuBlockThreads = 32;
    static constexpr size_t laneWidth = 64;

    std::vector<UnifiedCudaArray<LaneModelParameters<PBHalf, laneWidth>>> models;

    LaneModelParameters<PBHalf, laneWidth> referenceModel;
    referenceModel.BaselineMode().SetAllMeans(baselineMeta.mean).SetAllVars(baselineMeta.var);
    for (int i = 0; i < 4; ++i)
    {
        referenceModel.AnalogMode(i).SetAllMeans(meta[i].mean).SetAllVars(meta[i].var);
    }

    const auto numBatches = dataParams.numZmwLanes / dataParams.kernelLanes;

    FrameLabeler::Configure(meta, dataParams.kernelLanes, dataParams.blockLength);
    std::vector<FrameLabeler> frameLabelers(numBatches);
    models.reserve(numBatches);
    for (size_t i = 0; i < numBatches; ++i)
    {

        models.emplace_back(dataParams.kernelLanes, SyncDirection::HostWriteDeviceRead);
        auto modelView = models.back().GetHostView();
        for (size_t j = 0; j < dataParams.kernelLanes; ++j)
        {
            modelView[j] = referenceModel;
        }
    }

    auto tmp = [&models, &dataParams, &frameLabelers](
        const TraceBatch<int16_t>& batch,
        size_t batchIdx,
        TraceBatch<int16_t>& ret)
    {
        if (dataParams.laneWidth != 2*gpuBlockThreads) throw PBException("Lane width not currently configurable.  Must be 64 zmw");
        frameLabelers[batchIdx].ProcessBatch(models[batchIdx], batch, ret);
    };

    ZmwDataManager<int16_t, int16_t> manager(dataParams, MakeDataGenerator(dataParams, picketParams, traceParams));
    RunThreads(simulKernels, manager, tmp);

    FrameLabeler::Finalize();
}

}}
