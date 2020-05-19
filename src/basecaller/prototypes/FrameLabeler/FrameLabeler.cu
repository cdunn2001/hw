#include "FrameLabeler.h"

#include <common/cuda/PBCudaSimd.cuh>
#include <common/cuda/memory/DeviceOnlyObject.cuh>
#include <common/cuda/utility/CudaArray.h>
#include <common/KernelThreadPool.h>

#include <dataTypes/BatchData.cuh>

#include "SubframeScorer.cuh"
#include "FrameLabelerKernels.cuh"

using namespace PacBio::Cuda::Memory;
using namespace PacBio::Cuda::Data;
using namespace PacBio::Cuda::Utility;
using namespace PacBio::Cuda::Subframe;
using namespace PacBio::Mongo::Data;

namespace PacBio {
namespace Cuda {


static std::unique_ptr<GeneratorBase<int16_t>> MakeDataGenerator(
        const Data::DataManagerParams& dataParams,
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
    SetGlobalAllocationMode(CachingMode::ENABLED, AllocatorMode::CUDA);

    static constexpr size_t gpuBlockThreads = laneSize/2;

    std::vector<UnifiedCudaArray<LaneModelParameters<PBHalf, laneSize>>> models;

    LaneModelParameters<PBHalf, laneSize> referenceModel;
    referenceModel.BaselineMode().SetAllMeans(baselineMeta.mean).SetAllVars(baselineMeta.var);
    for (int i = 0; i < 4; ++i)
    {
        referenceModel.AnalogMode(i).SetAllMeans(meta[i].mean).SetAllVars(meta[i].var);
    }

    BatchDimensions latBatchDims;
    latBatchDims.framesPerBatch = dataParams.blockLength;
    latBatchDims.laneWidth = laneSize;
    latBatchDims.lanesPerBatch = dataParams.kernelLanes;
    std::vector<Data::BatchData<int16_t>> latTrace;

    const auto numBatches = dataParams.numZmwLanes / dataParams.kernelLanes;

    FrameLabeler::Configure(meta, dataParams.kernelLanes, dataParams.blockLength);
    std::vector<FrameLabeler> frameLabelers(numBatches);
    models.reserve(numBatches);
    for (size_t i = 0; i < numBatches; ++i)
    {

        models.emplace_back(dataParams.kernelLanes, SyncDirection::HostWriteDeviceRead, SOURCE_MARKER());
        auto modelView = models.back().GetHostView();
        for (size_t j = 0; j < dataParams.kernelLanes; ++j)
        {
            modelView[j] = referenceModel;
        }

        latTrace.emplace_back(latBatchDims, SyncDirection::HostReadDeviceWrite, SOURCE_MARKER());
    }

    std::vector<FrameLabelerMetrics> frameLabelerMetrics;
    for (size_t i = 0; i < numBatches; ++i)
    {
        frameLabelerMetrics.emplace_back(
                latBatchDims, SyncDirection::HostReadDeviceWrite, SOURCE_MARKER());
    }

    auto tmp = [&models, &dataParams, &frameLabelers, &latTrace, &frameLabelerMetrics](
        const TraceBatch<int16_t>& batch,
        size_t batchIdx,
        TraceBatch<int16_t>& ret)
    {
        if (dataParams.laneWidth != 2*gpuBlockThreads) throw PBException("Lane width not currently configurable.  Must be 64 zmw");
        frameLabelers[batchIdx].ProcessBatch(models[batchIdx], batch, latTrace[batchIdx], ret,
                                             frameLabelerMetrics[batchIdx]);
        (void)latTrace[batchIdx].GetBlockView(0);
    };

    ZmwDataManager<int16_t, int16_t> manager(dataParams, MakeDataGenerator(dataParams, picketParams, traceParams));
    RunThreads(simulKernels, manager, tmp);

    FrameLabeler::Finalize();

    Memory::DisableAllCaching();
}

}}
