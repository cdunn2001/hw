#include "FrameLabeler.h"

#include <common/cuda/PBCudaSimd.cuh>
#include <common/cuda/memory/DeviceOnlyObject.cuh>
#include <common/cuda/utility/CudaArray.cuh>
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

    DeviceOnlyObj<TransitionMatrix> trans(Utility::CudaArray<AnalogMeta, 4>{meta});
    std::vector<UnifiedCudaArray<LaneModelParameters<gpuBlockThreads>>> models;
    std::vector<DeviceOnlyObj<LatentViterbi<32>>> latent;
    std::vector<ViterbiDataHost<short2, gpuBlockThreads>> labels;

    LaneModelParameters<gpuBlockThreads> referenceModel;
    referenceModel.BaselineMode().SetAllMeans(baselineMeta.mean).SetAllVars(baselineMeta.var);
    for (int i = 0; i < 4; ++i)
    {
        referenceModel.AnalogMode(i).SetAllMeans(meta[i].mean).SetAllVars(meta[i].var);
    }

    const auto numBatches = dataParams.numZmwLanes / dataParams.kernelLanes;
    models.reserve(numBatches);
    latent.reserve(numBatches);
    // TODO this only needs one per active batch, not all batches
    labels.reserve(numBatches);
    for (size_t i = 0; i < numBatches; ++i)
    {
        latent.emplace_back();
        labels.emplace_back(dataParams.blockLength + Viterbi::lookbackDist);
        models.emplace_back(dataParams.kernelLanes, SyncDirection::HostWriteDeviceRead);
        auto modelView = models.back().GetHostView();
        for (size_t j = 0; j < dataParams.kernelLanes; ++j)
        {
            modelView[j] = referenceModel;
        }
    }

    auto tmp = [&trans, &models, &dataParams, &latent, &labels](
        TraceBatch<int16_t>& batch,
        size_t batchIdx,
        TraceBatch<int16_t>& ret)
    {
        if (dataParams.laneWidth != 2*gpuBlockThreads) throw PBException("Lane width not currently configurable.  Must be 64 zmw");
        FrameLabelerKernel<<<dataParams.kernelLanes, gpuBlockThreads>>>(
            trans.GetDevicePtr(),
            models[batchIdx].GetDeviceHandle(),
            latent[batchIdx].GetDevicePtr(),
            labels[batchIdx],
            batch,
            ret);
    };

    ZmwDataManager<int16_t, int16_t> manager(dataParams, MakeDataGenerator(dataParams, picketParams, traceParams));
    RunThreads(simulKernels, manager, tmp);
}

}}
