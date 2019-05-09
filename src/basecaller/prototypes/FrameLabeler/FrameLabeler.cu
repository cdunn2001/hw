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
         size_t simulKernels)
{
    static constexpr size_t gpuBlockThreads = 32;

    CudaArray<float, 4> pw(0.2f);
    CudaArray<float, 4> ipd(0.5f);
    CudaArray<float, 4> pwss(3.2f);
    CudaArray<float, 4> ipdss(0.0f);

    DeviceOnlyObj<TransitionMatrix> trans(pw, ipd, pwss, ipdss);
    std::vector<UnifiedCudaArray<LaneModelParameters<gpuBlockThreads>>> models;
    std::vector<ViterbiDataHost<PBHalf2, gpuBlockThreads>> scores;
    std::vector<ViterbiDataHost<short2, gpuBlockThreads>> labels;

    const auto numBatches = dataParams.numZmwLanes / dataParams.kernelLanes;
    models.reserve(numBatches);
    scores.reserve(numBatches);
    labels.reserve(numBatches);
    for (size_t i = 0; i < numBatches; ++i)
    {
        // ------------------------------------------------------
        // TODO we need to set our initial conditions somehow!!!
        // Currently hard coding to start in zero state.  Should be fine,
        // but we need to make sure it's built properly into the API
        // ------------------------------------------------------
        scores.emplace_back(dataParams.blockLength+1, PBHalf2(0.0f));
        labels.emplace_back(dataParams.blockLength);
        models.emplace_back(dataParams.kernelLanes, SyncDirection::HostWriteDeviceRead);
        // TODO populate models
    }

    auto tmp = [&trans, &models, &dataParams, &scores, &labels](
        TraceBatch<int16_t>& batch,
        size_t batchIdx,
        TraceBatch<int16_t>& ret)
    {
        if (dataParams.laneWidth != 2*gpuBlockThreads) throw PBException("Lane width not currently configurable.  Must be 64 zmw");
        FrameLabelerKernel<<<dataParams.kernelLanes, gpuBlockThreads>>>(
            trans.GetDevicePtr(),
            models[batchIdx].GetDeviceHandle(),
            scores[batchIdx],
            labels[batchIdx],
            batch,
            ret);
    };

    ZmwDataManager<int16_t, int16_t> manager(dataParams, MakeDataGenerator(dataParams, picketParams, traceParams));
    RunThreads(simulKernels, manager, tmp);
}

}}
