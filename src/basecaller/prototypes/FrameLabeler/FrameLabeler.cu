#include "FrameLabeler.h"

#include <common/cuda/memory/DeviceOnlyArray.cuh>
#include <common/KernelThreadPool.h>
#include <dataTypes/TraceBatch.cuh>

using namespace PacBio::Cuda::Memory;
using namespace PacBio::Cuda::Data;
using namespace PacBio::Mongo::Data;

namespace PacBio {
namespace Cuda {

template <typename T>
class DevicePtr
{
public:
    DevicePtr(T* data, detail::DataManagerKey)
        : data_(data)
    {}

    __device__ T* operator->() { return data_; }
    __device__ const T* operator->() const { return data_; }
private:
    T* data_;
};

template <typename T>
class DeviceOnlyObj : private detail::DataManager
{
public:
    template <typename... Args>
    DeviceOnlyObj(Args&&... args)
        : data_(1, std::forward<Args>(args)...)
    {}

    DevicePtr<T> GetDevicePtr()
    {
        return DevicePtr<T>(data_.GetDeviceView().Data(DataKey()), DataKey());
    }

private:
    Memory::DeviceOnlyArray<T> data_;
};

struct TransitionMatrix
{

};

struct BlockStateScorer
{

};

struct BlockModelParameters
{

};

// First arg should be const?
__global__ void FrameLabelerKernel(DevicePtr<TransitionMatrix> trans,
                                   DeviceView<BlockModelParameters>,
                                   GpuBatchData<short2> input,
                                   GpuBatchData<short2> output)
{

}

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

    DeviceOnlyObj<TransitionMatrix> trans;
    std::vector<UnifiedCudaArray<BlockModelParameters>> models;
    const auto numBatches = dataParams.numZmwLanes / dataParams.kernelLanes;
    for (size_t i = 0; i < numBatches; ++i)
    {
        models.emplace_back(dataParams.kernelLanes, SyncDirection::HostWriteDeviceRead);
        // TODO populate models
    }

    auto tmp = [&trans, &models, &dataParams](TraceBatch<int16_t>& batch, size_t batchIdx, TraceBatch<int16_t>& ret){
        FrameLabelerKernel<<<dataParams.kernelLanes, dataParams.laneWidth/2>>>(trans.GetDevicePtr(), models[batchIdx].GetDeviceHandle(), batch, ret);
    };

    ZmwDataManager<int16_t, int16_t> manager(dataParams, MakeDataGenerator(dataParams, picketParams, traceParams));
    RunThreads(simulKernels, manager, tmp);
}

}}
