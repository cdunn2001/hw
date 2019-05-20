#include <common/cuda/DisableBoostOnDevice.h>

#include "BaselineRunner.h"
#include "BaselineFilter.cuh"
#include "BaselineFilterKernels.cuh"
#include "ExtremaFilter.cuh"
#include "ExtremaFilterKernels.cuh"

#include <common/cuda/memory/DeviceOnlyArray.cuh>
#include <common/cuda/memory/UnifiedCudaArray.h>
#include <common/cuda/utility/CudaTuple.cuh>
#include <common/DataGenerators/SawtoothGenerator.h>
#include <common/KernelThreadPool.h>

#include <pacbio/PBException.h>

#include <utility>
#include <vector_types.h>

using namespace PacBio::Cuda::Memory;
using namespace PacBio::Cuda::Data;
using namespace PacBio::Cuda;
using namespace PacBio::Mongo::Data;

namespace PacBio {
namespace Cuda {

std::unique_ptr<GeneratorBase<short2>> MakeDataGenerator(const Data::DataManagerParams& dataParams,
                                                         const Data::PicketFenceParams& picketParams,
                                                         const Data::TraceFileParams& traceParams)
{
    return traceParams.traceFileName.empty()
        ? std::unique_ptr<GeneratorBase<short2>>(new PicketFenceGenerator(dataParams, picketParams))
        : std::unique_ptr<GeneratorBase<short2>>(new SignalGenerator(dataParams, traceParams));

}

template <size_t laneWidth>
void RunGlobalBaselineFilter(
        const Data::DataManagerParams& dataParams,
        const Data::PicketFenceParams& picketParams,
        const Data::TraceFileParams& traceParams,
        size_t simulKernels)
{
    using Filter = BaselineFilter<laneWidth, IntSeq<2,8>, IntSeq<9,31>>;
    DeviceOnlyArray<Filter> filterData(dataParams.numZmwLanes, 0);

    auto tmp = [dataParams,&filterData](TraceBatch<short2>& batch, size_t batchIdx, TraceBatch<short2>& ret){
        GlobalBaselineFilter<<<dataParams.kernelLanes, dataParams.gpuLaneWidth>>>(
                batch,
                filterData.GetDeviceView(batchIdx * dataParams.kernelLanes, dataParams.kernelLanes),
                ret);
        auto view = ret.GetBlockView(0);
    };

    ZmwDataManager<short2> manager(dataParams, MakeDataGenerator(dataParams, picketParams, traceParams));
    RunThreads(simulKernels, manager, tmp);
}

template <size_t laneWidth>
void RunSharedBaselineFilter(
        const Data::DataManagerParams& dataParams,
        const Data::PicketFenceParams& picketParams,
        const Data::TraceFileParams& traceParams,
        size_t simulKernels)
{
    using Filter = BaselineFilter<laneWidth, IntSeq<2,8>, IntSeq<9,31>>;
    DeviceOnlyArray<Filter> filterData(dataParams.numZmwLanes, 0);

    auto tmp = [dataParams,&filterData](TraceBatch<short2>& batch, size_t batchIdx, TraceBatch<short2>& ret){
        SharedBaselineFilter<<<dataParams.kernelLanes, dataParams.gpuLaneWidth>>>(
                batch,
                filterData.GetDeviceView(batchIdx * dataParams.kernelLanes, dataParams.kernelLanes),
                ret);
        auto view = ret.GetBlockView(0);
    };

    ZmwDataManager<short2> manager(dataParams, MakeDataGenerator(dataParams, picketParams, traceParams));
    RunThreads(simulKernels, manager, tmp);
}

template <size_t laneWidth>
void RunCompressedBaselineFilter(
        const Data::DataManagerParams& dataParams,
        const Data::PicketFenceParams& picketParams,
        const Data::TraceFileParams& traceParams,
        size_t simulKernels)
{
    using Filter = BaselineFilter<laneWidth, IntSeq<2,8>, IntSeq<9,31>>;
    using Lower1 = ErodeDilate<laneWidth, 9>;
    using Lower2 = ErodeDilate<laneWidth, 31>;
    using Upper1 = DilateErode<laneWidth, 9>;
    using Upper2 = ErodeDilate<laneWidth, 31>;

    DeviceOnlyArray<Lower1> lower1(dataParams.numZmwLanes, 0);
    DeviceOnlyArray<Lower2> lower2(dataParams.numZmwLanes, 0);
    DeviceOnlyArray<Upper1> upper1(dataParams.numZmwLanes, 0);
    DeviceOnlyArray<Upper2> upper2(dataParams.numZmwLanes, 0);

    BatchDimensions dims;
    dims.framesPerBatch = dataParams.blockLength;
    dims.laneWidth = dataParams.gpuLaneWidth;
    dims.lanesPerBatch = dataParams.kernelLanes;
    std::vector<BatchData<short2>> work1;
    std::vector<BatchData<short2>> work2;
    for (size_t i = 0; i < dataParams.numZmwLanes / dataParams.kernelLanes; ++i)
    {
        work1.emplace_back(dims, SyncDirection::HostReadDeviceWrite, nullptr);
        work2.emplace_back(dims, SyncDirection::HostReadDeviceWrite, nullptr);
    }

    auto tmp = [dataParams, &upper1, &upper2, &lower1, &lower2, &work1, &work2]
        (TraceBatch<short2>& batch, size_t batchIdx, TraceBatch<short2>& ret) {
        CompressedBaselineFilter<laneWidth, 9, 31, 2, 8><<<dataParams.kernelLanes, dataParams.gpuLaneWidth>>>(
                batch,
                lower1.GetDeviceView(batchIdx * dataParams.kernelLanes, dataParams.kernelLanes),
                lower2.GetDeviceView(batchIdx * dataParams.kernelLanes, dataParams.kernelLanes),
                upper1.GetDeviceView(batchIdx * dataParams.kernelLanes, dataParams.kernelLanes),
                upper2.GetDeviceView(batchIdx * dataParams.kernelLanes, dataParams.kernelLanes),
                work1[batchIdx],
                work2[batchIdx],
                ret);
        auto view = ret.GetBlockView(0);
    };

    ZmwDataManager<short2> manager(dataParams, MakeDataGenerator(dataParams, picketParams, traceParams));
    RunThreads(simulKernels, manager, tmp);
}

template <size_t laneWidth>
void RunMultipleBaselineFilter(
        const Data::DataManagerParams& dataParams,
        const Data::PicketFenceParams& picketParams,
        const Data::TraceFileParams& traceParams,
        size_t simulKernels)
{
    using Filter = BaselineFilter<laneWidth, IntSeq<2,8>, IntSeq<9,31>>;
    using Filter2 = ComposedFilter<laneWidth, 9, 31, 2, 8>;

    DeviceOnlyArray<Filter> full(dataParams.numZmwLanes, 0);

    BatchDimensions dims;
    dims.laneWidth = dataParams.gpuLaneWidth;
    dims.framesPerBatch = dataParams.blockLength;
    dims.lanesPerBatch = dataParams.kernelLanes;

    std::vector<BatchData<short2>> work1;
    std::vector<BatchData<short2>> work2;
    std::vector<Filter2> filters;
    filters.reserve(dataParams.numZmwLanes / dataParams.kernelLanes);
    for (size_t i = 0; i < dataParams.numZmwLanes / dataParams.kernelLanes; ++i)
    {
        work1.emplace_back(dims, SyncDirection::HostReadDeviceWrite, nullptr);
        work2.emplace_back(dims, SyncDirection::HostReadDeviceWrite, nullptr);
        filters.emplace_back(dataParams.kernelLanes, 0);
    }

    auto tmp = [dataParams, &work1, &work2, &filters, &full]
        (TraceBatch<short2>& batch, size_t batchIdx, TraceBatch<short2>& ret) {

        filters[batchIdx].RunComposedFilter(batch, ret, work1[batchIdx], work2[batchIdx]);
        auto view = ret.GetBlockView(0);
    };

    ZmwDataManager<short2> manager(dataParams, MakeDataGenerator(dataParams, picketParams, traceParams));
    RunThreads(simulKernels, manager, tmp);
}

template <size_t laneWidth>
void RunMaxFilter(const Data::DataManagerParams& params, size_t simulKernels, BaselineFilterMode mode)
{
    static constexpr int FilterWidth = 7;

    using Filter = ExtremaFilter<laneWidth, FilterWidth>;
    DeviceOnlyArray<Filter> filterData(params.numZmwLanes, 0);

    auto tmp = [params,&filterData, mode](TraceBatch<short2>& batch, size_t batchIdx, TraceBatch<short2>& ret){
        switch (mode)
        {
        case BaselineFilterMode::GlobalMax:
        {
            MaxGlobalFilter<laneWidth,FilterWidth><<<params.kernelLanes, params.gpuLaneWidth>>>(
                    batch,
                    filterData.GetDeviceView(batchIdx*params.kernelLanes, params.kernelLanes),
                    ret);
            break;
        }
        case BaselineFilterMode::SharedMax:
        {
            MaxSharedFilter<laneWidth,FilterWidth><<<params.kernelLanes, params.gpuLaneWidth>>>(
                    batch,
                    filterData.GetDeviceView(batchIdx*params.kernelLanes, params.kernelLanes),
                    ret);
            break;
        }
        case BaselineFilterMode::LocalMax:
        {
            MaxLocalFilter<laneWidth,FilterWidth><<<params.kernelLanes, params.gpuLaneWidth>>>(
                    batch,
                    filterData.GetDeviceView(batchIdx*params.kernelLanes, params.kernelLanes),
                    ret);
            break;
        }
        default:
        {
            throw PBException("Unexpected baseline filter mode");
        }
        }
        auto view = ret.GetBlockView(0);
    };

    ZmwDataManager<short2> manager(params, std::make_unique<SawtoothGenerator>(params));
    RunThreads(simulKernels, manager, tmp);
}

void run(const Data::DataManagerParams& dataParams,
         const Data::PicketFenceParams& picketParams,
         const Data::TraceFileParams& traceParams,
         size_t simulKernels,
         BaselineFilterMode mode)
{
    if (mode == BaselineFilterMode::GlobalFull)
    {
        switch (dataParams.gpuLaneWidth)
        {
        case 32:
        {
            RunGlobalBaselineFilter<32>(dataParams, picketParams, traceParams, simulKernels);
            break;
        }
        case 64:
        {
            RunGlobalBaselineFilter<64>(dataParams, picketParams, traceParams, simulKernels);
            break;
        }
        case 128:
        {
            RunGlobalBaselineFilter<128>(dataParams, picketParams, traceParams, simulKernels);
            break;
        }
        case 256:
        {
            RunGlobalBaselineFilter<256>(dataParams, picketParams, traceParams, simulKernels);
            break;
        }
        case 512:
        {
            RunGlobalBaselineFilter<512>(dataParams, picketParams, traceParams, simulKernels);
            break;
        }
        case 1024:
        {
            RunGlobalBaselineFilter<1024>(dataParams, picketParams, traceParams, simulKernels);
            break;
        }
        default:
        {
            throw PBException("Invalid lane size\n");
        }
        }
    }
    else if (mode == BaselineFilterMode::SharedFullCompressed)
    {
        switch (dataParams.gpuLaneWidth)
        {
        case 32:
        {
            RunCompressedBaselineFilter<32>(dataParams, picketParams, traceParams, simulKernels);
            break;
        }
        case 64:
        {
            RunCompressedBaselineFilter<64>(dataParams, picketParams, traceParams, simulKernels);
            break;
        }
        case 128:
        {
            RunCompressedBaselineFilter<128>(dataParams, picketParams, traceParams, simulKernels);
            break;
        }
        default:
        {
            throw PBException("Invalid lane size\n");
        }
        }
    }
    else if (mode == BaselineFilterMode::MultipleFull)
    {
        switch (dataParams.gpuLaneWidth)
        {
        case 32:
        {
            RunMultipleBaselineFilter<32>(dataParams, picketParams, traceParams, simulKernels);
            break;
        }
        case 64:
        {
            RunMultipleBaselineFilter<64>(dataParams, picketParams, traceParams, simulKernels);
            break;
        }
        default:
        {
            throw PBException("Invalid lane size\n");
        }
        }
    }
    else if (mode == BaselineFilterMode::SharedFull)
    {
        switch (dataParams.gpuLaneWidth)
        {
        case 32:
        {
            RunSharedBaselineFilter<32>(dataParams, picketParams, traceParams, simulKernels);
            break;
        }
        case 64:
        {
            RunSharedBaselineFilter<64>(dataParams, picketParams, traceParams, simulKernels);
            break;
        }
        default:
        {
            throw PBException("Invalid lane size\n");
        }
        }
    }
    else
    {
        switch (dataParams.gpuLaneWidth)
        {
        case 32:
        {
            RunMaxFilter<32>(dataParams, simulKernels, mode);
            break;
        }
        case 64:
        {
            RunMaxFilter<64>(dataParams, simulKernels, mode);
            break;
        }
        case 128:
        {
            RunMaxFilter<128>(dataParams, simulKernels, mode);
            break;
        }
        case 256:
        {
            RunMaxFilter<256>(dataParams, simulKernels, mode);
            break;
        }
        case 512:
        {
            RunMaxFilter<512>(dataParams, simulKernels, mode);
            break;
        }
        case 1024:
        {
            RunMaxFilter<1024>(dataParams, simulKernels, mode);
            break;
        }
        default:
        {
            throw PBException("Invalid lane size\n");
        }
        }
    }
}

}}
