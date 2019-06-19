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

static std::unique_ptr<GeneratorBase<int16_t>> MakeDataGenerator(
        const Data::DataManagerParams& dataParams,
        const Data::PicketFenceParams& picketParams,
        const Data::TraceFileParams& traceParams)
{
    return traceParams.traceFileName.empty()
        ? std::unique_ptr<GeneratorBase<int16_t>>(new PicketFenceGenerator(dataParams, picketParams))
        : std::unique_ptr<GeneratorBase<int16_t>>(new SignalGenerator(dataParams, traceParams));

}

template <size_t laneWidth>
void RunGlobalBaselineFilter(
        const Data::DataManagerParams& dataParams,
        const Data::PicketFenceParams& picketParams,
        const Data::TraceFileParams& traceParams,
        size_t simulKernels)
{
    using Filter = BaselineFilter<laneWidth, IntSeq<2,8>, IntSeq<9,31>>;
    std::vector<DeviceOnlyArray<Filter>> filterData;
    for (int i = 0; i < dataParams.numZmwLanes / dataParams.kernelLanes; ++i)
    {
        filterData.emplace_back(dataParams.kernelLanes, 0);
    }

    auto tmp = [dataParams,&filterData](const TraceBatch<int16_t>& batch, size_t batchIdx, TraceBatch<int16_t>& ret){
        GlobalBaselineFilter<<<dataParams.kernelLanes, dataParams.laneWidth/2>>>(
                batch,
                filterData[batchIdx].GetDeviceView(),
                ret);
        auto view = ret.GetBlockView(0);
    };

    ZmwDataManager<int16_t> manager(dataParams, MakeDataGenerator(dataParams, picketParams, traceParams));
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
    std::vector<DeviceOnlyArray<Filter>> filterData;
    for (int i = 0; i < dataParams.numZmwLanes / dataParams.kernelLanes; ++i)
    {
        filterData.emplace_back(dataParams.kernelLanes, 0);
    }

    auto tmp = [dataParams,&filterData](const TraceBatch<int16_t>& batch, size_t batchIdx, TraceBatch<int16_t>& ret){
        SharedBaselineFilter<<<dataParams.kernelLanes, dataParams.laneWidth/2>>>(
                batch,
                filterData[batchIdx].GetDeviceView(),
                ret);
        auto view = ret.GetBlockView(0);
    };

    ZmwDataManager<int16_t> manager(dataParams, MakeDataGenerator(dataParams, picketParams, traceParams));
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

    std::vector<DeviceOnlyArray<Lower1>> lower1;
    std::vector<DeviceOnlyArray<Lower2>> lower2;
    std::vector<DeviceOnlyArray<Upper1>> upper1;
    std::vector<DeviceOnlyArray<Upper2>> upper2;
    for (int i = 0; i < dataParams.numZmwLanes / dataParams.kernelLanes; ++i)
    {
        lower1.emplace_back(dataParams.kernelLanes, 0);
        lower2.emplace_back(dataParams.kernelLanes, 0);
        upper1.emplace_back(dataParams.kernelLanes, 0);
        upper2.emplace_back(dataParams.kernelLanes, 0);
    }

    BatchDimensions dims;
    dims.framesPerBatch = dataParams.blockLength;
    dims.laneWidth = dataParams.laneWidth;
    dims.lanesPerBatch = dataParams.kernelLanes;
    std::vector<BatchData<int16_t>> work1;
    std::vector<BatchData<int16_t>> work2;
    for (size_t i = 0; i < dataParams.numZmwLanes / dataParams.kernelLanes; ++i)
    {
        work1.emplace_back(dims, SyncDirection::HostReadDeviceWrite, nullptr);
        work2.emplace_back(dims, SyncDirection::HostReadDeviceWrite, nullptr);
    }

    auto tmp = [dataParams, &upper1, &upper2, &lower1, &lower2, &work1, &work2]
        (const TraceBatch<int16_t>& batch, size_t batchIdx, TraceBatch<int16_t>& ret) {
        CompressedBaselineFilter<laneWidth, 9, 31, 2, 8><<<dataParams.kernelLanes, dataParams.laneWidth/2>>>(
                batch,
                lower1[batchIdx].GetDeviceView(),
                lower2[batchIdx].GetDeviceView(),
                upper1[batchIdx].GetDeviceView(),
                upper2[batchIdx].GetDeviceView(),
                work1[batchIdx],
                work2[batchIdx],
                ret);
        auto view = ret.GetBlockView(0);
    };

    ZmwDataManager<int16_t> manager(dataParams, MakeDataGenerator(dataParams, picketParams, traceParams));
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
    dims.laneWidth = dataParams.laneWidth;
    dims.framesPerBatch = dataParams.blockLength;
    dims.lanesPerBatch = dataParams.kernelLanes;

    std::vector<BatchData<int16_t>> work1;
    std::vector<BatchData<int16_t>> work2;
    std::vector<Filter2> filters;
    filters.reserve(dataParams.numZmwLanes / dataParams.kernelLanes);
    for (size_t i = 0; i < dataParams.numZmwLanes / dataParams.kernelLanes; ++i)
    {
        work1.emplace_back(dims, SyncDirection::HostReadDeviceWrite, nullptr);
        work2.emplace_back(dims, SyncDirection::HostReadDeviceWrite, nullptr);
        filters.emplace_back(dataParams.kernelLanes, 0);
    }

    auto tmp = [dataParams, &work1, &work2, &filters, &full]
        (const TraceBatch<int16_t>& batch, size_t batchIdx, TraceBatch<int16_t>& ret) {

        filters[batchIdx].RunComposedFilter(batch, ret, work1[batchIdx], work2[batchIdx]);
        auto view = ret.GetBlockView(0);
    };

    ZmwDataManager<int16_t> manager(dataParams, MakeDataGenerator(dataParams, picketParams, traceParams));
    RunThreads(simulKernels, manager, tmp);
}

template <size_t laneWidth>
void RunMaxFilter(const Data::DataManagerParams& params, size_t simulKernels, BaselineFilterMode mode)
{
    static constexpr int FilterWidth = 7;

    using Filter = ExtremaFilter<laneWidth, FilterWidth>;
    std::vector<DeviceOnlyArray<Filter>> filterData;
    for (int i = 0; i < params.numZmwLanes / params.kernelLanes; ++i)
    {
        filterData.emplace_back(params.kernelLanes, 0);
    }

    auto tmp = [params,&filterData, mode](const TraceBatch<int16_t>& batch, size_t batchIdx, TraceBatch<int16_t>& ret){
        switch (mode)
        {
        case BaselineFilterMode::GlobalMax:
        {
            MaxGlobalFilter<laneWidth,FilterWidth><<<params.kernelLanes, params.laneWidth/2>>>(
                    batch,
                    filterData[batchIdx].GetDeviceView(),
                    ret);
            break;
        }
        case BaselineFilterMode::SharedMax:
        {
            MaxSharedFilter<laneWidth,FilterWidth><<<params.kernelLanes, params.laneWidth/2>>>(
                    batch,
                    filterData[batchIdx].GetDeviceView(),
                    ret);
            break;
        }
        case BaselineFilterMode::LocalMax:
        {
            MaxLocalFilter<laneWidth,FilterWidth><<<params.kernelLanes, params.laneWidth/2>>>(
                    batch,
                    filterData[batchIdx].GetDeviceView(),
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

    ZmwDataManager<int16_t> manager(params, std::make_unique<SawtoothGenerator>(params));
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
        switch (dataParams.laneWidth)
        {
        case 64:
        {
            RunGlobalBaselineFilter<32>(dataParams, picketParams, traceParams, simulKernels);
            break;
        }
        case 128:
        {
            RunGlobalBaselineFilter<64>(dataParams, picketParams, traceParams, simulKernels);
            break;
        }
        case 256:
        {
            RunGlobalBaselineFilter<128>(dataParams, picketParams, traceParams, simulKernels);
            break;
        }
        case 512:
        {
            RunGlobalBaselineFilter<256>(dataParams, picketParams, traceParams, simulKernels);
            break;
        }
        case 1024:
        {
            RunGlobalBaselineFilter<512>(dataParams, picketParams, traceParams, simulKernels);
            break;
        }
        case 2048:
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
        switch (dataParams.laneWidth)
        {
        case 64:
        {
            RunCompressedBaselineFilter<32>(dataParams, picketParams, traceParams, simulKernels);
            break;
        }
        case 128:
        {
            RunCompressedBaselineFilter<64>(dataParams, picketParams, traceParams, simulKernels);
            break;
        }
        case 256:
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
        switch (dataParams.laneWidth)
        {
        case 64:
        {
            RunMultipleBaselineFilter<32>(dataParams, picketParams, traceParams, simulKernels);
            break;
        }
        case 128:
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
        switch (dataParams.laneWidth)
        {
        case 64:
        {
            RunSharedBaselineFilter<32>(dataParams, picketParams, traceParams, simulKernels);
            break;
        }
        case 128:
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
        switch (dataParams.laneWidth)
        {
        case 64:
        {
            RunMaxFilter<32>(dataParams, simulKernels, mode);
            break;
        }
        case 128:
        {
            RunMaxFilter<64>(dataParams, simulKernels, mode);
            break;
        }
        case 256:
        {
            RunMaxFilter<128>(dataParams, simulKernels, mode);
            break;
        }
        case 512:
        {
            RunMaxFilter<256>(dataParams, simulKernels, mode);
            break;
        }
        case 1024:
        {
            RunMaxFilter<512>(dataParams, simulKernels, mode);
            break;
        }
        case 2048:
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
