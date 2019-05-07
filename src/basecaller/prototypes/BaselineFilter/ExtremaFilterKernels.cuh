#ifndef EXTREMA_FILTER_KERNELS_CUH
#define EXTREMA_FILTER_KERNELS_CUH

#include <dataTypes/TraceBatch.cuh>

#include "ExtremaFilter.cuh"

namespace PacBio {
namespace Cuda {

// Simply runs the filter in-place in global memory
template <size_t laneWidth, size_t filterWidth>
__global__ void MaxGlobalFilter(Mongo::Data::GpuBatchData<short2> in,
                                Memory::DeviceView<ExtremaFilter<laneWidth, filterWidth>> filters,
                                Mongo::Data::GpuBatchData<short2> out)
{
    assert(laneWidth == blockDim.x);

    const size_t numFrames = in.Dims().framesPerBatch;

    auto& myFilter = filters[blockIdx.x];
    auto inZmw  = in.ZmwData(blockIdx.x, threadIdx.x);
    auto outZmw = out.ZmwData(blockIdx.x, threadIdx.x);

    for (int i = 0; i < numFrames; ++i)
    {
        outZmw[i] = myFilter(inZmw[i]);
    }
}

// First coppies the filter to shared memory before processing data.
template <size_t laneWidth, size_t filterWidth>
__global__ void MaxSharedFilter(Mongo::Data::GpuBatchData<short2> in,
                                Memory::DeviceView<ExtremaFilter<laneWidth, filterWidth>> filters,
                                Mongo::Data::GpuBatchData<short2> out)
{
    assert(laneWidth == blockDim.x);

    const size_t numFrames = in.Dims().framesPerBatch;

    __shared__ ExtremaFilter<laneWidth, filterWidth> myFilter;
    myFilter = filters[blockIdx.x];
    auto inZmw  = in.ZmwData(blockIdx.x, threadIdx.x);
    auto outZmw = out.ZmwData(blockIdx.x, threadIdx.x);

    for (int i = 0; i < numFrames; ++i)
    {
        outZmw[i] = myFilter(inZmw[i]);
    }

    filters[blockIdx.x] = myFilter;
}

// Moves filter data to registers for processing
template <size_t laneWidth, size_t filterWidth>
__global__ void MaxLocalFilter(Mongo::Data::GpuBatchData<short2> in,
                               Memory::DeviceView<ExtremaFilter<laneWidth, filterWidth>> filters,
                               Mongo::Data::GpuBatchData<short2> out)
{

    const size_t numFrames = in.Dims().framesPerBatch;
    LocalExtremaFilter<laneWidth, filterWidth> myFilter(filters[blockIdx.x]);
    auto inZmw  = in.ZmwData(blockIdx.x, threadIdx.x);
    auto outZmw = out.ZmwData(blockIdx.x, threadIdx.x);

    for (int i = 0; i < numFrames; ++i)
    {
        outZmw[i] = myFilter(inZmw[i]);
    }

    myFilter.ReplaceShared(filters[blockIdx.x]);
}

}}

#endif //EXTREMA_FILTER_KERNELS_CUH
