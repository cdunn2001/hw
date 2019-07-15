#ifndef EXTREMA_FILTER_KERNELS_CUH
#define EXTREMA_FILTER_KERNELS_CUH

#include <dataTypes/BatchData.cuh>

#include "ExtremaFilter.cuh"

namespace PacBio {
namespace Cuda {

// Simply runs the filter in-place in global memory
template <size_t blockThreads, size_t filterWidth>
__global__ void MaxGlobalFilter(const Mongo::Data::GpuBatchData<const PBShort2> in,
                                Memory::DeviceView<ExtremaFilter<blockThreads, filterWidth>> filters,
                                Mongo::Data::GpuBatchData<PBShort2> out)
{
    assert(blockThreads == blockDim.x);

    const size_t numFrames = in.NumFrames();

    auto& myFilter = filters[blockIdx.x];
    const auto& inZmw  = in.ZmwData(blockIdx.x, threadIdx.x);
    auto outZmw = out.ZmwData(blockIdx.x, threadIdx.x);

    for (int i = 0; i < numFrames; ++i)
    {
        outZmw[i] = myFilter(inZmw[i]);
    }
}

// First coppies the filter to shared memory before processing data.
template <size_t blockThreads, size_t filterWidth>
__global__ void MaxSharedFilter(const Mongo::Data::GpuBatchData<const PBShort2> in,
                                Memory::DeviceView<ExtremaFilter<blockThreads, filterWidth>> filters,
                                Mongo::Data::GpuBatchData<PBShort2> out)
{
    assert(blockThreads == blockDim.x);

    const size_t numFrames = in.NumFrames();

    __shared__ ExtremaFilter<blockThreads, filterWidth> myFilter;
    myFilter = filters[blockIdx.x];
    const auto& inZmw  = in.ZmwData(blockIdx.x, threadIdx.x);
    auto outZmw = out.ZmwData(blockIdx.x, threadIdx.x);

    for (int i = 0; i < numFrames; ++i)
    {
        outZmw[i] = myFilter(inZmw[i]);
    }

    filters[blockIdx.x] = myFilter;
}

// Moves filter data to registers for processing
template <size_t blockThreads, size_t filterWidth>
__global__ void MaxLocalFilter(const Mongo::Data::GpuBatchData<const PBShort2> in,
                               Memory::DeviceView<ExtremaFilter<blockThreads, filterWidth>> filters,
                               Mongo::Data::GpuBatchData<PBShort2> out)
{

    const size_t numFrames = in.NumFrames();
    LocalExtremaFilter<blockThreads, filterWidth> myFilter(filters[blockIdx.x]);
    const auto& inZmw  = in.ZmwData(blockIdx.x, threadIdx.x);
    auto outZmw = out.ZmwData(blockIdx.x, threadIdx.x);

    for (int i = 0; i < numFrames; ++i)
    {
        outZmw[i] = myFilter(inZmw[i]);
    }

    myFilter.ReplaceShared(filters[blockIdx.x]);
}

}}

#endif //EXTREMA_FILTER_KERNELS_CUH
