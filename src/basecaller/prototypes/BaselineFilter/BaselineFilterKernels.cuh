#ifndef CUDA_BASELINE_FILTER_KERNELS_CUH
#define CUDA_BASELINE_FILTER_KERNELS_CUH

#include "BaselineFilter.cuh"

#include <common/cuda/memory/DeviceOnlyArray.cuh>
#include <dataTypes/TraceBatch.cuh>

namespace PacBio {
namespace Cuda {

template <typename Filter>
__global__ void GlobalBaselineFilter(Mongo::Data::GpuBatchData<short2> in,
                                     Memory::DeviceView<Filter> filters,
                                     Mongo::Data::GpuBatchData<short2> out)
{
    const size_t numFrames = in.Dims().framesPerBatch;
    auto& myFilter = filters[blockIdx.x];
    const auto inZmw  = in.ZmwData(blockIdx.x, threadIdx.x);
    auto outZmw = out.ZmwData(blockIdx.x, threadIdx.x);

    for (int i = 0; i < numFrames; ++i)
    {
        outZmw[i] = myFilter(inZmw[i]);
    }
}

template <typename Filter>
__global__ void SharedBaselineFilter(Mongo::Data::GpuBatchData<short2> in,
                                     Memory::DeviceView<Filter> filters,
                                     Mongo::Data::GpuBatchData<short2> out)
{
    const size_t numFrames = in.Dims().framesPerBatch;
    __shared__ Filter myFilter;
    myFilter = filters[blockIdx.x];
    const auto inZmw  = in.ZmwData(blockIdx.x, threadIdx.x);
    auto outZmw = out.ZmwData(blockIdx.x, threadIdx.x);

    for (int i = 0; i < numFrames; ++i)
    {
        outZmw[i] = myFilter(inZmw[i]);
    }

    filters[blockIdx.x] = myFilter;
}

__device__ int constexpr constexprMax(int a, int b)
{
    return a > b ? a : b;
}

// Tries to implement the full baseline filter in a single kernel, without using shared memory
// for inactive stages.  It may be alright for some settings, but for widths of 9 and 31 it
// was not the best approach.  Loading one filter at a time instead of *all* filters does let us
// use less shared memory and increase our occupancy by a factor of 2, but the fact that we now
// have to have multiple passes over the data kills those gains.  Instead having separate
// kernels for each filter stage actually performs better.  We still have multiple passes over the
// data, but now the kernels with smaller filters can use a lot less shared memory and get a lot
// more increased occupancy, making them run significantly faster.
template <size_t laneWidth, size_t width1, size_t width2, size_t stride1, size_t stride2>
__global__ void CompressedBaselineFilter(Mongo::Data::GpuBatchData<short2> in,
                                         Memory::DeviceView<ErodeDilate<laneWidth, width1>> lower1,
                                         Memory::DeviceView<ErodeDilate<laneWidth, width2>> lower2,
                                         Memory::DeviceView<DilateErode<laneWidth, width1>> upper1,
                                         Memory::DeviceView<ErodeDilate<laneWidth, width2>> upper2,
                                         Mongo::Data::GpuBatchData<short2> workspace1,
                                         Mongo::Data::GpuBatchData<short2> workspace2,
                                         Mongo::Data::GpuBatchData<short2> out)
{
    const size_t numFrames = in.Dims().framesPerBatch;

    // Grab a swath of memory that can fit our largest filter
    constexpr size_t maxSize = constexprMax(sizeof(ErodeDilate<laneWidth, width1>), sizeof(ErodeDilate<laneWidth, width2>));
    __shared__  char mempool[maxSize];

    assert(numFrames % stride1*stride2 == 0);

    // Get block specific workspaces
    const auto inb  = in.ZmwData(blockIdx.x, threadIdx.x);
    auto outb = out.ZmwData(blockIdx.x, threadIdx.x);
    auto workb1 = workspace1.ZmwData(blockIdx.x, threadIdx.x);
    auto workb2 = workspace2.ZmwData(blockIdx.x, threadIdx.x);

    // --------------Lower filter-------------------
    ErodeDilate<laneWidth, width1>* low1 = new (&mempool[0])ErodeDilate<laneWidth, width1>(lower1[blockIdx.x]);
    for (int i = 0; i < numFrames; i += stride1)
    {
        workb1[i/stride1] = (*low1)(inb[i]);
    }
    lower1[blockIdx.x] = *low1;
    low1->~ErodeDilate<laneWidth, width1>();
    low1 = nullptr;

    ErodeDilate<laneWidth, width2>* low2 = new (&mempool[0])ErodeDilate<laneWidth, width2>(lower2[blockIdx.x]);
    int limit = numFrames / stride1;
    for (int i = 0; i < limit; i += stride2)
    {
        workb1[i/stride2] = (*low2)(workb1[i]);
    }
    lower2[blockIdx.x] = *low2;
    low2->~ErodeDilate<laneWidth, width2>();
    low2 = nullptr;

    // --------------Upper filter-------------------
    DilateErode<laneWidth, width1>* up1 = new (&mempool[0])DilateErode<laneWidth, width1>(upper1[blockIdx.x]);
    for (int i = 0; i < numFrames; i += stride1)
    {
        workb2[i/stride1] = (*up1)(inb[i]);

    }
    upper1[blockIdx.x] = *up1;
    up1->~DilateErode<laneWidth, width1>();
    up1 = nullptr;

    ErodeDilate<laneWidth, width2>* up2 = new (&mempool[0])ErodeDilate<laneWidth, width2>(upper2[blockIdx.x]);
    limit = numFrames / stride1;
    for (int i = 0; i < limit; i += stride2)
    {
        workb2[i/stride2] = (*up2)(workb2[i]);
    }
    upper2[blockIdx.x] = *up2;
    up2->~ErodeDilate<laneWidth, width2>();
    up2 = nullptr;
}

// Runs a filter over the input data, but only outputs 1/stride of the
// results.
template <size_t laneWidth, size_t stride, typename Filter>
__global__ void StridedFilter(Mongo::Data::GpuBatchData<short2> in,
                              Memory::DeviceView<Filter> filters,
                              int numFrames,
                              Mongo::Data::GpuBatchData<short2> out)
{
    const size_t maxFrames = in.Dims().framesPerBatch;

    assert(laneWidth == blockDim.x);
    assert(numFrames <= out.Dims().framesPerBatch);
    assert(numFrames <= maxFrames);
    assert(numFrames % 4 == 0);
    assert((numFrames/4) % stride == 0);

    __shared__ Filter myFilter;
    myFilter = filters[blockIdx.x];
    const auto inZmw = in.ZmwData(blockIdx.x, threadIdx.x);
    auto outZmw = out.ZmwData(blockIdx.x, threadIdx.x);

    for (int i = 0; i < numFrames; i += 4*stride)
    {
        Bundle b;
        b.data0 = inZmw[i];
        b.data1 = inZmw[i + stride];
        b.data2 = inZmw[i + 2*stride];
        b.data3 = inZmw[i + 3*stride];
        myFilter(b);

        outZmw[i/stride]   = b.data0;
        outZmw[i/stride+1] = b.data1;
        outZmw[i/stride+2] = b.data2;
        outZmw[i/stride+3] = b.data3;
    }
    filters[blockIdx.x] = myFilter;
}

// Averages the output of the lower and upper filters, and expands the data
// to back to the original (unstrided) size
template <size_t laneWidth, size_t stride>
__global__ void AverageAndExpand(Mongo::Data::GpuBatchData<short2> in1,
                                 Mongo::Data::GpuBatchData<short2> in2,
                                 Mongo::Data::GpuBatchData<short2> out)
{
    const size_t numFrames = out.Dims().framesPerBatch;

    assert(numFrames % stride == 0);
    int inputCount = numFrames / stride;

    const auto inZmw1 = in1.ZmwData(blockIdx.x, threadIdx.x);
    const auto inZmw2 = in2.ZmwData(blockIdx.x, threadIdx.x);
    auto outZmw = out.ZmwData(blockIdx.x, threadIdx.x);

    for (int i = 0; i < inputCount; ++i)
    {
        short2 val;
        val.x = (inZmw1[i].x + inZmw2[i].x)/2;
        val.y = (inZmw1[i].y + inZmw2[i].y)/2;

        for (int j = i*stride; j < stride; ++j)
        {
            outZmw[j] = val;
        }
    }
}

template <size_t laneWidth, size_t width1, size_t width2, size_t stride1, size_t stride2>
class ComposedFilter
{
    using Lower1 = ErodeDilate<laneWidth, width1>;
    using Lower2 = ErodeDilate<laneWidth, width2>;
    using Upper1 = DilateErode<laneWidth, width1>;
    using Upper2 = ErodeDilate<laneWidth, width2>;

    Memory::DeviceOnlyArray<Lower1> lower1;
    Memory::DeviceOnlyArray<Lower2> lower2;
    Memory::DeviceOnlyArray<Upper1> upper1;
    Memory::DeviceOnlyArray<Upper2> upper2;
    size_t numLanes_;

public:
    __host__ ComposedFilter(size_t numLanes, short val)
        : lower1(numLanes, val)
        , lower2(numLanes, val)
        , upper1(numLanes, val)
        , upper2(numLanes, val)
        , numLanes_(numLanes)
    {}

    __host__ void RunComposedFilter(Mongo::Data::TraceBatch<short2>& input,
                                    Mongo::Data::TraceBatch<short2>& output,
                                    Mongo::Data::BatchData<short2>& workspace1,
                                    Mongo::Data::BatchData<short2>& workspace2)
    {
        const uint64_t numFrames = input.Dimensions().framesPerBatch;

        assert(input.Dimensions().laneWidth == laneWidth);
        assert(input.Dimensions().lanesPerBatch == numLanes_);

        StridedFilter<laneWidth, 2, Lower1><<<numLanes_, laneWidth>>>(
            input,
            lower1.GetDeviceView(),
            numFrames,
            workspace1);
        StridedFilter<laneWidth, 8, Lower2><<<numLanes_, laneWidth>>>(
            workspace1,
            lower2.GetDeviceView(),
            numFrames/2,
            workspace1);

        StridedFilter<laneWidth, 2, Upper1><<<numLanes_, laneWidth>>>(
            input,
            upper1.GetDeviceView(),
            numFrames,
            workspace2);
        StridedFilter<laneWidth, 8, Upper2><<<numLanes_, laneWidth>>>(
            workspace2,
            upper2.GetDeviceView(),
            numFrames/2,
            workspace2);

        AverageAndExpand<laneWidth, 16><<<numLanes_, laneWidth>>>(workspace1, workspace2, output);
    }

};

}}

#endif //CUDA_BASELINE_FILTER_KERNELS_CUH
