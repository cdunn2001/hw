#ifndef CUDA_CIRCULAR_BUFFER_H
#define CUDA_CIRCULAR_BUFFER_H

#include <dataTypes/BatchData.cuh>
#include <common/cuda/PBCudaSimd.cuh>

namespace PacBio {
namespace Cuda {

template<size_t blockThreads, size_t lag, template<size_t,size_t> class CircularBuffer>
__global__ void GlobalMemCircularBuffer(const Mongo::Data::GpuBatchData<const PBShort2> in,
                                        Memory::DeviceView<CircularBuffer<blockThreads,lag>> circularBuffers,
                                        Mongo::Data::GpuBatchData<PBShort2> out)
{
    assert(blockThreads == blockDim.x);

    const size_t numFrames = in.NumFrames();
    auto& circularBuffer = circularBuffers[blockIdx.x];
    const auto& inZmw = in.ZmwData(blockIdx.x, threadIdx.x);
    auto outZmw = out.ZmwData(blockIdx.x, threadIdx.x);

    for (int i = 0; i < numFrames; ++i)
    {
        outZmw[i] = ToShort(circularBuffer.Front());
        circularBuffer.PushBack(PBHalf2(inZmw[i]));
    }
}

template<size_t blockThreads, size_t lag, template<size_t,size_t> class CircularBuffer>
__global__ void SharedMemCircularBuffer(const Mongo::Data::GpuBatchData<const PBShort2> in,
                                        Memory::DeviceView <CircularBuffer<blockThreads,lag>> circularBuffers,
                                        Mongo::Data::GpuBatchData <PBShort2> out)
{
    assert(blockThreads == blockDim.x);

    const size_t numFrames = in.NumFrames();
    __shared__ CircularBuffer<blockThreads,lag> circularBuffer;
    circularBuffer = circularBuffers[blockIdx.x];
    const auto& inZmw = in.ZmwData(blockIdx.x, threadIdx.x);
    auto outZmw = out.ZmwData(blockIdx.x, threadIdx.x);

    for (int i = 0; i < numFrames; ++i)
    {
        outZmw[i] = ToShort(circularBuffer.Front());
        circularBuffer.PushBack(PBHalf2(inZmw[i]));
    }
    circularBuffers[blockIdx.x] = circularBuffer;
}

template <size_t lag, template <size_t> class CircularBuffer>
__global__ void LocalMemCircularBuffer(const Mongo::Data::GpuBatchData<const PBShort2> in,
                                       Memory::DeviceView<CircularBuffer<lag>> circularBuffers,
                                       Mongo::Data::GpuBatchData<PBShort2> out)
{
    const size_t numFrames = in.NumFrames();
    CircularBuffer<lag> circularBuffer = circularBuffers[blockIdx.x];
    const auto& inZmw  = in.ZmwData(blockIdx.x, threadIdx.x);
    auto outZmw = out.ZmwData(blockIdx.x, threadIdx.x);

    for (int i = 0; i < numFrames; ++i)
    {
        outZmw[i] = ToShort(circularBuffer.Front());
        circularBuffer.PushBack(PBHalf2(inZmw[i]));
    }
    circularBuffers[blockIdx.x] = circularBuffer;
}

}} // PacBio::Cuda

#endif // CUDA_CIRCULAR_BUFFER_H