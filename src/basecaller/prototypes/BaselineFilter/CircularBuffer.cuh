#ifndef CIRCULAR_BUFFER_CUH
#define CIRCULAR_BUFFER_CUH

#include <cassert>
#include <dataTypes/BatchData.cuh>
#include <common/cuda/PBCudaSimd.cuh>

namespace PacBio {
namespace Cuda {

static constexpr short Capacity = 4;

template <size_t blockThreads>
struct __align__(128) CircularBuffer
{
    CircularBuffer() = default;

    __device__ CircularBuffer(short dummyVal)
    {
        // Pre-load the buffer and treat it as initially starting out full.
        for (size_t i = 0; i < blockThreads; ++i)
        {
            front[i] = 0;
        }
        for (size_t i = 0; i < Capacity; ++i)
        {
            for (size_t j = 0; j < blockThreads; ++j)
            {
                data[i][j] = 0;
            }
        }
    }

    __device__ PBHalf2 Front()
    {
        return data[front[threadIdx.x]][threadIdx.x];
    }

    __device__ void PushBack(PBHalf2 val)
    {
        data[front[threadIdx.x]][threadIdx.x] = val;
        front[threadIdx.x] = (front[threadIdx.x] + 1) % Capacity;
    }

    __device__ CircularBuffer& operator=(const CircularBuffer& cb)
    {
        for (size_t i = 0; i < Capacity; ++i)
        {
            data[i][threadIdx.x] = cb.data[i][threadIdx.x];
        }
        front[threadIdx.x] = cb.front[threadIdx.x];
        __syncthreads();
        return *this;
    }

    using row = PBHalf2[blockThreads];
    row data[Capacity];
    short front[blockThreads];
};

template <size_t blockThreads>
struct CircularBufferLocal
{
    __device__ CircularBufferLocal(const CircularBuffer<blockThreads>& cb)
    {
        #pragma unroll(Capacity)
        for (size_t i = 0; i < Capacity; ++i)
        {
            data[i] = cb.data[i][threadIdx.x];
        }
        front = cb.front[threadIdx.x];
        __syncthreads();
    }

    __device__ PBHalf2 Front()
    {
        switch (front)
        {
            static_assert(Capacity == 4, "Only supported value of Capacity=4");
            case 0:
                return data[0];
            case 1:
                return data[1];
            case 2:
                return data[2];
            default:
                assert(front == 3);
                return data[3];
        }
    }

    __device__ void PushBack(PBHalf2 val)
    {
        switch (front)
        {
            static_assert(Capacity == 4, "Only supported value of Capacity=4");
            case 0:
                data[0] = val;
                break;
            case 1:
                data[1] = val;
                break;
            case 2:
                data[2] = val;
                break;
            default:
                assert(front == 3);
                data[3] = val;
        }
        front = (front + 1) % Capacity;
    }

    __device__ void ReplaceShared(CircularBuffer<blockThreads>& cb)
    {
        #pragma unroll(Capacity)
        for (size_t i = 0; i < Capacity; ++i)
        {
            cb.data[i][threadIdx.x] = data[i];
        }
        cb.front[threadIdx.x] = front;
        __syncthreads();
    }

    PBHalf2 data[Capacity];
    short front;
};

template <size_t blockThreads>
struct __align__(128) CircularBufferShift
{
    CircularBufferShift() = default;

    __device__ CircularBufferShift(short dummyVal)
    {
        // Pre-load the buffer and treat it as initially starting out full.
        for (size_t i = 0; i < Capacity; ++i)
        {
            for (size_t j = 0; j < blockThreads; ++j)
            {
                data[i][j] = 0;
            }
        }
    }

    __device__ PBHalf2 Front()
    {
        return data[0][threadIdx.x];
    }

    __device__ void PushBack(PBHalf2 val)
    {
        for (size_t i = 1; i < Capacity; i++)
        {
            data[i-1][threadIdx.x] = data[i][threadIdx.x];
        }
        data[Capacity-1][threadIdx.x] = val;
    }

    __device__ CircularBufferShift& operator=(const CircularBufferShift& cb)
    {
        for (size_t i = 0; i < Capacity; ++i)
        {
            data[i][threadIdx.x] = cb.data[i][threadIdx.x];
        }
        __syncthreads();
        return *this;
    }

    using row = PBHalf2[blockThreads];
    row data[Capacity];
};

template <size_t blockThreads>
struct CircularBufferShiftLocal
{
    __device__ CircularBufferShiftLocal(const CircularBufferShift<blockThreads>& cb)
    {
        // Pre-load the buffer and treat it as initially starting out full.
        #pragma unroll(Capacity)
        for (size_t i = 0; i < Capacity; ++i)
        {
            data[i] = cb.data[i][threadIdx.x];
        }
        __syncthreads();
    }

    __device__ PBHalf2 Front()
    {
        return data[0];
    }

    __device__ void PushBack(PBHalf2 val)
    {
        #pragma unroll(Capacity)
        for (size_t i = 1; i < Capacity; i++)
        {
            data[i-1] = data[i];
        }
        data[Capacity-1] = val;
    }

    __device__ void ReplaceShared(CircularBufferShift<blockThreads>& cb)
    {
        #pragma unroll(Capacity)
        for (size_t i = 0; i < Capacity; ++i)
        {
            cb.data[i][threadIdx.x] = data[i];
        }
        __syncthreads();
    }

    PBHalf2 data[Capacity];
};

template <size_t blockThreads, template <size_t> class CircularBuffer>
__global__ void GlobalCircularBuffer(const Mongo::Data::GpuBatchData<const PBShort2> in,
                                     Memory::DeviceView<CircularBuffer<blockThreads>> circularBuffers,
                                     Mongo::Data::GpuBatchData<PBShort2> out)
{
    assert(blockThreads == blockDim.x);

    const size_t numFrames = in.NumFrames();
    auto& circularBuffer = circularBuffers[blockIdx.x];
    const auto& inZmw  = in.ZmwData(blockIdx.x, threadIdx.x);
    auto outZmw = out.ZmwData(blockIdx.x, threadIdx.x);

    for (int i = 0; i < numFrames; ++i)
    {
        outZmw[i] = ToShort(circularBuffer.Front());
        circularBuffer.PushBack(PBHalf2(inZmw[i]));
    }
}

template <size_t blockThreads, template <size_t> class CircularBuffer>
__global__ void SharedCircularBuffer(const Mongo::Data::GpuBatchData<const PBShort2> in,
                                     Memory::DeviceView<CircularBuffer<blockThreads>> circularBuffers,
                                     Mongo::Data::GpuBatchData<PBShort2> out)
{
    assert(blockThreads == blockDim.x);

    const size_t numFrames = in.NumFrames();
    __shared__ CircularBuffer<blockThreads> circularBuffer;
    circularBuffer = circularBuffers[blockIdx.x];
    const auto& inZmw  = in.ZmwData(blockIdx.x, threadIdx.x);
    auto outZmw = out.ZmwData(blockIdx.x, threadIdx.x);

    for (int i = 0; i < numFrames; ++i)
    {
        outZmw[i] = ToShort(circularBuffer.Front());
        circularBuffer.PushBack(PBHalf2(inZmw[i]));
    }
    circularBuffers[blockIdx.x] = circularBuffer;
}

template <size_t blockThreads, template <size_t> class CircularBuffer, template <size_t> class CircularBufferLocal>
__global__ void LocalCircularBuffer(const Mongo::Data::GpuBatchData<const PBShort2> in,
                                    Memory::DeviceView<CircularBuffer<blockThreads>> circularBuffers,
                                    Mongo::Data::GpuBatchData<PBShort2> out)
{
    assert(blockThreads == blockDim.x);

    const size_t numFrames = in.NumFrames();
    CircularBufferLocal<blockThreads> circularBuffer(circularBuffers[blockIdx.x]);
    const auto& inZmw  = in.ZmwData(blockIdx.x, threadIdx.x);
    auto outZmw = out.ZmwData(blockIdx.x, threadIdx.x);

    for (int i = 0; i < numFrames; ++i)
    {
        outZmw[i] = ToShort(circularBuffer.Front());
        circularBuffer.PushBack(PBHalf2(inZmw[i]));
    }
    circularBuffer.ReplaceShared(circularBuffers[blockIdx.x]);
}

}}  // namespace PacBio::Cuda

#endif // CIRCULAR_BUFFER_CUH