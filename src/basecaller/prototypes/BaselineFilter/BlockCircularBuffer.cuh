// Copyright (c) 2019, Pacific Biosciences of California, Inc.
//
// All rights reserved.
//
// THIS SOFTWARE CONSTITUTES AND EMBODIES PACIFIC BIOSCIENCES' CONFIDENTIAL
// AND PROPRIETARY INFORMATION.
//
// Disclosure, redistribution and use of this software is subject to the
// terms and conditions of the applicable written agreement(s) between you
// and Pacific Biosciences, where "you" refers to you or your company or
// organization, as applicable.  Any other disclosure, redistribution or
// use is prohibited.
//
// THIS SOFTWARE IS PROVIDED BY PACIFIC BIOSCIENCES AND ITS CONTRIBUTORS "AS
// IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
// THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL PACIFIC BIOSCIENCES OR ITS
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
// OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
// WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
// OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
// ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#ifndef BLOCK_CIRCULAR_BUFFER_CUH
#define BLOCK_CIRCULAR_BUFFER_CUH

#include <common/cuda/PBCudaSimd.cuh>

namespace PacBio {
namespace Cuda {

// Implementation of circular buffer that only utilizes a
// head pointer as elements are only added to the buffer
// and popped from the front. This version is meant to
// stored in either global or shared memory.
template<size_t blockThreads, size_t Capacity>
struct __align__(128) BlockCircularBuffer
{
    // This default constructor does leave the object in an invalid state
    // since front will not be initialized but is necessary since
    // a default constructor is needed for any variable stored using the
    // __shared__ qualifier. Users should call the ParallelAssign()
    // below which actually initializes the object before first use.
    BlockCircularBuffer() = default;

    __device__ BlockCircularBuffer(PBHalf2 dummyVal)
    {
        // We treat the circular buffer as always being full so we initialize
        // it with the dummy value.
        for (size_t i = 0; i < blockThreads; ++i)
        {
            front[i] = 0;
        }
        for (size_t i = 0; i < Capacity; ++i)
        {
            for (size_t j = 0; j < blockThreads; ++j)
            {
                data[i][j] = dummyVal;
            }
        }
    }

    __device__ PBHalf2 Front() const
    {
        return data[front[threadIdx.x]][threadIdx.x];
    }

    __device__ void PushBack(PBHalf2 val)
    {
        data[front[threadIdx.x]][threadIdx.x] = val;
        front[threadIdx.x] = (front[threadIdx.x] + 1) % Capacity;
    }

    __device__ BlockCircularBuffer& ParallelAssign(const BlockCircularBuffer& cb)
    {
        for (size_t i = 0; i < Capacity; ++i)
        {
            data[i][threadIdx.x] = cb.data[i][threadIdx.x];
        }
        front[threadIdx.x] = cb.front[threadIdx.x];
        __syncthreads();
        return *this;
    }

    __device__ PBHalf2& operator[](size_t idx)
    {
        return data[idx][threadIdx.x];
    }

    __device__ const PBHalf2& operator[](size_t idx) const
    {
        return data[idx][threadIdx.x];
    }

    __device__ short FrontIdx() const
    {
        return front[threadIdx.x];
    }

    __device__ BlockCircularBuffer& FrontIdx(short val)
    {
        front[threadIdx.x] = val;
        return *this;
    }

private:
    using Row = PBHalf2[blockThreads];
    Row data[Capacity];
    short front[blockThreads];
};

}}  // PacBio::Cuda

#endif  // BLOCK_CIRCULAR_BUFFER_CUH
