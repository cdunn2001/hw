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

#ifndef LOCAL_CIRCULAR_BUFFER_H
#define LOCAL_CIRCULAR_BUFFER_H

#include "BlockCircularBuffer.cuh"

#include <common/cuda/PBCudaSimd.cuh>
#include <common/cuda/PBCudaSimd.cuh>

namespace PacBio {
namespace Cuda {

// Implementation of circular buffer where the elements
// are shifted to the front to make room for a newly
// added element. This version is meant to be used in local
// memory.
template <size_t blockThreads, size_t Capacity>
struct LocalCircularBuffer
{
    LocalCircularBuffer() = default;

    __device__ PBHalf2 Front() const
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

    __device__ void Init(const BlockCircularBuffer<blockThreads, Capacity>& cb)
    {
        #pragma unroll(Capacity)
        for (size_t i = 0; i < Capacity; i++)
        {
            data[i] = cb[(i+cb.FrontIdx()) % Capacity];
        }
        __syncthreads();
    }

    __device__ void ReplaceShared(BlockCircularBuffer<blockThreads, Capacity>& cb)
    {
        #pragma unroll(Capacity)
        for (size_t i = 0; i < Capacity; i++)
        {
            cb[i] = data[i];
        }
        cb.FrontIdx(0);
        __syncthreads();
    }

private:
    PBHalf2 data[Capacity];
};

}}  // PacBio::Cuda

#endif // LOCAL_CIRCULAR_BUFFER_H