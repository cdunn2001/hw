// Copyright (c) 2021, Pacific Biosciences of California, Inc.
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

#ifndef PACBIO_MONGO_CUDA_LANE_ARRAY_CUH
#define PACBIO_MONGO_CUDA_LANE_ARRAY_CUH

#include <cstddef>

#include <common/cuda/utility/CudaArray.h>

namespace PacBio {
namespace Cuda {

struct SerialConstruct {};
struct ParallelConstruct {};

template <typename T, size_t Len>
class CudaLaneArray
{
    // Noop function in release, but in debug will hopefully help
    // flush out any instances where threading model assumptions
    // are wrong.
    __device__ static void Validate() { assert(blockDim.x == Len); }
public:
    // Trivial default construction is mandatory.  The only places a
    // CudaLaneArray makes sense is in global, shared, or constant memory,
    // which often requires trivial default construction (as that
    // makes more sense than trying to force a specific threading model
    // for initialization)
    CudaLaneArray() = default;

    // Specifically kill of other constructors.  It would be a mistake
    // to create this class as a local variable, so we need to make it
    // hard to do so.;
    CudaLaneArray(const T&) = delete;
    CudaLaneArray(const CudaLaneArray&) = delete;
    CudaLaneArray(CudaLaneArray&&) = delete;

    // We do allow some nontrivial constructors where both the
    // required parallelism is explicit and there is no chance of
    // accidentally creating a CudaLaneArray local variable from a scalar
    // value
    template <typename U>
    __device__ CudaLaneArray(const U& u, SerialConstruct)
    {
        for (auto& d : data_) d = u;
    }
    template <typename U>
    __device__ CudaLaneArray(const U& u, ParallelConstruct)
    {
        Validate();
        data_[threadIdx.x] = u;
    }

    // Assignment makes sense, but we need to make sure we only copy
    // our thread's slot.  We lose trivially_copyable which
    // is unfortunate, but I really don't see any way around that.
    __device__ CudaLaneArray& operator=(const T& t)
    {
        Validate();
        data_[threadIdx.x] = t; return *this;
    }
    __device__ CudaLaneArray& operator=(const CudaLaneArray& a)
    {
        Validate();
        data_[threadIdx.x] = a.data_[threadIdx.x]; return *this;
    }
    __device__ CudaLaneArray& operator=(const Cuda::Utility::CudaArray<T, Len>& a)
    {
        Validate();
        data_[threadIdx.x] = a[threadIdx.x]; return *this;
    }

    // Helper to make it easier to extract from CudaArrays when we don't necessarily
    // have a CudaLaneArray to assign it to.
    template <typename U>
    __device__ static T FromArray(const Cuda::Utility::CudaArray<U, Len>& arr)
    {
        Validate();
        return arr[threadIdx.x];
    }

    // Implicit convsersions to extract out our thread's slot automatically
    // give us all the arithmetic operations we want, including the usual
    // automatic promotions from things like int to float.
    __device__ operator const T&() const
    {
        Validate();
        return data_[threadIdx.x];
    }
    __device__ operator T&()
    {
        Validate();
        return data_[threadIdx.x];
    }
private:
    Cuda::Utility::CudaArray<T, Len> data_;
};

}} // ::PacBio::Simd

#endif //PACBIO_MONGO_CUDA_LANE_ARRAY_CUH
