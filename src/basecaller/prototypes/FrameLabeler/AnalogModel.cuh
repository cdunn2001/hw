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

#ifndef PACBIO_CUDA_MODEL_CUH_
#define PACBIO_CUDA_MODEL_CUH_

#include <common/cuda/PBCudaSimd.cuh>
#include <common/cuda/utility/CudaArray.cuh>

namespace PacBio {
namespace Cuda {

// Analog information for an entire lane of zmw.
template <size_t laneWidth>
struct __align__(128) LaneAnalogMode
{
    __host__ __device__ LaneAnalogMode& operator=(const LaneAnalogMode other)
    {
#ifdef __CUDA_ARCH__
        means[threadIdx.x] = other.means[threadIdx.x];
        vars[threadIdx.x] = other.vars[threadIdx.x];
        return  *this;
#else
        for (int i = 0; i < laneWidth; ++i)
        {
            means[i] = other.means[i];
            vars[i] = other.vars[i];
        }
#endif
    }

    __host__ LaneAnalogMode& SetAllMeans(float val)
    {
        std::fill(means.data(), means.data() + laneWidth, PBHalf2(val));
        return *this;
    }
    __host__ LaneAnalogMode& SetAllVars(float val)
    {
        std::fill(vars.data(), vars.data() + laneWidth, PBHalf2(val));
        return *this;
    }
    using Row = Utility::CudaArray<PBHalf2, laneWidth>;
    Row means;
    Row vars;
};

// Full model for baseline + analogs for a lane of zmw.
template <size_t laneWidth>
struct __align__(128) LaneModelParameters
{
    static constexpr unsigned int numAnalogs = 4;

    __host__ __device__ LaneModelParameters& operator=(const LaneModelParameters& other)
    {
        baseline_ = other.baseline_;
        for (int i = 0; i < numAnalogs; ++i)
        {
            analogs_[i] = other.analogs_[i];
        }
        return *this;
    }

    __host__ __device__ const LaneAnalogMode<laneWidth>& BaselineMode() const
    {
        return baseline_;
    }
    __host__ __device__ LaneAnalogMode<laneWidth>& BaselineMode()
    {
        return baseline_;
    }

    __host__ __device__ const LaneAnalogMode<laneWidth>& AnalogMode(unsigned i) const
    {
        return analogs_[i];
    }
    __host__ __device__ LaneAnalogMode<laneWidth>& AnalogMode(unsigned i)
    {
        return analogs_[i];
    }

 private:
    Utility::CudaArray<LaneAnalogMode<laneWidth>, numAnalogs> analogs_;
    LaneAnalogMode<laneWidth> baseline_;
};

}}

#endif //PACBIO_CUDA_MODEL_CUH_
