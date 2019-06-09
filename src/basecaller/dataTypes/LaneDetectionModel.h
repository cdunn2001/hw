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

#ifndef PACBIO_MONGO_DATA_ANALOG_MODEL_H_
#define PACBIO_MONGO_DATA_ANALOG_MODEL_H_


#include <common/cuda/PBCudaSimd.h>
#include <common/cuda/utility/CudaArray.h>
#include <common/cuda/CudaFunctionDecorators.h>

#include <common/cuda/memory/UnifiedCudaArray.h>

namespace PacBio {
namespace Mongo {
namespace Data {

// Analog information for an entire lane of zmw.
template <typename T, int laneWidth>
struct __align__(128) LaneAnalogMode
{
    static_assert(std::is_same<T, Cuda::PBHalf>::value ||
                  std::is_same<T, Cuda::PBHalf2>::value,
                  "Invalid type for LaneAnalogMode");

    CUDA_ENABLED LaneAnalogMode& operator=(const LaneAnalogMode& other)
    {
#ifdef __CUDA_ARCH__
        assert(blockDim.x == laneWidth);
        means[threadIdx.x] = other.means[threadIdx.x];
        vars[threadIdx.x] = other.vars[threadIdx.x];
        return  *this;
#else
        for (int i = 0; i < laneWidth; ++i)
        {
            means[i] = other.means[i];
            vars[i] = other.vars[i];
        }
        return *this;
#endif
    }

    LaneAnalogMode& SetAllMeans(float val)
    {
        std::fill(means.data(), means.data() + laneWidth, T(val));
        return *this;
    }
    LaneAnalogMode& SetAllVars(float val)
    {
        std::fill(vars.data(), vars.data() + laneWidth, T(val));
        return *this;
    }
    using Row = Cuda::Utility::CudaArray<T, laneWidth>;
    Row means;
    Row vars;
};

// Full model for baseline + analogs for a lane of zmw.
template <typename T, size_t laneWidth>
struct __align__(128) LaneModelParameters
{
    static constexpr int numAnalogs = 4;

    __host__ __device__ LaneModelParameters& operator=(const LaneModelParameters& other)
    {
        baseline_ = other.baseline_;
        for (int i = 0; i < numAnalogs; ++i)
        {
            analogs_[i] = other.analogs_[i];
        }
        return *this;
    }

    __host__ __device__ const LaneAnalogMode<T, laneWidth>& BaselineMode() const
    {
        return baseline_;
    }
    __host__ __device__ LaneAnalogMode<T, laneWidth>& BaselineMode()
    {
        return baseline_;
    }

    __host__ __device__ const LaneAnalogMode<T, laneWidth>& AnalogMode(unsigned i) const
    {
        return analogs_[i];
    }
    __host__ __device__ LaneAnalogMode<T, laneWidth>& AnalogMode(unsigned i)
    {
        return analogs_[i];
    }

 private:
    Cuda::Utility::CudaArray<LaneAnalogMode<T, laneWidth>, numAnalogs> analogs_;
    LaneAnalogMode<T, laneWidth> baseline_;
};

static_assert(sizeof(LaneModelParameters<Cuda::PBHalf, 64>) == 128*10, "Unexpected size");
static_assert(sizeof(LaneModelParameters<Cuda::PBHalf2, 32>) == 128*10, "Unexpected size");

}}} // ::PacBio::Mongo::Data

namespace PacBio {
namespace Cuda {
namespace Memory {

//  Allow conversion between PBHalf and PBHalf2
template <size_t laneWidth>
struct gpu_type<Mongo::Data::LaneModelParameters<PBHalf, laneWidth>>
{
    static_assert(laneWidth % 2 == 0, "Invalid lane width");
    using type = Mongo::Data::LaneModelParameters<PBHalf2, laneWidth/2>;
};

}}}

#endif // PACBIO_MONGO_DATA_ANALOG_MODEL_H_
