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

#ifndef mongo_dataTypes_LaneDetectionModel_H_
#define mongo_dataTypes_LaneDetectionModel_H_


#include <common/cuda/PBCudaSimd.h>
#include <common/cuda/utility/CudaArray.h>
#include <common/cuda/CudaFunctionDecorators.h>
#include <common/cuda/memory/UnifiedCudaArray.h>

#include <common/MongoConstants.h>

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

#ifdef __CUDA_ARCH__
    __device__ LaneAnalogMode& ParallelAssign(const LaneAnalogMode& other)
    {
        assert(blockDim.x == laneWidth);
        means[threadIdx.x] = other.means[threadIdx.x];
        vars[threadIdx.x] = other.vars[threadIdx.x];
        return  *this;
    }
#endif

    LaneAnalogMode& operator=(const LaneAnalogMode& other) = default;

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

#ifdef __CUDA_ARCH__
    __device__ LaneModelParameters& ParallelAssign(const LaneModelParameters& other)
    {
        baseline_.ParallelAssign(other.baseline_);
        for (int i = 0; i < numAnalogs; ++i)
        {
            analogs_[i].ParallelAssign(other.analogs_[i]);
        }
        baselineWeight_[threadIdx.x] = other.baselineWeight_[threadIdx.x];
        return *this;
    }
#endif

    LaneModelParameters& operator=(const LaneModelParameters& other) = default;

    CUDA_ENABLED const LaneAnalogMode<T, laneWidth>& BaselineMode() const
    {
        return baseline_;
    }
    CUDA_ENABLED LaneAnalogMode<T, laneWidth>& BaselineMode()
    {
        return baseline_;
    }

    CUDA_ENABLED const Cuda::Utility::CudaArray<T, laneWidth>& BaselineWeight() const
    {
        return baselineWeight_;
    }

    CUDA_ENABLED Cuda::Utility::CudaArray<T, laneWidth>& BaselineWeight()
    {
        return baselineWeight_;
    }

    CUDA_ENABLED const LaneAnalogMode<T, laneWidth>& AnalogMode(unsigned i) const
    {
        return analogs_[i];
    }
    CUDA_ENABLED LaneAnalogMode<T, laneWidth>& AnalogMode(unsigned i)
    {
        return analogs_[i];
    }
    void SetAllBaselineWeight(float val)
    {
        baselineWeight_ = T(val);
    }

 private:
    Cuda::Utility::CudaArray<LaneAnalogMode<T, laneWidth>, numAnalogs> analogs_;
    Cuda::Utility::CudaArray<T, laneWidth> baselineWeight_;
    LaneAnalogMode<T, laneWidth> baseline_;
};

static_assert(sizeof(LaneModelParameters<Cuda::PBHalf, 64>) == 128*(10+1), "Unexpected size");
static_assert(sizeof(LaneModelParameters<Cuda::PBHalf2, 32>) == 128*(10+1), "Unexpected size");

/// A bundle of model parameters for a normal mixture representing the
/// baselined trace data for a lane of ZMWs.
/// \tparam T is the elemental data type (e.g., float).
template <typename T>
using LaneDetectionModel = LaneModelParameters<T, laneSize>;

}}}     // namespace PacBio::Mongo::Data


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

}}}     // namespace PacBio::Cuda::Memory

#endif  // mongo_dataTypes_LaneDetectionModel_H_
