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

#include <array>

#include <common/cuda/PBCudaSimd.h>
#include <common/cuda/utility/CudaArray.h>
#include <common/cuda/CudaFunctionDecorators.h>
#include <common/cuda/memory/UnifiedCudaArray.h>

#include <common/IntInterval.h>
#include <common/MongoConstants.h>
#include <dataTypes/BasicTypes.h>

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
        weights[threadIdx.x] = other.weights[threadIdx.x];
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
    LaneAnalogMode& SetAllWeights(float val)
    {
        std::fill(weights.data(), weights.data() + laneWidth, T(val));
        return *this;
    }
    using Row = Cuda::Utility::CudaArray<T, laneWidth>;
    Row means;
    Row vars;
    Row weights;
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
        confidence_ = other.confidence_;  // TODO: Should this be parallelized?
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
    CUDA_ENABLED const LaneAnalogMode<T, laneWidth>& AnalogMode(unsigned i) const
    {
        return analogs_[i];
    }
    CUDA_ENABLED LaneAnalogMode<T, laneWidth>& AnalogMode(unsigned i)
    {
        return analogs_[i];
    }

    CUDA_ENABLED const Cuda::Utility::CudaArray<T, laneWidth>& Confidence() const
    {
        return confidence_;
    }

    CUDA_ENABLED Cuda::Utility::CudaArray<T, laneWidth>& Confidence()
    {
        return confidence_;
    }

private:
    Cuda::Utility::CudaArray<LaneAnalogMode<T, laneWidth>, numAnalogs> analogs_;
    LaneAnalogMode<T, laneWidth> baseline_;
    Cuda::Utility::CudaArray<T, laneWidth> confidence_;
};

static_assert(sizeof(LaneModelParameters<Cuda::PBHalf, 64>) == 128*(5*3 + 1), "Unexpected size");
static_assert(sizeof(LaneModelParameters<Cuda::PBHalf2, 32>) == 128*(5*3 + 1), "Unexpected size");

/// A bundle of model parameters for a normal mixture representing the
/// baselined trace data for a lane of ZMWs.
/// \tparam T is the elemental data type (e.g., float).
template <typename T>
using LaneDetectionModel = LaneModelParameters<T, laneSize>;


template <typename T>
struct IntervalData
{
    IntervalData(size_t count,
               Cuda::Memory::SyncDirection syncDir,
               const Cuda::Memory::AllocationMarker& allocMarker)
        : data {count, syncDir, allocMarker}
        , frameInterval {}
    { }

    Cuda::Memory::UnifiedCudaArray<T> data;
    IntInterval<FrameIndexType> frameInterval;
};

template <typename T>
using DetectionModelPool = IntervalData<LaneDetectionModel<T>>;


/// Creates a mostly arbitrary instance of LaneDetectionModel<T>.
/// Primarily useful for unit tests.
template <typename T>
LaneDetectionModel<T> MockLaneDetectionModel()
{
    LaneDetectionModel<T> ldm;

    {
        auto& bm = ldm.BaselineMode();
        bm.SetAllWeights(0.40f);
        bm.SetAllMeans(0.0f);
        bm.SetAllVars(1.0f);
    }

    using AnalogArray = std::array<float, LaneDetectionModel<T>::numAnalogs>;
    const AnalogArray mean {4.0f, 6.0f, 9.0f, 13.0f};
    const AnalogArray var {1.1f, 1.2f, 1.3f, 1.4f};
    for (unsigned int i = 0; i < ldm.numAnalogs; ++i)
    {
        auto& am = ldm.AnalogMode(i);
        am.SetAllWeights(0.15f);
        am.SetAllMeans(mean.at(i));
        am.SetAllVars(var.at(i));
    }

    ldm.Confidence() = 0.0f;

    return ldm;
}

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
