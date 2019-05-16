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

#ifndef PACBIO_CUDA_FRAME_LABELER_KERNELS_CUH_
#define PACBIO_CUDA_FRAME_LABELER_KERNELS_CUH_

#include <common/cuda/PBCudaSimd.cuh>
#include <common/cuda/memory/DeviceOnlyArray.cuh>
#include <common/cuda/memory/DeviceOnlyObject.cuh>

#include <dataTypes/TraceBatch.cuh>

#include "SubframeScorer.cuh"

namespace PacBio {
namespace Cuda {

// TODO move?
namespace Viterbi {
static constexpr unsigned int lookbackDist = 16;
}

template <size_t laneWidth>
struct __align__(128) LatentViterbi
{
 public:
    __device__ LatentViterbi()
        : boundary_(make_short2(0,0))
        , numFrames_(0)
    {}

    __device__ void SetBoundary(short2 boundary) { boundary_ = boundary; }
    __device__ short2 GetBoundary() const { return boundary_; }

    __device__ const LaneModelParameters<laneWidth>& GetModel() const { return oldModel; }
    __device__ void SetModel(const LaneModelParameters<laneWidth>& model)
    {
        oldModel = model;
    }

    __device__ void SetData(Mongo::Data::StridedBlockView<short2>& block)
    {
        numFrames_ = Viterbi::lookbackDist;
        auto start = block.size() - Viterbi::lookbackDist;
        for (int i = 0; i < Viterbi::lookbackDist; ++i)
        {
            oldData_[i*laneWidth + threadIdx.x] = block[start+i];
        }
    }

    __device__ short2& FrameData(int idx) { return oldData_[idx*laneWidth + threadIdx.x]; }

    __device__ int NumFrames() const { return numFrames_; }

private:
    LaneModelParameters<laneWidth> oldModel;
    short2 oldData_[laneWidth * Viterbi::lookbackDist];
    short2 boundary_;
    int numFrames_;
};

// TODO this needs cleanup.  If numLanes doesn't match what is actually used on the gpu, we're dead
template <typename T, size_t laneWidth>
struct ViterbiDataHost
{
    ViterbiDataHost(size_t numFrames, size_t numLanes, T val = T{})
        : data_(numFrames*numLanes*laneWidth*Subframe::numStates, val)
        , numFrames_(numFrames)
    {}

    Memory::DeviceView<T> Data(Memory::detail::DataManagerKey)
    {
        return data_.GetDeviceView();
    }
    int NumFrames() const { return numFrames_; }
 private:
    Memory::DeviceOnlyArray<T> data_;
    int numFrames_;
};

// TODO I don't like this API...  Should have a separate ViterbiData object, either
// per thread or per block.
template <typename T, size_t laneWidth>
struct ViterbiData : private Memory::detail::DataManager
{
    ViterbiData(ViterbiDataHost<T, laneWidth>& hostData)
        : data_(hostData.Data(DataKey()))
        , numFrames_(hostData.NumFrames())
    {}

    __device__ T& operator()(int frame, int state)
    {
        return data_[numFrames_*laneWidth*Subframe::numStates*blockIdx.x
                     + frame*laneWidth*Subframe::numStates
                     + state*laneWidth +  threadIdx.x];
    }
 private:
    Memory::DeviceView<T> data_;
    int numFrames_;
};

// First arg should be const?
__global__ void FrameLabelerKernel(Memory::DevicePtr<Subframe::TransitionMatrix> trans,
                                   Memory::DeviceView<LaneModelParameters<32>> models,
                                   Memory::DeviceView<LatentViterbi<32>> latent,
                                   ViterbiData<short2, 32> labels,
                                   Mongo::Data::GpuBatchData<short2> input,
                                   Mongo::Data::GpuBatchData<short2> output);

}}

#endif //PACBIO_CUDA_FRAME_LABELER_KERNELS_CUH_
