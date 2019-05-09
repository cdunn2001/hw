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

template <typename T, size_t laneWidth>
struct ViterbiDataHost
{
    static constexpr int numStates = 13;
    ViterbiDataHost(size_t numFrames, T val = T{})
        : data_(numFrames*laneWidth*numStates, val)
    {}

    Memory::DeviceView<T> Data(Memory::detail::DataManagerKey)
    {
        return data_.GetDeviceView();
    }
 private:
    Memory::DeviceOnlyArray<T> data_;
};

template <typename T, size_t laneWidth>
struct ViterbiData : private Memory::detail::DataManager
{
    static constexpr int numStates = 13;
    ViterbiData(ViterbiDataHost<T, laneWidth>& hostData)
        : data_(hostData.Data(DataKey()))
    {}

    __device__ T& operator()(int frame, int state)
    {
        return data_[frame*laneWidth*numStates +  state*laneWidth +  threadIdx.x];
    }
 private:
    Memory::DeviceView<T> data_;
};

// First arg should be const?
__global__ void FrameLabelerKernel(Memory::DevicePtr<Subframe::TransitionMatrix> trans,
                                   Memory::DeviceView<LaneModelParameters<32>> models,
                                   ViterbiData<PBHalf2, 32> recur,
                                   ViterbiData<short2, 32> labels,
                                   Mongo::Data::GpuBatchData<short2> input,
                                   Mongo::Data::GpuBatchData<short2> output);

}}

#endif //PACBIO_CUDA_FRAME_LABELER_KERNELS_CUH_
