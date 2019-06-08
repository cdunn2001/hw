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
#include <common/MongoConstants.h>

#include <dataTypes/TraceBatch.cuh>

#include "SubframeScorer.cuh"

using namespace PacBio::Mongo;

namespace PacBio {
namespace Cuda {

template <size_t laneWidth>
struct __align__(128) LatentViterbi
{
    using LaneModelParameters = Mongo::Data::LaneModelParameters<PBHalf2, laneWidth>;
 public:
    __device__ LatentViterbi()
        : boundary_(make_short2(0,0))
        , numFrames_(0)
    {}

    __device__ void SetBoundary(short2 boundary) { boundary_ = boundary; }
    __device__ short2 GetBoundary() const { return boundary_; }

    __device__ const LaneModelParameters& GetModel() const { return oldModel; }
    __device__ void SetModel(const LaneModelParameters& model)
    {
        oldModel = model;
    }

    __device__ void SetData(const Mongo::Data::StridedBlockView<const short2>& block)
    {
        numFrames_ = ViterbiStitchLookback;
        auto start = block.size() - ViterbiStitchLookback;
        for (int i = 0; i < ViterbiStitchLookback; ++i)
        {
            oldData_[i*laneWidth + threadIdx.x] = block[start+i];
        }
    }

    __device__ short2& FrameData(int idx) { return oldData_[idx*laneWidth + threadIdx.x]; }

    __device__ int NumFrames() const { return numFrames_; }

private:
    LaneModelParameters oldModel;
    short2 oldData_[laneWidth * ViterbiStitchLookback];
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

class FrameLabeler
{
    static constexpr size_t BlockThreads = 32;
public:

    // Helpers to provide scratch space data.  Used to pool allocations so we
    // only need enough to satisfy the current active batches, not one for
    // each possible pool.
    static void Configure(const std::array<Subframe::AnalogMeta, 4>& meta,
                          int32_t lanesPerPool, int32_t framesPerChunk);
private:
    static std::unique_ptr<ViterbiDataHost<short2, BlockThreads>> BorrowScratch();
    static void ReturnScratch(std::unique_ptr<ViterbiDataHost<short2, BlockThreads>> data);

public:
    // This is necessary to call, if we wait until the C++ runtime is tearing down, the static scratch data
    // may be freed after the cuda runtime is torn down, which causes problems
    static void Finalize();

public:
    FrameLabeler();

    FrameLabeler(const FrameLabeler&) = delete;
    FrameLabeler(FrameLabeler&&) = default;
    FrameLabeler& operator=(const FrameLabeler&) = delete;
    FrameLabeler& operator=(FrameLabeler&&) = default;


    void ProcessBatch(const Memory::UnifiedCudaArray<Mongo::Data::LaneModelParameters<PBHalf, 64>>& models,
                      const Mongo::Data::TraceBatch<int16_t>& input,
                      Mongo::Data::TraceBatch<int16_t>& output);
private:
    Memory::DeviceOnlyArray<LatentViterbi<BlockThreads>> latent_;

    static int32_t lanesPerPool_;
    static int32_t framesPerChunk_;
    static std::unique_ptr<Memory::DeviceOnlyObj<Subframe::TransitionMatrix>> trans_;
    static ThreadSafeQueue<std::unique_ptr<ViterbiDataHost<short2, BlockThreads>>> scratchData_;
};

}}

#endif //PACBIO_CUDA_FRAME_LABELER_KERNELS_CUH_
