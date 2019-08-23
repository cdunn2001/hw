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

#include <pacbio/ipc/ThreadSafeQueue.h>

#include <common/cuda/PBCudaSimd.cuh>
#include <common/cuda/streams/KernelLaunchInfo.h>
#include <common/cuda/memory/DeviceOnlyArray.cuh>
#include <common/cuda/memory/DeviceOnlyObject.cuh>
#include <common/MongoConstants.h>

#include <dataTypes/BatchData.cuh>
#include <dataTypes/BatchMetrics.h>

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
    {
        auto SetAll = [](auto& arr, const auto& val) {
            for (int i = 0; i < laneWidth; ++i)
                arr[i] = val;
        };

        SetAll(boundary_, PBShort2(0));

        // I'm a little uncertain how to initialize this model.  I'm tryin to
        // make it work with latent data that we are guaranteed to be all zeroes,
        // and needs to produce baseline labels.
        //
        // I'm not even sure we need this `oldModel` it may be sufficient to use
        // the current block's model while processing the few frames of latent data
        //
        // In either case, any issues here are flushed out after only a few frames
        // (16) and we hit steady state, so I'm not hugely concerned.
        SetAll(oldModel.BaselineMode().means, 0);
        SetAll(oldModel.BaselineMode().vars, 50);

        for (int a = 0; a < numAnalogs; ++a)
        {
            SetAll(oldModel.AnalogMode(a).means, 1000);
            SetAll(oldModel.AnalogMode(a).vars, 10);
        }
    }

    __device__ void SetBoundary(PBShort2 boundary) { boundary_[threadIdx.x] = boundary; }
    __device__ PBShort2 GetBoundary() const { return boundary_[threadIdx.x]; }

    __device__ const LaneModelParameters& GetModel() const { return oldModel; }
    __device__ void SetModel(const LaneModelParameters& model)
    {
        oldModel.ParallelAssign(model);
    }

private:
    LaneModelParameters oldModel;
    Utility::CudaArray<PBShort2, laneWidth> boundary_;
};

// TODO this needs cleanup.  If numLanes doesn't match what is actually used on the gpu, we're dead
template <size_t laneWidth>
struct ViterbiDataHost
{
    using T = uint32_t;
    ViterbiDataHost(size_t numFrames, size_t numLanes)
        : data_(SOURCE_MARKER(), numFrames*numLanes*laneWidth*4, 0)
        , numFrames_(numFrames)
    {}

    Memory::DeviceView<T> Data(const KernelLaunchInfo& info)
    {
        return data_.GetDeviceView(info);
    }
    int NumFrames() const { return numFrames_; }
 private:
    Memory::DeviceOnlyArray<T> data_;
    int numFrames_;
};

// TODO I don't like this API...  Should have a separate ViterbiData object, either
// per thread or per block.
template <size_t laneWidth>
struct ViterbiData : private Memory::detail::DataManager
{
    using T = uint32_t;
    ViterbiData(ViterbiDataHost<laneWidth>& hostData, const KernelLaunchInfo& info)
        : data_(hostData.Data(info))
        , numFrames_(hostData.NumFrames())
    {}

    __device__ T& operator()(int frame, int state)
    {
        return data_[numFrames_*laneWidth*4*blockIdx.x
                     + frame*laneWidth*4
                     + state*laneWidth +  threadIdx.x];
    }
 private:
    Memory::DeviceView<T> data_;
    int numFrames_;
};

// Define overloads for this function, so that we can track kernel invocations, and
// so that we can be converted to our gpu specific representation
template <size_t laneWidth>
ViterbiData<laneWidth> KernelArgConvert(ViterbiDataHost<laneWidth>& v, const KernelLaunchInfo& info) { return ViterbiData<laneWidth>(v, info); }



class FrameLabeler
{
    static constexpr size_t BlockThreads = laneSize/2;
public:

    // Helpers to provide scratch space data.  Used to pool allocations so we
    // only need enough to satisfy the current active batches, not one for
    // each possible pool.
    static void Configure(const std::array<Subframe::AnalogMeta, 4>& meta,
                          int32_t lanesPerPool, int32_t framesPerChunk);
private:
    static std::unique_ptr<ViterbiDataHost<BlockThreads>> BorrowScratch();
    static void ReturnScratch(std::unique_ptr<ViterbiDataHost<BlockThreads>> data);

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


    void ProcessBatch(const Memory::UnifiedCudaArray<Mongo::Data::LaneModelParameters<PBHalf, laneSize>>& models,
                      const Mongo::Data::BatchData<int16_t>& input,
                      Mongo::Data::BatchData<int16_t>& latOut,
                      Mongo::Data::BatchData<int16_t>& output,
                      Mongo::Data::FrameLabelerMetrics& metricsOutput);
private:
    Memory::DeviceOnlyArray<LatentViterbi<BlockThreads>> latent_;
    Mongo::Data::BatchData<int16_t> prevLat_;

    static int32_t lanesPerPool_;
    static int32_t framesPerChunk_;
    static std::unique_ptr<Memory::DeviceOnlyObj<const Subframe::TransitionMatrix>> trans_;
    static ThreadSafeQueue<std::unique_ptr<ViterbiDataHost<BlockThreads>>> scratchData_;
};

}}

#endif //PACBIO_CUDA_FRAME_LABELER_KERNELS_CUH_
