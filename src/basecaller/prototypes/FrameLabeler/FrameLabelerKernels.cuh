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

// This class represents compressed frame labels, where
// each label only takes up four bits.  It works with
// paired 'x' and 'y' values to correspond with the
// ushort2 type used to produce/consume non-packed labels.
//
// The class has a very 'circular buffer'-esque interface,
// as the bit operations for always dealing with the front
// and back proved cheaper than trying to insert/extract
// items in the middle.
class PackedLabels
{
public:
    static constexpr int bitsPerValue = 4;
    static constexpr int numValues = 32 / bitsPerValue;
    static constexpr int numPairs = numValues/2;

    // Push back an x,y pair into the back two slots.
    // The first two slots will roll off
    __device__ void PushBack(const ushort2& val)
    {
        assert(val.x < (1 << bitsPerValue));
        assert(val.y < (1 << bitsPerValue));

        data_  = data_ >> 2*bitsPerValue;
        data_ |= (val.x << ((numValues - 2) * bitsPerValue));
        data_ |= (val.y << ((numValues - 1) * bitsPerValue));
    }

    // Push back the number of zero pairs specified by count
    template <int count>
    __device__ void PushBackZeroes()
    {
        data_ = data_ >> (2*bitsPerValue*count);
    }

    // extracts the front x,y pair, effectively pushes
    // zeroes into the back two slots.
    __device__ ushort2 PopFront()
    {
        auto ret = make_ushort2(data_ & 0xF, (data_ >> bitsPerValue) & 0xF);
        data_ = data_ >> 2*bitsPerValue;
        return ret;
    }

    // Access an x value of an x,y pair by index
    __device__ short XAt(ushort idx) const
    {
        return (data_ >> (2*bitsPerValue * idx)) & 0xF;
    }
    // Access a y value of an x,y pair by index
    __device__ short YAt(ushort idx) const
    {
        return (data_ >> (2*bitsPerValue * idx + bitsPerValue)) & 0xF;
    }
private:
    uint32_t data_;
};

template <size_t laneWidth>
struct ViterbiDataHost
{
    using T = PackedLabels;
    static constexpr int numPackedLabels = (Subframe::numStates + PackedLabels::numPairs - 1) / PackedLabels::numPairs;
    ViterbiDataHost(size_t numFrames, size_t numLanes)
        : data_(SOURCE_MARKER(), numFrames * numLanes * laneWidth * numPackedLabels)
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

template <size_t laneWidth>
struct ViterbiData : private Memory::detail::DataManager
{
    using T = PackedLabels;
    static constexpr int numPackedLabels = ViterbiDataHost<laneWidth>::numPackedLabels;
    ViterbiData(ViterbiDataHost<laneWidth>& hostData, const KernelLaunchInfo& info)
        : data_(hostData.Data(info))
        , numFrames_(hostData.NumFrames())
    {}

    class ViterbiBlockData
    {
    public:
        __device__ ViterbiBlockData(T* data)
            : data_(data)
        {}

        __device__ T& operator()(int frame, int state)
        {
            return data_[frame * laneWidth * numPackedLabels + state * laneWidth];
        }
    private:
        T* data_;
    };
    __device__ ViterbiBlockData BlockData()
    {
        return ViterbiBlockData(&data_[numFrames_ * laneWidth * numPackedLabels * blockIdx.x
                                       + threadIdx.x]);
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
    static ThreadSafeQueue<std::unique_ptr<ViterbiDataHost<BlockThreads>>> scratchData_;
};

}}

#endif //PACBIO_CUDA_FRAME_LABELER_KERNELS_CUH_
