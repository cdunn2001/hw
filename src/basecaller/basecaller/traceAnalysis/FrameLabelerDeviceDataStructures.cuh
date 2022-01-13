// Copyright (c) 2022, Pacific Biosciences of California, Inc.
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

#ifndef PACBIO_MONGO_BASECALLER_FRAME_DATA_STRUCTURES_H
#define PACBIO_MONGO_BASECALLER_FRAME_DATA_STRUCTURES_H

#include <common/cuda/CudaLaneArray.cuh>
#include <common/cuda/memory/DeviceOnlyArray.cuh>
#include <common/cuda/PBCudaSimd.cuh>

#include <dataTypes/BatchData.cuh>
#include <dataTypes/LaneDetectionModel.h>

#include <basecaller/traceAnalysis/SubframeScorer.h>

namespace PacBio::Mongo::Basecaller {

using namespace Cuda;

// Persistent data we need to hold on to, between consecutive runs of the
// FrameLabeler
template <size_t laneWidth, int32_t latentFrames>
struct __align__(128) LatentViterbi
{
    using LaneModelParameters = Mongo::Data::LaneModelParameters<PBHalf2, laneWidth>;
    using LaneArr = CudaLaneArray<PBShort2, laneWidth>;
 public:
    __device__ LatentViterbi()
        : labelsBoundary_(0, SerialConstruct{})
        , roiBoundary_(0, SerialConstruct{})
    {
        auto SetAll = [](auto& arr, const auto& val) {
            for (int i = 0; i < laneWidth; ++i)
                arr[i] = val;
        };

        for (auto& val : latentTrc_) val.SerialAssign(0);

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

    __device__ void SetLabelsBoundary(PBShort2 boundary) { labelsBoundary_ = boundary; }
    __device__ PBShort2 GetLabelsBoundary() const { return labelsBoundary_; }

    __device__ void SetRoiBoundary(PBShort2 boundary) { roiBoundary_ = boundary; }
    __device__ PBShort2 GetRoiBoundary() const { return roiBoundary_; }

    __device__ const LaneModelParameters& GetModel() const { return oldModel; }
    __device__ void SetModel(const LaneModelParameters& model)
    {
        oldModel.ParallelAssign(model);
    }

    __device__ const Utility::CudaArray<LaneArr, latentFrames>& GetLatentTraces() const
    {
        return latentTrc_;
    }
    // Accepts an iterator to the last frame emitted during an analysis.
    // This iterator will be *decremented* to extract/store as much latent
    // data as is necessary to enable the roi computation of the next block
    __device__ void SetLatentTraces(Data::StrideIterator<const PBShort2> trc)
    {
        for (int idx = latentFrames - 1; idx >= 0; --idx)
        {
            latentTrc_[idx] = *trc;
            --trc;
        }
    }

private:

    // Model for the previous block of data
    LaneModelParameters oldModel;
    // the label for the last frame emitted in the
    // previous block
    LaneArr labelsBoundary_;
    // the ROI determination for the last frame
    // emitted in the previous block
    LaneArr roiBoundary_;
    // Latent traces, which already had labels emitted
    // in the previous block, but are still needed
    // for the roi computation of the current block
    Utility::CudaArray<LaneArr, latentFrames> latentTrc_;
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
    __device__ PBShort2 PopFront()
    {
        auto ret = PBShort2(data_ & 0xF, (data_ >> bitsPerValue) & 0xF);
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

// Host handle for the workspace required for the Viterbi algorithm
template <size_t laneWidth>
struct ViterbiDataHost
{
    using T = PackedLabels;
    static constexpr int numPackedLabels = (Subframe::numStates + PackedLabels::numPairs - 1) / PackedLabels::numPairs;
    ViterbiDataHost(size_t numFrames, size_t numLanes)
        : data_(SOURCE_MARKER(), numFrames * numLanes * laneWidth * numPackedLabels)
        , numFrames_(numFrames)
    {}

    Cuda::Memory::DeviceView<T> Data(const KernelLaunchInfo& info)
    {
        return data_.GetDeviceView(info);
    }
    int NumFrames() const { return numFrames_; }
 private:
    Cuda::Memory::DeviceOnlyArray<T> data_;
    int numFrames_;
};

// Workspace for the viterbi algorith, to hold the traceback information
template <size_t laneWidth>
struct ViterbiData : private Cuda::Memory::detail::DataManager
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
    Cuda::Memory::DeviceView<T> data_;
    int numFrames_;
};

// Define overloads for this function, so that we can track kernel invocations, and
// so that we can be converted to our gpu specific representation
template <size_t laneWidth>
ViterbiData<laneWidth> KernelArgConvert(ViterbiDataHost<laneWidth>& v, const KernelLaunchInfo& info) { return ViterbiData<laneWidth>(v, info); }

}

#endif /*PACBIO_MONGO_BASECALLER_FRAME_DATA_STRUCTURES_H*/
