//  Copyright (c) 2019, Pacific Biosciences of California, Inc.
//
//  All rights reserved.
//
//  Redistribution and use in source and binary forms, with or without
//  modification, are permitted provided that the following conditions are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of Pacific Biosciences nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.
//
//  NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
//  THIS LICENSE.  THIS SOFTWARE IS PROVIDED BY PACIFIC BIOSCIENCES AND ITS
//  CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
//  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
//  PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL PACIFIC BIOSCIENCES OR
//  ITS CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
//  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
//  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
//  BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
//  IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
//  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
//  POSSIBILITY OF SUCH DAMAGE.

#include <basecaller/traceAnalysis/DevicePulseAccumulator.h>

#include <dataTypes/BatchData.cuh>
#include <dataTypes/BatchVectors.cuh>

#include <common/cuda/memory/DeviceOnlyArray.cuh>
#include <common/cuda/PBCudaSimd.cuh>
#include <common/MongoConstants.h>

using namespace PacBio::Cuda;
using namespace PacBio::Cuda::Memory;
using namespace PacBio::Cuda::Utility;
using namespace PacBio::Mongo::Data;

namespace PacBio {
namespace Mongo {
namespace Basecaller {

namespace {

// I don't want to promote this overload to a broader scope until I'm sure we need it
// I don't know if we want a full PBUInt2 type added, or if using raw uint2 is
// preferrable, or if using a pair of ints is just an implementation quirk of this
// particular file
inline __device__ uint2 Blend(PBBool2 cond, uint l, uint2 r) {
    uint2 ret;
    ret.x = cond.X() ? l : r.x;
    ret.y = cond.Y() ? l : r.y;
    return ret;
};


template <size_t blockThreads>
class Segment
{
public:
    __device__  Segment()
    {
        for (int i = 0; i < blockThreads; ++i)
        {
            startFrame_[i] = make_uint2(0,0);
            endFrame_[i] = make_uint2(0,0);
            signalFrstFrame_[i] = PBShort2(0);
            signalLastFrame_[i] = PBShort2(0);
            signalMax_[i] = PBShort2(0);
            signalTotal_[i] = PBShort2(0);
            signalM2_[i] = PBHalf2(0.0f);
            label_[i] = PBShort2(0);
        }
    }

    static __device__ PBBool2 IsPulseUpState(PBShort2 label)
    {
        return (numAnalogs < label) && (label <= 2*numAnalogs);
    }

    static __device__ PBBool2 IsPulseDownState(PBShort2 label)
    {
        return (2*numAnalogs < label);
    }

    __device__ PBBool2 IsNewSegment(PBShort2 label) const
    {
        return IsPulseUpState(label) || ((label == 0) && (label_[threadIdx.x] != 0));
    }

    __device__ PBBool2 IsPulse() const
    {
        return label_[threadIdx.x] != 0;
    }


    __device__ PBShort2 FullFrameLabel() const
    {
        PBShort2 ret = label_[threadIdx.x];
        ret = Blend(IsPulseDownState(ret), ret - 2*numAnalogs, ret);
        ret = Blend(IsPulseUpState(ret), ret - numAnalogs, ret);
        return ret;
    }

    template <int id>
    __device__ Data::Pulse ToPulse(uint32_t frameIndex)
    {
        static_assert(id < 2 && id >= 0, "Invalid index");
        using NucleotideLabel = Data::Pulse::NucleotideLabel;

        // This is potentially a short term hack.  The types like PBShort2 have
        // a Get function to handle getting the 0th or 1st element.  It would be
        // nice to handle this semmetrically, but right now we don't have a
        // PBUint2 type, and I don't think this is enough motivation to add one
        auto Get = [](uint2& var) -> uint& { return id ? var.x : var.y; };

        Data::Pulse ret;
        Get(endFrame_[threadIdx.x]) = frameIndex;
        auto start = Get(startFrame_[threadIdx.x]);
        short width = frameIndex - start;

        auto raw_mean = PBHalf2(signalTotal_[threadIdx.x] + signalLastFrame_[threadIdx.x] + signalFrstFrame_[threadIdx.x]) / width;
        auto raw_mid = PBHalf2(signalTotal_[threadIdx.x]) / (width - 2);

        const auto maxSignal = Data::Pulse::SignalMax();


        // TODO: This is a hard coded analog mapping that really needs to be handed in somehow
        auto LabelToAnalog = [](short label) {
            switch (label)
            {
            case 0:
                return NucleotideLabel::NONE;
            case 1:
                return NucleotideLabel::A;
            case 2:
                return NucleotideLabel::C;
            case 3:
                return NucleotideLabel::G;
            default :
            assert(label == 4);
                return NucleotideLabel::T;
            }
        };

        ret.Start(start)
            .Width(width)
            .MeanSignal(min(maxSignal, max(0.0f, raw_mean.Get<id>())))
            .MidSignal(width < 3 ? 0.0f : min(maxSignal, max(0.0f, raw_mid.Get<id>())))
            .MaxSignal(min(maxSignal, max(0.0f, signalMax_[threadIdx.x].template Get<id>())))
            .SignalM2(signalM2_[threadIdx.x].template Get<id>())
            .Label(LabelToAnalog(FullFrameLabel().template Get<id>()));

        return ret;
    }

    __device__ void ResetSegment(PBBool2 boundaryMask, uint32_t frameIndex,
                      PBShort2 label, PBShort2 signal)
    {
        startFrame_[threadIdx.x] = Blend(boundaryMask, frameIndex, startFrame_[threadIdx.x]);
        signalFrstFrame_[threadIdx.x] = Blend(boundaryMask, signal, signalFrstFrame_[threadIdx.x]);
        signalLastFrame_[threadIdx.x] = Blend(boundaryMask, 0, signalLastFrame_[threadIdx.x]);
        signalMax_[threadIdx.x] = Blend(boundaryMask, signal, signalMax_[threadIdx.x]);
        signalTotal_[threadIdx.x] = Blend(boundaryMask, 0, signalTotal_[threadIdx.x]);
        signalM2_[threadIdx.x] = Blend(boundaryMask, 0, signalM2_[threadIdx.x]);
        label_[threadIdx.x] = Blend(boundaryMask, label, label_[threadIdx.x]);
    }

    __device__ void AddSignal(PBBool2 update, PBShort2 signal)
    {
        signalTotal_[threadIdx.x] = Blend(update,
                                          signalTotal_[threadIdx.x] + signalLastFrame_[threadIdx.x],
                                          signalTotal_[threadIdx.x]);
        signalM2_[threadIdx.x] = Blend(update,
                                       signalM2_[threadIdx.x] + pow2(signalLastFrame_[threadIdx.x]),
                                       signalM2_[threadIdx.x]);
        signalLastFrame_[threadIdx.x] = Blend(update, signal, signalLastFrame_[threadIdx.x]);
        signalMax_[threadIdx.x] = Blend(update, max(signal, signalMax_[threadIdx.x]), signalMax_[threadIdx.x]);
    }

private:
    CudaArray<uint2, blockThreads> startFrame_;        // Needed because of partial segments
    CudaArray<uint2, blockThreads> endFrame_;          // 1 + the last frame included in the segment

    CudaArray<PBShort2, blockThreads> signalFrstFrame_;   // Signal of the most recent frame added
    CudaArray<PBShort2, blockThreads> signalLastFrame_;   // Signal recorded for the last frame in the segment
    CudaArray<PBHalf2, blockThreads> signalMax_;         // Max signal over all frames in segment
    CudaArray<PBShort2, blockThreads> signalTotal_;       // Signal total, excluding the first and last frame
    CudaArray<PBHalf2, blockThreads> signalM2_;          // Sum of squared signals, excluding the first and last frame

    CudaArray<PBShort2, blockThreads> label_;             // // Internal label ID corresponding to detection modes
};

template <size_t blockThreads>
__launch_bounds__(32, 32)
__global__ void ProcessLabels(GpuBatchData<const PBShort2> labels,
                              GpuBatchData<const PBShort2> signal,
                              GpuBatchData<const PBShort2> latSignal,
                              uint32_t firstFrameIdx,
                              DeviceView<Segment<blockThreads>> workingSegments,
                              GpuBatchVectors<Data::Pulse> pulsesOut)
{
    assert(labels.NumFrames() == signal.NumFrames() + latSignal.NumFrames());

    // TODO: move this to shared mem?
    auto& segment = workingSegments[blockIdx.x];

    // each thread handles 2 zmw, which normally are interleaved in something like PBShort2,
    // but cannot be for pulses
    auto pulsesZmw1 = pulsesOut.GetVector(blockIdx.x*2*blockDim.x + threadIdx.x*2);
    auto pulsesZmw2 = pulsesOut.GetVector(blockIdx.x*2*blockDim.x + threadIdx.x*2+1);
    pulsesZmw1.Reset();
    pulsesZmw2.Reset();

    auto labelZmw = labels.ZmwData(blockIdx.x, threadIdx.x);
    auto signalZmw = latSignal.ZmwData(blockIdx.x, threadIdx.x);

    auto HandleFrame = [&](PBShort2 label, PBShort2 signal, uint32_t frame) {
        auto boundaryMask = segment.IsNewSegment(label);
        auto pulseMask = segment.IsPulse();

        auto emit = boundaryMask && pulseMask;
        if (emit.X())
        {
            pulsesZmw1.push_back(segment.ToPulse<0>(frame));
        }
        if (emit.Y())
        {
            pulsesZmw2.push_back(segment.ToPulse<1>(frame));
        }

        segment.ResetSegment(boundaryMask, frame, label, signal);
        segment.AddSignal(!boundaryMask, signal);
    };

    const int latFrames = latSignal.NumFrames();
    for (int i = 0; i < latFrames; i++)
    {
        HandleFrame(labelZmw[i], signalZmw[i], i + firstFrameIdx);
    }

    signalZmw = signal.ZmwData(blockIdx.x, threadIdx.x);
    for (int i = 0; i < signal.NumFrames(); i++)
    {
        HandleFrame(labelZmw[i+latFrames], signalZmw[i], i + latFrames + firstFrameIdx);
    }
}

}

class DevicePulseAccumulator::AccumImpl
{
    static constexpr size_t blockThreads = laneSize / 2;
public:
    AccumImpl(size_t lanesPerPool)
        : workingSegments_(lanesPerPool)
    {
    }

    PulseBatch Process(const PulseBatchFactory& factory, LabelsBatch labels)
    {
        static constexpr size_t threadsPerBlock = 32;
        assert(threadsPerBlock*2 == labels.LaneWidth());
        auto ret = factory.NewBatch(labels.Metadata());
        ProcessLabels<threadsPerBlock><<<labels.LanesPerBatch(),threadsPerBlock>>>(
                labels,
                labels.TraceData(),
                labels.LatentTrace(),
                labels.Metadata().FirstFrame(),
                workingSegments_.GetDeviceView(),
                ret.Pulses());

        Cuda::CudaSynchronizeDefaultStream();
        return ret;
    }

private:
    DeviceOnlyArray<Segment<blockThreads>> workingSegments_;
};

void DevicePulseAccumulator::Configure(size_t maxCallsPerZmw)
{
    constexpr bool hostExecution = false;
    PulseAccumulator::InitAllocationPools(hostExecution, maxCallsPerZmw);
}

void DevicePulseAccumulator::Finalize()
{
    PulseAccumulator::DestroyAllocationPools();
}

DevicePulseAccumulator::DevicePulseAccumulator(uint32_t poolId, uint32_t lanesPerPool)
    : PulseAccumulator(poolId)
    , impl_(std::make_unique<AccumImpl>(lanesPerPool))
{

}

DevicePulseAccumulator::~DevicePulseAccumulator() = default;

Data::PulseBatch DevicePulseAccumulator::Process(Data::LabelsBatch labels)
{
    return impl_->Process(*batchFactory_, std::move(labels));
}

}}}     // namespace PacBio::Mongo::Basecaller

