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
#include <basecaller/traceAnalysis/SubframeLabelManager.h>

#include <dataTypes/BatchData.cuh>
#include <dataTypes/BatchVectors.cuh>
#include <dataTypes/configs/MovieConfig.h>

#include <common/cuda/memory/DeviceOnlyArray.cuh>
#include <common/cuda/memory/DeviceOnlyObject.cuh>
#include <common/cuda/PBCudaSimd.cuh>
#include <common/cuda/streams/LaunchManager.cuh>
#include <common/MongoConstants.h>
#include <common/StatAccumState.h>

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
inline __device__ uint2 Blend(PBShort2 cond, uint l, uint2 r) {
    uint2 ret;
    ret.x = cond.X() ? l : r.x;
    ret.y = cond.Y() ? l : r.y;
    return ret;
};

template <typename LabelManager, size_t blockThreads>
class Segment
{
public:
    Segment() = default;

    __device__  Segment(short initialState, float ampThr, float widThr)
        : ampThr_(__float2half(ampThr))
        , widThr_(__float2half(widThr))
    {
        for (int i = 0; i < blockThreads; ++i)
        {
            startFrame_[i] = make_uint2(0,0);
            endFrame_[i] = make_uint2(0,0);
            signalFrstFrame_[i] = PBShort2(0);
            signalLastFrame_[i] = PBShort2(0);
            signalMax_[i] = PBShort2(0);
            signalTotal_[i] = PBShort2(0);
            signalM2_[i] = PBFloat2(0.0f);
            label_[i] = PBShort2(initialState);
            m0_[i] = PBHalf2(0.0f);
            m1_[i] = PBHalf2(0.0f);
            m2_[i] = PBFloat2(0.0f);
        }
    }

    __device__ void SharedCopy(const Segment& other)
    {
        ampThr_ = other.ampThr_;
        widThr_ = other.widThr_;

        startFrame_[threadIdx.x] = other.startFrame_[threadIdx.x];
        endFrame_[threadIdx.x] = other.endFrame_[threadIdx.x];
        signalFrstFrame_[threadIdx.x] = other.signalFrstFrame_[threadIdx.x];
        signalLastFrame_[threadIdx.x] = other.signalLastFrame_[threadIdx.x];
        signalMax_[threadIdx.x] = other.signalMax_[threadIdx.x];
        signalTotal_[threadIdx.x] = other.signalTotal_[threadIdx.x];
        signalM2_[threadIdx.x] = other.signalM2_[threadIdx.x];
        label_[threadIdx.x] = other.label_[threadIdx.x];
        m0_[threadIdx.x] = other.m0_[threadIdx.x];
        m1_[threadIdx.x] = other.m1_[threadIdx.x];
        m2_[threadIdx.x] = other.m2_[threadIdx.x];
    }

    __device__ PBShort2 IsNewSegment(PBShort2 label) const
    {
        return LabelManager::IsNewSegment(label_[threadIdx.x], label);
    }

    __device__ PBShort2 IsPulse() const
    {
        return LabelManager::BaselineLabel() != label_[threadIdx.x];
    }

    // Pulse is an in/out reference to avoid memory churn. We're going to populate it's
    // values directly in it's final destination.
    template <int id>
    __device__ void ToPulse(uint32_t frameIndex, PBHalf2 minMean, const LabelManager& manager, Data::Pulse& pulse)
    {
        static_assert(id < 2 && id >= 0, "Invalid index");

        // This is potentially a short term hack.  The types like PBShort2 have
        // a Get function to handle getting the 0th or 1st element.  It would be
        // nice to handle this semmetrically, but right now we don't have a
        // PBUint2 type, and I don't think this is enough motivation to add one
        auto Get = [](uint2& var) -> uint& { return id == 0 ? var.x : var.y; };

        const float maxSignal = Data::Pulse::SignalMax();
        const float minSignal = 0.0f;

        Get(endFrame_[threadIdx.x]) = frameIndex;
        auto start = Get(startFrame_[threadIdx.x]);
        short width = frameIndex - start;
        half widthHalf(width);

        PBShort2 rawMeanShort = signalTotal_[threadIdx.x] + signalLastFrame_[threadIdx.x] + signalFrstFrame_[threadIdx.x];
        half raw_mean = half(rawMeanShort.template Get<id>()) / widthHalf;
        float raw_mid = float(signalTotal_[threadIdx.x].template Get<id>()) / (width - 2);

        half lowAmp = minMean.Get<id>();
        auto keep = (widthHalf >= widThr_) || (raw_mean * widthHalf >= ampThr_ * lowAmp);

        pulse.Start(start)
            .Width(width)
            .MeanSignal(min(maxSignal, max(minSignal, __half2float(raw_mean))))
            .MidSignal(width < 3 ? 0.0f : min(maxSignal, max(minSignal, raw_mid)))
            .MaxSignal(min(maxSignal, max(minSignal, signalMax_[threadIdx.x].template Get<id>())))
            .SignalM2(signalM2_[threadIdx.x].template Get<id>())
            .Label(manager.Nucleotide(label_[threadIdx.x].template Get<id>()))
            .IsReject(!keep);
    }

    __device__ void ResetSegment(PBShort2 boundaryMask, uint32_t frameIndex,
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

    __device__ void AddSignal(PBShort2 update, PBShort2 signal)
    {
        signalTotal_[threadIdx.x] = Blend(update,
                                          signalTotal_[threadIdx.x] + signalLastFrame_[threadIdx.x],
                                          signalTotal_[threadIdx.x]);
        PBFloat2 s{PBHalf2(signalLastFrame_[threadIdx.x])};
        signalM2_[threadIdx.x] = Blend(update,
                                       signalM2_[threadIdx.x] + s*s,
                                       signalM2_[threadIdx.x]);
        signalLastFrame_[threadIdx.x] = Blend(update, signal, signalLastFrame_[threadIdx.x]);
        signalMax_[threadIdx.x] = Blend(update, max(signal, signalMax_[threadIdx.x]), signalMax_[threadIdx.x]);
    }

    __device__ void AddBaseline(PBShort2 baselineMask, PBShort2 signal)
    {
        PBHalf2 zero(0.0f);
        PBHalf2 one(1.0f);

        m0_[threadIdx.x] += Blend(baselineMask, one, zero);
        m1_[threadIdx.x] += Blend(baselineMask, signal, zero);
        m2_[threadIdx.x] += Blend(baselineMask, pow2(signal), zero);
    }

    __device__ void FillBaselineStats(Mongo::StatAccumState& stats)
    {
        stats.moment0[2*threadIdx.x] = m0_[threadIdx.x].FloatX();
        stats.moment0[2*threadIdx.x+1] = m0_[threadIdx.x].FloatY();
        stats.moment1[2*threadIdx.x] = m1_[threadIdx.x].FloatX();
        stats.moment1[2*threadIdx.x+1] = m1_[threadIdx.x].FloatY();
        stats.moment2[2*threadIdx.x] = m2_[threadIdx.x].X();
        stats.moment2[2*threadIdx.x+1] = m2_[threadIdx.x].Y();
    }

private:
    CudaArray<uint2, blockThreads> startFrame_;        // Needed because of partial segments
    CudaArray<uint2, blockThreads> endFrame_;          // 1 + the last frame included in the segment

    CudaArray<PBShort2, blockThreads> signalFrstFrame_;   // Signal of the most recent frame added
    CudaArray<PBShort2, blockThreads> signalLastFrame_;   // Signal recorded for the last frame in the segment
    CudaArray<PBHalf2, blockThreads> signalMax_;          // Max signal over all frames in segment
    CudaArray<PBShort2, blockThreads> signalTotal_;       // Signal total, excluding the first and last frame
    CudaArray<PBFloat2, blockThreads> signalM2_;          // Sum of squared signals, excluding the first and last frame

    CudaArray<PBShort2, blockThreads> label_;             // Internal label ID corresponding to detection modes

    // Baseline stats computed from frames marked as baseline
    CudaArray<PBHalf2,  blockThreads> m0_;
    CudaArray<PBHalf2,  blockThreads> m1_;
    CudaArray<PBFloat2, blockThreads> m2_;
    half ampThr_;
    half widThr_;
};

template <typename LabelManager, size_t blockThreads>
__launch_bounds__(blockThreads, 32)
__global__ void ProcessLabels(GpuBatchData<const PBShort2> labels,
                              GpuBatchData<const PBShort2> signal,
                              GpuBatchData<const PBShort2> latSignal,
                              uint32_t firstFrameIdx,
                              DeviceView<Segment<LabelManager, blockThreads>> workingSegments,
                              DeviceView<const Data::LaneModelParameters<Cuda::PBHalf2, blockThreads>> models,
                              DevicePtr<const LabelManager> manager,
                              GpuBatchVectors<Data::Pulse> pulsesOut,
                              DeviceView<Mongo::StatAccumState> stats)
{
    assert(labels.NumFrames() == signal.NumFrames() + latSignal.NumFrames());

    __shared__ Segment<LabelManager, blockThreads> segment;

    segment.SharedCopy(workingSegments[blockIdx.x]);

    // each thread handles 2 zmw, which normally are interleaved in something like PBShort2,
    // but cannot be for pulses
    auto pulsesZmw1 = pulsesOut.GetVector(blockIdx.x*2*blockDim.x + threadIdx.x*2);
    auto pulsesZmw2 = pulsesOut.GetVector(blockIdx.x*2*blockDim.x + threadIdx.x*2+1);
    pulsesZmw1.Reset();
    pulsesZmw2.Reset();

    auto HandleFrame = [&](PBShort2 label, PBShort2 signal, uint32_t frame) {
        auto boundaryMask = segment.IsNewSegment(label);
        auto pulseMask = segment.IsPulse();

        // Last analog channel is the darkest (lowest amplitude)
        auto minMean = models[blockIdx.x].AnalogMode(numAnalogs-1).means[threadIdx.x];

        auto emit = boundaryMask && pulseMask;
        if (emit.X())
        {
            pulsesZmw1.emplace_back_default();
            segment.ToPulse<0>(frame, minMean, *manager, pulsesZmw1.back());
        }
        if (emit.Y())
        {
            pulsesZmw2.emplace_back_default();
            segment.ToPulse<1>(frame, minMean, *manager, pulsesZmw2.back());
        }

        segment.ResetSegment(boundaryMask, frame, label, signal);
        segment.AddSignal(!boundaryMask, signal);

        // TODO: The below excludes the baseline frame succeeding a pulse
        // but we also want to exclude the baseline frame preceding a pulse
        // to not contaminate the baseline statistics by frames that
        // might be partially influenced by an adjacent pulse frame.
        auto baseline = (!pulseMask) && (!boundaryMask);
        segment.AddBaseline(baseline, signal);
    };

    const int latFrames = latSignal.NumFrames();
    auto labelZmw  = labels.ZmwData(blockIdx.x, threadIdx.x);
    auto signalZmw = latSignal.ZmwData(blockIdx.x, threadIdx.x);

    for (int i = 0; i < latFrames; i++)
    {
        HandleFrame(labelZmw[i], signalZmw[i], i + firstFrameIdx);
    }

    signalZmw = signal.ZmwData(blockIdx.x, threadIdx.x);
    for (int i = 0; i < signal.NumFrames(); i++)
    {
        HandleFrame(labelZmw[i+latFrames], signalZmw[i], i + latFrames + firstFrameIdx);
    }

    workingSegments[blockIdx.x].SharedCopy(segment);
    workingSegments[blockIdx.x].FillBaselineStats(stats[blockIdx.x]);
}

}

template <typename LabelManager>
class DevicePulseAccumulator<LabelManager>::AccumImpl
{
    static constexpr size_t blockThreads = laneSize / 2;
public:
    AccumImpl(size_t lanesPerPool, StashableAllocRegistrar* registrar)
        : workingSegments_(registrar, SOURCE_MARKER(),
                           lanesPerPool, LabelManager::BaselineLabel(), ampThresh_, widthThresh_)
    {
    }

    std::pair<PulseBatch, PulseDetectorMetrics>
    Process(const PulseBatchFactory& factory, LabelsBatch labels, const PoolModelParameters& models)
    {
        assert(blockThreads*2 == labels.LaneWidth());
        auto ret = factory.NewBatch(labels.Metadata(), labels.StorageDims());

        const auto& launcher = PBLauncher(
            ProcessLabels<LabelManager, blockThreads>,
            labels.LanesPerBatch(),
            blockThreads);
        launcher(labels,
                 labels.TraceData(),
                 labels.LatentTrace(),
                 labels.Metadata().FirstFrame(),
                 workingSegments_,
                 models,
                 *manager_,
                 ret.first.Pulses(),
                 ret.second.baselineStats);

        Cuda::CudaSynchronizeDefaultStream();
        return ret;
    }

    static void Configure(CudaArray<Data::Pulse::NucleotideLabel, numAnalogs>& analogMap)
    {
        manager_ = std::make_unique<DeviceOnlyObj<const LabelManager>>(SOURCE_MARKER(), analogMap);
    }

    static void Finalize()
    {
        manager_.release();
    }

private:
    DeviceOnlyArray<Segment<LabelManager, blockThreads>> workingSegments_;
    static std::unique_ptr<DeviceOnlyObj<const LabelManager>> manager_;
};

template <typename LabelManager>
constexpr size_t DevicePulseAccumulator<LabelManager>::AccumImpl::blockThreads;

template <typename LabelManager>
std::unique_ptr<DeviceOnlyObj<const LabelManager>> DevicePulseAccumulator<LabelManager>::AccumImpl::manager_;

template <typename LabelManager>
void DevicePulseAccumulator<LabelManager>::Configure(const Data::MovieConfig& movieConfig,
                                                     const Data::BasecallerPulseAccumConfig& pulseConfig)
{
    constexpr bool hostExecution = false;
    PulseAccumulator::InitFactory(hostExecution, pulseConfig);

    CudaArray<Data::Pulse::NucleotideLabel, numAnalogs> analogMap;

    for(size_t i = 0; i < analogMap.size(); i++)
    {
        analogMap[i] = Data::mapToNucleotideLabel(movieConfig.analogs[i].baseLabel);
    }

    AccumImpl::Configure(analogMap);
}

template <typename LabelManager>
void DevicePulseAccumulator<LabelManager>::Finalize()
{
    AccumImpl::Finalize();
}

template <typename LabelManager>
DevicePulseAccumulator<LabelManager>::DevicePulseAccumulator(
        uint32_t poolId,
        uint32_t lanesPerPool,
        StashableAllocRegistrar* registrar)
    : PulseAccumulator(poolId)
    , impl_(std::make_unique<AccumImpl>(lanesPerPool, registrar))
{

}

template <typename LabelManager>
DevicePulseAccumulator<LabelManager>::~DevicePulseAccumulator() = default;

template <typename LabelManager>
std::pair<Data::PulseBatch, Data::PulseDetectorMetrics>
DevicePulseAccumulator<LabelManager>::Process(Data::LabelsBatch labels, const PoolModelParameters& models)
{
    return impl_->Process(*batchFactory_, std::move(labels), models);
}

// explicit instantiations
template class DevicePulseAccumulator<SubframeLabelManager>;

}}}     // namespace PacBio::Mongo::Basecaller

