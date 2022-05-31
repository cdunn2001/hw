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
//  Description:
/// \file   HFMetricsFilterDevice.cu
/// \brief  A filter for computing or aggregating trace- and pulse-metrics
///         on a time scale equal to or greater than the standard block size.


#include "HFMetricsFilterDevice.h"
#include <dataTypes/BatchData.cuh>
#include <dataTypes/BatchVectors.cuh>
#include <dataTypes/HQRFPhysicalStates.h>
#include <dataTypes/LaneDetectionModel.h>
#include <common/AutocorrAccumState.h>
#include <common/cuda/memory/DeviceOnlyArray.cuh>
#include <common/cuda/PBCudaRuntime.h>
#include <common/cuda/PBCudaSimd.cuh>
#include <common/cuda/streams/LaunchManager.cuh>

using namespace PacBio::Cuda;
using namespace PacBio::Mongo::Data;
using namespace PacBio::Cuda::Memory;

namespace PacBio {
namespace Mongo {
namespace Basecaller {

HFMetricsFilterDevice::~HFMetricsFilterDevice() = default;

using DeviceStatAccumState = StatAccumStateT<PBFloat2, laneSize/2>;

namespace {

__constant__ TrainedCartDevice trainedCartParams;

struct alignas(64) BasecallingMetricsAccumulatorDevice
{
public: // types
    using InputPulses = Data::LaneVectorView<const Data::Pulse>;
    using InputBaselineStats = Data::BaselinerStatAccumState;
    using InputModelsT = Data::LaneModelParameters<Cuda::PBHalf, laneSize>;

    template <typename T>
    using SingleMetric = Cuda::Utility::CudaArray<T, laneSize/2>;
    template <typename T>
    using AnalogMetric = Cuda::Utility::CudaArray<SingleMetric<T>,
                                                  numAnalogs>;

public: // ctors
    __device__ BasecallingMetricsAccumulatorDevice()
    { }

public: // metrics
    SingleMetric<PBShort2> numPulseFrames;
    SingleMetric<PBShort2> numBaseFrames;
    SingleMetric<PBShort2> numSandwiches;
    SingleMetric<PBShort2> numHalfSandwiches;
    SingleMetric<PBShort2> numPulseLabelStutters;
    SingleMetric<PBShort2> activityLabel;
    AnalogMetric<PBFloat2> pkMidSignal;
    AnalogMetric<PBHalf2> bpZvar;
    AnalogMetric<PBHalf2> pkZvar;
    AnalogMetric<PBHalf2> pkMax;
    AnalogMetric<PBHalf2> modelVariance;
    AnalogMetric<PBHalf2> modelMean;
    AnalogMetric<PBShort2> numPkMidFrames;
    AnalogMetric<PBShort2> numPkMidBasesByAnalog;
    AnalogMetric<PBShort2> numBasesByAnalog;
    AnalogMetric<PBShort2> numPulsesByAnalog;
    SingleMetric<uint2> startFrame;
    SingleMetric<uint2> numFrames;
    SingleMetric<PBShort2> pixelChecksum;
    SingleMetric<PBHalf2> pulseDetectionScore;
    AnalogMetric<PBFloat2> bpZvarAcc;
    AnalogMetric<PBFloat2> pkZvarAcc;

    // The baseline stat accumulator:
    DeviceStatAccumState baselineStats;

    // The autocorrelation accumulator:
    SingleMetric<float2> traceM0;
    SingleMetric<float2> traceM1;
    SingleMetric<float2>  traceM2;
    SingleMetric<float2>  autocorrM2;
    Cuda::Utility::CudaArray<SingleMetric<PBHalf2>, AutocorrAccumState::lag> fBuf; // left region buffer
    Cuda::Utility::CudaArray<SingleMetric<PBHalf2>, AutocorrAccumState::lag> bBuf; // right region buffer
    SingleMetric<PBShort2> bIdx;  // left (X) and right (Y) buffer indices

public: // state trackers
    // These are the right size, but doesn't follow the <Pair, laneSize/2>
    // pattern above:
    Cuda::Utility::CudaArray<Data::Pulse, laneSize> prevBasecallCache;
    Cuda::Utility::CudaArray<Data::Pulse, laneSize> prevprevBasecallCache;
};

__device__ float variance(const float M0, const float M1, const float M2)
{
    if (M0 < 2.0f)
        return std::numeric_limits<half>::quiet_NaN();
    float var = M1 * M1 / M0;
    var = (M2 - var) / (M0 - 1.0f);
    var = max(var , 0.0f);
    return var;
}

__device__ float2 variance(const float2 M0, const float2 M1, const float2 M2)
{
    return make_float2(variance(M0.x, M1.x, M2.x), variance(M0.y, M1.y, M2.y));
}

__device__ PBFloat2 variance2(const PBFloat2 M0, const PBFloat2 M1, const PBFloat2 M2)
{
    return PBFloat2(variance(M0.X(), M1.X(), M2.X()), variance(M0.Y(), M1.Y(), M2.Y()));
}

__device__ uint2 getWideLoad(const Cuda::Utility::CudaArray<uint16_t, laneSize>& load)
{ return make_uint2(load[threadIdx.x * 2], load[threadIdx.x * 2 + 1]); };

__device__ float2 getWideLoad(const Cuda::Utility::CudaArray<float, laneSize>& load)
{ return make_float2(load[threadIdx.x * 2], load[threadIdx.x * 2 + 1]); };

__device__ PBFloat2 getWideLoad2(const Cuda::Utility::CudaArray<float, laneSize>& load)
{
    return PBFloat2(load[threadIdx.x * 2], load[threadIdx.x * 2 + 1]); 
};


__device__ PBHalf2 replaceNans(PBHalf2 vals)
{ return Blend(vals == vals, vals, PBHalf2(0.0)); };

__device__ PBFloat2 replaceNans(PBFloat2 vals)
{ return Blend(vals == vals, vals, PBFloat2(0.0)); };

__device__ float2& operator+=(float2& l, const float2 r)
{
    l.x += r.x;
    l.y += r.y;
    return l;
}

__device__ uint2 operator+(uint2 l, uint2 r)
{ return make_uint2(l.x + r.x, l.y + r.y); }

__device__ float2 operator-(float2 l, float2 r)
{ return make_float2(l.x - r.x, l.y - r.y); }

__device__ float2 operator*(float2 l, float2 r)
{ return make_float2(l.x * r.x, l.y * r.y); }

__device__ float2 operator/(float2 l, float2 r)
{ return make_float2(l.x / r.x, l.y / r.y); }

__device__ float2 asFloat2(PBHalf2 val)
{ return make_float2(val.FloatX(), val.FloatY()); }

__device__ void ResetStats(DeviceStatAccumState& stats)
{
    const auto pbzero = PBFloat2(0.0f);
    stats.moment0[threadIdx.x] = pbzero;
    stats.moment1[threadIdx.x] = pbzero;
    stats.moment2[threadIdx.x] = pbzero;
    // todo: is this correct, or not?
    //stats.offset[threadIdx.x] = pbzero;
}

__device__ PBFloat2 Mean(const DeviceStatAccumState& stats)
{
    return stats.moment1[threadIdx.x] / stats.moment0[threadIdx.x] + stats.offset[threadIdx.x];
}

__device__ PBFloat2 Mean(const StatAccumState& stats)
{
    const PBFloat2 offset = getWideLoad2(stats.offset);
    const PBFloat2 m0 = getWideLoad2(stats.moment0);
    const PBFloat2 m1 = getWideLoad2(stats.moment1);
    return m1 / m0 + offset;
}

__device__ PBFloat2 Variance(const DeviceStatAccumState& stats)
{
    return variance2(stats.moment0[threadIdx.x],stats.moment1[threadIdx.x],stats.moment2[threadIdx.x]);
}

// GPU implementation of much of the functionality of the host StatAccumulator. The purpose of this
// class is to make GPU code look more like CPU code, hopefully reducing (numerical) differences
// in results.
// todo: unify with host code, if/where feasible.
class QuickStats {
    PBFloat2 m0_;
    PBFloat2 m1_;
    PBFloat2 m2_;
    PBFloat2 offset_;

    public:
    __device__ QuickStats(const PacBio::Mongo::StatAccumState& stats)
    {
        offset_ = getWideLoad2(stats.offset);
        m0_ = getWideLoad2(stats.moment0);
        m1_ = getWideLoad2(stats.moment1);
        m2_ = getWideLoad2(stats.moment2);
    }

    __device__ QuickStats(const DeviceStatAccumState& stats)
    {
        offset_ = stats.offset[threadIdx.x];
        m0_ = stats.moment0[threadIdx.x];
        m1_ = stats.moment1[threadIdx.x];
        m2_ = stats.moment2[threadIdx.x];
    }

    __device__ void Output(DeviceStatAccumState& stats) const
    {
        stats.offset[threadIdx.x] = offset_;
        stats.moment0[threadIdx.x] = m0_;
        stats.moment1[threadIdx.x] = m1_;
        stats.moment2[threadIdx.x] = m2_;
    }

    /// see StatAccumulator.Shift()
    __device__ void Shift(const PBFloat2& shift)
    {
        offset_ += shift;
    }

    __device__ QuickStats& operator+=(QuickStats other)
    {
        const auto one = PBFloat2(1.0f);

        const PBFloat2 w = other.m0_ / (m0_ + other.m0_);
        PBFloat2 offsetNew = (one - w)*offset_ + w*other.offset_;
        offsetNew = Blend(isnan(offsetNew), offset_, offsetNew);
        other.Offset(offsetNew);
        this->Offset(offsetNew);
        Merge(other);
        return *this;

    }

        /// Merge another instance with the same offset into this one.
    __device__ QuickStats& Merge(const QuickStats& other)
    {
        m0_ += other.m0_;
        m1_ += other.m1_;
        m2_ += other.m2_;
        return *this;
    }

    __device__ void Offset(const PBFloat2& value)
    {
        const auto zero = PBFloat2(0.0f);
        //if (all(value == offset_)) return;

        const auto m1new = m1_ + m0_*(offset_ - value);
        const auto m2new = m2_ + (pow2f(m1new) - pow2f(m1_)) / m0_;

        offset_ = value;
        m1_ = m1new;
        // Guard against NaN.
        m2_ = Blend(m0_ == zero,  zero, m2new);
    }

};

template<int id>
__device__ float2 blendFloat0(float val)
{
    static_assert(id < 2, "Out of bounds access in blendFloat0");
    if (id == 0) return make_float2(val, 0.0f);
    else return make_float2(0.0f, val);
}

template<int id>
__device__ PBFloat2 blendFloat0f(float val)
{
    static_assert(id < 2, "Out of bounds access in blendFloat0");
    if (id == 0) return PBFloat2(val, 0.0f);
    else return PBFloat2(0.0f, val);
}

template<int id>
__device__ PBHalf2 blendHalf0(float val)
{
    static_assert(id < 2, "Out of bounds access in blendHalf0");
    if (id == 0) return PBHalf2(val, 0.0f);
    else return PBHalf2(0.0f, val);
}

template<int id>
__device__ PBShort2 blendShort0(const uint16_t val)
{
    static_assert(id < 2, "Out of bounds access in blendShort0");
    if (id == 0) return PBShort2(val, 0);
    else return PBShort2(0, val);
}

__device__ PBHalf2 autocorrelation(const BasecallingMetricsAccumulatorDevice& blockMetrics)
{
    const uint32_t lag = AutocorrAccumState::lag;
    // math in float2 for additional range
    const auto nmk = blockMetrics.traceM0[threadIdx.x] - make_float2(lag, lag);
    PBHalf2 ac = [&blockMetrics](float2 nmk)
    {
        float2 mu = blockMetrics.traceM1[threadIdx.x] / blockMetrics.traceM0[threadIdx.x];
        float2 m1x2 = make_float2(2.0f, 2.0f) * blockMetrics.traceM1[threadIdx.x];
        for (auto k = 0u; k < lag; ++k)
        {
            m1x2 = m1x2 - asFloat2(blockMetrics.fBuf[k][threadIdx.x]
                                   + blockMetrics.bBuf[k][threadIdx.x]);
        }
        float2 ac_internal = mu*(m1x2 - nmk*mu);
        nmk = nmk - asFloat2(PBHalf2(1.0f));
        ac_internal = (blockMetrics.autocorrM2[threadIdx.x] - ac_internal)
             / (nmk * variance(blockMetrics.traceM0[threadIdx.x],
                               blockMetrics.traceM1[threadIdx.x],
                               blockMetrics.traceM2[threadIdx.x]));
        return ac_internal;
    }(nmk);
    const PBBool2 nanMask = !(ac == ac);
    const PBHalf2 nans(std::numeric_limits<half2>::quiet_NaN());
    ac = min(max(ac, -1.0f), 1.0f);  // clamp ac in [-1..1]
    return Blend(nmk < 1.0f || nanMask, nans, ac);
}

template <size_t size>
__device__ PBHalf2 evaluatePolynomial(const Cuda::Utility::CudaArray<float, size>& coeff, PBHalf2 x)
{
    static_assert(size > 0);
    PBHalf2 y(coeff[0]);
    for (size_t i = 1; i < size; ++i)
        y = y * x + PBHalf2(coeff[i]);
    return y;
}

template <int id, typename FeaturesT>
__device__ uint16_t traverseCart(const FeaturesT& features)
{
    size_t current = 0;
    while (trainedCartParams.feature[current] >= 0)
    {
        const auto& featVal = features[trainedCartParams.feature[current]].template Get<id>();
        if (featVal <= trainedCartParams.threshold[current])
        {
            current = trainedCartParams.childrenLeft[current];
        }
        else
        {
            current = trainedCartParams.childrenRight[current];
        }
    }
    return trainedCartParams.value[current];
};

// Necessary evil: we don't construct a new DeviceOnlyArray for each hfmetric
// block, but rather re-initialize
__global__ void InitializeMetrics(
        DeviceView<BasecallingMetricsAccumulatorDevice> metrics,
        bool initialize)
{
    // Some members aren't accumulators, and don't need initialization. They
    // aren't included here:
    BasecallingMetricsAccumulatorDevice& blockMetrics = metrics[blockIdx.x];
    float2 zero = make_float2(0.0f, 0.0f);
    PBFloat2 zero2 = PBFloat2(0.0f);

    if (!initialize)
    {
        blockMetrics.startFrame[threadIdx.x] = blockMetrics.startFrame[threadIdx.x]
                                               + blockMetrics.numFrames[threadIdx.x];
    }
    else
    {
        blockMetrics.startFrame[threadIdx.x] = make_uint2(0, 0);
    }
    blockMetrics.numFrames[threadIdx.x] = make_uint2(0, 0);

    blockMetrics.numPulseFrames[threadIdx.x] = 0;
    blockMetrics.numBaseFrames[threadIdx.x] = 0;
    blockMetrics.numSandwiches[threadIdx.x] = 0;
    blockMetrics.numHalfSandwiches[threadIdx.x] = 0;
    blockMetrics.numPulseLabelStutters[threadIdx.x] = 0;
    blockMetrics.pulseDetectionScore[threadIdx.x] = 0.0f;

    ResetStats(blockMetrics.baselineStats);

    blockMetrics.traceM0[threadIdx.x] = zero;
    blockMetrics.traceM1[threadIdx.x] = zero;
    blockMetrics.traceM2[threadIdx.x] = zero;

    blockMetrics.autocorrM2[threadIdx.x] = zero;

    auto lag = AutocorrAccumState::lag;
    for (auto k = 0u; k < lag; ++k) blockMetrics.fBuf[k][threadIdx.x] = 0.0f;
    for (auto k = 0u; k < lag; ++k) blockMetrics.bBuf[k][threadIdx.x] = 0.0f;
    blockMetrics.bIdx[threadIdx.x] = 0;

    for (size_t a = 0; a < numAnalogs; ++a)
    {
        blockMetrics.pkMidSignal[a][threadIdx.x] = zero2;
        blockMetrics.bpZvarAcc[a][threadIdx.x] = zero2;
        blockMetrics.pkZvarAcc[a][threadIdx.x] = zero2;
        blockMetrics.pkMax[a][threadIdx.x] = 0.0f;
        blockMetrics.numPkMidFrames[a][threadIdx.x] = 0;
        blockMetrics.numPkMidBasesByAnalog[a][threadIdx.x] = 0;
        blockMetrics.numBasesByAnalog[a][threadIdx.x] = 0;
        blockMetrics.numPulsesByAnalog[a][threadIdx.x] = 0;
    }

    blockMetrics.prevBasecallCache[threadIdx.x * 2] = Pulse().Start(0).Width(0).Label(Pulse::NucleotideLabel::NONE);
    blockMetrics.prevBasecallCache[threadIdx.x * 2 + 1] = Pulse().Start(0).Width(0).Label(Pulse::NucleotideLabel::NONE);
    blockMetrics.prevprevBasecallCache[threadIdx.x * 2] = Pulse().Start(0).Width(0).Label(Pulse::NucleotideLabel::NONE);
    blockMetrics.prevprevBasecallCache[threadIdx.x * 2 + 1] = Pulse().Start(0).Width(0).Label(Pulse::NucleotideLabel::NONE);
}

template<int id>
__device__ void stutterSandwich(const Pulse* pulse,
                                const Pulse* prevPulse,
                                const Pulse* prevprevPulse,
                                BasecallingMetricsAccumulatorDevice& blockMetrics)
{
    const auto& inc = blendShort0<id>(1);
    if (prevPulse->Label() != Pulse::NucleotideLabel::NONE)
    {
        if (prevPulse->Label() == pulse->Label())
        {
            blockMetrics.numPulseLabelStutters[threadIdx.x] += inc;
        }
        const bool abutted = (pulse->Start() == prevPulse->Stop());
        if (abutted && prevPulse->Label() != pulse->Label())
        {
            blockMetrics.numHalfSandwiches[threadIdx.x] += inc;
        }
        if (prevprevPulse->Label() != Pulse::NucleotideLabel::NONE)
        {
            const bool prevAbutted = (prevPulse->Start() == prevprevPulse->Stop());
            if (prevAbutted && abutted
                    && prevprevPulse->Label() == pulse->Label()
                    && prevprevPulse->Label() != prevPulse->Label())
            {
                blockMetrics.numSandwiches[threadIdx.x] += inc;
            }
        }
    }
};

template<int id>
__device__ void goodBaseMetrics(
        const Pulse* pulse,
        BasecallingMetricsAccumulatorDevice& blockMetrics)
{
    if (!pulse->IsReject())
    {
        const auto& inc = blendShort0<id>(1);
        const auto& label = static_cast<uint8_t>(pulse->Label());
        blockMetrics.numBaseFrames[threadIdx.x] += blendShort0<id>(pulse->Width());
        blockMetrics.numBasesByAnalog[label][threadIdx.x] += inc;

        if (!isnan(pulse->MidSignal()))
        {
            const auto& midWidth = pulse->Width() - 2;
            const auto& midSignal = pulse->MidSignal();
            blockMetrics.numPkMidBasesByAnalog[label][threadIdx.x] += inc;
            blockMetrics.numPkMidFrames[label][threadIdx.x] += blendShort0<id>(midWidth);
            blockMetrics.pkMidSignal[label][threadIdx.x] += blendFloat0f<id>(midSignal * midWidth);
            blockMetrics.bpZvarAcc[label][threadIdx.x] += blendFloat0f<id>(midSignal * midSignal * midWidth);
            blockMetrics.pkZvarAcc[label][threadIdx.x] += blendFloat0f<id>(pulse->SignalM2());
        }
    }
};

template <int id>
__device__ void processPulse(
        const Pulse* pulse,
        const Pulse* prevPulse,
        const Pulse* prevprevPulse,
        BasecallingMetricsAccumulatorDevice& blockMetrics)
{
    const auto& label = static_cast<uint8_t>(pulse->Label());
    blockMetrics.numPulseFrames[threadIdx.x] += blendShort0<id>(pulse->Width());
    blockMetrics.numPulsesByAnalog[label][threadIdx.x] += blendShort0<id>(1);
    blockMetrics.pkMax[label][threadIdx.x] = Blend(
            blendShort0<id>(1),
            max(blockMetrics.pkMax[label][threadIdx.x].Get<id>(),
                pulse->MaxSignal()),
            blockMetrics.pkMax[label][threadIdx.x]);
    stutterSandwich<id>(pulse, prevPulse, prevprevPulse, blockMetrics);
    goodBaseMetrics<id>(pulse, blockMetrics);
};


__global__ void ProcessChunk(
        const DeviceView<const HFMetricsFilterDevice::BaselinerStatsT> baselinerStats,
        const DeviceView<const Data::LaneModelParameters<Cuda::PBHalf2, laneSize/2>> models,
        const Data::GpuBatchVectors<const Data::Pulse> pulses,
        const DeviceView<const PacBio::Cuda::Utility::CudaArray<float, laneSize>> flMetrics,
        const DeviceView<const PacBio::Mongo::StatAccumState> pdMetrics,
        uint32_t numFrames,
        DeviceView<BasecallingMetricsAccumulatorDevice> metrics)
{
    auto& blockMetrics = metrics[blockIdx.x];

    blockMetrics.numFrames[threadIdx.x].x += numFrames;
    blockMetrics.numFrames[threadIdx.x].y += numFrames;

    // Pull the last block's pulses out of the cache
    const Pulse* prevPulseX = &blockMetrics.prevBasecallCache[threadIdx.x * 2];
    const Pulse* prevPulseY = &blockMetrics.prevBasecallCache[threadIdx.x * 2 + 1];
    const Pulse* prevprevPulseX = &blockMetrics.prevprevBasecallCache[threadIdx.x * 2];
    const Pulse* prevprevPulseY = &blockMetrics.prevprevBasecallCache[threadIdx.x * 2 + 1];

    // Setup the iteration
    const auto& pulsesX = pulses.GetVector(blockIdx.x*2*blockDim.x + threadIdx.x*2);
    const auto& pulsesY = pulses.GetVector(blockIdx.x*2*blockDim.x + threadIdx.x*2+1);

    uint32_t numPulsesX = pulsesX.size();
    uint32_t numPulsesY = pulsesY.size();
    for (uint32_t pIdx = 0; pIdx < numPulsesX; ++pIdx)
    {
        const Pulse* pulseX = &pulsesX[pIdx];
        processPulse<0>(pulseX, prevPulseX, prevprevPulseX, blockMetrics);
        prevprevPulseX = prevPulseX;
        prevPulseX = pulseX;
    }
    for (uint32_t pIdx = 0; pIdx < numPulsesY; ++pIdx)
    {
        const Pulse* pulseY = &pulsesY[pIdx];
        processPulse<1>(pulseY, prevPulseY, prevprevPulseY, blockMetrics);
        prevprevPulseY = prevPulseY;
        prevPulseY = pulseY;
    }

    // Cache this block's final pulses
    blockMetrics.prevBasecallCache[threadIdx.x * 2] = *prevPulseX;
    blockMetrics.prevBasecallCache[threadIdx.x * 2 + 1] = *prevPulseY;
    blockMetrics.prevprevBasecallCache[threadIdx.x * 2] = *prevprevPulseX;
    blockMetrics.prevprevBasecallCache[threadIdx.x * 2 + 1] = *prevprevPulseY;

    // AddModels: update model information
    for (size_t ai = 0; ai < numAnalogs; ++ai)
    {
        blockMetrics.modelVariance[ai][threadIdx.x] = models[blockIdx.x].AnalogMode(ai).vars[threadIdx.x];
        blockMetrics.modelMean[ai][threadIdx.x] = models[blockIdx.x].AnalogMode(ai).means[threadIdx.x];
    }

    // AddMetrics: accumulate metrics from other stages
    { // baseline metrics (correctly taken from pulse accumulator)
        // collect the subtracted baseline and shift the pulse detection baseline.
        // see the equivalent for the host code in BasecallingMetricsAccumulator::AddBatchMetrics
        // extract subtracted baseline mean and count
        const PBFloat2 subtractedBaseline = Mean(baselinerStats[blockIdx.x].backgroundStats);
        // shift the pulse detector metrics by the subtracted baseline
        // and merge into a working copy of the baseline metrics
        QuickStats pdMetricsStats( pdMetrics[blockIdx.x]);
        QuickStats metricsStats( blockMetrics.baselineStats);
        pdMetricsStats.Shift(subtractedBaseline);
        metricsStats += pdMetricsStats;
        // and transfer the results back to the metrics
        metricsStats.Output(blockMetrics.baselineStats);
     }

    { // Autocorrelation basic metrics (correctly taken from baseliner)
        blockMetrics.traceM0[threadIdx.x] += getWideLoad(
            baselinerStats[blockIdx.x].fullAutocorrState.basicStats.moment0);
        blockMetrics.traceM1[threadIdx.x] += getWideLoad(
            baselinerStats[blockIdx.x].fullAutocorrState.basicStats.moment1);
        blockMetrics.traceM2[threadIdx.x] += getWideLoad(
            baselinerStats[blockIdx.x].fullAutocorrState.basicStats.moment2);
    }

    { // Autocorrelation lag metrics (correctly taken from baseliner)
        auto lag = AutocorrAccumState::lag;
        auto& that = baselinerStats[blockIdx.x].fullAutocorrState;

        uint16_t fbi = blockMetrics.bIdx[threadIdx.x].X();
        uint16_t bbi = blockMetrics.bIdx[threadIdx.x].Y();
        uint16_t that_fbi = that.bIdx[0][threadIdx.x];
        uint16_t that_bbi = that.bIdx[1][threadIdx.x];

        blockMetrics.autocorrM2[threadIdx.x] += getWideLoad(that.moment2);

        auto n1 = lag - that_fbi;  // that fBuf may be not filled up
        for (uint16_t k = 0; k < lag - n1; k++)
        {
            // Sum of muls of overlapping elements
            blockMetrics.autocorrM2[threadIdx.x]      +=
                getWideLoad(that.fBuf[k]) * 
                asFloat2(blockMetrics.bBuf[(bbi+k)%lag][threadIdx.x]);
            // Accept the whole back buffer
            blockMetrics.bBuf[(bbi+k)%lag][threadIdx.x] =
                getWideLoad(that.bBuf[(that_bbi+n1+k)%lag]);
        }

        auto n2 = lag - fbi;      // this fBuf may be not filled up
        for (uint16_t k = 0; k < n2; ++k)
        {
            // No need to adjust m2_ as excessive values were mul by 0
            blockMetrics.fBuf[fbi+k][threadIdx.x] = getWideLoad(that.fBuf[k]);
        }

        // Advance buffer indices
        blockMetrics.bIdx[threadIdx.x] = PBShort2(fbi + n2, bbi + (lag-n1) % lag);
    }

    { // FrameLabeler metrics
        blockMetrics.pulseDetectionScore[threadIdx.x] += getWideLoad(flMetrics[blockIdx.x]);
    }
}

__device__ PBShort2 labelBlock(
        const BasecallingMetricsAccumulatorDevice& blockMetrics,
        const BasecallingMetrics& outMetrics,
        //const DeviceView<const TrainedCartDevice> cartParams,
        float frameRate)
{
    using AnalogVals = Utility::CudaArray<PBHalf2, numAnalogs>;

    const PBHalf2 zeros(0);
    const auto& stdDev = sqrtf(Variance(blockMetrics.baselineStats));
    const PBHalf2& numBases = getWideLoad(outMetrics.numBases);
    const PBHalf2& numPulses = getWideLoad(outMetrics.numPulses);
    const PBHalf2& pulseWidth = replaceNans(
        numPulses / blockMetrics.numPulseFrames[threadIdx.x]);
    const AnalogVals& pkmid = [&blockMetrics]()
    {
        AnalogVals ret;
        for (size_t ai = 0; ai < numAnalogs; ++ai)
        {
            ret[ai] = ToHalf2(replaceNans(blockMetrics.pkMidSignal[ai][threadIdx.x]
                                  / blockMetrics.numPkMidFrames[ai][threadIdx.x]));
        }
        return ret;
    }();

    Utility::CudaArray<PBHalf2, ActivityLabeler::NUM_FEATURES> features;
    for (auto& val : features)
    {
        val = zeros;
    }

    const PBHalf2& seconds = PBHalf2(blockMetrics.numFrames[threadIdx.x]) / frameRate;

    features[ActivityLabeler::PULSERATE] = numPulses / seconds;
    features[ActivityLabeler::SANDWICHRATE] = replaceNans(
        blockMetrics.numSandwiches[threadIdx.x] / numPulses);

    const PBHalf2& hswr = replaceNans(blockMetrics.numHalfSandwiches[threadIdx.x] / numPulses);
    auto hswrExp = evaluatePolynomial(trainedCartParams.hswCurve,
                                      features[ActivityLabeler::PULSERATE]);
    hswrExp = Blend(hswrExp > trainedCartParams.maxAcceptableHalfsandwichRate,
                    PBHalf2(trainedCartParams.maxAcceptableHalfsandwichRate),
                    hswrExp);
    features[ActivityLabeler::LOCALHSWRATENORM] = hswr - hswrExp;

    features[ActivityLabeler::VITERBISCORE] = blockMetrics.pulseDetectionScore[threadIdx.x];
    features[ActivityLabeler::MEANPULSEWIDTH] = pulseWidth;
    features[ActivityLabeler::LABELSTUTTERRATE] = replaceNans(
        blockMetrics.numPulseLabelStutters[threadIdx.x] / numPulses);

    PBShort2 lowAmpIndex(0);
    PBHalf2 minamp(1.0f); // min relative amp, therefore 1.0 is the maximum amplitude
    AnalogVals relamps;
    { // populate the above:
        PBHalf2 maxamp(0.0f);
        for (size_t ai = 0; ai < numAnalogs; ++ai)
        {
            relamps[ai] = blockMetrics.modelMean[ai][threadIdx.x];
            maxamp = max(relamps[ai], maxamp);
        }
        for (size_t ai = 0; ai < numAnalogs; ++ai)
        {
            relamps[ai] /= maxamp;
            const PBShort2 thisAnalogIndex(ai);
            lowAmpIndex = Blend(relamps[ai] < minamp, thisAnalogIndex, lowAmpIndex);
            minamp = min(relamps[ai], minamp);
        }
    }

    for (size_t i = 0; i < numAnalogs; ++i)
    {
        features[ActivityLabeler::BLOCKLOWSNR] += ToHalf2(pkmid[i] / stdDev
                                                * blockMetrics.numBasesByAnalog[i][threadIdx.x] / numBases
                                                * minamp / relamps[i]);
        features[ActivityLabeler::MAXPKMAXNORM] = ToHalf2(max(
            features[ActivityLabeler::MAXPKMAXNORM],
            (blockMetrics.pkMax[i][threadIdx.x] - pkmid[i]) / stdDev));
    }
    features[ActivityLabeler::BLOCKLOWSNR] =
        replaceNans(features[ActivityLabeler::BLOCKLOWSNR]);
    features[ActivityLabeler::MAXPKMAXNORM] =
        replaceNans(features[ActivityLabeler::MAXPKMAXNORM]);
    features[ActivityLabeler::AUTOCORRELATION] =
        getWideLoad(outMetrics.autocorrelation);

    PBHalf2 lowbp(0);
    PBHalf2 lowpk(0);

    for (size_t i = 0; i < numAnalogs; ++i)
    {
        const auto& bpZvar = replaceNans(blockMetrics.bpZvar[i][threadIdx.x]);
        features[ActivityLabeler::BPZVARNORM] += bpZvar;
        lowbp = Blend(lowAmpIndex == i, bpZvar, lowbp);

        const auto& pkZvar = replaceNans(blockMetrics.pkZvar[i][threadIdx.x]);
        features[ActivityLabeler::PKZVARNORM] += pkZvar;
        lowpk = Blend(lowAmpIndex == i, pkZvar, lowpk);
    }

    features[ActivityLabeler::BPZVARNORM] -= lowbp;
    features[ActivityLabeler::BPZVARNORM] /= PBHalf2(3.0f);
    features[ActivityLabeler::PKZVARNORM] -= lowpk;
    features[ActivityLabeler::PKZVARNORM] /= PBHalf2(3.0f);

#ifndef NDEBUG
    for (size_t i = 0; i < features.size(); ++i)
    {
        const auto& nanMask = features[i] == features[i];
        (void)nanMask;
        assert(nanMask.X());
        assert(nanMask.Y());
    }
#endif

    return PBShort2(traverseCart<0>(features), traverseCart<1>(features));
}

__global__ void FinalizeMetrics(
        bool realtimeActivityLabels,
        float frameRate,
        DeviceView<BasecallingMetricsAccumulatorDevice> metrics,
        DeviceView<BasecallingMetrics> outBatchMetrics)
{
    auto& blockMetrics = metrics[blockIdx.x];

    const PBHalf2 nans(std::numeric_limits<half2>::quiet_NaN());
    const PBHalf2 zeros(0.0f);
    const PBFloat2 nansf2(std::numeric_limits<float2>::quiet_NaN());

    for (size_t pulseLabel = 0; pulseLabel < numAnalogs; pulseLabel++)
    {
        const PBHalf2 nf = blockMetrics.numPkMidFrames[pulseLabel][threadIdx.x];
        const PBFloat2 nff = PBFloat2(nf);
        const PBHalf2 baselineVariance = ToHalf2(Variance(blockMetrics.baselineStats));
        const PBFloat2 pkMidSignal(blockMetrics.pkMidSignal[pulseLabel][threadIdx.x]);
        const PBFloat2 pkMidSignalSqr = pkMidSignal * pkMidSignal;

        { // Convert moments to interpulse variance
            const PBHalf2& nb = blockMetrics.numPkMidBasesByAnalog[pulseLabel][threadIdx.x];
            blockMetrics.bpZvar[pulseLabel][threadIdx.x] =
                ToHalf2(
                (blockMetrics.bpZvarAcc[pulseLabel][threadIdx.x]
                 - pkMidSignalSqr / nff)
                / nff);

            // Bessel's correction with num bases, not frames
            blockMetrics.bpZvar[pulseLabel][threadIdx.x] *= nb / (nb - 1.0f);

            blockMetrics.bpZvar[pulseLabel][threadIdx.x] -= baselineVariance / (nf / nb);

            blockMetrics.pkZvar[pulseLabel][threadIdx.x] = ToHalf2(
                (blockMetrics.pkZvarAcc[pulseLabel][threadIdx.x] - (pkMidSignalSqr / nff))
                 / (nff - PBFloat2(1.0f, 1.0f)));
        }

        // pkzvar up to this point contains total signal variance. We
        // subtract out interpulse variance and baseline variance to leave
        // intrapulse variance.
        blockMetrics.pkZvar[pulseLabel][threadIdx.x] -=
            blockMetrics.bpZvar[pulseLabel][threadIdx.x] + baselineVariance;

        const auto& modelIntraVars = blockMetrics.modelVariance[pulseLabel][threadIdx.x];
        // the model intrapulse variance still contains baseline variance,
        // remove before normalizing
        auto mask = (modelIntraVars <= baselineVariance);
        blockMetrics.pkZvar[pulseLabel][threadIdx.x] /= modelIntraVars - baselineVariance;
        mask = mask || (blockMetrics.pkZvar[pulseLabel][threadIdx.x] < PBHalf2(0.0f));
        blockMetrics.pkZvar[pulseLabel][threadIdx.x] = Blend(
                mask, 0.0f, blockMetrics.pkZvar[pulseLabel][threadIdx.x]);


        const auto& pkMid = pkMidSignal / nf;
        mask = (pkMid < PBFloat2(0.0f));
        const auto tmp = PBFloat2(blockMetrics.bpZvar[pulseLabel][threadIdx.x]);
        blockMetrics.bpZvar[pulseLabel][threadIdx.x] = ToHalf2(tmp/(pkMid * pkMid));
        mask = mask || (blockMetrics.bpZvar[pulseLabel][threadIdx.x] < PBHalf2(0.0f));
        blockMetrics.bpZvar[pulseLabel][threadIdx.x] = Blend(
                mask, nans, blockMetrics.bpZvar[pulseLabel][threadIdx.x]);

        { // Set nans as appropriate
            auto emptyMask = (blockMetrics.numPkMidBasesByAnalog[pulseLabel][threadIdx.x] == 0);
            blockMetrics.pkMidSignal[pulseLabel][threadIdx.x] = Blend(
                    emptyMask, nansf2, pkMidSignal);
            emptyMask = emptyMask || (blockMetrics.numPkMidBasesByAnalog[pulseLabel][threadIdx.x] < 2)
                                  || (blockMetrics.numPkMidFrames[pulseLabel][threadIdx.x] < 2);
            blockMetrics.bpZvar[pulseLabel][threadIdx.x] = Blend(
                    emptyMask, nans, blockMetrics.bpZvar[pulseLabel][threadIdx.x]);
            blockMetrics.pkZvar[pulseLabel][threadIdx.x] = Blend(
                    emptyMask, nans, blockMetrics.pkZvar[pulseLabel][threadIdx.x]);
        }
    }

    { // Normalize, and replace nans with zeros
        const PBHalf2 numFrames(blockMetrics.numFrames[threadIdx.x].x,
                                 blockMetrics.numFrames[threadIdx.x].y);
        blockMetrics.pulseDetectionScore[threadIdx.x] = Blend(
                numFrames == 0.0f,
                zeros,
                blockMetrics.pulseDetectionScore[threadIdx.x] / numFrames);
    }

    // Expand the half/short accumulator into float/short output
    BasecallingMetrics& outMetrics = outBatchMetrics[blockIdx.x];

    const auto& indX = threadIdx.x * 2;
    const auto& indY = threadIdx.x * 2 + 1;
    const auto& autocorr = autocorrelation(blockMetrics);
    // Populate the rest of the metrics that weren't populated during operations above
    outMetrics.numPulseFrames[indX] = blockMetrics.numPulseFrames[threadIdx.x].X();
    outMetrics.numPulseFrames[indY] = blockMetrics.numPulseFrames[threadIdx.x].Y();
    outMetrics.numBaseFrames[indX] = blockMetrics.numBaseFrames[threadIdx.x].X();
    outMetrics.numBaseFrames[indY] = blockMetrics.numBaseFrames[threadIdx.x].Y();
    outMetrics.numSandwiches[indX] = blockMetrics.numSandwiches[threadIdx.x].X();
    outMetrics.numSandwiches[indY] = blockMetrics.numSandwiches[threadIdx.x].Y();
    outMetrics.numHalfSandwiches[indX] = blockMetrics.numHalfSandwiches[threadIdx.x].X();
    outMetrics.numHalfSandwiches[indY] = blockMetrics.numHalfSandwiches[threadIdx.x].Y();
    outMetrics.numPulseLabelStutters[indX] = blockMetrics.numPulseLabelStutters[threadIdx.x].X();
    outMetrics.numPulseLabelStutters[indY] = blockMetrics.numPulseLabelStutters[threadIdx.x].Y();
    outMetrics.startFrame[indX] = blockMetrics.startFrame[threadIdx.x].x;
    outMetrics.startFrame[indY] = blockMetrics.startFrame[threadIdx.x].y;
    outMetrics.numFrames[indX] = blockMetrics.numFrames[threadIdx.x].x;
    outMetrics.numFrames[indY] = blockMetrics.numFrames[threadIdx.x].y;
    outMetrics.autocorrelation[indX] = autocorr.X();
    outMetrics.autocorrelation[indY] = autocorr.Y();
    outMetrics.pulseDetectionScore[indX] = blockMetrics.pulseDetectionScore[threadIdx.x].X();
    outMetrics.pulseDetectionScore[indY] = blockMetrics.pulseDetectionScore[threadIdx.x].Y();
    outMetrics.pixelChecksum[indX] = blockMetrics.pixelChecksum[threadIdx.x].X();
    outMetrics.pixelChecksum[indY] = blockMetrics.pixelChecksum[threadIdx.x].Y();

    const auto frameBaselineDWS = Mean(blockMetrics.baselineStats); 
    outMetrics.frameBaselineDWS[indX] = frameBaselineDWS.X();
    outMetrics.frameBaselineDWS[indY] = frameBaselineDWS.Y();
    const auto& var = Variance(blockMetrics.baselineStats);
    outMetrics.frameBaselineVarianceDWS[indX] = var.X();
    outMetrics.frameBaselineVarianceDWS[indY] = var.Y();

    const auto numFramesBaseline = blockMetrics.baselineStats.moment0[threadIdx.x];
    outMetrics.numFramesBaseline[indX] = numFramesBaseline.X();
    outMetrics.numFramesBaseline[indY] = numFramesBaseline.Y();

    outMetrics.numPulses[indX] = 0;
    outMetrics.numPulses[indY] = 0;
    outMetrics.numBases[indX] = 0;
    outMetrics.numBases[indY] = 0;
    for (size_t a = 0; a < numAnalogs; ++a)
    {
        outMetrics.numPulses[indX] += blockMetrics.numPulsesByAnalog[a][threadIdx.x].X();
        outMetrics.numPulses[indY] += blockMetrics.numPulsesByAnalog[a][threadIdx.x].Y();
        outMetrics.numPulsesByAnalog[a][indX] = blockMetrics.numPulsesByAnalog[a][threadIdx.x].X();
        outMetrics.numPulsesByAnalog[a][indY] = blockMetrics.numPulsesByAnalog[a][threadIdx.x].Y();

        outMetrics.numBases[indX] += blockMetrics.numBasesByAnalog[a][threadIdx.x].X();
        outMetrics.numBases[indY] += blockMetrics.numBasesByAnalog[a][threadIdx.x].Y();
        outMetrics.numBasesByAnalog[a][indX] = blockMetrics.numBasesByAnalog[a][threadIdx.x].X();
        outMetrics.numBasesByAnalog[a][indY] = blockMetrics.numBasesByAnalog[a][threadIdx.x].Y();

        outMetrics.pkMidSignal[a][indX] = blockMetrics.pkMidSignal[a][threadIdx.x].X();
        outMetrics.pkMidSignal[a][indY] = blockMetrics.pkMidSignal[a][threadIdx.x].Y();
        outMetrics.bpZvar[a][indX] = blockMetrics.bpZvar[a][threadIdx.x].X();
        outMetrics.bpZvar[a][indY] = blockMetrics.bpZvar[a][threadIdx.x].Y();
        outMetrics.pkZvar[a][indX] = blockMetrics.pkZvar[a][threadIdx.x].X();
        outMetrics.pkZvar[a][indY] = blockMetrics.pkZvar[a][threadIdx.x].Y();
        outMetrics.pkMax[a][indX] = blockMetrics.pkMax[a][threadIdx.x].X();
        outMetrics.pkMax[a][indY] = blockMetrics.pkMax[a][threadIdx.x].Y();
        outMetrics.numPkMidFrames[a][indX] = blockMetrics.numPkMidFrames[a][threadIdx.x].X();
        outMetrics.numPkMidFrames[a][indY] = blockMetrics.numPkMidFrames[a][threadIdx.x].Y();
        outMetrics.numPkMidBasesByAnalog[a][indX] = blockMetrics.numPkMidBasesByAnalog[a][threadIdx.x].X();
        outMetrics.numPkMidBasesByAnalog[a][indY] = blockMetrics.numPkMidBasesByAnalog[a][threadIdx.x].Y();
    }

    if (realtimeActivityLabels)
    {
        PBShort2 activityLabels = labelBlock(blockMetrics, outMetrics, frameRate);
        outMetrics.activityLabel[indX] = static_cast<HQRFPhysicalStates>(activityLabels.X());
        outMetrics.activityLabel[indY] = static_cast<HQRFPhysicalStates>(activityLabels.Y());
    }
}

} // anonymous

class HFMetricsFilterDevice::AccumImpl
{
public:
    AccumImpl(size_t lanesPerPool, StashableAllocRegistrar* registrar)
        : metrics_(registrar, SOURCE_MARKER(), lanesPerPool)
        , framesSeen_(0)
        , lanesPerBatch_(lanesPerPool)
    { };

    std::unique_ptr<UnifiedCudaArray<Data::BasecallingMetrics>>
    Process(const Data::PulseBatch& pulseBatch,
            const Data::BaselinerMetrics& baselinerMetrics,
            const UnifiedCudaArray<Data::LaneModelParameters<Cuda::PBHalf, laneSize>>& models,
            const Data::FrameLabelerMetrics& flMetrics,
            const Data::PulseDetectorMetrics& pdMetrics)
    {
        if (framesSeen_ == 0)
        {
            const auto& initLauncher = Cuda::PBLauncher(InitializeMetrics,
                                                        lanesPerBatch_,
                                                        threadsPerBlock_);
            initLauncher(metrics_, !initialized_);
            initialized_ = true;
        }
        const auto& processLauncher = Cuda::PBLauncher(ProcessChunk,
                                                       lanesPerBatch_,
                                                       threadsPerBlock_);
        processLauncher(baselinerMetrics.baselinerStats,
                        models,
                        pulseBatch.Pulses(),
                        flMetrics.viterbiScore,
                        pdMetrics.baselineStats,
                        pulseBatch.Dims().framesPerBatch,
                        metrics_);
        framesSeen_ += pulseBatch.Dims().framesPerBatch;

        if (pulseBatch.GetMeta().LastFrame() % framesPerHFMetricBlock_ < pulseBatch.Dims().framesPerBatch)
        {
            auto ret = metricsFactory_->NewBatch(pulseBatch.Dims().lanesPerBatch);
            const auto& finalizeLauncher = Cuda::PBLauncher(
                    FinalizeMetrics,
                    lanesPerBatch_,
                    threadsPerBlock_);
            finalizeLauncher(realtimeActivityLabels_,
                             static_cast<float>(frameRate_),
                             metrics_,
                             *(ret.get()));
            framesSeen_ = 0;
            Cuda::CudaSynchronizeDefaultStream();
            return ret;
        }
        Cuda::CudaSynchronizeDefaultStream();
        return std::unique_ptr<UnifiedCudaArray<Data::BasecallingMetrics>>();
    }

private:
    static constexpr size_t threadsPerBlock_ = laneSize / 2;

private:
    DeviceOnlyArray<BasecallingMetricsAccumulatorDevice> metrics_;
    uint32_t framesSeen_;
    uint32_t lanesPerBatch_;
    bool initialized_ = false;

};

constexpr size_t HFMetricsFilterDevice::AccumImpl::threadsPerBlock_;

HFMetricsFilterDevice::HFMetricsFilterDevice(uint32_t poolId,
                                             uint32_t lanesPerPool,
                                             StashableAllocRegistrar* registrar)
    : HFMetricsFilter(poolId)
    , impl_(std::make_unique<AccumImpl>(lanesPerPool, registrar))
{ };

std::unique_ptr<UnifiedCudaArray<Data::BasecallingMetrics>>
HFMetricsFilterDevice::Process(
        const Data::PulseBatch& pulseBatch,
        const Data::BaselinerMetrics& baselinerMetrics,
        const UnifiedCudaArray<Data::LaneModelParameters<Cuda::PBHalf, 64>>& models,
        const Data::FrameLabelerMetrics& flMetrics,
        const Data::PulseDetectorMetrics& pdMetrics)
{
    return impl_->Process(pulseBatch, baselinerMetrics, models, flMetrics,
            pdMetrics);
}

void HFMetricsFilterDevice::Configure(uint32_t sandwichTolerance,
                                      uint32_t framesPerHFMetricBlock,
                                      double frameRate,
                                      bool realtimeActivityLabels)
{
    HFMetricsFilter::Configure(sandwichTolerance,
                               framesPerHFMetricBlock,
                               frameRate,
                               realtimeActivityLabels,
                               false);
    auto trainedCartHost = TrainedCartDevice::PopulatedModel();
    CudaCopyToSymbol(&trainedCartParams, &trainedCartHost);
}

}}} // PacBio::Mongo::Basecaller
