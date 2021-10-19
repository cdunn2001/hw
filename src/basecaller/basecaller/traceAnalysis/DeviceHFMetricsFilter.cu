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
/// \file   DeviceHFMetricsFilter.cu
/// \brief  A filter for computing or aggregating trace- and pulse-metrics
///         on a time scale equal to or greater than the standard block size.


#include "DeviceHFMetricsFilter.h"
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

DeviceHFMetricsFilter::~DeviceHFMetricsFilter() = default;

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
    AnalogMetric<float2> pkMidSignal;
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
    AnalogMetric<float2> bpZvarAcc;
    AnalogMetric<float2> pkZvarAcc;

    // The baseline stat accumulator:
    SingleMetric<float2> baselineM0;
    SingleMetric<float2>  baselineM1;
    SingleMetric<float2>  baselineM2;

    // The autocorrelation accumulator:
    SingleMetric<PBHalf2> traceM0;
    SingleMetric<PBHalf2> traceM1;
    SingleMetric<float2>  traceM2;
    SingleMetric<PBHalf2> autocorrM1First;
    SingleMetric<PBHalf2> autocorrM1Last;
    SingleMetric<float2>  autocorrM2;

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

__device__ PBHalf2 variance(const PBHalf2 M0, const PBHalf2 M1, const float2 M2)
{
    return PBHalf2(variance(M0.FloatX(), M1.FloatX(), M2.x), variance(M0.FloatY(), M1.FloatY(), M2.y));
}

__device__ uint2 getWideLoad(const Cuda::Utility::CudaArray<uint16_t, laneSize>& load)
{ return make_uint2(load[threadIdx.x * 2], load[threadIdx.x * 2 + 1]); };

__device__ float2 getWideLoad(const Cuda::Utility::CudaArray<float, laneSize>& load)
{ return make_float2(load[threadIdx.x * 2], load[threadIdx.x * 2 + 1]); };

__device__ PBHalf2 replaceNans(PBHalf2 vals)
{ return Blend(vals == vals, vals, PBHalf2(0.0)); };

__device__ float2& operator+=(float2& l, const float2 r)
{
    l.x += r.x;
    l.y += r.y;
    return l;
}

__device__ uint2 operator+(uint2 l, uint2 r)
{ return make_uint2(l.x - r.x, l.y - r.y); }

__device__ float2 operator-(float2 l, float2 r)
{ return make_float2(l.x - r.x, l.y - r.y); }

__device__ float2 operator*(float2 l, float2 r)
{ return make_float2(l.x * r.x, l.y * r.y); }

__device__ float2 operator/(float2 l, float2 r)
{ return make_float2(l.x / r.x, l.y / r.y); }

__device__ float2 asFloat2(PBHalf2 val)
{ return make_float2(val.FloatX(), val.FloatY()); }

template<int id>
__device__ float2 blendFloat0(float val)
{
    static_assert(id < 2, "Out of bounds access in blendFloat0");
    if (id == 0) return make_float2(val, 0.0f);
    else return make_float2(0.0f, val);
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

__device__ PBHalf2 autocorrelation(const BasecallingMetricsAccumulatorDevice& blockMetrics,
                                   uint32_t lag)
{
    PBHalf2 nans(std::numeric_limits<half2>::quiet_NaN());
    const auto& nmk = blockMetrics.traceM0[threadIdx.x] - PBHalf2(lag);
    // math in float2 for additional range
    PBHalf2 ac = [&blockMetrics](float2 nmk)
    {
        auto ac = asFloat2(blockMetrics.autocorrM1First[threadIdx.x])
                  * asFloat2(blockMetrics.autocorrM1Last[threadIdx.x])
                  / nmk;
        ac = (asFloat2(blockMetrics.autocorrM2[threadIdx.x]) - ac)
             / (nmk * asFloat2(variance(blockMetrics.traceM0[threadIdx.x],
                                        blockMetrics.traceM1[threadIdx.x],
                                        blockMetrics.traceM2[threadIdx.x])));
        return ac;
    }(asFloat2(nmk));
    const PBBool2 nanMask = !(ac == ac);
    ac = max(ac, -1.0f);
    ac = min(ac, 1.0f);
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

    if (initialize)
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

    blockMetrics.baselineM0[threadIdx.x] = zero;
    blockMetrics.baselineM1[threadIdx.x] = zero;
    blockMetrics.baselineM2[threadIdx.x] = zero;

    blockMetrics.traceM0[threadIdx.x] = 0.0f;
    blockMetrics.traceM1[threadIdx.x] = 0.0f;
    blockMetrics.traceM2[threadIdx.x] = zero;

    blockMetrics.autocorrM1First[threadIdx.x] = 0.0f;
    blockMetrics.autocorrM1Last[threadIdx.x] = 0.0f;
    blockMetrics.autocorrM2[threadIdx.x] = zero;

    for (size_t a = 0; a < numAnalogs; ++a)
    {
        blockMetrics.pkMidSignal[a][threadIdx.x] = zero;
        blockMetrics.bpZvarAcc[a][threadIdx.x] = zero;
        blockMetrics.pkZvarAcc[a][threadIdx.x] = zero;
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
            blockMetrics.pkMidSignal[label][threadIdx.x] += blendFloat0<id>(midSignal * midWidth);
            blockMetrics.bpZvarAcc[label][threadIdx.x] += blendFloat0<id>(midSignal * midSignal * midWidth);
            blockMetrics.pkZvarAcc[label][threadIdx.x] += blendFloat0<id>(pulse->SignalM2());
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
        const DeviceView<const DeviceHFMetricsFilter::BaselinerStatsT> baselinerStats,
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
        blockMetrics.baselineM0[threadIdx.x] += getWideLoad(pdMetrics[blockIdx.x].moment0);
        blockMetrics.baselineM1[threadIdx.x] += getWideLoad(pdMetrics[blockIdx.x].moment1);
        blockMetrics.baselineM2[threadIdx.x] += getWideLoad(pdMetrics[blockIdx.x].moment2);
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
        blockMetrics.autocorrM1First[threadIdx.x] += getWideLoad(
            baselinerStats[blockIdx.x].fullAutocorrState.moment1First);
        blockMetrics.autocorrM1Last[threadIdx.x] += getWideLoad(
            baselinerStats[blockIdx.x].fullAutocorrState.moment1Last);
        blockMetrics.autocorrM2[threadIdx.x] += getWideLoad(
            baselinerStats[blockIdx.x].fullAutocorrState.moment2);
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

    const auto& stdDev = sqrt(variance(blockMetrics.baselineM0[threadIdx.x],
                                       blockMetrics.baselineM1[threadIdx.x],
                                       blockMetrics.baselineM2[threadIdx.x]));
    const PBHalf2& numBases = getWideLoad(outMetrics.numBases);
    const PBHalf2& numPulses = getWideLoad(outMetrics.numPulses);
    const PBHalf2& pulseWidth = replaceNans(
        numPulses / blockMetrics.numPulseFrames[threadIdx.x]);
    const AnalogVals& pkmid = [&blockMetrics]()
    {
        AnalogVals ret;
        for (size_t ai = 0; ai < numAnalogs; ++ai)
        {
            ret[ai] = replaceNans(blockMetrics.pkMidSignal[ai][threadIdx.x]
                                  / blockMetrics.numPkMidFrames[ai][threadIdx.x]);
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
        features[ActivityLabeler::BLOCKLOWSNR] += pkmid[i] / stdDev
                                                * blockMetrics.numBasesByAnalog[i][threadIdx.x] / numBases
                                                * minamp / relamps[i];
        features[ActivityLabeler::MAXPKMAXNORM] = max(
            features[ActivityLabeler::MAXPKMAXNORM],
            (blockMetrics.pkMax[i][threadIdx.x] - pkmid[i]) / stdDev);
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
        uint32_t autocorrAccumLag,
        bool realtimeActivityLabels,
        float frameRate,
        DeviceView<BasecallingMetricsAccumulatorDevice> metrics,
        DeviceView<BasecallingMetrics> outBatchMetrics)
{
    auto& blockMetrics = metrics[blockIdx.x];

    const PBHalf2 nans(std::numeric_limits<half2>::quiet_NaN());
    const PBHalf2 zeros(0.0f);
    const float2 nansf(std::numeric_limits<float2>::quiet_NaN());

    for (size_t pulseLabel = 0; pulseLabel < numAnalogs; pulseLabel++)
    {
        const PBHalf2 nf = blockMetrics.numPkMidFrames[pulseLabel][threadIdx.x];
        const float2 nff = asFloat2(nf);
        const PBHalf2 baselineVariance = variance(blockMetrics.baselineM0[threadIdx.x],
                                                  blockMetrics.baselineM1[threadIdx.x],
                                                  blockMetrics.baselineM2[threadIdx.x]);
        const float2 pkMidSignal(blockMetrics.pkMidSignal[pulseLabel][threadIdx.x]);
        const float2 pkMidSignalSqr = asFloat2(pkMidSignal) * asFloat2(pkMidSignal);

        { // Convert moments to interpulse variance
            const PBHalf2& nb = blockMetrics.numPkMidBasesByAnalog[pulseLabel][threadIdx.x];
            blockMetrics.bpZvar[pulseLabel][threadIdx.x] =
                (blockMetrics.bpZvarAcc[pulseLabel][threadIdx.x]
                 - pkMidSignalSqr / nff)
                / nff;

            // Bessel's correction with num bases, not frames
            blockMetrics.bpZvar[pulseLabel][threadIdx.x] *= nb / (nb - 1.0f);

            blockMetrics.bpZvar[pulseLabel][threadIdx.x] -= baselineVariance / (nf / nb);

            blockMetrics.pkZvar[pulseLabel][threadIdx.x] =
                (blockMetrics.pkZvarAcc[pulseLabel][threadIdx.x] - (pkMidSignalSqr / nff))
                 / (nff - make_float2(1.0f, 1.0f));
        }

        // pkzvar up to this point contains total signal variance. We
        // subtract out interpulse variance and baseline variance to leave
        // intrapulse variance.
        blockMetrics.pkZvar[pulseLabel][threadIdx.x] -=
            blockMetrics.bpZvar[pulseLabel][threadIdx.x] + baselineVariance;

        const auto& modelIntraVars = blockMetrics.modelVariance[pulseLabel][threadIdx.x];
        // the model intrapulse variance still contains baseline variance,
        // remove before normalizing
        auto mask = (modelIntraVars < baselineVariance);
        blockMetrics.pkZvar[pulseLabel][threadIdx.x] /= modelIntraVars - baselineVariance;
        mask = mask || (blockMetrics.pkZvar[pulseLabel][threadIdx.x] < 0.0f);
        blockMetrics.pkZvar[pulseLabel][threadIdx.x] = Blend(
                mask, 0.0f, blockMetrics.pkZvar[pulseLabel][threadIdx.x]);


        const auto& pkMid = pkMidSignal / nf;
        mask = (pkMid < 0);
        blockMetrics.bpZvar[pulseLabel][threadIdx.x] /= (pkMid * pkMid);
        mask = mask || (blockMetrics.bpZvar[pulseLabel][threadIdx.x] < 0.0f);
        blockMetrics.bpZvar[pulseLabel][threadIdx.x] = Blend(
                mask, nans, blockMetrics.bpZvar[pulseLabel][threadIdx.x]);

        { // Set nans as appropriate
            auto emptyMask = (blockMetrics.numPkMidBasesByAnalog[pulseLabel][threadIdx.x] == 0);
            blockMetrics.pkMidSignal[pulseLabel][threadIdx.x] = Blend(
                    emptyMask, nansf, pkMidSignal);
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
    const auto& autocorr = autocorrelation(blockMetrics, autocorrAccumLag);
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

    outMetrics.frameBaselineDWS[indX] = blockMetrics.baselineM1[threadIdx.x].x / blockMetrics.baselineM0[threadIdx.x].x;
    outMetrics.frameBaselineDWS[indY] = blockMetrics.baselineM1[threadIdx.x].y / blockMetrics.baselineM0[threadIdx.x].y;
    const auto& var = variance(blockMetrics.baselineM0[threadIdx.x],
                               blockMetrics.baselineM1[threadIdx.x],
                               blockMetrics.baselineM2[threadIdx.x]);
    outMetrics.frameBaselineVarianceDWS[indX] = var.x;
    outMetrics.frameBaselineVarianceDWS[indY] = var.y;

    outMetrics.numFramesBaseline[indX] = blockMetrics.baselineM0[threadIdx.x].x;
    outMetrics.numFramesBaseline[indY] = blockMetrics.baselineM0[threadIdx.x].y;

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

        outMetrics.pkMidSignal[a][indX] = blockMetrics.pkMidSignal[a][threadIdx.x].x;
        outMetrics.pkMidSignal[a][indY] = blockMetrics.pkMidSignal[a][threadIdx.x].y;
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

class DeviceHFMetricsFilter::AccumImpl
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

        if (framesSeen_ >= framesPerHFMetricBlock_)
        {
            auto ret = metricsFactory_->NewBatch(pulseBatch.Dims().lanesPerBatch);
            const auto& finalizeLauncher = Cuda::PBLauncher(
                    FinalizeMetrics,
                    lanesPerBatch_,
                    threadsPerBlock_);
            finalizeLauncher(AutocorrAccumState::lag,
                             realtimeActivityLabels_,
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

constexpr size_t DeviceHFMetricsFilter::AccumImpl::threadsPerBlock_;

DeviceHFMetricsFilter::DeviceHFMetricsFilter(uint32_t poolId,
                                             uint32_t lanesPerPool,
                                             StashableAllocRegistrar* registrar)
    : HFMetricsFilter(poolId)
    , impl_(std::make_unique<AccumImpl>(lanesPerPool, registrar))
{ };

std::unique_ptr<UnifiedCudaArray<Data::BasecallingMetrics>>
DeviceHFMetricsFilter::Process(
        const Data::PulseBatch& pulseBatch,
        const Data::BaselinerMetrics& baselinerMetrics,
        const UnifiedCudaArray<Data::LaneModelParameters<Cuda::PBHalf, 64>>& models,
        const Data::FrameLabelerMetrics& flMetrics,
        const Data::PulseDetectorMetrics& pdMetrics)
{
    return impl_->Process(pulseBatch, baselinerMetrics, models, flMetrics,
            pdMetrics);
}

void DeviceHFMetricsFilter::Configure(uint32_t sandwichTolerance,
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
    CudaCopyToSymbol(trainedCartParams, &trainedCartHost);
}

}}} // PacBio::Mongo::Basecaller
