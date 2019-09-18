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
/// \file   DeviceHFMetricsFilter.cpp
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
    {
        // Everything else will be set to zero in InitializeMetrics, but these
        // two matter from the very beginning (
        for (int i = 0; i < blockDim.x; ++i)
        {
            startFrame[i] = make_uint2(0,0);
            numFrames[i] = make_uint2(0,0);
        }
    }

public: // metrics
    SingleMetric<PBShort2> numPulseFrames;
    SingleMetric<PBShort2> numBaseFrames;
    SingleMetric<PBShort2> numSandwiches;
    SingleMetric<PBShort2> numHalfSandwiches;
    SingleMetric<PBShort2> numPulseLabelStutters;
    SingleMetric<PBShort2> activityLabel;
    AnalogMetric<PBHalf2> pkMidSignal;
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
    SingleMetric<PBHalf2> baselineM0;
    SingleMetric<PBHalf2> baselineM1;
    SingleMetric<float2> baselineM2;

    // The autocorrelation accumulator:
    SingleMetric<PBHalf2> traceM0;
    SingleMetric<PBHalf2> traceM1;
    SingleMetric<float2> traceM2;
    SingleMetric<PBHalf2> autocorrM1First;
    SingleMetric<PBHalf2> autocorrM1Last;
    SingleMetric<PBHalf2> autocorrM2;

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

__device__ PBHalf2 variance(const PBHalf2 M0, const PBHalf2 M1, const float2 M2)
{
    return PBHalf2(variance(M0.FloatX(), M1.FloatX(), M2.x), variance(M0.FloatY(), M1.FloatY(), M2.y));
}

__device__ float2& operator+=(float2& l, const PBHalf2 r)
{
    l.x += r.FloatX();
    l.y += r.FloatY();
    return l;
}

__device__ float2 operator-(float2 l, float2 r)
{ return make_float2(l.x - r.x, l.y - r.y); }

__device__ float2 operator*(float2 l, float2 r)
{ return make_float2(l.x * r.x, l.y * r.y); }

__device__ float2 operator/(float2 l, float2 r)
{ return make_float2(l.x / r.x, l.y / r.y); }

__device__ float2 asFloat2(PBHalf2 val)
{ return make_float2(val.FloatX(), val.FloatY()); }

__device__ PBHalf2 asPBHalf2(float2 val)
{ return PBHalf2(val.x, val.y); }

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
        return asPBHalf2(ac);
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
        Cuda::Memory::DeviceView<BasecallingMetricsAccumulatorDevice> metrics)
{
    BasecallingMetricsAccumulatorDevice& blockMetrics = metrics[blockIdx.x];

    blockMetrics.startFrame[threadIdx.x].x = blockMetrics.startFrame[threadIdx.x].x
                                           + blockMetrics.numFrames[threadIdx.x].x;
    blockMetrics.startFrame[threadIdx.x].y = blockMetrics.startFrame[threadIdx.x].y
                                           + blockMetrics.numFrames[threadIdx.x].y;
    blockMetrics.numFrames[threadIdx.x].x = 0;
    blockMetrics.numFrames[threadIdx.x].y = 0;

    blockMetrics.numPulseFrames[threadIdx.x] = 0;
    blockMetrics.numBaseFrames[threadIdx.x] = 0;
    blockMetrics.numSandwiches[threadIdx.x] = 0;
    blockMetrics.numHalfSandwiches[threadIdx.x] = 0;
    blockMetrics.numPulseLabelStutters[threadIdx.x] = 0;
    blockMetrics.activityLabel[threadIdx.x] = 0;
    blockMetrics.pixelChecksum[threadIdx.x] = 0;
    blockMetrics.pulseDetectionScore[threadIdx.x] = 0.0f;

    blockMetrics.baselineM0[threadIdx.x] = 0.0f;
    blockMetrics.baselineM1[threadIdx.x] = 0.0f;
    blockMetrics.baselineM2[threadIdx.x].x = 0.0f;
    blockMetrics.baselineM2[threadIdx.x].y = 0.0f;

    blockMetrics.traceM0[threadIdx.x] = 0.0f;
    blockMetrics.traceM1[threadIdx.x] = 0.0f;
    blockMetrics.traceM2[threadIdx.x].x = 0.0f;
    blockMetrics.traceM2[threadIdx.x].y = 0.0f;

    blockMetrics.autocorrM1First[threadIdx.x] = 0.0f;
    blockMetrics.autocorrM1Last[threadIdx.x] = 0.0f;
    blockMetrics.autocorrM2[threadIdx.x] = 0.0f;

    for (size_t a = 0; a < numAnalogs; ++a)
    {
        blockMetrics.pkMidSignal[a][threadIdx.x] = 0.0f;
        blockMetrics.bpZvarAcc[a][threadIdx.x].x = 0.0f;
        blockMetrics.bpZvarAcc[a][threadIdx.x].y = 0.0f;
        blockMetrics.pkZvarAcc[a][threadIdx.x].x = 0.0f;
        blockMetrics.pkZvarAcc[a][threadIdx.x].y = 0.0f;
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

    // Aren't accumulators / Don't need initialization:
    //blockMetrics.modelVariance = 0.0f;
    //blockMetrics.modelMean = 0.0f;
    //blockMetrics.bpZvar[a][threadIdx.x] = 0.0f;
    //blockMetrics.pkZvar[a][threadIdx.x] = 0.0f;
}

__global__ void ProcessChunk(
        const Cuda::Memory::DeviceView<const DeviceHFMetricsFilter::BaselinerStatsT> baselinerStats,
        const Cuda::Memory::DeviceView<const Data::LaneModelParameters<Cuda::PBHalf2, laneSize/2>> models,
        const Data::GpuBatchVectors<const Data::Pulse> pulses,
        const Cuda::Memory::DeviceView<const PacBio::Cuda::Utility::CudaArray<float, laneSize>> flMetrics,
        const Cuda::Memory::DeviceView<const PacBio::Mongo::StatAccumState> pdMetrics,
        uint32_t numFrames,
        Cuda::Memory::DeviceView<BasecallingMetricsAccumulatorDevice> metrics)
{
    // AddPulses: loop over the pulses in each zmw

    BasecallingMetricsAccumulatorDevice& blockMetrics = metrics[blockIdx.x];

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

    const PBBool2 maskX(1, 0);
    const PBBool2 maskY(0, 1);
    const PBShort2 incX(1, 0);
    const PBShort2 incY(0, 1);
    const PBShort2 inc(1, 1);

    // Shove the parts with a lot of branching into lambdas.
    // Results are generally saved to analog specific vectors, so there isn't much to be gained by "vectorizing"
    auto stutterSandwich = [&blockMetrics](const Pulse* pulse,
                                           const Pulse* prevPulse,
                                           const Pulse* prevprevPulse,
                                           const PBShort2 inc) -> void
    {
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

    auto goodBaseMetricsX = [&blockMetrics](const Pulse* pulse) -> void
    {
        if (!pulse->IsReject())
        {
            const auto& label = static_cast<uint8_t>(pulse->Label());
            blockMetrics.numBaseFrames[threadIdx.x].X(
                blockMetrics.numBaseFrames[threadIdx.x].X() + pulse->Width());
            blockMetrics.numBasesByAnalog[label][threadIdx.x].X(
                blockMetrics.numBasesByAnalog[label][threadIdx.x].X() + 1);

            if (!isnan(pulse->MidSignal()))
            {
                const auto& midWidth = pulse->Width() - 2;
                const auto& midSignal = pulse->MidSignal();
                blockMetrics.numPkMidBasesByAnalog[label][threadIdx.x].X(
                    blockMetrics.numPkMidBasesByAnalog[label][threadIdx.x].X() + 1);
                blockMetrics.numPkMidFrames[label][threadIdx.x].X(
                    blockMetrics.numPkMidFrames[label][threadIdx.x].X() + midWidth);
                blockMetrics.pkMidSignal[label][threadIdx.x].X(
                    blockMetrics.pkMidSignal[label][threadIdx.x].X()
                    + static_cast<half>(midSignal * midWidth));
                blockMetrics.bpZvarAcc[label][threadIdx.x].x =
                    blockMetrics.bpZvarAcc[label][threadIdx.x].x + midSignal * midSignal * midWidth;
                blockMetrics.pkZvarAcc[label][threadIdx.x].x =
                    blockMetrics.pkZvarAcc[label][threadIdx.x].x + pulse->SignalM2();
            }
        }
    };

    auto goodBaseMetricsY = [&blockMetrics](const Pulse* pulse) -> void
    {
        if (!pulse->IsReject())
        {
            const auto& label = static_cast<uint8_t>(pulse->Label());
            blockMetrics.numBaseFrames[threadIdx.x].Y(
                blockMetrics.numBaseFrames[threadIdx.x].Y() + pulse->Width());
            blockMetrics.numBasesByAnalog[label][threadIdx.x].Y(
                blockMetrics.numBasesByAnalog[label][threadIdx.x].Y() + 1);

            if (!isnan(pulse->MidSignal()))
            {
                const auto& midWidth = pulse->Width() - 2;
                const auto& midSignal = pulse->MidSignal();
                blockMetrics.numPkMidBasesByAnalog[label][threadIdx.x].Y(
                    blockMetrics.numPkMidBasesByAnalog[label][threadIdx.x].Y() + 1);
                blockMetrics.numPkMidFrames[label][threadIdx.x].Y(
                    blockMetrics.numPkMidFrames[label][threadIdx.x].Y() + midWidth);
                blockMetrics.pkMidSignal[label][threadIdx.x].Y(
                    blockMetrics.pkMidSignal[label][threadIdx.x].Y()
                    + static_cast<half>(midSignal * midWidth));
                blockMetrics.bpZvarAcc[label][threadIdx.x].y =
                    blockMetrics.bpZvarAcc[label][threadIdx.x].y + midSignal * midSignal * midWidth;
                blockMetrics.pkZvarAcc[label][threadIdx.x].y =
                    blockMetrics.pkZvarAcc[label][threadIdx.x].y + pulse->SignalM2();
            }
        }
    };

    auto processPulseX = [&blockMetrics, &stutterSandwich, &goodBaseMetricsX](
            const Pulse* pulse,
            const Pulse* prevPulse,
            const Pulse* prevprevPulse)
    {
        const auto& label = static_cast<uint8_t>(pulse->Label());
        blockMetrics.numPulseFrames[threadIdx.x].X(
            blockMetrics.numPulseFrames[threadIdx.x].X() + pulse->Width());
        blockMetrics.numPulsesByAnalog[label][threadIdx.x].X(
            blockMetrics.numPulsesByAnalog[label][threadIdx.x].X() + 1);
        blockMetrics.pkMax[label][threadIdx.x].X(
               max(blockMetrics.pkMax[label][threadIdx.x].X(), pulse->MaxSignal()));

        stutterSandwich(pulse, prevPulse, prevprevPulse, PBShort2(1, 0));

        goodBaseMetricsX(pulse);
    };

    auto processPulseY = [&blockMetrics, &stutterSandwich, &goodBaseMetricsY](
            const Pulse* pulse,
            const Pulse* prevPulse,
            const Pulse* prevprevPulse)
    {
        const auto& label = static_cast<uint8_t>(pulse->Label());
        blockMetrics.numPulseFrames[threadIdx.x].Y(
            blockMetrics.numPulseFrames[threadIdx.x].Y() + pulse->Width());
        blockMetrics.numPulsesByAnalog[label][threadIdx.x].Y(
            blockMetrics.numPulsesByAnalog[label][threadIdx.x].Y() + 1);
        blockMetrics.pkMax[label][threadIdx.x].Y(
               max(blockMetrics.pkMax[label][threadIdx.x].Y(), pulse->MaxSignal()));

        stutterSandwich(pulse, prevPulse, prevprevPulse, PBShort2(0, 1));

        goodBaseMetricsY(pulse);
    };

    uint32_t numPulsesX = pulsesX.size();
    uint32_t numPulsesY = pulsesY.size();
    auto iterCount = min(numPulsesX, numPulsesY);
    // Iterate over both as long as possible:
    for (uint32_t pIdx = 0; pIdx < iterCount; ++pIdx)
    {
        const Pulse* pulseX = &pulsesX[pIdx];
        const Pulse* pulseY = &pulsesY[pIdx];
        const uint8_t labelX = static_cast<uint8_t>(pulseX->Label());
        const uint8_t labelY = static_cast<uint8_t>(pulseY->Label());

        blockMetrics.numPulseFrames[threadIdx.x] += PBShort2(pulseX->Width(), pulseY->Width());
        blockMetrics.numPulsesByAnalog[labelX][threadIdx.x] += incX;
        blockMetrics.numPulsesByAnalog[labelY][threadIdx.x] += incY;

        const PBShort2 thatPkMax(blockMetrics.pkMax[labelX][threadIdx.x].X(),
                                 blockMetrics.pkMax[labelY][threadIdx.x].Y());
        const PBShort2 thisPkMax(pulseX->MaxSignal(), pulseY->MaxSignal());
        const PBShort2 pkMax = max(thatPkMax, thisPkMax);
        blockMetrics.pkMax[labelX][threadIdx.x].X(pkMax.X());
        blockMetrics.pkMax[labelY][threadIdx.x].Y(pkMax.Y());

        stutterSandwich(pulseX, prevPulseX, prevprevPulseX, incX);
        stutterSandwich(pulseY, prevPulseY, prevprevPulseY, incY);

        goodBaseMetricsX(pulseX);
        goodBaseMetricsY(pulseY);

        prevprevPulseX = prevPulseX;
        prevPulseX = pulseX;
        prevprevPulseY = prevPulseY;
        prevPulseY = pulseY;
    }
    // Only one of the two loops below will happen:
    for (uint32_t pIdx = iterCount; pIdx < numPulsesX; ++pIdx)
    {
        const Pulse* pulseX = &pulsesX[pIdx];
        processPulseX(pulseX, prevPulseX, prevprevPulseX);
        prevprevPulseX = prevPulseX;
        prevPulseX = pulseX;
    }
    for (uint32_t pIdx = iterCount; pIdx < numPulsesY; ++pIdx)
    {
        const Pulse* pulseY = &pulsesY[pIdx];
        processPulseY(pulseY, prevPulseY, prevprevPulseY);
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
    { // baseline metrics (currently erroneously taken from the baseliner,
      // should be taken from pulse accumulator, but those aren't available yet
        PBHalf2 tmpM0(baselinerStats[blockIdx.x].baselineStats.moment0[threadIdx.x * 2],
                      baselinerStats[blockIdx.x].baselineStats.moment0[threadIdx.x * 2 + 1]);
        blockMetrics.baselineM0[threadIdx.x] += tmpM0;
        PBHalf2 tmpM1(baselinerStats[blockIdx.x].baselineStats.moment1[threadIdx.x * 2],
                      baselinerStats[blockIdx.x].baselineStats.moment1[threadIdx.x * 2 + 1]);
        blockMetrics.baselineM1[threadIdx.x] += tmpM1;
        PBHalf2 tmpM2(baselinerStats[blockIdx.x].baselineStats.moment2[threadIdx.x * 2],
                      baselinerStats[blockIdx.x].baselineStats.moment2[threadIdx.x * 2 + 1]);
        blockMetrics.baselineM2[threadIdx.x] += tmpM2;
    }

    { // Autocorrelation basic metrics (correctly taken from baseliner)
        PBHalf2 tmpM0(baselinerStats[blockIdx.x].fullAutocorrState.basicStats.moment0[threadIdx.x * 2],
                      baselinerStats[blockIdx.x].fullAutocorrState.basicStats.moment0[threadIdx.x * 2 + 1]);
        blockMetrics.traceM0[threadIdx.x] += tmpM0;
        PBHalf2 tmpM1(baselinerStats[blockIdx.x].fullAutocorrState.basicStats.moment1[threadIdx.x * 2],
                      baselinerStats[blockIdx.x].fullAutocorrState.basicStats.moment1[threadIdx.x * 2 + 1]);
        blockMetrics.traceM1[threadIdx.x] += tmpM1;
        PBHalf2 tmpM2(baselinerStats[blockIdx.x].fullAutocorrState.basicStats.moment2[threadIdx.x * 2],
                      baselinerStats[blockIdx.x].fullAutocorrState.basicStats.moment2[threadIdx.x * 2 + 1]);
        blockMetrics.traceM2[threadIdx.x] += tmpM2;
    }

    { // Autocorrelation lag metrics (correctly taken from baseliner)
        PBHalf2 tmpM1F(baselinerStats[blockIdx.x].fullAutocorrState.moment1First[threadIdx.x * 2],
                       baselinerStats[blockIdx.x].fullAutocorrState.moment1First[threadIdx.x * 2 + 1]);
        blockMetrics.autocorrM1First[threadIdx.x] += tmpM1F;
        PBHalf2 tmpM1L(baselinerStats[blockIdx.x].fullAutocorrState.moment1Last[threadIdx.x * 2],
                       baselinerStats[blockIdx.x].fullAutocorrState.moment1Last[threadIdx.x * 2 + 1]);
        blockMetrics.autocorrM1Last[threadIdx.x] += tmpM1L;
        PBHalf2 tmpM2(baselinerStats[blockIdx.x].fullAutocorrState.moment2[threadIdx.x * 2],
                      baselinerStats[blockIdx.x].fullAutocorrState.moment2[threadIdx.x * 2 + 1]);
        blockMetrics.autocorrM2[threadIdx.x] += tmpM2;
    }

    { // FrameLabeler metrics
        PBHalf2 pdTmp(flMetrics[blockIdx.x][threadIdx.x * 2],
                      flMetrics[blockIdx.x][threadIdx.x * 2 + 1]);
        blockMetrics.pulseDetectionScore[threadIdx.x] += pdTmp;
    }
}

__device__ PBShort2 labelBlock(
        const BasecallingMetricsAccumulatorDevice& blockMetrics,
        const BasecallingMetrics& outMetrics,
        //const Cuda::Memory::DeviceView<const TrainedCartDevice> cartParams,
        float frameRate)
{
    // Lets see if we can pass in just the lane object
    //const auto& blockMetrics = metrics[blockIdx.x];
    //const auto& outMetrics = outBatchMetrics[blockIdx.x];

    using AnalogVals = Utility::CudaArray<PBHalf2, numAnalogs>;

    const PBHalf2 zeros(0);

    auto getNativeWideLoad = [](Utility::CudaArray<uint2, laneSize/2> load) -> PBHalf2
    {
        return PBHalf2(load[threadIdx.x].x, load[threadIdx.x].y);
    };

    auto getWideLoad = [](const auto& load) -> PBHalf2
    {
        return PBHalf2(load[threadIdx.x * 2], load[threadIdx.x * 2 + 1]);
    };

    auto deNan = [&zeros](PBHalf2 vals) -> PBHalf2
    {
        return Blend(vals == vals, vals, zeros);
    };

    const auto& stdDev = sqrt(variance(blockMetrics.baselineM0[threadIdx.x],
                                       blockMetrics.baselineM1[threadIdx.x],
                                       blockMetrics.baselineM2[threadIdx.x]));
    const PBHalf2& numBases = getWideLoad(outMetrics.numBases);
    const PBHalf2& numPulses = getWideLoad(outMetrics.numPulses);
    const PBHalf2& pulseWidth = deNan(numPulses / blockMetrics.numPulseFrames[threadIdx.x]);
    const AnalogVals& pkmid = [&blockMetrics, &deNan]()
    {
        AnalogVals ret;
        for (size_t ai = 0; ai < numAnalogs; ++ai)
        {
            ret[ai] = deNan(blockMetrics.pkMidSignal[ai][threadIdx.x]
                            / blockMetrics.numPkMidFrames[ai][threadIdx.x]);
        }
        return ret;
    }();

    Utility::CudaArray<PBHalf2, ActivityLabeler::NUM_FEATURES> features;
    for (auto& analog : features)
    {
        analog = zeros;
    }

    const PBHalf2& seconds = getNativeWideLoad(blockMetrics.numFrames) / frameRate;

    features[ActivityLabeler::PULSERATE] = numPulses / seconds;
    features[ActivityLabeler::SANDWICHRATE] = deNan(blockMetrics.numSandwiches[threadIdx.x] / numPulses);

    const PBHalf2& hswr = deNan(blockMetrics.numHalfSandwiches[threadIdx.x] / numPulses);
    auto hswrExp = evaluatePolynomial(trainedCartParams.hswCurve,
                                      features[ActivityLabeler::PULSERATE]);
    hswrExp = Blend(hswrExp > trainedCartParams.maxAcceptableHalfsandwichRate,
                    PBHalf2(trainedCartParams.maxAcceptableHalfsandwichRate),
                    hswrExp);

    features[ActivityLabeler::LOCALHSWRATENORM] = hswr - hswrExp;

    features[ActivityLabeler::VITERBISCORE] = blockMetrics.pulseDetectionScore[threadIdx.x];
    features[ActivityLabeler::MEANPULSEWIDTH] = pulseWidth;
    features[ActivityLabeler::LABELSTUTTERRATE] = deNan(blockMetrics.numPulseLabelStutters[threadIdx.x] / numPulses);

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
    features[ActivityLabeler::BLOCKLOWSNR] = deNan(features[ActivityLabeler::BLOCKLOWSNR]);
    features[ActivityLabeler::MAXPKMAXNORM] = deNan(features[ActivityLabeler::MAXPKMAXNORM]);
    features[ActivityLabeler::AUTOCORRELATION] = getWideLoad(outMetrics.autocorrelation);

    PBHalf2 lowbp(0);
    PBHalf2 lowpk(0);
    for (size_t i = 0; i < numAnalogs; ++i)
    {
        const auto& bpZvar = deNan(blockMetrics.bpZvar[i][threadIdx.x]);
        features[ActivityLabeler::BPZVARNORM] += bpZvar;
        lowbp = Blend(lowAmpIndex == i, bpZvar, lowbp);

        const auto& pkZvar = deNan(blockMetrics.pkZvar[i][threadIdx.x]);
        features[ActivityLabeler::PKZVARNORM] += pkZvar;
        lowpk = Blend(lowAmpIndex == i, pkZvar, lowpk);
    }
    features[ActivityLabeler::BPZVARNORM] -= lowbp;
    features[ActivityLabeler::BPZVARNORM] /= PBHalf2(3.0f);
    features[ActivityLabeler::PKZVARNORM] -= lowpk;
    features[ActivityLabeler::PKZVARNORM] /= PBHalf2(3.0f);

    for (size_t i = 0; i < features.size(); ++i)
    {
        const auto& nanMask = features[i] == features[i];
        assert(nanMask.X());
        assert(nanMask.Y());
    }


    return PBShort2(traverseCart<0>(features), traverseCart<1>(features));
}

__global__ void FinalizeMetrics(
        uint32_t autocorrAccumLag,
        bool realtimeActivityLabels,
        float frameRate,
        Cuda::Memory::DeviceView<BasecallingMetricsAccumulatorDevice> metrics,
        Cuda::Memory::DeviceView<BasecallingMetrics> outBatchMetrics)
{
    BasecallingMetricsAccumulatorDevice& blockMetrics = metrics[blockIdx.x];

    const PBHalf2 nans(std::numeric_limits<half2>::quiet_NaN());
    const PBHalf2 zeros(0.0f);

    for (size_t pulseLabel = 0; pulseLabel < numAnalogs; pulseLabel++)
    {
        const PBHalf2 nf = blockMetrics.numPkMidFrames[pulseLabel][threadIdx.x];

        const PBHalf2 baselineVariance = variance(blockMetrics.baselineM0[threadIdx.x],
                                                  blockMetrics.baselineM1[threadIdx.x],
                                                  blockMetrics.baselineM2[threadIdx.x]);
        const PBHalf2 pkMidSignal(blockMetrics.pkMidSignal[pulseLabel][threadIdx.x]);
        // For squaring:
        const float pkMidSignalXsqr = [&blockMetrics, &pulseLabel]()
        {
            auto tmp = static_cast<float>(blockMetrics.pkMidSignal[pulseLabel][threadIdx.x].X());
            return tmp * tmp;
        }();
        const float pkMidSignalYsqr = [&blockMetrics, &pulseLabel]()
        {
            auto tmp = static_cast<float>(blockMetrics.pkMidSignal[pulseLabel][threadIdx.x].Y());
            return tmp * tmp;
        }();
        const float nfX = static_cast<float>(nf.X());
        const float nfY = static_cast<float>(nf.X());

        { // Convert moments to interpulse variance
            const PBHalf2& nb = blockMetrics.numPkMidBasesByAnalog[pulseLabel][threadIdx.x];
            blockMetrics.bpZvar[pulseLabel][threadIdx.x].X(
                (static_cast<float>(blockMetrics.bpZvarAcc[pulseLabel][threadIdx.x].x)
                 - pkMidSignalXsqr / nfX)
                / nfX);
            blockMetrics.bpZvar[pulseLabel][threadIdx.x].Y(
                (static_cast<float>(blockMetrics.bpZvarAcc[pulseLabel][threadIdx.x].y)
                 - pkMidSignalYsqr / nfY)
                / nfY);

            // Bessel's correction with num bases, not frames
            blockMetrics.bpZvar[pulseLabel][threadIdx.x] *= nb / (nb - 1.0f);

            blockMetrics.bpZvar[pulseLabel][threadIdx.x] -= baselineVariance / (nf / nb);

            blockMetrics.pkZvar[pulseLabel][threadIdx.x].X(
                (blockMetrics.pkZvarAcc[pulseLabel][threadIdx.x].x - (pkMidSignalXsqr / nfX))
                 / (nfX - 1.0f));
            blockMetrics.pkZvar[pulseLabel][threadIdx.x].Y(
                (blockMetrics.pkZvarAcc[pulseLabel][threadIdx.x].y - (pkMidSignalYsqr / nfY))
                 / (nfY - 1.0f));
            //blockMetrics.pkZvar[pulseLabel][threadIdx.x].Y(
            //    (static_cast<float>(blockMetrics.pkZvar[pulseLabel][threadIdx.x].Y())
            //     - (pkMidSignalYsqr / nfY)) / (nfY - 1.0f));
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
                    emptyMask, nans, pkMidSignal);
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

class DeviceHFMetricsFilter::AccumImpl
{
public:
    AccumImpl(size_t lanesPerPool)
        : metrics_(SOURCE_MARKER(), lanesPerPool)
        , framesSeen_(0)
        , lanesPerBatch_(lanesPerPool)
    { };

    std::unique_ptr<Cuda::Memory::UnifiedCudaArray<Data::BasecallingMetrics>>
    Process(const Data::PulseBatch& pulseBatch,
            const Data::BaselinerMetrics& baselinerMetrics,
            const Cuda::Memory::UnifiedCudaArray<Data::LaneModelParameters<Cuda::PBHalf, laneSize>>& models,
            const Data::FrameLabelerMetrics& flMetrics,
            const Data::PulseDetectorMetrics& pdMetrics)
    {
        if (framesSeen_ == 0)
        {
            const auto& initLauncher = Cuda::PBLauncher(InitializeMetrics,
                                                        lanesPerBatch_,
                                                        threadsPerBlock_);
            initLauncher(metrics_);
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
            auto ret = metricsFactory_->NewBatch();
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
            return ret;
        }
        return std::unique_ptr<Cuda::Memory::UnifiedCudaArray<Data::BasecallingMetrics>>();
    }

private:
    static constexpr size_t threadsPerBlock_ = laneSize / 2;

private:
    Cuda::Memory::DeviceOnlyArray<BasecallingMetricsAccumulatorDevice> metrics_;
    uint32_t framesSeen_;
    uint32_t lanesPerBatch_;

};

constexpr size_t DeviceHFMetricsFilter::AccumImpl::threadsPerBlock_;

DeviceHFMetricsFilter::DeviceHFMetricsFilter(uint32_t poolId,
                                             uint32_t lanesPerPool)
    : HFMetricsFilter(poolId)
    , impl_(std::make_unique<AccumImpl>(lanesPerPool))
{ };

std::unique_ptr<Cuda::Memory::UnifiedCudaArray<Data::BasecallingMetrics>>
DeviceHFMetricsFilter::Process(
        const Data::PulseBatch& pulseBatch,
        const Data::BaselinerMetrics& baselinerMetrics,
        const Cuda::Memory::UnifiedCudaArray<Data::LaneModelParameters<Cuda::PBHalf, 64>>& models,
        const Data::FrameLabelerMetrics& flMetrics,
        const Data::PulseDetectorMetrics& pdMetrics)
{
    return impl_->Process(pulseBatch, baselinerMetrics, models, flMetrics,
            pdMetrics);
}

void DeviceHFMetricsFilter::Configure(uint32_t sandwichTolerance,
                                      uint32_t framesPerHFMetricBlock,
                                      uint32_t framesPerChunk,
                                      double frameRate,
                                      bool realtimeActivityLabels,
                                      uint32_t lanesPerBatch)
{
    HFMetricsFilter::Configure(sandwichTolerance,
                               framesPerHFMetricBlock,
                               framesPerChunk,
                               frameRate,
                               realtimeActivityLabels,
                               lanesPerBatch,
                               false);
    TrainedCartDevice trainedCartHost(42);
    CudaCopyToSymbol(trainedCartParams, &trainedCartHost);
}

}}} // PacBio::Mongo::Basecaller
