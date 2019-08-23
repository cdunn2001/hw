#ifndef CUDA_BASELINE_FILTER_KERNELS_CUH
#define CUDA_BASELINE_FILTER_KERNELS_CUH

#include "BaselineFilter.cuh"

#include <common/cuda/memory/DeviceOnlyArray.cuh>
#include <common/cuda/PBCudaSimd.cuh>
#include <common/cuda/streams/LaunchManager.cuh>
#include <common/cuda/utility/CudaArray.h>
#include <dataTypes/BatchData.cuh>
#include <dataTypes/BaselinerStatAccumState.h>
#include <dataTypes/TraceBatch.h>

namespace PacBio {
namespace Cuda {

template <typename Filter>
__global__ void GlobalBaselineFilter(const Mongo::Data::GpuBatchData<const PBShort2> in,
                                     Memory::DeviceView<Filter> filters,
                                     Mongo::Data::GpuBatchData<PBShort2> out)
{
    const size_t numFrames = in.NumFrames();
    auto& myFilter = filters[blockIdx.x];
    const auto& inZmw  = in.ZmwData(blockIdx.x, threadIdx.x);
    auto outZmw = out.ZmwData(blockIdx.x, threadIdx.x);

    for (int i = 0; i < numFrames; ++i)
    {
        outZmw[i] = myFilter(inZmw[i]);
    }
}

template <typename Filter>
__global__ void SharedBaselineFilter(const Mongo::Data::GpuBatchData<const PBShort2> in,
                                     Memory::DeviceView<Filter> filters,
                                     Mongo::Data::GpuBatchData<PBShort2> out)
{
    const size_t numFrames = in.NumFrames();
    __shared__ Filter myFilter;
    myFilter = filters[blockIdx.x];
    const auto& inZmw  = in.ZmwData(blockIdx.x, threadIdx.x);
    auto outZmw = out.ZmwData(blockIdx.x, threadIdx.x);

    for (int i = 0; i < numFrames; ++i)
    {
        outZmw[i] = myFilter(inZmw[i]);
    }

    filters[blockIdx.x] = myFilter;
}

__device__ int constexpr constexprMax(int a, int b)
{
    return a > b ? a : b;
}

// Tries to implement the full baseline filter in a single kernel, without using shared memory
// for inactive stages.  It may be alright for some settings, but for widths of 9 and 31 it
// was not the best approach.  Loading one filter at a time instead of *all* filters does let us
// use less shared memory and increase our occupancy by a factor of 2, but the fact that we now
// have to have multiple passes over the data kills those gains.  Instead having separate
// kernels for each filter stage actually performs better.  We still have multiple passes over the
// data, but now the kernels with smaller filters can use a lot less shared memory and get a lot
// more increased occupancy, making them run significantly faster.
template <size_t blockThreads, size_t width1, size_t width2, size_t stride1, size_t stride2>
__global__ void CompressedBaselineFilter(const Mongo::Data::GpuBatchData<const PBShort2> in,
                                         Memory::DeviceView<ErodeDilate<blockThreads, width1>> lower1,
                                         Memory::DeviceView<ErodeDilate<blockThreads, width2>> lower2,
                                         Memory::DeviceView<DilateErode<blockThreads, width1>> upper1,
                                         Memory::DeviceView<ErodeDilate<blockThreads, width2>> upper2,
                                         Mongo::Data::GpuBatchData<PBShort2> workspace1,
                                         Mongo::Data::GpuBatchData<PBShort2> workspace2,
                                         Mongo::Data::GpuBatchData<PBShort2> out)
{
    const size_t numFrames = in.NumFrames();

    // Grab a swath of memory that can fit our largest filter
    constexpr size_t maxSize = constexprMax(sizeof(ErodeDilate<blockThreads, width1>), sizeof(ErodeDilate<blockThreads, width2>));
    __shared__  char mempool[maxSize];

    assert(numFrames % stride1*stride2 == 0);

    // Get block specific workspaces
    const auto& inb  = in.ZmwData(blockIdx.x, threadIdx.x);
    auto outb = out.ZmwData(blockIdx.x, threadIdx.x);
    auto workb1 = workspace1.ZmwData(blockIdx.x, threadIdx.x);
    auto workb2 = workspace2.ZmwData(blockIdx.x, threadIdx.x);

    // --------------Lower filter-------------------
    ErodeDilate<blockThreads, width1>* low1 = new (&mempool[0])ErodeDilate<blockThreads, width1>(lower1[blockIdx.x]);
    for (int i = 0; i < numFrames; i += stride1)
    {
        workb1[i/stride1] = (*low1)(inb[i]);
    }
    lower1[blockIdx.x] = *low1;
    low1->~ErodeDilate<blockThreads, width1>();
    low1 = nullptr;

    ErodeDilate<blockThreads, width2>* low2 = new (&mempool[0])ErodeDilate<blockThreads, width2>(lower2[blockIdx.x]);
    int limit = numFrames / stride1;
    for (int i = 0; i < limit; i += stride2)
    {
        workb1[i/stride2] = (*low2)(workb1[i]);
    }
    lower2[blockIdx.x] = *low2;
    low2->~ErodeDilate<blockThreads, width2>();
    low2 = nullptr;

    // --------------Upper filter-------------------
    DilateErode<blockThreads, width1>* up1 = new (&mempool[0])DilateErode<blockThreads, width1>(upper1[blockIdx.x]);
    for (int i = 0; i < numFrames; i += stride1)
    {
        workb2[i/stride1] = (*up1)(inb[i]);

    }
    upper1[blockIdx.x] = *up1;
    up1->~DilateErode<blockThreads, width1>();
    up1 = nullptr;

    ErodeDilate<blockThreads, width2>* up2 = new (&mempool[0])ErodeDilate<blockThreads, width2>(upper2[blockIdx.x]);
    limit = numFrames / stride1;
    for (int i = 0; i < limit; i += stride2)
    {
        workb2[i/stride2] = (*up2)(workb2[i]);
    }
    upper2[blockIdx.x] = *up2;
    up2->~ErodeDilate<blockThreads, width2>();
    up2 = nullptr;
}

// Runs a filter over the input data, but only outputs 1/stride of the
// results.
template <size_t blockThreads, size_t stride, typename Filter>
__global__ void StridedFilter(const Mongo::Data::GpuBatchData<const PBShort2> in,
                              Memory::DeviceView<Filter> filters,
                              int numFrames,
                              Mongo::Data::GpuBatchData<PBShort2> out)
{
    const size_t maxFrames = in.NumFrames();

    assert(blockThreads == blockDim.x);
    assert(numFrames <= out.NumFrames());
    assert(numFrames <= maxFrames);
    assert(numFrames % 4 == 0);
    assert((numFrames/4) % stride == 0);

    __shared__ Filter myFilter;
    myFilter = filters[blockIdx.x];
    const auto& inZmw = in.ZmwData(blockIdx.x, threadIdx.x);
    auto outZmw = out.ZmwData(blockIdx.x, threadIdx.x);

    for (int i = 0; i < numFrames; i += 4*stride)
    {
        Bundle b;
        b.data0 = inZmw[i];
        b.data1 = inZmw[i + stride];
        b.data2 = inZmw[i + 2*stride];
        b.data3 = inZmw[i + 3*stride];
        myFilter(b);

        outZmw[i/stride]   = b.data0;
        outZmw[i/stride+1] = b.data1;
        outZmw[i/stride+2] = b.data2;
        outZmw[i/stride+3] = b.data3;
    }
    filters[blockIdx.x] = myFilter;
}

// Averages the output of the lower and upper filters, and expands the data
// to back to the original (unstrided) size
template <size_t blockThreads, size_t stride>
__global__ void AverageAndExpand(const Mongo::Data::GpuBatchData<const PBShort2> in1,
                                 const Mongo::Data::GpuBatchData<const PBShort2> in2,
                                 Mongo::Data::GpuBatchData<PBShort2> out)
{
    const size_t numFrames = out.NumFrames();

    assert(numFrames % stride == 0);
    int inputCount = numFrames / stride;

    const auto& inZmw1 = in1.ZmwData(blockIdx.x, threadIdx.x);
    const auto& inZmw2 = in2.ZmwData(blockIdx.x, threadIdx.x);
    auto outZmw = out.ZmwData(blockIdx.x, threadIdx.x);

    for (int i = 0; i < inputCount; ++i)
    {
        PBShort2 val((inZmw1[i].X() + inZmw2[i].X()) / 2,
                     (inZmw1[i].Y() + inZmw2[i].Y()) / 2);
        for (int j = i*stride; j < stride; ++j)
        {
            outZmw[j] = val;
        }
    }
}

// TODO support autocorrelation
template <size_t blockThreads>
class StatAccumulator
{
public:
    __device__ void Reset()
    {
        auto largest = std::numeric_limits<int16_t>::max();
        auto lowest = std::numeric_limits<int16_t>::min();
        min_[threadIdx.x] = PBShort2(largest);
        max_[threadIdx.x] = PBShort2(lowest);

        rawSum_[threadIdx.x] = 0.0f;
        m0_[threadIdx.x] = 0.0f;
        m1_[threadIdx.x] = 0.0f;
        m2_[threadIdx.x] = 0.0f;
    }

    __device__ void AddData(PBHalf2 raw,
                            PBHalf2 bs, //baseline subtracted
                            PBBool2 baselineMask)
    {
        PBHalf2 zero(0.0f);
        PBHalf2 one(1.0f);

        min_[threadIdx.x] = min(bs, min_[threadIdx.x]);
        max_[threadIdx.x] = max(bs, max_[threadIdx.x]);
        rawSum_[threadIdx.x] += Blend(baselineMask, raw, zero);

        m0_[threadIdx.x] += Blend(baselineMask, one, zero);
        m1_[threadIdx.x] += Blend(baselineMask, bs, zero);
        m2_[threadIdx.x] += Blend(baselineMask, bs*bs, zero);
    }

    __device__ void FillOutputStats(Mongo::Data::BaselinerStatAccumState& stats)
    {
        stats.baselineStats.moment0[2*threadIdx.x] = m0_[threadIdx.x].FloatX();
        stats.baselineStats.moment0[2*threadIdx.x+1] = m0_[threadIdx.x].FloatY();
        stats.baselineStats.moment1[2*threadIdx.x] = m1_[threadIdx.x].FloatX();
        stats.baselineStats.moment1[2*threadIdx.x+1] = m1_[threadIdx.x].FloatY();
        stats.baselineStats.moment2[2*threadIdx.x] = m2_[threadIdx.x].FloatX();
        stats.baselineStats.moment2[2*threadIdx.x+1] = m2_[threadIdx.x].FloatY();

        // TODO weird float/short conversions going on.  Should make these consistently shorts probably
        stats.rawBaselineSum[2*threadIdx.x] = rawSum_[threadIdx.x].FloatX();
        stats.rawBaselineSum[2*threadIdx.x+1] = rawSum_[threadIdx.x].FloatY();
        stats.traceMin[2*threadIdx.x] = min_[threadIdx.x].FloatX();
        stats.traceMin[2*threadIdx.x+1] = min_[threadIdx.x].FloatY();
        stats.traceMax[2*threadIdx.x] = max_[threadIdx.x].FloatX();
        stats.traceMax[2*threadIdx.x+1] = max_[threadIdx.x].FloatY();
    }

private:
    Utility::CudaArray<PBHalf2, blockThreads> min_;
    Utility::CudaArray<PBHalf2, blockThreads> max_;
    Utility::CudaArray<PBHalf2, blockThreads> rawSum_;
    Utility::CudaArray<PBHalf2, blockThreads> m0_;
    Utility::CudaArray<PBHalf2, blockThreads> m1_;
    Utility::CudaArray<PBHalf2, blockThreads> m2_;
};

template <size_t blockThreads>
struct LatentBaselineData
{
    __device__ LatentBaselineData(PBHalf2 sig)
    {
        for (int i = 0; i < blockThreads; ++i)
        {
            bgSigma[i] = sig;
        }
    }
    class LocalLatent
    {
        friend LatentBaselineData;

        PBHalf2 bgSigma;
        PBHalf2 latData;
        PBHalf2 latRawData;
        PBBool2 latLMask;
        PBBool2 latHMask1;
        PBBool2 latHMask2;

        PBHalf2 thrHigh;
        PBHalf2 thrLow;

        // TODO unify these somehow with the host multiscale implementation
        static constexpr float sigmaThrL = 4.5f;
        static constexpr float sigmaThrH = 4.5f;
        static constexpr float alphaFactor = 0.7f;

    public:
        __device__ PBHalf2 SmoothedSigma(PBHalf2 frameSigma)
        {
            static constexpr float minSigma = .288675135f; // sqrt(1.0f/12.0f);
            bgSigma = (1.0f - alphaFactor) * bgSigma + alphaFactor * max(frameSigma, minSigma);

            thrLow = bgSigma * sigmaThrL;
            thrHigh = bgSigma * sigmaThrH;

            return bgSigma;
        }

        __device__ void ProcessFrame(
                PBHalf2 raw,
                PBHalf2 subtracted,
                StatAccumulator<blockThreads>& stats)
        {
            auto maskHp1 = subtracted < thrHigh;
            auto mask = latHMask1 && latLMask && maskHp1;

            latLMask = subtracted < thrLow;
            latHMask2 = latHMask1;
            latHMask1 = maskHp1;

            stats.AddData(latRawData, latData, mask);

            latRawData = raw;
            latData = subtracted;
        }
    };

    __device__ LocalLatent GetLocal()
    {
        LocalLatent local;

        local.bgSigma    = bgSigma[threadIdx.x];
        local.latData    = latData[threadIdx.x];
        local.latRawData = latRawData[threadIdx.x];
        local.latLMask   = latLMask[threadIdx.x];
        local.latHMask1  = latHMask1[threadIdx.x];
        local.latHMask2  = latHMask2[threadIdx.x];

        return local;
    }

    __device__ void StoreLocal(const LocalLatent& local)
    {
        bgSigma[threadIdx.x]     = local.bgSigma;
        latData[threadIdx.x]     = local.latData;
        latRawData[threadIdx.x]  = local.latRawData;
        latLMask[threadIdx.x]    = local.latLMask;
        latHMask1[threadIdx.x]   = local.latHMask1;
        latHMask2[threadIdx.x]   = local.latHMask2;
    }
private:
    // TODO init
    Utility::CudaArray<PBHalf2, blockThreads> bgSigma;
    Utility::CudaArray<PBHalf2, blockThreads> latData;
    Utility::CudaArray<PBHalf2, blockThreads> latRawData;
    Utility::CudaArray<PBBool2, blockThreads> latLMask;
    Utility::CudaArray<PBBool2, blockThreads> latHMask1;
    Utility::CudaArray<PBBool2, blockThreads> latHMask2;
};

template <size_t blockThreads, size_t stride>
__global__ void SubtractBaseline(const Mongo::Data::GpuBatchData<const PBShort2> input,
                                 Memory::DeviceView<LatentBaselineData<blockThreads>> latent,
                                 const Mongo::Data::GpuBatchData<const PBShort2> lower,
                                 const Mongo::Data::GpuBatchData<const PBShort2> upper,
                                 Mongo::Data::GpuBatchData<PBShort2> out,
                                 Memory::DeviceView<Mongo::Data::BaselinerStatAccumState> outputStats)
{
    __shared__ StatAccumulator<blockThreads> stats;
    stats.Reset();

    auto localLatent = latent[blockIdx.x].GetLocal();

    // For bias estimate
    // TODO: these need to be set consistent with filter widths
    PBHalf2 cSigmaBias(2.44f);
    PBHalf2 cMeanBias(0.5f);

    const size_t numFrames = out.NumFrames();

    assert(numFrames % stride == 0);
    int inputCount = numFrames / stride;

    const auto& inZmw = input.ZmwData(blockIdx.x, threadIdx.x);
    const auto& lowerZmw = lower.ZmwData(blockIdx.x, threadIdx.x);
    const auto& upperZmw = upper.ZmwData(blockIdx.x, threadIdx.x);
    auto outZmw = out.ZmwData(blockIdx.x, threadIdx.x);

    for (int i = 0; i < inputCount; ++i)
    {
        auto baseline = (PBHalf2(upperZmw[i]) + PBHalf2(lowerZmw[i])) / PBHalf2(2.0f);
        auto sigma = localLatent.SmoothedSigma((PBHalf2(upperZmw[i]) - PBHalf2(lowerZmw[i])) / cSigmaBias);
        auto frameBiasEstimate = cMeanBias * sigma;

        auto start = i*stride;
        auto end = (i+1)*stride;
        for (int j = start; j < end; ++j)
        {
            auto raw = PBHalf2(inZmw[j]);
            auto val = raw - baseline - frameBiasEstimate;
            localLatent.ProcessFrame(raw, val, stats);

            outZmw[j] = ToShort(val);
        }
    }

    stats.FillOutputStats(outputStats[blockIdx.x]);
    latent[blockIdx.x].StoreLocal(localLatent);
}

template <size_t blockThreads, size_t width1, size_t width2, size_t stride1, size_t stride2>
class ComposedFilter
{
    using Lower1 = ErodeDilate<blockThreads, width1>;
    using Lower2 = ErodeDilate<blockThreads, width2>;
    using Upper1 = DilateErode<blockThreads, width1>;
    using Upper2 = ErodeDilate<blockThreads, width2>;

    Memory::DeviceOnlyArray<Lower1> lower1;
    Memory::DeviceOnlyArray<Lower2> lower2;
    Memory::DeviceOnlyArray<Upper1> upper1;
    Memory::DeviceOnlyArray<Upper2> upper2;
    Memory::DeviceOnlyArray<LatentBaselineData<blockThreads>> latent;
    size_t numLanes_;

public:
    __host__ ComposedFilter(size_t numLanes, short val)
        : lower1(numLanes, val)
        , lower2(numLanes, val)
        , upper1(numLanes, val)
        , upper2(numLanes, val)
        , numLanes_(numLanes)
        , latent(numLanes, 0.0f)
    {}

    // TODO should probably rename or remove.  Computes a naive baseline, but does
    // not do the actual baseline subtraction, nor does it do any bias corrections
    __host__ void RunComposedFilter(const Mongo::Data::TraceBatch<int16_t>& input,
                                    Mongo::Data::TraceBatch<int16_t>& output,
                                    Mongo::Data::BatchData<int16_t>& workspace1,
                                    Mongo::Data::BatchData<int16_t>& workspace2)
    {
        const uint64_t numFrames = input.NumFrames();

        assert(input.LaneWidth() == 2*blockThreads);
        assert(input.LanesPerBatch() == numLanes_);

        const auto& L1 = PBLauncher(StridedFilter<blockThreads, 2, Lower1>,
                                  numLanes_,
                                  blockThreads);
        L1(input,
           lower1,
           numFrames,
           workspace1);
        const auto& L2 = PBLauncher(StridedFilter<blockThreads, 8, Lower2>,
                                  numLanes_,
                                  blockThreads);
        L2(workspace1,
           lower2,
           numFrames/2,
           workspace1);

        const auto& U1 = PBLauncher(StridedFilter<blockThreads, 2, Upper1>,
                                  numLanes_,
                                  blockThreads);
        U1(input,
           upper1,
           numFrames,
           workspace2);
        const auto& U2 = PBLauncher(StridedFilter<blockThreads, 8, Upper2>,
                                  numLanes_,
                                  blockThreads);
        U2(workspace2,
           upper2,
           numFrames/2,
           workspace2);

        const auto& average = PBLauncher(AverageAndExpand<blockThreads, 16>,
                                       numLanes_,
                                       blockThreads);
        average(workspace1, workspace2, output);
    }

    __host__ void RunBaselineFilter(const Mongo::Data::TraceBatch<int16_t>& input,
                                    Mongo::Data::TraceBatch<int16_t>& output,
                                    Memory::UnifiedCudaArray<Mongo::Data::BaselinerStatAccumState>& stats,
                                    Mongo::Data::BatchData<int16_t>& workspace1,
                                    Mongo::Data::BatchData<int16_t>& workspace2)
    {
        const uint64_t numFrames = input.NumFrames();

        assert(input.LaneWidth() == 2*blockThreads);
        assert(input.LanesPerBatch() == numLanes_);

        const auto& L1 = PBLauncher(StridedFilter<blockThreads, 2, Lower1>, numLanes_, blockThreads);
        L1(input,
           lower1,
           numFrames,
           workspace1);
        const auto& L2 = PBLauncher(StridedFilter<blockThreads, 8, Lower2>, numLanes_, blockThreads);
        L2(workspace1,
           lower2,
           numFrames/2,
           workspace1);

        const auto& U1 = PBLauncher(StridedFilter<blockThreads, 2, Upper1>, numLanes_, blockThreads);
        U1(input,
           upper1,
           numFrames,
           workspace2);
        const auto& U2 = PBLauncher(StridedFilter<blockThreads, 8, Upper2>, numLanes_, blockThreads);
        U2(workspace2,
           upper2,
           numFrames/2,
           workspace2);

        const auto& Subtract = PBLauncher(SubtractBaseline<blockThreads, 16>, numLanes_, blockThreads);
        Subtract(input,
                 latent,
                 workspace1,
                 workspace2,
                 output,
                 stats);
    }

};

}}

#endif //CUDA_BASELINE_FILTER_KERNELS_CUH
