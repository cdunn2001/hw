#ifndef CUDA_BASELINE_FILTER_KERNELS_CUH
#define CUDA_BASELINE_FILTER_KERNELS_CUH

#include <numeric>
#include "BaselineFilter.cuh"
#include "BlockCircularBuffer.cuh"
#include "LocalCircularBuffer.cuh"

#include <common/cuda/memory/DeviceOnlyArray.cuh>
#include <common/cuda/PBCudaSimd.cuh>
#include <common/cuda/streams/LaunchManager.cuh>
#include <common/cuda/utility/CudaArray.h>
#include <dataTypes/BatchData.cuh>
#include <dataTypes/BaselinerStatAccumState.h>
#include <dataTypes/TraceBatch.h>

#include <basecaller/traceAnalysis/BaselinerParams.h>

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
template <size_t blockThreads>
__global__ void AverageAndExpand(const Mongo::Data::GpuBatchData<const PBShort2> in1,
                                 const Mongo::Data::GpuBatchData<const PBShort2> in2,
                                 Mongo::Data::GpuBatchData<PBShort2> out,
                                 size_t stride)
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

template <size_t blockThreads, size_t lag>
class StatAccumulator
{
public:
    __device__ void Reset()
    {
        constexpr auto largest = std::numeric_limits<int16_t>::max();
        constexpr auto lowest = std::numeric_limits<int16_t>::min();
        minB_[threadIdx.x] = PBShort2(largest);
        maxB_[threadIdx.x] = PBShort2(lowest);

        rawSumB_[threadIdx.x] = 0.0f;
        m0B_[threadIdx.x] = 0.0f;
        m1B_[threadIdx.x] = 0.0f;
        m2B_[threadIdx.x] = 0.0f;

        m0_[threadIdx.x] = 0.0f;
        m1_[threadIdx.x] = 0.0f;
        m2_[threadIdx.x] = 0.0f;
        m1LagFirst_[threadIdx.x] = 0.0f;
        m1LagLast_[threadIdx.x] = 0.0f;
        m2Lag_[threadIdx.x] = 0.0f;
    }

    __device__ void AddBaselineData(PBHalf2 raw,
                                    PBHalf2 bs, //baseline subtracted
                                    PBBool2 baselineMask)
    {
        PBHalf2 zero(0.0f);
        PBHalf2 one(1.0f);

        minB_[threadIdx.x] = min(bs, minB_[threadIdx.x]);
        maxB_[threadIdx.x] = max(bs, maxB_[threadIdx.x]);
        rawSumB_[threadIdx.x] += Blend(baselineMask, raw, zero);

        m0B_[threadIdx.x] += Blend(baselineMask, one, zero);
        m1B_[threadIdx.x] += Blend(baselineMask, bs, zero);
        m2B_[threadIdx.x] += Blend(baselineMask, bs*bs, zero);
    }

    __device__ void AddAutoCorrData(PBHalf2 bs,
                                    PBHalf2 lagVal)
    {
        PBHalf2 one(1.0f);

        m0_[threadIdx.x] += one;
        m1_[threadIdx.x] += bs;
        m2_[threadIdx.x] += bs*bs;

        m1LagFirst_[threadIdx.x] += lagVal;
        m1LagLast_[threadIdx.x] += bs;
        m2Lag_[threadIdx.x] += bs * lagVal;
    }

    __device__ void FillOutputStats(Mongo::Data::BaselinerStatAccumState& stats)
    {
        // Baseline stats
        stats.baselineStats.moment0[2*threadIdx.x] = m0B_[threadIdx.x].FloatX();
        stats.baselineStats.moment0[2*threadIdx.x+1] = m0B_[threadIdx.x].FloatY();
        stats.baselineStats.moment1[2*threadIdx.x] = m1B_[threadIdx.x].FloatX();
        stats.baselineStats.moment1[2*threadIdx.x+1] = m1B_[threadIdx.x].FloatY();
        stats.baselineStats.moment2[2*threadIdx.x] = m2B_[threadIdx.x].X();
        stats.baselineStats.moment2[2*threadIdx.x+1] = m2B_[threadIdx.x].Y();

        // TODO weird float/short conversions going on.  Should make these consistently shorts probably
        stats.rawBaselineSum[2*threadIdx.x] = rawSumB_[threadIdx.x].X();
        stats.rawBaselineSum[2*threadIdx.x+1] = rawSumB_[threadIdx.x].Y();
        stats.traceMin[2*threadIdx.x] = minB_[threadIdx.x].FloatX();
        stats.traceMin[2*threadIdx.x+1] = minB_[threadIdx.x].FloatY();
        stats.traceMax[2*threadIdx.x] = maxB_[threadIdx.x].FloatX();
        stats.traceMax[2*threadIdx.x+1] = maxB_[threadIdx.x].FloatY();

        // Auto-correlation stats
        stats.fullAutocorrState.moment1First[2*threadIdx.x] = m1LagFirst_[threadIdx.x].FloatX();
        stats.fullAutocorrState.moment1First[2*threadIdx.x+1] = m1LagFirst_[threadIdx.x].FloatY();
        stats.fullAutocorrState.moment1Last[2*threadIdx.x] = m1LagLast_[threadIdx.x].FloatX();
        stats.fullAutocorrState.moment1Last[2*threadIdx.x+1] = m1LagLast_[threadIdx.x].FloatY();
        stats.fullAutocorrState.moment2[2*threadIdx.x] = m2Lag_[threadIdx.x].X();
        stats.fullAutocorrState.moment2[2*threadIdx.x+1] = m2Lag_[threadIdx.x].Y();

        stats.fullAutocorrState.basicStats.moment0[2*threadIdx.x] = m0_[threadIdx.x].FloatX();
        stats.fullAutocorrState.basicStats.moment0[2*threadIdx.x+1] = m0_[threadIdx.x].FloatY();
        stats.fullAutocorrState.basicStats.moment1[2*threadIdx.x] = m1_[threadIdx.x].FloatX();
        stats.fullAutocorrState.basicStats.moment1[2*threadIdx.x+1] = m1_[threadIdx.x].FloatY();
        stats.fullAutocorrState.basicStats.moment2[2*threadIdx.x] = m2_[threadIdx.x].X();
        stats.fullAutocorrState.basicStats.moment2[2*threadIdx.x+1] = m2_[threadIdx.x].Y();
    }

private:
    // Min/max over all baseline-subtracted frames
    Utility::CudaArray<PBHalf2,  blockThreads> minB_;
    Utility::CudaArray<PBHalf2,  blockThreads> maxB_;
    // Sum over all baseline frames before baseline subtraction
    Utility::CudaArray<PBFloat2,  blockThreads> rawSumB_;
    // Baseline stats computed from baseline-subtracted frames classified as baseline
    Utility::CudaArray<PBHalf2,  blockThreads> m0B_;
    Utility::CudaArray<PBHalf2,  blockThreads> m1B_;
    Utility::CudaArray<PBFloat2, blockThreads> m2B_;
    // Auto-correlation stats computed from all frames
    Utility::CudaArray<PBHalf2,  blockThreads> m0_;
    Utility::CudaArray<PBHalf2,  blockThreads> m1_;
    Utility::CudaArray<PBFloat2, blockThreads> m2_;
    Utility::CudaArray<PBHalf2,  blockThreads> m1LagFirst_;
    Utility::CudaArray<PBHalf2,  blockThreads> m1LagLast_;
    Utility::CudaArray<PBFloat2, blockThreads> m2Lag_;
};

template <size_t blockThreads, size_t lag>
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

        LocalCircularBuffer<blockThreads, lag> circularBuffer;

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

        __device__ void ProcessFrame(PBHalf2 raw,
                                     PBHalf2 subtracted,
                                     StatAccumulator<blockThreads, lag>& stats)
        {
            auto maskHp1 = subtracted < thrHigh;
            auto mask = latHMask1 && latLMask && maskHp1;

            latLMask = subtracted < thrLow;
            latHMask2 = latHMask1;
            latHMask1 = maskHp1;

            stats.AddBaselineData(latRawData, latData, mask);

            latRawData = raw;
            latData = subtracted;

            auto lagVal = circularBuffer.Front();
            circularBuffer.PushBack(subtracted);
            stats.AddAutoCorrData(subtracted, lagVal);
        }
    };

    __device__ LocalLatent GetLocal()
    {
        LocalLatent local;

        local.bgSigma        = bgSigma[threadIdx.x];
        local.latData        = latData[threadIdx.x];
        local.latRawData     = latRawData[threadIdx.x];
        local.latLMask       = latLMask[threadIdx.x];
        local.latHMask1      = latHMask1[threadIdx.x];
        local.latHMask2      = latHMask2[threadIdx.x];
        local.circularBuffer.Init(circularBuffer);

        return local;
    }

    __device__ void StoreLocal(LocalLatent& local)
    {
        bgSigma[threadIdx.x]        = local.bgSigma;
        latData[threadIdx.x]        = local.latData;
        latRawData[threadIdx.x]     = local.latRawData;
        latLMask[threadIdx.x]       = local.latLMask;
        latHMask1[threadIdx.x]      = local.latHMask1;
        latHMask2[threadIdx.x]      = local.latHMask2;
        local.circularBuffer.ReplaceShared(circularBuffer);
    }
private:
    // TODO init
    Utility::CudaArray<PBHalf2, blockThreads> bgSigma;
    Utility::CudaArray<PBHalf2, blockThreads> latData;
    Utility::CudaArray<PBHalf2, blockThreads> latRawData;
    Utility::CudaArray<PBBool2, blockThreads> latLMask;
    Utility::CudaArray<PBBool2, blockThreads> latHMask1;
    Utility::CudaArray<PBBool2, blockThreads> latHMask2;
    BlockCircularBuffer<blockThreads, lag> circularBuffer;
};

struct SubtractParams
{
    PBHalf2 cSigmaBias;
    PBHalf2 cMeanBias;
    PBHalf2 scale;
};
template <size_t blockThreads, size_t lag>
__global__ void SubtractBaseline(const Mongo::Data::GpuBatchData<const PBShort2> input,
                                 size_t stride,
                                 SubtractParams sParams,
                                 Memory::DeviceView<LatentBaselineData<blockThreads,lag>> latent,
                                 const Mongo::Data::GpuBatchData<const PBShort2> lower,
                                 const Mongo::Data::GpuBatchData<const PBShort2> upper,
                                 Mongo::Data::GpuBatchData<PBShort2> out,
                                 Memory::DeviceView<Mongo::Data::BaselinerStatAccumState> outputStats)
{
    __shared__ StatAccumulator<blockThreads,lag> stats;
    stats.Reset();

    auto localLatent = latent[blockIdx.x].GetLocal();

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
        auto sigma = localLatent.SmoothedSigma((PBHalf2(upperZmw[i]) - PBHalf2(lowerZmw[i])) / sParams.cSigmaBias);
        auto frameBiasEstimate = sParams.cMeanBias * sigma;

        auto start = i*stride;
        auto end = (i+1)*stride;
        for (int j = start; j < end; ++j)
        {
            auto raw = PBHalf2(inZmw[j]);
            auto val = (raw - baseline - frameBiasEstimate) * sParams.scale;
            localLatent.ProcessFrame(raw, val, stats);

            outZmw[j] = ToShort(val);
        }
    }

    stats.FillOutputStats(outputStats[blockIdx.x]);
    latent[blockIdx.x].StoreLocal(localLatent);
}

class FilterStage
{
public:
    virtual void
    RunFilter(const Mongo::Data::BatchData<int16_t>& in, int numFrames, Mongo::Data::BatchData<int16_t>& out) = 0;
    virtual size_t Stride() const = 0;

    virtual ~FilterStage() = default;
};

template <typename Filter, size_t stride, size_t blockThreads>
class FilterStageImpl : public FilterStage
{
public:
    __host__ FilterStageImpl(const Memory::AllocationMarker& marker,
                             size_t numLanes,
                             short val,
                             Memory::StashableAllocRegistrar* registrar)
        : filterData_(registrar, marker, numLanes, val)
    {}

    void
    RunFilter(const Mongo::Data::BatchData<int16_t>& in, int numFrames, Mongo::Data::BatchData<int16_t>& out) override
    {
        const auto& launcher = PBLauncher(
            StridedFilter<blockThreads, stride, Filter>, filterData_.Size(), blockThreads);
        launcher(in, filterData_, numFrames, out);
    }

    size_t Stride() const override { return stride; }

private:
    Memory::DeviceOnlyArray<Filter> filterData_;
};

template <size_t blockThreads, size_t lag>
class ComposedFilter
{
    template <template <size_t, size_t> class FilterType>
    std::unique_ptr<FilterStage> CreateFilter(size_t width,
                                              size_t stride,
                                              const Memory::AllocationMarker& marker,
                                              size_t numLanes,
                                              short val,
                                              Memory::StashableAllocRegistrar* registrar)
    {
    #define ReturnIf(s, w)                                                                          \
        if (s == stride && width == w)                                                              \
            return std::make_unique<FilterStageImpl<FilterType<blockThreads, w>, s, blockThreads>>( \
                marker, numLanes, val, registrar);

        ReturnIf(1, 7);
        ReturnIf(1, 9);
        ReturnIf(1, 11);
        ReturnIf(2, 7);
        ReturnIf(2, 9);
        ReturnIf(2, 11);
        ReturnIf(2, 17);
        ReturnIf(8, 17);
        ReturnIf(8, 31);
        ReturnIf(8, 61);

        throw PBException("Unsupported and unexpected baseline filter stide/width combo");
    }
public:
    using TraceBatch = Mongo::Data::TraceBatch<int16_t>;
    using BatchData = Mongo::Data::BatchData<int16_t>;

    __host__ ComposedFilter(const Mongo::Basecaller::BaselinerParams& params,
                            const Memory::AllocationMarker& marker,
                            size_t numLanes,
                            short val,
                            Memory::StashableAllocRegistrar* registrar = nullptr)
        : numLanes_(numLanes)
        , latent(registrar, marker, numLanes, 0.0f)
    {
        const auto& widths = params.Widths();
        const auto& strides = params.Strides();

        sParams_.cMeanBias = params.MeanBias();
        sParams_.cSigmaBias = params.SigmaBias();
        // TODO this needs to be plumbed through properly still
        sParams_.scale = 1.0f;

        fullStride_ = std::accumulate(strides.begin(), strides.end(), 1, std::multiplies{});;

        lower_.push_back(CreateFilter<ErodeDilate>(widths[0], strides[0], marker, numLanes, val, registrar));
        upper_.push_back(CreateFilter<DilateErode>(widths[0], strides[0], marker, numLanes, val, registrar));
        for (size_t i = 1; i < widths.size(); ++i)
        {
            lower_.push_back(CreateFilter<ErodeDilate>(widths[i], strides[i], marker, numLanes, val, registrar));
            upper_.push_back(CreateFilter<ErodeDilate>(widths[i], strides[i], marker, numLanes, val, registrar));
        }
    }

    // TODO should probably rename or remove.  Computes a naive baseline, but does
    // not do the actual baseline subtraction, nor does it do any bias corrections
    __host__ void
    RunComposedFilter(const TraceBatch& input, TraceBatch& output, BatchData& workspace1, BatchData& workspace2)
    {
        RunMorpho(input, output, workspace1, workspace2);

        const auto& average = PBLauncher(AverageAndExpand<blockThreads>, numLanes_, blockThreads);
        average(workspace1, workspace2, output, fullStride_);
    }

    __host__ void RunBaselineFilter(const Mongo::Data::TraceBatch<int16_t>& input,
                                    Mongo::Data::TraceBatch<int16_t>& output,
                                    Memory::UnifiedCudaArray<Mongo::Data::BaselinerStatAccumState>& stats,
                                    Mongo::Data::BatchData<int16_t>& workspace1,
                                    Mongo::Data::BatchData<int16_t>& workspace2)
    {
        RunMorpho(input, output, workspace1, workspace2);

        const auto& Subtract = PBLauncher(
            SubtractBaseline<blockThreads, lag>, numLanes_, blockThreads);
        Subtract(input, fullStride_, sParams_, latent, workspace1, workspace2, output, stats);
    }

private:
    __host__ void RunMorpho(const TraceBatch& input, TraceBatch& output, BatchData& workspace1, BatchData& workspace2)
    {
        uint64_t numFrames = input.NumFrames();

        assert(input.LaneWidth() == 2 * blockThreads);
        assert(input.LanesPerBatch() == numLanes_);

        for (size_t i = 0; i < lower_.size(); ++i)
        {
            auto* lowerIn = i == 0 ? static_cast<const BatchData*>(&input) : &workspace1;
            auto* upperIn = i == 0 ? static_cast<const BatchData*>(&input) : &workspace2;
            lower_[i]->RunFilter(*lowerIn, numFrames, workspace1);
            upper_[i]->RunFilter(*upperIn, numFrames, workspace2);
            numFrames /= lower_[i]->Stride();
            assert(upper_[i]->Stride() == lower_[i]->Stride());
        }
    }

    std::vector<std::unique_ptr<FilterStage>> lower_;
    std::vector<std::unique_ptr<FilterStage>> upper_;
    using LatentBaselineData = LatentBaselineData<blockThreads, lag>;

    Memory::DeviceOnlyArray<LatentBaselineData> latent;
    size_t numLanes_;
    size_t fullStride_;
    SubtractParams sParams_;
};

}}

#endif  // CUDA_BASELINE_FILTER_KERNELS_CUH
