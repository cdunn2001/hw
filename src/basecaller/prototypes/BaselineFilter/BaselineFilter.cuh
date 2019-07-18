#ifndef CUDA_BASELINE_FILTER_CUH
#define CUDA_BASELINE_FILTER_CUH

#include <common/cuda/PBCudaSimd.cuh>
#include <common/cuda/utility/CudaTuple.cuh>

#include <prototypes/BaselineFilter/ExtremaFilter.cuh>

namespace PacBio {
namespace Cuda {

// Manual grouping of 4 data entries.  Specifically not using a fixed
// size array, because its very easy to accidentally push arrays
// out of registers and into device memory.
struct Bundle
{
    PBShort2 data0;
    PBShort2 data1;
    PBShort2 data2;
    PBShort2 data3;
};

template <size_t laneWidth, size_t width>
struct ErodeDilate
{
    // default ctor leaves things unitialized!  Intended to facilitate use of this
    // class in __shared__ storage, but manual initialization will have to be performed
    ErodeDilate() = default;

    // ctor intended to be called by a single cuda thread
    __device__ ErodeDilate(short val)
        : f1(val)
        , f2(val)
    {}

    __device__ PBShort2 operator()(PBShort2 in)
    {
        return f2(f1(in));
    }
    __device__ void operator()(Bundle& in)
    {
        in.data0 = f1(in.data0);
        in.data1 = f1(in.data1);
        in.data2 = f1(in.data2);
        in.data3 = f1(in.data3);

        in.data0 = f2(in.data0);
        in.data1 = f2(in.data1);
        in.data2 = f2(in.data2);
        in.data3 = f2(in.data3);
    }
    ExtremaFilter<laneWidth, width, MinOp> f1;
    ExtremaFilter<laneWidth, width, MaxOp> f2;
};
template <size_t laneWidth, size_t width>
struct DilateErode
{
    // default ctor leaves things unitialized!  Intended to facilitate use of this
    // class in __shared__ storage, but manual initialization will have to be performed
    DilateErode() = default;

    // ctor intended to be called by a single cuda thread
    __device__ DilateErode(short val)
        : f1(val)
        , f2(val)
    {}

    __device__ PBShort2 operator()(PBShort2 in)
    {
        return f2(f1(in));
    }
    __device__ void operator()(Bundle& in)
    {
        in.data0 = f1(in.data0);
        in.data1 = f1(in.data1);
        in.data2 = f1(in.data2);
        in.data3 = f1(in.data3);

        in.data0 = f2(in.data0);
        in.data1 = f2(in.data1);
        in.data2 = f2(in.data2);
        in.data3 = f2(in.data3);
    }
    ExtremaFilter<laneWidth, width, MaxOp> f1;
    ExtremaFilter<laneWidth, width, MinOp> f2;
};

// Need this to help with some template futzing.  Helps keep
// separate multiple variadic lists (e.g. widths and strides)
template <int... Is>
using IntSeq = std::integer_sequence<int, Is...>;

template <size_t laneWidth, typename params, class... Filters>
struct MultiFilter;

// Chains together a sequence of filter states.  Can take a variadic list
// of filters to chain, and requires a "stride" value for each filter
// A filter stage requires "stride" inputs before generating a single output
// but the chain as a whole will still output a value for each input, by
// repeating the last value to come out the last stage until enough inputs
// have been provided to generate a new output.
template <size_t laneWidth, int... Strides, class... Filters>
struct __align__(128) MultiFilter<laneWidth, IntSeq<Strides...>, Filters...>
{
    static_assert(sizeof...(Strides) == sizeof...(Filters), "Invalid filter specifications");

    // default ctor leaves things unitialized!  Intended to facilitate use of this
    // class in __shared__ storage, but manual initialization will have to be performed
    MultiFilter() = default;

    // ctor intended to be called by a single cuda thread
    __device__ MultiFilter(short val)
        : filters{val}
        , stridMax{Strides...}
    {
        for (int i = 0; i < sizeof...(Strides); ++i)
        {
            for (int j = 0; j < laneWidth; ++j)
            {
                strideIdx[i][j] = 0;
            }
        }
    }

    // Setting up some types to help with template recursion.  Using types instead of
    // values to control our recursion allows us to use function overloads to control
    // recursion termination.  A previous incarnation used values and in order to
    // terminate the final iteration had a call to the 0th iteration again, but
    // was disabled by a conditional that was always false.  That approach does
    // get optimized away in release builds, but debug builds couldn't tell the
    // conditional was always false, and emitted numerous warnings about being
    // unable to determine the stack size due to the apparrent runtime (not compile time)
    // recursion.  Those warnings were mostly harmless, but annoying, so this alternate
    // version was written instead.
    template <size_t idx>
    struct IdxWrapper
    {
        static constexpr size_t value = idx;
    };
    struct Terminus{};

    __device__ void InvokeNext(PBShort2 val, Terminus){}
    template <typename Idx>
    __device__ void InvokeNext(PBShort2 val, Idx)
    {
        ApplyFilterStage(val, Idx{});
    }

    template <typename Idx, size_t idx = Idx::value>
    __device__ void ApplyFilterStage(PBShort2 val, Idx)
    {
        assert(blockDim.x == laneWidth);
        using NextIdx = typename std::conditional<idx < length-1, IdxWrapper<idx+1> , Terminus>::type;

        auto lidx = strideIdx[idx][threadIdx.x];
        const bool cont = (lidx == 0);
        lidx++;
        if (lidx >= stridMax[idx]) lidx = 0;
        strideIdx[idx][threadIdx.x] = lidx;

        if (cont)
        {
            auto tmp = filters.template Invoke<idx>(val);
            if (idx == length-1)
            {
                ret[threadIdx.x] = tmp;
            }
            else
            {
                InvokeNext(tmp, NextIdx{});
            }
        }
    }

    __device__ PBShort2 operator()(PBShort2 val)
    {
        ApplyFilterStage(val, IdxWrapper<0>{});
        return ret[threadIdx.x];
    }

    static constexpr int length = sizeof...(Filters);
    template <typename T>
    using row = T[laneWidth];

    row<int> strideIdx[length];
    row<PBShort2> ret;
    Utility::CudaTuple<Filters...> filters;
    int stridMax[length];
};

template <size_t laneWidth, typename... params>
struct BaselineFilter;

template <size_t laneWidth, int... Latencies, int width1, int... Widths>
struct BaselineFilter<laneWidth, IntSeq<Latencies...>, IntSeq<width1, Widths...>>
{
    BaselineFilter() = default;
    __device__ BaselineFilter(short val)
        : lower(val)
        , upper(val)
    {}

    __device__ PBShort2 operator()(PBShort2 val)
    {
        auto tmp1 = upper(val);
        auto tmp2 = lower(val);
        return PBShort2((tmp1.X() + tmp2.X()) / 2,
                        (tmp1.Y() + tmp2.Y()) / 2);
    }

    MultiFilter<laneWidth, IntSeq<Latencies...>, ErodeDilate<laneWidth, width1>, ErodeDilate<laneWidth, Widths>...> lower;
    MultiFilter<laneWidth, IntSeq<Latencies...>, DilateErode<laneWidth, width1>, ErodeDilate<laneWidth, Widths>...> upper;
};

}}

#endif //CUDA_BASELINE_FILTER_CUH
