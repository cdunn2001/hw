#ifndef EXTREMA_FILTER_CUH
#define EXTREMA_FILTER_CUH

#include <common/cuda/PBCudaSimd.cuh>

#include <cassert>
#include <cstdlib>
#include <utility>

namespace PacBio {
namespace Cuda {

struct MaxOp
{
    template <typename T>
    __device__ static T op(T v1, T v2)
    {
        return max(v1, v2);
    }
};
struct MinOp
{
    template <typename T>
    __device__ static T op(T v1, T v2)
    {
        return min(v1, v2);
    }
};

// Baseline filter for one (cuda) block of threads.  Meant to
// be allocated in global memory.
template <size_t blockThreads, size_t filterWidth, typename Op = MaxOp, typename T = PBShort2>
struct __align__(128) ExtremaFilter
{
    // default ctor leaves things unitialized!  Intended to facilitate use of this
    // class in __shared__ storage, but manual initialization will have to be performed
    ExtremaFilter() = default;

    // ctor intended to be called by a single cuda thread
    __device__ ExtremaFilter(short val)
    {
        for (size_t i = 0; i < blockThreads; ++i)
        {
            idx[i] = 0;
        }
        static_assert(sizeof(ExtremaFilter) % 128 == 0, "Alignment request not respected");
        for (size_t i = 0; i < filterWidth; ++i)
        {
            for (size_t j = 0; j < blockThreads; ++j)
            {
                data[i][j] = T(0);
            }
        }
    }

    // Copy is meant to be done by an entire cuda block, with one thread per zmw in the lane
    __device__ ExtremaFilter& operator=(const ExtremaFilter& f)
    {
        for (size_t i = 0; i < filterWidth; ++i)
        {
            data[i][threadIdx.x] = f.data[i][threadIdx.x];
        }
        s[threadIdx.x] = f.s[threadIdx.x];
        idx[threadIdx.x] = f.idx[threadIdx.x];
        __syncthreads();
        return *this;
    }

    __device__ T operator()(T val)
    {
        assert(blockDim.x == blockThreads);

        auto myIdx = idx[threadIdx.x];
        if (myIdx == 0)
        {
            for (int i = filterWidth-1; i > 0; --i)
            {
                data[i-1][threadIdx.x] = Op::op(data[i][threadIdx.x], data[i-1][threadIdx.x]);
            }
            s[threadIdx.x] = data[filterWidth-1][threadIdx.x];
        }
        else
        {
            s[threadIdx.x] = Op::op(s[threadIdx.x], data[myIdx-1][threadIdx.x]);
        }

        T tmp = data[myIdx][threadIdx.x];
        data[myIdx][threadIdx.x] = val;

        myIdx++;
        if (myIdx >= filterWidth) myIdx=0;
        idx[threadIdx.x] = myIdx;

        return Op::op(s[threadIdx.x], tmp);
    }

    using row = T[blockThreads];
    // TODO create cuda version of std::array
    row data[filterWidth];
    row s;
    int idx[blockThreads];
};

// Experimental version that tries to push the extrema filter into registers.  Did not really
// pan out, but leaving here for educational reasons
template <size_t blockThreads, size_t filterWidth, typename Op = MaxOp, typename T = PBShort2>
struct LocalExtremaFilter
{
    // Does a rolling swap of `data`, to keep the active element in data[0]
    __device__ void Rotate()
    {
        auto tmp = data[0];
        // All loops over `data` *must* be unrolled to allow storage in register
        #pragma unroll(filterWidth-1)
        for (int i = 0; i < filterWidth-1; ++i)
        {
            data[i] = data[i+1];
        }
        data[filterWidth-1] = tmp;
    }

    __device__ LocalExtremaFilter(const ExtremaFilter<blockThreads, filterWidth, Op, T>& f)
        : idx(0)
    {
        // All loops over `data` *must* be unrolled to allow storage in register
        #pragma unroll(filterWidth)
        for (int i = 0; i < filterWidth; ++i)
        {
            data[i] = f.data[i][threadIdx.x];
        }
        s = f.s[threadIdx.x];
        idx = f.idx[threadIdx.x];
        __syncthreads();
    }

    __device__ void ReplaceShared(ExtremaFilter<blockThreads, filterWidth, Op, T> & f)
    {
        // All loops over `data` *must* be unrolled to allow storage in register
        #pragma unroll(filterWidth)
        for (int i = 0; i < filterWidth; ++i)
        {
            f.data[i][threadIdx.x] = data[i];
        }
        f.s[threadIdx.x] = s;
        f.idx[threadIdx.x] = idx;
        __syncthreads();
    }

    __device__ T operator()(T val)
    {
        if (idx == 0)
        {
            // All loops over `data` *must* be unrolled to allow storage in register
            #pragma unroll(filterWidth)
            for (int i = filterWidth-1; i > 0; --i)
            {
                data[i-1] = Op::op(data[i], data[i-1]);
            }
            s = data[filterWidth-1];
        }
        else s = Op::op(s, data[filterWidth-1]);

        auto tmp = data[0];
        data[0] = val;
        Rotate();
        idx++;
        if (idx >= filterWidth) idx = 0;
        return Op::op(s, tmp);
    }

    T data[filterWidth];
    T s;
    int idx;
};

}}

#endif // EXTREMA_FILTER_CUH
