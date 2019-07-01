#ifndef EXTREMA_FILTER_CUH
#define EXTREMA_FILTER_CUH

#include <utility>
#include <vector_types.h>

#include <cassert>
#include <cstdlib>

namespace PacBio {
namespace Cuda {

struct MaxOp
{
    __device__ static short2 op(short2 v1, short2 v2)
    {
        return make_short2(max(v1.x, v2.x), max(v1.y, v2.y));
    }
};
struct MinOp
{
    __device__ static short2 op(short2 v1, short2 v2)
    {
        return make_short2(min(v1.x, v2.x), min(v1.y, v2.y));
    }
};

// Baseline filter for one (cuda) block of threads.  Meant to
// be allocated in global memory.
template <size_t blockThreads, size_t filterWidth, typename Op = MaxOp>
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
                data[i][j] = make_short2(0,0);
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

    __device__ short2 operator()(short2 val)
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

        short2 tmp = data[myIdx][threadIdx.x];
        data[myIdx][threadIdx.x] = val;

        myIdx++;
        if (myIdx >= filterWidth) myIdx=0;
        idx[threadIdx.x] = myIdx;

        return Op::op(s[threadIdx.x], tmp);
    }

    using row = short2[blockThreads];
    // TODO create cuda version of std::array
    row data[filterWidth];
    row s;
    int idx[blockThreads];
};

// Experimental version that tries to push the extrema filter into registers.  Did not really
// pan out, but leaving here for educational reasons
template <size_t blockThreads, size_t filterWidth, typename Op = MaxOp>
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

    __device__ LocalExtremaFilter(const ExtremaFilter<blockThreads, filterWidth, Op>& f)
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

    __device__ void ReplaceShared(ExtremaFilter<blockThreads, filterWidth, Op> & f)
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

    __device__ short2 operator()(short2 val)
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

    short2 data[filterWidth];
    short2 s;
    int idx;
};

}}

#endif // EXTREMA_FILTER_CUH
