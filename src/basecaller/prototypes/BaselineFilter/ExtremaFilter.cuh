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
    // Compile time loop, which is necessary to keep `data` in registers
    // instead of pushed out to device memory
    template <size_t id>
    __device__ void RotateLoop()
    {
        data[id] = data[id+1];
        static constexpr size_t nextId = (id < filterWidth-2) ? id+1 : 0;
        if (id < filterWidth-2) RotateLoop<nextId>();
    }
    // Does a rolling swap of `data`, to keep the active element in data[0]
    __device__ void Rotate()
    {
        auto tmp = data[0];
        RotateLoop<0>();
        data[filterWidth-1] = tmp;
    }

    // Another compile time loop, to keep `data` in registers
    template <size_t id>
    __device__
    void ExtractZmwData(const ExtremaFilter<blockThreads, filterWidth, Op>& f)
    {
        data[id] = f.data[id][threadIdx.x];

        static constexpr size_t next = (id+1 < filterWidth) ? id+1 : filterWidth-1;
        if (id == filterWidth-1) return;
        else ExtractZmwData<next>(f);
    }
    __device__ LocalExtremaFilter(const ExtremaFilter<blockThreads, filterWidth, Op>& f)
        : idx(0)
    {
        ExtractZmwData<0>(f);
        s = f.s[threadIdx.x];
        idx = f.idx[threadIdx.x];
        __syncthreads();
    }

    // compile time loop, to keep `data` in registers
    template <size_t id>
    __device__
    void ReplaceZmwData(ExtremaFilter<blockThreads, filterWidth, Op>& f)
    {
        f.data[id][threadIdx.x] = data[id];

        static constexpr size_t next = (id+1 < filterWidth) ? id+1 : filterWidth-1;
        if (id == filterWidth-1) return;
        else ReplaceZmwData<next>(f);
    }
    __device__ void ReplaceShared(ExtremaFilter<blockThreads, filterWidth, Op> & f)
    {
        ReplaceZmwData<0>(f);
        f.s[threadIdx.x] = s;
        f.idx[threadIdx.x] = idx;
        __syncthreads();
    }

    template <int id>
    __device__ void OpLoop()
    {
        data[id-1] = Op::op(data[id] , data[id-1]);
        static constexpr int nextId = (id > 1) ? id-1 : 1;
        if (id > 1) OpLoop<nextId>();
    }
    __device__ short2 operator()(short2 val)
    {
        if (idx == 0)
        {
            OpLoop<filterWidth-1>();
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
