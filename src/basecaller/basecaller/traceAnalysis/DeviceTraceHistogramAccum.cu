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

#include <cuda_runtime.h>

#include <cstdint>

#include <basecaller/traceAnalysis/DeviceTraceHistogramAccum.h>

#include <common/cuda/PBCudaSimd.cuh>
#include <common/cuda/memory/DeviceOnlyArray.cuh>
#include <common/cuda/memory/UnifiedCudaArray.h>
#include <common/cuda/streams/LaunchManager.cuh>

#include <dataTypes/BatchData.cuh>
#include <dataTypes/configs/BasecallerTraceHistogramConfig.h>

using namespace PacBio::Cuda::Memory;
using namespace PacBio::Cuda::Utility;
using namespace PacBio::Cuda;
using namespace PacBio::Mongo::Data;

namespace PacBio {
namespace Mongo {
namespace Basecaller {

namespace {

struct StaticConfig
{
    float binSizeCoeff_;
    unsigned int baselineStatMinFrameCount_;
    float fallBackBaselineSigma_;
};

__constant__ StaticConfig staticConfig;

}

// This is essentially just a copy of LaneHistogram with the storage
// order of binCount swapped.  Could conceivably unify the two with
// a little template work, but it's probably not worth doing.  This
// is only used in implementations that are not the fastest, and are
// only kept around for reference, to be kept around as a reminder
// of what has been tried, and for re-evaluation if we get new hardware
// with different characteristics.
struct LaneHistogramTrans
{
    using DataT = float;
    using CountT = uint16_t;
    template <typename T>
    using Array = Cuda::Utility::CudaArray<T, laneSize>;

    using DataType = DataT;
    using CountType = CountT;

    // This constant must be large enough to accomodate high SNR data.
    // Ideally, it would be a function of BinSizeCoeff and the SNR and excess-
    // noise CV of the brightest analog.
    // nBins(snr, bsc, xsn) = (snr + 4(1 + sqrt(1 + snr + xsn*snr^2))) / bsc,
    // where bsc = binSizeCoeff, and xsn = excess noise CV.
    // In practice, the value will probably be determined somewhat empirically.
    static constexpr unsigned int numBins = Data::LaneHistogram<float, uint16_t>::numBins;

    /// The lower bound of the lowest bin.
    Array<DataT> lowBound;

    /// The size of all bins for each ZMW.
    Array<DataT> binSize;

    /// The number of data less than lowBound.
    Array<CountT> outlierCountLow;

    /// The number of data >= the high bound = lowBound + numBins*binSize.
    Array<CountT> outlierCountHigh;

    /// The number of data in each bin.
    Array<CudaArray<CountT, numBins>> binCount;
};

__global__ void BinningGlobalInterleaved(Data::GpuBatchData<const PBShort2> traces,
                                         DeviceView<Data::LaneHistogram<float, uint16_t>> hists)
{
    auto& hist = hists[blockIdx.x];

    float lowBound = hist.lowBound[threadIdx.x];
    float binSize = hist.binSize[threadIdx.x];
    ushort lowOutlier = hist.outlierCountLow[threadIdx.x];
    ushort highOutlier = hist.outlierCountHigh[threadIdx.x];

    constexpr int16_t numBins = LaneHistogramTrans::numBins;

    auto zmw = traces.ZmwData(blockIdx.x, threadIdx.x/2);
    for (int i = 0; i < traces.NumFrames(); ++i)
    {
        auto idx = threadIdx.x;
        auto val = (threadIdx.x % 2 == 0) ? zmw[i].X() : zmw[i].Y();
        int bin = (val - lowBound) / binSize;
        if (bin < 0) lowOutlier++;
        else if (bin >= numBins) highOutlier++;
        else hist.binCount[bin][idx]++;
    }

    hist.outlierCountLow[threadIdx.x] = lowOutlier;
    hist.outlierCountHigh[threadIdx.x] = highOutlier;
}


// Notes: I had been hoping that un-coallesced reads would still trigger a whole cacheline
//        load into L2 (and maybe even L1).  Thus by having bins contiguous I was hoping I'd
//        have better cache hits since the series is generally time correlated.
//
//        That said, things run slower.  Cases of 1 signal are drastically slower, which makes
//        sense as we gave up very good warp convergence.  1 signal of course is not a good
//        and representative case.  Even a mix of different signals however shows a performance
//        degredation.  Eithegiving up the minor read-coalescing we randomly get isn't worth it,
//        or the L2 doesn't work as I'd hoped.
//
//        Need to profile more and do some reasearch, to test the many assumptions in the original
//        theory
__global__ void BinningGlobalContig(Data::GpuBatchData<const PBShort2> traces,
                                    DeviceView<LaneHistogramTrans> hists)
{
    auto& hist = hists[blockIdx.x];

    float lowBound = hist.lowBound[threadIdx.x];
    float binSize = hist.binSize[threadIdx.x];
    ushort lowOutlier = hist.outlierCountLow[threadIdx.x];
    ushort highOutlier = hist.outlierCountHigh[threadIdx.x];

    constexpr int16_t numBins = LaneHistogramTrans::numBins;

    auto zmw = traces.ZmwData(blockIdx.x, threadIdx.x/2);
    for (int i = 0; i < traces.NumFrames(); ++i)
    {
        auto idx = threadIdx.x;
        auto val = (threadIdx.x % 2 == 0 ? zmw[i].X() : zmw[i].Y());
        int bin = (val - lowBound) / binSize;
        if (bin < 0) lowOutlier++;
        else if (bin >= numBins) highOutlier++;
        else hist.binCount[idx][bin]++;
    }

    hist.outlierCountLow[threadIdx.x] = lowOutlier;
    hist.outlierCountHigh[threadIdx.x] = highOutlier;
}

// Notes: This function isn't too hard to tweak so that multiple warps participate, each warp working
//        on a disjoint subset of zmw.

//        There is some slight evidece that having 2-4 warps active in the same block have a marginal
//        benefit in some cases and a marginal detrement in others.  Having too many warps definitely
//        does hurt performance, presumably because we're back to having a large enough portion of the
//        histogram being accessed, causing more cache trashing.
__global__ void BinningGlobalContigCoopWarps(Data::GpuBatchData<const PBShort2> traces,
                                             DeviceView<LaneHistogramTrans> hists)
{
    assert(traces.NumFrames() % blockDim.x == 0);
    auto& hist = hists[blockIdx.x];

    constexpr int16_t numBins = LaneHistogramTrans::numBins;
    auto traceItrs = traces.NumFrames() / blockDim.x;
    for (int zmw = 0; zmw < laneSize/2; ++zmw)
    {
        float2 lowBound = {hist.lowBound[2*zmw], hist.lowBound[2*zmw+1]};
        float2 binSize = {hist.binSize[2*zmw], hist.binSize[2*zmw+1]};

        auto dat = traces.ZmwData(blockIdx.x, zmw);
        for (int i = threadIdx.x; i < traces.NumFrames(); i+=blockDim.x)
        {
            auto val = dat[i];
            int bin = (val.X() - lowBound.x) / binSize.x;
            if (bin < 0) bin = -1;
            else if (bin > numBins) bin = numBins;

            auto same = __match_any_sync(0xFFFFFFFF, bin);
            // Number of threads with the same bin
            auto count = __popc(same);
            // Thread with the most significant bit gets to own the update
            bool owner = (32 - __clz(same) -1 ) == threadIdx.x;

            if (owner)
            {
                if (bin < 0) hist.outlierCountLow[2*zmw] += count;
                else if (bin == numBins) hist.outlierCountHigh[2*zmw] += count;
                else hist.binCount[2*zmw][bin] += count;
            }

            bin = (val.Y() - lowBound.y) / binSize.y;
            if (bin < 0) bin = -1;
            else if (bin > numBins) bin = numBins;

            same = __match_any_sync(0xFFFFFFFF, bin);
            // Number of threads with the same bin
            count = __popc(same);
            // Thread with the most significant bit gets to own the update
            owner = (32 - __clz(same) -1 ) == threadIdx.x;

            if (owner)
            {
                if (bin < 0) hist.outlierCountLow[2*zmw+1] +=count;
                else if (bin == numBins) hist.outlierCountHigh[2*zmw+1] += count;
                else hist.binCount[2*zmw+1][bin] += count;
            }
        }
    }
}

__global__ void BinningSharedContigCoopWarps(Data::GpuBatchData<const PBShort2> traces,
                                             DeviceView<LaneHistogramTrans> hists)
{
    constexpr int16_t numBins = LaneHistogramTrans::numBins;
    __shared__ uint16_t localHist[numBins+2][2];
    assert(traces.NumFrames() % blockDim.x == 0);
    auto& hist = hists[blockIdx.x];

    for (int zmw = 0; zmw < laneSize/2; ++zmw)
    {
        float2 lowBound = {hist.lowBound[2*zmw], hist.lowBound[2*zmw+1]};
        float2 binSize = {hist.binSize[2*zmw], hist.binSize[2*zmw+1]};

        for (int i = threadIdx.x; i < numBins+2; i+=blockDim.x)
        {
            localHist[i][0] = 0;
            localHist[i][1] = 0;
        }
        __syncwarp(0xFFFFFFFF);

        auto dat = traces.ZmwData(blockIdx.x, zmw);
        for (int i = threadIdx.x; i < traces.NumFrames(); i+=blockDim.x)
        {
            auto val = dat[i];
            int bin = (val.X() - lowBound.x) / binSize.x;
            if (bin >= numBins) bin = numBins+1;
            else if (bin < 0) bin = numBins;

            auto same = __match_any_sync(0xFFFFFFFF, bin);
            // Number of threads with the same bin
            auto count = __popc(same);
            // Thread with the most significant bit gets to own the update
            bool owner = (32 - __clz(same) -1 ) == threadIdx.x;

            if (owner)
            {
                localHist[bin][0] += count;
            }

            bin = (val.Y() - lowBound.y) / binSize.y;
            if (bin >= numBins) bin = numBins+1;
            else if (bin < 0) bin = numBins;

            same = __match_any_sync(0xFFFFFFFF, bin);
            // Number of threads with the same bin
            count = __popc(same);
            // Thread with the most significant bit gets to own the update
            owner = (32 - __clz(same) -1 ) == threadIdx.x;

            if (owner)
            {
                localHist[bin][1] += count;
            }
            __syncwarp();
        }

        for (int i = threadIdx.x; i < numBins; i+=blockDim.x)
        {
            hist.binCount[2*zmw][i] += localHist[i][0];
            hist.binCount[2*zmw+1][i] += localHist[i][1];
        }
        if (threadIdx.x == 0)
        {
            hist.outlierCountHigh[2*zmw] += localHist[numBins+1][0];
            hist.outlierCountHigh[2*zmw+1] += localHist[numBins+1][1];
        }
        else if (threadIdx.x == 1)
        {
            hist.outlierCountLow[2*zmw] += localHist[numBins][0];
            hist.outlierCountLow[2*zmw+1] += localHist[numBins][1];
        }
        __syncwarp(0xFFFFFFFF);
    }
}

__global__ void BinningSharedContig2DBlock(Data::GpuBatchData<const PBShort2> traces,
                                           DeviceView<LaneHistogramTrans> hists)
{
    constexpr int16_t numBins = LaneHistogramTrans::numBins;

    struct SharedData
    {
        uint16_t localHist[32][numBins+2][2];
        PBShort2 trans[32][33];
    };

    assert(traces.NumFrames() % blockDim.x == 0);
    assert(blockDim.x == 32);
    assert(blockDim.y == 32);
    assert(blockDim.z == 1);

    __shared__ SharedData shared;
    assert(traces.NumFrames() % blockDim.x == 0);
    auto& hist = hists[blockIdx.x];

    const auto zmw = threadIdx.y;

    float2 lowBound = {hist.lowBound[2*zmw], hist.lowBound[2*zmw+1]};
    float2 binSize = {hist.binSize[2*zmw], hist.binSize[2*zmw+1]};

    for (int i = threadIdx.x; i < numBins+2; i+=blockDim.x)
    {
        shared.localHist[zmw][i][0] = 0;
        shared.localHist[zmw][i][1] = 0;
    }
    __syncwarp(0xFFFFFFFF);

    auto dat = traces.ZmwData(blockIdx.x, threadIdx.x);
    for (int i = threadIdx.y; i < traces.NumFrames(); i+=blockDim.y)
    {
        __syncthreads();
        shared.trans[threadIdx.y][threadIdx.x] = dat[i];
        __syncthreads();
        auto val = shared.trans[threadIdx.x][threadIdx.y];

        int bin = (val.X() - lowBound.x) / binSize.x;
        if (bin >= numBins) bin = numBins+1;
        else if (bin < 0) bin = numBins;

        auto same = __match_any_sync(0xFFFFFFFF, bin);
        // Number of threads with the same bin
        auto count = __popc(same);
        // Thread with the most significant bit gets to own the update
        bool owner = (32 - __clz(same) -1 ) == threadIdx.x;

        if (owner)
        {
            shared.localHist[zmw][bin][0] += count;
        }

        bin = (val.Y() - lowBound.y) / binSize.y;
        if (bin >= numBins) bin = numBins+1;
        else if (bin < 0) bin = numBins;

        same = __match_any_sync(0xFFFFFFFF, bin);
        // Number of threads with the same bin
        count = __popc(same);
        // Thread with the most significant bit gets to own the update
        owner = (32 - __clz(same) -1 ) == threadIdx.x;

        if (owner)
        {
            shared.localHist[zmw][bin][1] += count;
        }
    }

    for (int i = threadIdx.x; i < numBins; i+=blockDim.x)
    {
        hist.binCount[2*zmw][i] += shared.localHist[zmw][i][0];
        hist.binCount[2*zmw+1][i] += shared.localHist[zmw][i][1];
    }
    if (threadIdx.x == 0)
    {
        hist.outlierCountHigh[2*zmw] += shared.localHist[zmw][numBins+1][0];
        hist.outlierCountHigh[2*zmw+1] += shared.localHist[zmw][numBins+1][1];
    }
    else if (threadIdx.x == 1)
    {
        hist.outlierCountLow[2*zmw] += shared.localHist[zmw][numBins][0];
        hist.outlierCountLow[2*zmw+1] += shared.localHist[zmw][numBins][1];
    }
}

__global__ void BinningSharedInterleaved2DBlock(Data::GpuBatchData<const PBShort2> traces,
                                                DeviceView<Data::LaneHistogram<float, uint16_t>> hists)
{
    assert(blockDim.x == 32);
    auto& ghist = hists[blockIdx.x];
    const auto zmw = threadIdx.x;

    PBHalf2 lowBound = {ghist.lowBound[2*zmw], ghist.lowBound[2*zmw+1]};
    PBHalf2 binSize = {ghist.binSize[2*zmw], ghist.binSize[2*zmw+1]};

    constexpr int16_t numBins = LaneHistogramTrans::numBins;
    __shared__ PBShort2 lhist[numBins+2][32];

    for (int i = threadIdx.y; i < numBins+2; i+=blockDim.y)
    {
        lhist[i][threadIdx.x] = 0;
    }

    __syncthreads();

    auto trace = traces.ZmwData(blockIdx.x, threadIdx.x);
    for (int i = threadIdx.y; i < traces.NumFrames(); i+=blockDim.y)
    {
        auto val = trace[i];

        auto bin = (val - lowBound) / binSize;
        bin = Blend(bin >= numBins, numBins+1, bin);
        bin = Blend(bin < 0, numBins, bin);

        atomicAdd(&lhist[bin.IntX()][threadIdx.x].data(), 1);
        atomicAdd(&lhist[bin.IntY()][threadIdx.x].data(), 1<<16);
    }

    __syncthreads();

    for (int i = threadIdx.y; i < numBins; i+=blockDim.y)
    {
        ghist.binCount[i][2*threadIdx.x] += lhist[i][threadIdx.x].X();
        ghist.binCount[i][2*threadIdx.x+1] += lhist[i][threadIdx.x].Y();
    }

    if (threadIdx.y == 0)
    {
        ghist.outlierCountLow[2*threadIdx.x] += lhist[numBins][threadIdx.x].X();
        ghist.outlierCountLow[2*threadIdx.x+1] += lhist[numBins][threadIdx.x].Y();
    }
    else if (threadIdx.y == 1 % blockDim.y)
    {
        ghist.outlierCountHigh[2*threadIdx.x] += lhist[numBins+1][threadIdx.x].X();
        ghist.outlierCountHigh[2*threadIdx.x+1] += lhist[numBins+1][threadIdx.x].Y();
    }
}

__global__ void CopyToContig(DeviceView<LaneHistogramTrans> source,
                             DeviceView<Data::LaneHistogram<float, uint16_t>> dest)
{
    assert(blockDim.x == 64);
    auto& sBlock = source[blockIdx.x];
    auto& dBlock = dest[blockIdx.x];

    for (int i = 0; i < dBlock.numBins; ++i)
    {
        dBlock.binCount[i][threadIdx.x] = sBlock.binCount[threadIdx.x][i];
    }
    dBlock.outlierCountHigh[threadIdx.x] = sBlock.outlierCountHigh[threadIdx.x];
    dBlock.outlierCountLow[threadIdx.x] = sBlock.outlierCountLow[threadIdx.x];
    dBlock.lowBound[threadIdx.x] = sBlock.lowBound[threadIdx.x];
    dBlock.binSize[threadIdx.x] = sBlock.binSize[threadIdx.x];
}

__global__ void CopyToInterleaved(DeviceView<Data::LaneHistogram<float, uint16_t>> source,
                                  DeviceView<Data::LaneHistogram<float, uint16_t>> dest)
{
    assert(blockDim.x == 64);
    auto& sBlock = source[blockIdx.x];
    auto& dBlock = dest[blockIdx.x];

    for (int i = 0; i < dBlock.numBins; ++i)
    {
        dBlock.binCount[i][threadIdx.x] = sBlock.binCount[i][threadIdx.x];
    }
    dBlock.outlierCountHigh[threadIdx.x] = sBlock.outlierCountHigh[threadIdx.x];
    dBlock.outlierCountLow[threadIdx.x] = sBlock.outlierCountLow[threadIdx.x];
    dBlock.lowBound[threadIdx.x] = sBlock.lowBound[threadIdx.x];
    dBlock.binSize[threadIdx.x] = sBlock.binSize[threadIdx.x];
}

__global__ void ResetHistsContigBounds(DeviceView<LaneHistogramTrans> hists,
                                       DeviceView<const LaneHistBounds> bounds)
{
    assert(blockDim.x == 64);
    auto& hist = hists[blockIdx.x];
    auto& bound = bounds[blockIdx.x];

    for (int i = 0; i < hist.numBins; ++i)
    {
        hist.binCount[threadIdx.x][i] = 0;
    }
    hist.outlierCountHigh[threadIdx.x] = 0;
    hist.outlierCountLow[threadIdx.x] = 0;
    hist.lowBound[threadIdx.x] = bound.lowerBounds[threadIdx.x];
    hist.binSize[threadIdx.x] = (bound.upperBounds[threadIdx.x] - bound.lowerBounds[threadIdx.x])
                              / static_cast<float>(hist.numBins);
}

__global__ void ResetHistsInterleavedBounds(DeviceView<Data::LaneHistogram<float, uint16_t>> hists,
                                            DeviceView<const LaneHistBounds> bounds)
{
    assert(blockDim.x == 64);
    auto& hist = hists[blockIdx.x];
    auto& bound = bounds[blockIdx.x];

    for (int i = 0; i < hist.numBins; ++i)
    {
        hist.binCount[i][threadIdx.x] = 0;
    }
    hist.outlierCountHigh[threadIdx.x] = 0;
    hist.outlierCountLow[threadIdx.x] = 0;
    hist.lowBound[threadIdx.x] = bound.lowerBounds[threadIdx.x];
    hist.binSize[threadIdx.x] = (bound.upperBounds[threadIdx.x] - bound.lowerBounds[threadIdx.x])
                              / static_cast<float>(hist.numBins);
}

__global__ void ResetHistsInterleavedStats(DeviceView<Data::LaneHistogram<float, uint16_t>> hists,
                                           DeviceView<const Data::BaselinerStatAccumState> stats)
{
    // Determine histogram parameters.
    const auto& laneBlStats = stats[blockIdx.x].baselineStats;

    const auto blCount = laneBlStats.moment0[threadIdx.x];
    const auto mom1 = laneBlStats.moment1[threadIdx.x];
    auto blMean = mom1 / blCount + laneBlStats.offset[threadIdx.x];
    auto blSigma = laneBlStats.moment2[threadIdx.x] - blMean * mom1;
    blSigma = sqrt(blSigma / (blCount - 1));

    if (blCount < staticConfig.baselineStatMinFrameCount_)
    {
        blMean = 0.0f;
        blSigma = staticConfig.fallBackBaselineSigma_;
    }

    const auto binSize = staticConfig.binSizeCoeff_ * blSigma;
    const auto lower = blMean - 4.0f*blSigma;

    assert(blockDim.x == 64);
    auto& hist = hists[blockIdx.x];

    for (int i = 0; i < hist.numBins; ++i)
    {
        hist.binCount[i][threadIdx.x] = 0;
    }
    hist.outlierCountHigh[threadIdx.x] = 0;
    hist.outlierCountLow[threadIdx.x] = 0;
    hist.lowBound[threadIdx.x] = lower;
    hist.binSize[threadIdx.x] = binSize;
}

__global__ void ResetHistsContigStats(DeviceView<LaneHistogramTrans> hists,
                                      DeviceView<const Data::BaselinerStatAccumState> stats)
{
    // Determine histogram parameters.
    const auto& laneBlStats = stats[blockIdx.x].baselineStats;

    const auto blCount = laneBlStats.moment0[threadIdx.x];
    const auto mom1 = laneBlStats.moment1[threadIdx.x];
    auto blMean = mom1 / blCount + laneBlStats.offset[threadIdx.x];
    auto blSigma = laneBlStats.moment2[threadIdx.x] - blMean * mom1;
    blSigma = sqrt(blSigma / (blCount - 1));

    if (blCount < staticConfig.baselineStatMinFrameCount_)
    {
        blMean = 0.0f;
        blSigma = staticConfig.fallBackBaselineSigma_;
    }

    const auto binSize = staticConfig.binSizeCoeff_ * blSigma;
    const auto lower = blMean - 4.0f*blSigma;

    assert(blockDim.x == 64);
    auto& hist = hists[blockIdx.x];

    for (int i = 0; i < hist.numBins; ++i)
    {
        hist.binCount[threadIdx.x][i] = 0;
    }
    hist.outlierCountHigh[threadIdx.x] = 0;
    hist.outlierCountLow[threadIdx.x] = 0;
    hist.lowBound[threadIdx.x] = lower;
    hist.binSize[threadIdx.x] = binSize;
}

class DeviceTraceHistogramAccum::ImplBase
{
public:
    ImplBase(size_t poolSize, DeviceHistogramTypes type)
        : poolSize_(poolSize)
        , type_(type)
    {}

    virtual ~ImplBase() = default;

    virtual void AddBatchImpl(const Data::TraceBatch<DataType>& traces) = 0;

    virtual void ResetImpl(const Cuda::Memory::UnifiedCudaArray<LaneHistBounds>& bounds) = 0;

    virtual void ResetImpl(const Data::BaselinerMetrics& metrics) = 0;

    virtual void Copy(Data::PoolHistogram<HistDataType, HistCountType>& dest) = 0;

    size_t PoolSize() const { return poolSize_; }
    DeviceHistogramTypes Type() const { return type_; }
private:
    size_t poolSize_;
    DeviceHistogramTypes type_;
};

class HistInterleavedZmw : public DeviceTraceHistogramAccum::ImplBase
{
public:
    HistInterleavedZmw(unsigned int poolSize,
                       Cuda::Memory::StashableAllocRegistrar* registrar,
                       DeviceHistogramTypes type)
        : DeviceTraceHistogramAccum::ImplBase(poolSize, type)
        , data_(registrar, SOURCE_MARKER(), poolSize)
    {}

    void AddBatchImpl(const Data::TraceBatch<int16_t>& traces) override
    {
        switch (Type())
        {
        case DeviceHistogramTypes::GlobalInterleaved:
            {
                PBLauncher(BinningGlobalInterleaved, this->PoolSize(), laneSize)(traces, data_);
                break;
            }
        case DeviceHistogramTypes::SharedInterleaved2DBlock:
            {
                PBLauncher(BinningSharedInterleaved2DBlock, this->PoolSize(), dim3{laneSize/2, 32, 1})(traces, data_);
                break;
            }
        default:
            throw PBException("Unexpected device histogram type in HistInterleavedZmw");
        }
    }

    void ResetImpl(const Cuda::Memory::UnifiedCudaArray<LaneHistBounds>& bounds) override
    {
        assert(bounds.Size() == PoolSize());
        PBLauncher(ResetHistsInterleavedBounds, PoolSize(), laneSize)(data_, bounds);
    }

    void ResetImpl(const Data::BaselinerMetrics& metrics) override
    {
        assert(metrics.baselinerStats.Size() == PoolSize());
        PBLauncher(ResetHistsInterleavedStats, this->PoolSize(), laneSize)(data_, metrics.baselinerStats);
    }

    void Copy(Data::PoolHistogram<float, uint16_t>& dest) override
    {
        PBLauncher(CopyToInterleaved, this->PoolSize(), laneSize)(data_, dest.data);
    }

private:
    DeviceOnlyArray<LaneHistogram<float, uint16_t>> data_;
};

class HistContigZmw : public DeviceTraceHistogramAccum::ImplBase
{
public:
    HistContigZmw(unsigned int poolSize,
                  Cuda::Memory::StashableAllocRegistrar* registrar,
                  DeviceHistogramTypes type)
        : DeviceTraceHistogramAccum::ImplBase(poolSize, type)
        , data_(registrar, SOURCE_MARKER(), poolSize)
    {}

    void AddBatchImpl(const Data::TraceBatch<int16_t>& traces) override
    {
        switch (Type())
        {
        case DeviceHistogramTypes::GlobalContig:
            {
                PBLauncher(BinningGlobalContig, this->PoolSize(), laneSize)(traces, data_);
                break;
            }
        case DeviceHistogramTypes::GlobalContigCoopWarps:
            {
                PBLauncher(BinningGlobalContigCoopWarps, this->PoolSize(), laneSize/2)(traces, data_);
                break;
            }
        case DeviceHistogramTypes::SharedContigCoopWarps:
            {
                PBLauncher(BinningSharedContigCoopWarps, this->PoolSize(), laneSize/2)(traces, data_);
                break;
            }
        case DeviceHistogramTypes::SharedContig2DBlock:
            {
                PBLauncher(BinningSharedContig2DBlock, this->PoolSize(), dim3{laneSize/2, laneSize/2,1})(traces, data_);
                break;
            }
        default:
            throw PBException("Unexpected device histogram type in HistInterleavedZmw");
        }
    }

    void ResetImpl(const Cuda::Memory::UnifiedCudaArray<LaneHistBounds>& bounds) override
    {
        assert(bounds.Size() == PoolSize());
        PBLauncher(ResetHistsContigBounds, PoolSize(), laneSize)(data_, bounds);
    }

    void ResetImpl(const Data::BaselinerMetrics& metrics) override
    {
        assert(metrics.baselinerStats.Size() == PoolSize());
        PBLauncher(ResetHistsContigStats, this->PoolSize(), laneSize)(data_, metrics.baselinerStats);
    }

    void Copy(Data::PoolHistogram<float, uint16_t>& dest) override
    {
        PBLauncher(CopyToContig, this->PoolSize(), laneSize)(data_, dest.data);
    }

private:
    DeviceOnlyArray<LaneHistogramTrans> data_;
};

void DeviceTraceHistogramAccum::Configure(const Data::BasecallerTraceHistogramConfig& sigConfig)
{
    StaticConfig config;

    config.binSizeCoeff_ = sigConfig.BinSizeCoeff;
    PBLOG_INFO << "TraceHistogramAccumulator: BinSizeCoeff = "
               << config.binSizeCoeff_ << '.';
    if (config.binSizeCoeff_ <= 0.0f)
    {
        std::ostringstream msg;
        msg << "BinSizeCoeff must be positive.";
        throw PBException(msg.str());
    }

    config.baselineStatMinFrameCount_ = sigConfig.BaselineStatMinFrameCount;
    PBLOG_INFO << "TraceHistogramAccumulator: BaselineStatMinFrameCount = "
               << config.baselineStatMinFrameCount_ << '.';

    config.fallBackBaselineSigma_ = sigConfig.FallBackBaselineSigma;
    PBLOG_INFO << "TraceHistogramAccumulator: FallBackBaselineSigma = "
               << config.fallBackBaselineSigma_ << '.';
    if (config.fallBackBaselineSigma_ <= 0.0f)
    {
        std::ostringstream msg;
        msg << "FallBackBaselineSigma must be positive.";
        throw PBException(msg.str());
    }
    CudaRawCopyToSymbol(&staticConfig, &config, sizeof(StaticConfig));
}


void DeviceTraceHistogramAccum::AddBatchImpl(const Data::TraceBatch<DataType>& traces)
{
    impl_->AddBatchImpl(traces);
    CudaSynchronizeDefaultStream();
}

void DeviceTraceHistogramAccum::ResetImpl(const Cuda::Memory::UnifiedCudaArray<LaneHistBounds>& bounds)
{
    impl_->ResetImpl(bounds);
    CudaSynchronizeDefaultStream();
}

void DeviceTraceHistogramAccum::ResetImpl(const Data::BaselinerMetrics& metrics)
{
    impl_->ResetImpl(metrics);
    CudaSynchronizeDefaultStream();
}

DeviceTraceHistogramAccum::PoolHistType DeviceTraceHistogramAccum::HistogramImpl() const
{
    PoolHistType hists(PoolId(), PoolSize(), SyncDirection::HostReadDeviceWrite);
    impl_->Copy(hists);
    return hists;
}

DeviceTraceHistogramAccum::DeviceTraceHistogramAccum(unsigned int poolId,
                                                     unsigned int poolSize,
                                                     Cuda::Memory::StashableAllocRegistrar* registrar,
                                                     DeviceHistogramTypes type)
    : TraceHistogramAccumulator(poolId, poolSize)
{
    switch (type)
    {
    case DeviceHistogramTypes::GlobalInterleaved:
    case DeviceHistogramTypes::SharedInterleaved2DBlock:
        {
            impl_ = std::make_unique<HistInterleavedZmw>(poolSize, registrar, type);
            break;
        }
    case DeviceHistogramTypes::GlobalContig:
    case DeviceHistogramTypes::GlobalContigCoopWarps:
    case DeviceHistogramTypes::SharedContigCoopWarps:
    case DeviceHistogramTypes::SharedContig2DBlock:
        {
            impl_ = std::make_unique<HistContigZmw>(poolSize, registrar, type);
            break;
        }
    default:
        throw PBException("Unexpected value for DeviceHistogramType");
    }
    CudaSynchronizeDefaultStream();
}

DeviceTraceHistogramAccum::~DeviceTraceHistogramAccum() = default;

}}}
