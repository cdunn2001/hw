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

void DeviceTraceHistogramAccumHidden::Configure(const Data::BasecallerTraceHistogramConfig& sigConfig)
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

    auto ptr = reinterpret_cast<const int16_t*>(traces.BlockData(blockIdx.x)) + threadIdx.x;
    for (int i = 0; i < traces.NumFrames(); ++i)
    {
        auto idx = threadIdx.x;
        int bin = (*ptr - lowBound) / binSize;
        if (bin < 0) lowOutlier++;
        else if (bin >= hist.numBins) highOutlier++;
        else hist.binCount[idx][bin]++;

        ptr += blockDim.x;
    }

    hist.outlierCountLow[threadIdx.x] = lowOutlier;
    hist.outlierCountHigh[threadIdx.x] = highOutlier;
}

__global__ void BinningGlobalContigAtomic(Data::GpuBatchData<const PBShort2> traces,
                                          DeviceView<LaneHistogramTrans> hists)
{
    assert(traces.NumFrames() % blockDim.x == 0);
    auto& hist = hists[blockIdx.x];

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
            else if (bin > hist.numBins) bin = hist.numBins;

            auto same = __match_any_sync(0xFFFFFFFF, bin);
            // Number of threads with the same bin
            auto count = __popc(same);
            // Thread with the most significant bit gets to own the update
            bool owner = (32 - __clz(same) -1 ) == threadIdx.x;

            if (owner)
            {
                if (bin < 0) hist.outlierCountLow[2*zmw] += count;
                else if (bin == hist.numBins) hist.outlierCountHigh[2*zmw] += count;
                else hist.binCount[2*zmw][bin] += count;
            }

            bin = (val.Y() - lowBound.y) / binSize.y;
            if (bin < 0) bin = -1;
            else if (bin > hist.numBins) bin = hist.numBins;

            same = __match_any_sync(0xFFFFFFFF, bin);
            // Number of threads with the same bin
            count = __popc(same);
            // Thread with the most significant bit gets to own the update
            owner = (32 - __clz(same) -1 ) == threadIdx.x;

            if (owner)
            {
                if (bin < 0) hist.outlierCountLow[2*zmw+1] +=count;
                else if (bin == hist.numBins) hist.outlierCountHigh[2*zmw+1] += count;
                else hist.binCount[2*zmw+1][bin] += count;
            }
        }
    }
}


__global__ void BinningGlobalInterleaved(Data::GpuBatchData<const PBShort2> traces,
                                         DeviceView<Data::LaneHistogram<float, uint16_t>> hists)
{
    auto& hist = hists[blockIdx.x];

    float lowBound = hist.lowBound[threadIdx.x];
    float binSize = hist.binSize[threadIdx.x];
    ushort lowOutlier = hist.outlierCountLow[threadIdx.x];
    ushort highOutlier = hist.outlierCountHigh[threadIdx.x];

    auto ptr = reinterpret_cast<const int16_t*>(traces.BlockData(blockIdx.x)) + threadIdx.x;
    for (int i = 0; i < traces.NumFrames(); ++i)
    {
        auto idx = threadIdx.x;
        int bin = (*ptr - lowBound) / binSize;
        if (bin < 0) lowOutlier++;
        else if (bin >= hist.numBins) highOutlier++;
        else hist.binCount[bin][idx]++;

        ptr += blockDim.x;
    }

    hist.outlierCountLow[threadIdx.x] = lowOutlier;
    hist.outlierCountHigh[threadIdx.x] = highOutlier;
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
    auto blMean = mom1 / blCount;
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
    auto blMean = mom1 / blCount;
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

struct DeviceTraceHistogramAccumHidden::ImplBase
{
    ImplBase(size_t poolSize)
        : poolSize_(poolSize)
    {}

    virtual ~ImplBase() = default;

    virtual void AddBatchImpl(const Data::TraceBatch<DataType>& traces) = 0;

    virtual void ResetImpl(const Cuda::Memory::UnifiedCudaArray<LaneHistBounds>& bounds) = 0;

    virtual void ResetImpl(const Data::BaselinerMetrics& metrics) = 0;

    virtual void Copy(Data::PoolHistogram<HistDataType, HistCountType>& dest) = 0;

    size_t PoolSize() const { return poolSize_; }
private:
    size_t poolSize_;
};

struct HistGlobalInterleaved : public DeviceTraceHistogramAccumHidden::ImplBase
{
    HistGlobalInterleaved(unsigned int poolSize,
              Cuda::Memory::StashableAllocRegistrar* registrar)
        : DeviceTraceHistogramAccumHidden::ImplBase(poolSize)
        , data_(registrar, SOURCE_MARKER(), poolSize)
    {}

    void AddBatchImpl(const Data::TraceBatch<int16_t>& traces) override
    {
        PBLauncher(BinningGlobalInterleaved, this->PoolSize(), laneSize)(traces, data_);
    }

    void ResetImpl(const Cuda::Memory::UnifiedCudaArray<LaneHistBounds>& bounds) override
    {
        assert(bounds.Size() == PoolSize());
        PBLauncher(ResetHistsInterleavedBounds, this->PoolSize(), laneSize)(data_, bounds);
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

struct HistGlobalContig : public DeviceTraceHistogramAccumHidden::ImplBase
{
    HistGlobalContig(unsigned int poolSize,
              Cuda::Memory::StashableAllocRegistrar* registrar)
        : DeviceTraceHistogramAccumHidden::ImplBase(poolSize)
        , data_(registrar, SOURCE_MARKER(), poolSize)
    {}

    void AddBatchImpl(const Data::TraceBatch<int16_t>& traces) override
    {
        PBLauncher(BinningGlobalContig, this->PoolSize(), laneSize)(traces, data_);
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

struct HistGlobalContigAtomic : public DeviceTraceHistogramAccumHidden::ImplBase
{
    HistGlobalContigAtomic(unsigned int poolSize,
              Cuda::Memory::StashableAllocRegistrar* registrar)
        : DeviceTraceHistogramAccumHidden::ImplBase(poolSize)
        , data_(registrar, SOURCE_MARKER(), poolSize)
    {}

    void AddBatchImpl(const Data::TraceBatch<int16_t>& traces) override
    {
        PBLauncher(BinningGlobalContigAtomic, this->PoolSize(), laneSize/2)(traces, data_);
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

void DeviceTraceHistogramAccumHidden::AddBatchImpl(const Data::TraceBatch<DataType>& traces)
{
    impl_->AddBatchImpl(traces);
    CudaSynchronizeDefaultStream();
}

void DeviceTraceHistogramAccumHidden::ResetImpl(const Cuda::Memory::UnifiedCudaArray<LaneHistBounds>& bounds)
{
    impl_->ResetImpl(bounds);
    CudaSynchronizeDefaultStream();
}

void DeviceTraceHistogramAccumHidden::ResetImpl(const Data::BaselinerMetrics& metrics)
{
    impl_->ResetImpl(metrics);
    CudaSynchronizeDefaultStream();
}

DeviceTraceHistogramAccumHidden::PoolHistType DeviceTraceHistogramAccumHidden::HistogramImpl() const
{
    PoolHistType hists(PoolId(), PoolSize(), SyncDirection::HostReadDeviceWrite);
    impl_->Copy(hists);
    return hists;
}

DeviceTraceHistogramAccumHidden::DeviceTraceHistogramAccumHidden(unsigned int poolId,
                                                     unsigned int poolSize,
                                                     DeviceHistogramTypes type,
                                                     Cuda::Memory::StashableAllocRegistrar* registrar)
    : TraceHistogramAccumulator(poolId, poolSize)
{
    switch (type)
    {
    case DeviceHistogramTypes::GlobalInterleaved:
        {
            impl_ = std::make_unique<HistGlobalInterleaved>(poolSize, registrar);
            break;
        }
    case DeviceHistogramTypes::GlobalContig:
        {
            impl_ = std::make_unique<HistGlobalContig>(poolSize, registrar);
            break;
        }
    case DeviceHistogramTypes::GlobalContigAtomic:
        {
            impl_ = std::make_unique<HistGlobalContigAtomic>(poolSize, registrar);
            break;
        }
    default:
        throw PBException("Not supported implementation type");
    }
    CudaSynchronizeDefaultStream();
}

DeviceTraceHistogramAccumHidden::~DeviceTraceHistogramAccumHidden() = default;

}}}
