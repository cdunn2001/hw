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

__global__ void Binning1(Data::GpuBatchData<const PBShort2> traces,
                         DeviceView<Data::LaneHistogram<float, uint16_t>> hists
                         )
{
    auto zmw = traces.ZmwData(blockIdx.x, threadIdx.x);
    auto& hist = hists[blockIdx.x];

    float2 lowBound {hist.lowBound[2*threadIdx.x], hist.lowBound[2*threadIdx.x+1]};
    float2 binSize {hist.binSize[2*threadIdx.x], hist.binSize[2*threadIdx.x+1]};
    ushort2 lowOutlier{hist.outlierCountLow[2*threadIdx.x], hist.outlierCountLow[2*threadIdx.x+1]};
    ushort2 highOutlier{hist.outlierCountHigh[2*threadIdx.x], hist.outlierCountHigh[2*threadIdx.x+1]};

    for (const auto val : zmw)
    {
        auto idx = 2*threadIdx.x;
        int bin = (val.X() - lowBound.x) / binSize.x;
        if (bin < 0) lowOutlier.x++;
        else if (bin >= hist.numBins) highOutlier.x++;
        else hist.binCount[bin][idx]++;

        idx++;

        bin = (val.Y() - lowBound.y) / binSize.y;
        if (bin < 0) lowOutlier.y++;
        else if (bin >= hist.numBins) highOutlier.y++;
        else hist.binCount[bin][idx]++;
    }

    hist.outlierCountLow[2*threadIdx.x] = lowOutlier.x;
    hist.outlierCountLow[2*threadIdx.x+1] = lowOutlier.y;
    hist.outlierCountHigh[2*threadIdx.x] = highOutlier.x;
    hist.outlierCountHigh[2*threadIdx.x+1] = highOutlier.y;
}

__global__ void CopyTo(DeviceView<Data::LaneHistogram<float, uint16_t>> source,
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

__device__ void ResetHistsBounds(DeviceView<Data::LaneHistogram<float, uint16_t>> hists,
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

__global__ void ResetHistsStats(DeviceView<Data::LaneHistogram<float, uint16_t>> hists,
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

struct HistType1 : public DeviceTraceHistogramAccumHidden::ImplBase
{
    HistType1(unsigned int poolSize,
              Cuda::Memory::StashableAllocRegistrar* registrar)
        : DeviceTraceHistogramAccumHidden::ImplBase(poolSize)
        , data_(registrar, SOURCE_MARKER(), poolSize)
    {}

    void AddBatchImpl(const Data::TraceBatch<int16_t>& traces) override
    {
        PBLauncher(Binning1, this->PoolSize(), laneSize/2)(traces, data_);
    }

    void ResetImpl(const Cuda::Memory::UnifiedCudaArray<LaneHistBounds>& bounds) override
    {
        assert(bounds.Size() == PoolSize());
        PBLauncher(ResetHistsBounds, this->PoolSize(), laneSize)(data_, bounds);
    }


    void ResetImpl(const Data::BaselinerMetrics& metrics) override
    {
        assert(metrics.baselinerStats.Size() == PoolSize());
        PBLauncher(ResetHistsStats, this->PoolSize(), laneSize)(data_, metrics.baselinerStats);
    }

    void Copy(Data::PoolHistogram<float, uint16_t>& dest) override
    {
        PBLauncher(CopyTo, this->PoolSize(), laneSize)(data_, dest.data);
    }

private:
    DeviceOnlyArray<LaneHistogram<float, uint16_t>> data_;
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
            impl_ = std::make_unique<HistType1>(poolSize, registrar);
            break;
        }
    default:
        throw PBException("Not supported implementation type");
    }
    CudaSynchronizeDefaultStream();
}

DeviceTraceHistogramAccumHidden::~DeviceTraceHistogramAccumHidden() = default;

}}}
