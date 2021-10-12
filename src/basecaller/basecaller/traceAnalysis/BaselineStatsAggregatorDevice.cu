// Copyright (c) 2021, Pacific Biosciences of California, Inc.
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
//  Defines class BaselineStatsAggregatorDevice, which customizes
//  BaselineStatsAggregator.


#include <basecaller/traceAnalysis/BaselineStatsAggregatorDevice.h>

#include <common/cuda/memory/DeviceOnlyArray.cuh>
#include <common/cuda/streams/LaunchManager.cuh>

using namespace PacBio::Cuda;
using namespace PacBio::Cuda::Memory;
using namespace PacBio::Cuda::Utility;
using namespace PacBio::Mongo;
using namespace PacBio::Mongo::Data;

namespace {

// TODO: Move MergeAutocorr and MergeStat in a common functionality so that
// it would be shared among BaselineStatsAggregatorDevice, 
// BasecallingMetricsAccumulatorDevice and AutocorrAccumulator
__device__ void MergeStat(StatAccumState& l, const StatAccumState& r)
{
    l.moment0[threadIdx.x] += r.moment0[threadIdx.x];
    l.moment1[threadIdx.x] += r.moment1[threadIdx.x];
    l.moment2[threadIdx.x] += r.moment2[threadIdx.x];
    l.offset[threadIdx.x] += r.offset[threadIdx.x];
}

__device__ void MergeAutocorr(AutocorrAccumState& l, const AutocorrAccumState& r)
{
    auto i = threadIdx.x;
    auto lag = AutocorrAccumState::lag;

    auto lbi_ = l.bIdx[0][i];
    auto rbi_ = l.bIdx[1][i];
    auto that_lbi_ = r.bIdx[0][i];
    auto that_rbi_ = r.bIdx[1][i];

    // Merge common statistics before processing tails
    MergeStat(l.basicStats, r.basicStats);
    l.moment2[i]      += r.moment2[i];

    auto n1 = lag - that_lbi_;  // that lBuf may be not filled up
    for (uint16_t k = 0; k < lag - n1; k++)
    {
        // Sum of muls of overlapping elements
        l.moment2[i]      += r.lBuf[k][i] * l.rBuf[(rbi_+k)%lag][i];
        // Accept the whole right buffer
        l.rBuf[(rbi_+k)%lag][i] = r.rBuf[(that_rbi_+n1+k)%lag][i];
    }

    auto n2 = lag - lbi_;      // this lBuf may be not filled up
    for (uint16_t k = 0; k < n2; ++k)
    {
        // No need to adjust m2_ as excessive values were mul by 0
        l.lBuf[lbi_+k][i] = r.lBuf[k][i];
    }

    // Advance buffer indices
    l.bIdx[0][i] += n2;                              // lbi
    l.bIdx[1][i] += (lag-n1); l.bIdx[1][i] %= lag;   // rbi
}

__global__ void MergeBaselinerStats(DeviceView<BaselinerStatAccumState> l,
                                    DeviceView<const BaselinerStatAccumState> r)
{
    assert(blockDim.x == laneSize);
    auto& lb = l[blockIdx.x];
    const auto& rb = r[blockIdx.x];

    MergeAutocorr(lb.fullAutocorrState, rb.fullAutocorrState);
    MergeStat(lb.baselineStats, rb.baselineStats);

    lb.traceMin[threadIdx.x] = min(lb.traceMin[threadIdx.x], rb.traceMin[threadIdx.x]);
    lb.traceMax[threadIdx.x] = max(lb.traceMax[threadIdx.x], rb.traceMax[threadIdx.x]);
    lb.rawBaselineSum[threadIdx.x] += rb.rawBaselineSum[threadIdx.x];
}

template <typename T>
__device__ void ResetArray(CudaArray<T, laneSize>& arr, T val = 0)
{
    assert(blockDim.x == laneSize);
    arr[threadIdx.x] = val;
}
__device__ void ResetStat(StatAccumState& stat)
{
    ResetArray(stat.offset);
    ResetArray(stat.moment0);
    ResetArray(stat.moment1);
    ResetArray(stat.moment2);
}
__device__ void ResetAutoCorr(AutocorrAccumState& accum)
{
    ResetStat(accum.basicStats);
    ResetArray(accum.moment2);
    auto k = AutocorrAccumState::lag;
    k = AutocorrAccumState::lag; while (k--) ResetArray(accum.lBuf[k]);
    k = AutocorrAccumState::lag; while (k--) ResetArray(accum.rBuf[k]);
    ResetArray(accum.bIdx[0]);
    ResetArray(accum.bIdx[1]);
}
__global__ void ResetStats(DeviceView<BaselinerStatAccumState> stats)
{
    auto& blockStats = stats[blockIdx.x];
    ResetAutoCorr(blockStats.fullAutocorrState);
    ResetArray(blockStats.traceMin, std::numeric_limits<int16_t>::max());
    ResetArray(blockStats.traceMax, std::numeric_limits<int16_t>::lowest());
    ResetStat(blockStats.baselineStats);
    ResetArray(blockStats.rawBaselineSum);
}

}

namespace PacBio {
namespace Mongo {
namespace Basecaller {

class BaselineStatsAggregatorDevice::Impl
{
public:
    Impl(unsigned int poolSize,
         StashableAllocRegistrar* registrar)
        : data_(registrar, SOURCE_MARKER(), poolSize)
        , poolSize_(poolSize)
    {
        ResetImpl();
    }

    void AddMetricsImpl(const BaselinerMetrics& metrics)
    {
        PBLauncher(MergeBaselinerStats, poolSize_, laneSize)(data_, metrics.baselinerStats);
        CudaSynchronizeDefaultStream();
    }

    BaselinerMetrics TraceStatsImpl() const
    {
        auto ret = data_.CopyAsUnifiedCudaArray(SyncDirection::HostReadDeviceWrite,
                                                SOURCE_MARKER());
        return BaselinerMetrics(std::move(ret));
    }

    void ResetImpl()
    {
        PBLauncher(ResetStats, poolSize_, laneSize)(data_);
    }

private:
    DeviceOnlyArray<Data::BaselinerStatAccumState> data_;
    uint32_t poolSize_;
};

BaselineStatsAggregatorDevice::BaselineStatsAggregatorDevice(uint32_t poolId, unsigned int poolSize,
                                                             Cuda::Memory::StashableAllocRegistrar* registrar)
    : BaselineStatsAggregator(poolId, poolSize)
    , impl_(std::make_unique<Impl>(poolSize, registrar))
{}

BaselineStatsAggregatorDevice::~BaselineStatsAggregatorDevice() = default;

void BaselineStatsAggregatorDevice::AddMetricsImpl(const Data::BaselinerMetrics& metrics)
{
    impl_->AddMetricsImpl(metrics);
}

Data::BaselinerMetrics BaselineStatsAggregatorDevice::TraceStatsImpl() const
{
    return impl_->TraceStatsImpl();
}

void BaselineStatsAggregatorDevice::ResetImpl()
{
    impl_->ResetImpl();
}

}}}     // namespace PacBio::Mongo::Basecaller
