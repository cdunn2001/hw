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

__device__ void MergeStat(StatAccumState& l, const StatAccumState& r)
{
    l.moment0[threadIdx.x] += r.moment0[threadIdx.x];
    l.moment1[threadIdx.x] += r.moment1[threadIdx.x];
    l.moment2[threadIdx.x] += r.moment2[threadIdx.x];
    l.offset[threadIdx.x] += r.offset[threadIdx.x];
}
__device__ void MergeAutocorr(AutocorrAccumState& l, const AutocorrAccumState& r)
{
    MergeStat(l.basicStats, r.basicStats);
    l.moment1First[threadIdx.x] += r.moment1First[threadIdx.x];
    l.moment1Last[threadIdx.x] += r.moment1Last[threadIdx.x];
    l.moment2[threadIdx.x] += r.moment2[threadIdx.x];
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
__device__ void ResetArray(CudaArray<T, laneSize>& arr)
{
    assert(blockDim.x == laneSize);
    arr[threadIdx.x] = 0;
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
    ResetArray(accum.moment1First);
    ResetArray(accum.moment1Last);
    ResetArray(accum.moment2);
}
__global__ void ResetStats(DeviceView<BaselinerStatAccumState> stats)
{
    auto& blockStats = stats[blockIdx.x];
    ResetAutoCorr(blockStats.fullAutocorrState);
    ResetArray(blockStats.traceMin);
    ResetArray(blockStats.traceMax);
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
        // The 4th argument is there so that we initialize the array by
        // copying to each entry a value initialized `BaselinerStatAccumState`.
        // Otherwise the DeviceOnlyArray will do "default initialization" of each
        // entry.  Value initialization is necessary so that all the data is zeroed
        // out
        : data_(registrar, SOURCE_MARKER(), poolSize, Data::BaselinerStatAccumState{})
        , poolSize_(poolSize)
    {}

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
