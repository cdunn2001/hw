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
//  Description:
/// \file   HFMetricsFilter.cpp
/// \brief  A filter for computing or aggregating trace- and pulse-metrics
///         on a time scale equal to or greater than the standard block size.

#include "HFMetricsFilter.h"

namespace PacBio {
namespace Mongo {
namespace Basecaller {

uint32_t HFMetricsFilter::sandwichTolerance_ = 0;
uint32_t HFMetricsFilter::framesPerHFMetricBlock_ = 0;
double HFMetricsFilter::frameRate_ = 0;
bool HFMetricsFilter::realtimeActivityLabels_ = 0;
uint32_t HFMetricsFilter::lanesPerBatch_;
uint32_t HFMetricsFilter::zmwsPerBatch_;
std::unique_ptr<Data::BasecallingMetricsFactory<laneSize>> HFMetricsFilter::metricsFactory_;
std::unique_ptr<Data::BasecallingMetricsAccumulatorFactory<laneSize>> HFMetricsFilter::metricsAccumulatorFactory_;

void HFMetricsFilter::Configure(const Data::BasecallerMetricsConfig& config)
{
    framesPerHFMetricBlock_ = Data::GetPrimaryConfig().framesPerHFMetricBlock;
    if (framesPerHFMetricBlock_ < Data::GetPrimaryConfig().framesPerChunk)
        throw PBException("HFMetric frame block size cannot be smaller than "
                          "trace block size!");

    sandwichTolerance_ = config.sandwichTolerance;
    frameRate_ = Data::GetPrimaryConfig().sensorFrameRate;
    realtimeActivityLabels_ = Data::GetPrimaryConfig().realtimeActivityLabels;

    Data::BatchDimensions dims;
    dims.framesPerBatch = Data::GetPrimaryConfig().framesPerChunk;
    dims.laneWidth = laneSize;
    dims.lanesPerBatch = Data::GetPrimaryConfig().lanesPerPool;
    lanesPerBatch_ = dims.lanesPerBatch;
    zmwsPerBatch_ = dims.ZmwsPerBatch();

    constexpr bool hostExecution = true;

    InitAllocationPools(hostExecution);
}

void HFMetricsFilter::Finalize()
{
    DestroyAllocationPools();
}

void HFMetricsFilter::InitAllocationPools(bool hostExecution)
{
    using Cuda::Memory::SyncDirection;

    Data::BatchDimensions dims;
    dims.framesPerBatch = Data::GetPrimaryConfig().framesPerChunk;
    dims.lanesPerBatch = Data::GetPrimaryConfig().lanesPerPool;
    dims.laneWidth = laneSize;

    SyncDirection syncDir = hostExecution ? SyncDirection::HostWriteDeviceRead
                                          : SyncDirection::HostReadDeviceWrite;
    metricsFactory_ = std::make_unique<Data::BasecallingMetricsFactory<laneSize>>(
            dims, syncDir, true);
    metricsAccumulatorFactory_ = std::make_unique<
        Data::BasecallingMetricsAccumulatorFactory<laneSize>>(
            dims, syncDir, true);
}

void HFMetricsFilter::DestroyAllocationPools()
{
    metricsFactory_.release();
    metricsAccumulatorFactory_.release();
}

void HostHFMetricsFilter::FinalizeBlock()
{
    for (size_t l = 0; l < lanesPerBatch_; ++l)
    {
        metrics_->GetHostView()[l].FinalizeMetrics(realtimeActivityLabels_,
                                                   frameRate_);
    }
}

HostHFMetricsFilter::~HostHFMetricsFilter() = default;

void HFMetricsFilter::AddPulses(const Data::PulseBatch& pulseBatch)
{
    const auto& basecalls = pulseBatch.Pulses();
    for (size_t l = 0; l < lanesPerBatch_; l++)
    {
        const auto& laneCalls = basecalls.LaneView(l);
        metrics_->GetHostView()[l].Count(laneCalls,
                                         pulseBatch.Dims().framesPerBatch);
    }
}

void HostHFMetricsFilter::AddModels(const ModelsT& modelsBatch)
{
    for (size_t l = 0; l < lanesPerBatch_; l++)
    {
        const auto& models = modelsBatch.GetHostView()[l];
        metrics_->GetHostView()[l].AddModels(models);
    }
}

void HostHFMetricsFilter::AddBaselineStats(const BaselineStatsT& baselineStats)
{
    for (size_t l = 0; l < lanesPerBatch_; l++)
    {
        const auto& laneStats = baselineStats.GetHostView()[l];
        metrics_->GetHostView()[l].AddBaselineStats(laneStats);
    }
}

std::unique_ptr<HostHFMetricsFilter::BasecallingMetricsBatchT>
HostHFMetricsFilter::Process(
        const PulseBatchT& pulseBatch,
        const BaselineStatsT& baselineStats,
        const ModelsT& models)
{
    if (framesSeen_ == 0)
    {
        for (size_t l = 0; l < lanesPerBatch_; ++l)
        {
            metrics_->GetHostView()[l].Reset();
        }
    }

    AddPulses(pulseBatch);
    AddBaselineStats(baselineStats);
    AddModels(models);
    // TODO: AddPulseDetectionScore/Confidence
    // TODO: AddAutocorrelation
    framesSeen_ += pulseBatch.Dims().framesPerBatch;

    if (framesSeen_ >= framesPerHFMetricBlock_)
    {
        FinalizeBlock();
        framesSeen_ = 0;
        auto ret = metricsFactory_->NewBatch();
        for (size_t l = 0; l < lanesPerBatch_; ++l)
        {
            metrics_->GetHostView()[l].PopulateBasecallingMetrics(
                    ret->GetHostView()[l]);
        }
        return ret;
    }
    return std::unique_ptr<BasecallingMetricsBatchT>();
}


NoHFMetricsFilter::~NoHFMetricsFilter() = default;



}}} // PacBio::Mongo::Basecaller
