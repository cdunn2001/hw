// Copyright (c) 2019, Pacific Biosciences of California, Inc.
//
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted (subject to the limitations in the
// disclaimer below) provided that the following conditions are met:
//
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//
//  * Redistributions in binary form must reproduce the above
//    copyright notice, this list of conditions and the following
//    disclaimer in the documentation and/or other materials provided
//    with the distribution.
//
//  * Neither the name of Pacific Biosciences nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE
// GRANTED BY THIS LICENSE. THIS SOFTWARE IS PROVIDED BY PACIFIC
// BIOSCIENCES AND ITS CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
// OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL PACIFIC BIOSCIENCES OR ITS
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
// USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
// OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
// SUCH DAMAGE.

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
std::unique_ptr<Data::BasecallingMetricsFactory<HFMetricsFilter::BasecallingMetricsT, laneSize>> HFMetricsFilter::metricsFactory_;

void HFMetricsFilter::Configure(const Data::BasecallerMetricsConfig& config)
{
    framesPerHFMetricBlock_ = Data::GetPrimaryConfig().framesPerHFMetricBlock;
    if (framesPerHFMetricBlock_ < Data::GetPrimaryConfig().framesPerChunk)
        throw PBException("HFMetric frame block size cannot be smaller than trace block size!");

    PBLOG_INFO << "framesPerHFMetricBlock = " << framesPerHFMetricBlock_;
    sandwichTolerance_ = config.sandwichTolerance;
    PBLOG_INFO << "Definition of sandwich = " << sandwichTolerance_ << " ipd";
    frameRate_ = Data::GetPrimaryConfig().sensorFrameRate;
    realtimeActivityLabels_ = Data::GetPrimaryConfig().realtimeActivityLabels;

    Data::BatchDimensions dims;
    dims.framesPerBatch = Data::GetPrimaryConfig().framesPerChunk;
    dims.laneWidth = laneSize;
    dims.lanesPerBatch = Data::GetPrimaryConfig().lanesPerPool;
    lanesPerBatch_ = dims.lanesPerBatch;
    zmwsPerBatch_ = dims.ZmwsPerBatch();

    InitAllocationPools(true);
    Data::BasecallingMetrics<laneSize>::Configure(std::make_unique<MetricsAccumulatorT>());
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

    SyncDirection syncDir = hostExecution ? SyncDirection::HostWriteDeviceRead : SyncDirection::HostReadDeviceWrite;
    metricsFactory_ = std::make_unique<Data::BasecallingMetricsFactory<HFMetricsFilter::BasecallingMetricsT, laneSize>>(
            dims, syncDir, true);
}

void HFMetricsFilter::DestroyAllocationPools()
{
    metricsFactory_.release();
}

HFMetricsFilter::HFMetricsFilter(uint32_t poolId)
    : poolId_(poolId)
    , framesSeen_(0)
{}

void HostHFMetricsFilter::FinalizeBlock()
{
    PBLOG_INFO << "Finalizing HFMetricsBlock";
    for (size_t l = 0; l < lanesPerBatch_; ++l)
    {
        metrics_->GetHostView()[l].FinalizeMetrics();
    }

    if (realtimeActivityLabels_)
    {
        PBLOG_INFO << "Labeling Sequencing Activity";
        // Adds activity label directly to basecalling metrics object:
        for (size_t l = 0; l < lanesPerBatch_; ++l)
        {
            ActivityLabeler::LabelBlock(metrics_->GetHostView()[l], frameRate_, laneSize);
        }
    }
}

HostHFMetricsFilter::~HostHFMetricsFilter() = default;

void HFMetricsFilter::AddBasecalls(const Data::BasecallBatch& basecallBatch)
{
    const auto& basecalls = basecallBatch.Basecalls();
    for (size_t l = 0; l < lanesPerBatch_; l++)
    {
        const auto& laneCalls = basecalls.LaneView(l);
        metrics_->GetHostView()[l].Count(laneCalls, basecallBatch.Dims().framesPerBatch);
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

std::unique_ptr<HostHFMetricsFilter::ElementTypeOut> HostHFMetricsFilter::Process(
        const BasecallBatchT& basecallBatch,
        const BaselineStatsT& baselineStats)
{
    if (framesSeen_ == 0)
    {
        metrics_ = std::move(metricsFactory_->NewBatch());
        for (size_t l = 0; l < lanesPerBatch_; l++)
        {
            metrics_->GetHostView()[l].Initialize();
        }
    }

    AddBasecalls(basecallBatch);
    AddBaselineStats(baselineStats);
    // TODO: AddPulseDetectionScore/Confidence
    // TODO: AddAutocorrelation
    framesSeen_ += basecallBatch.Dims().framesPerBatch;

    if (framesSeen_ >= framesPerHFMetricBlock_)
    {
        FinalizeBlock();
        framesSeen_ = 0;
        return std::move(metrics_);
    }
    return std::unique_ptr<ElementTypeOut>();
}



// TODO: Finish "NoOp" version of metrics
uint32_t MinimalHFMetricsFilter::sandwichTolerance_ = 0;
uint32_t MinimalHFMetricsFilter::framesPerHFMetricBlock_ = 0;
double MinimalHFMetricsFilter::frameRate_ = 0;
bool MinimalHFMetricsFilter::realtimeActivityLabels_ = 0;
uint32_t MinimalHFMetricsFilter::zmwsPerBatch_;
std::unique_ptr<Data::BasecallingMetricsFactory<MinimalHFMetricsFilter::BasecallingMetricsT, laneSize>> MinimalHFMetricsFilter::metricsFactory_;

MinimalHFMetricsFilter::MinimalHFMetricsFilter(uint32_t poolId)
    : HFMetricsFilter(poolId)
    , poolId_(poolId)

{}

std::unique_ptr<MinimalHFMetricsFilter::ElementTypeOut> MinimalHFMetricsFilter::Process(
        const BasecallBatchT& basecallBatch,
        const BaselineStatsT& baselineStats)
{
    (void)baselineStats; // ignore
    metrics_ = std::move(metricsFactory_->NewBatch());
    AddBasecalls(basecallBatch);
    FinalizeBlock();
    return std::move(metrics_);
}

void MinimalHFMetricsFilter::FinalizeBlock()
{
    PBLOG_INFO << "Finalizing MinimalHFMetricsBlock";
}

MinimalHFMetricsFilter::~MinimalHFMetricsFilter() = default;



}}} // PacBio::Mongo::Basecaller
