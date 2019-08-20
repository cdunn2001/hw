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
uint32_t HFMetricsFilter::framesPerChunk_;
uint32_t HFMetricsFilter::lanesPerBatch_;
uint32_t HFMetricsFilter::zmwsPerBatch_;
std::unique_ptr<Data::BasecallingMetricsFactory<laneSize>> HFMetricsFilter::metricsFactory_;

void HFMetricsFilter::Configure(uint32_t sandwichTolerance,
                                uint32_t framesPerHFMetricBlock,
                                uint32_t framesPerChunk,
                                double frameRate,
                                bool realtimeActivityLabels,
                                uint32_t lanesPerBatch)
{
    framesPerHFMetricBlock_ = framesPerHFMetricBlock;
    framesPerChunk_ = framesPerChunk;
    if (framesPerHFMetricBlock_ < framesPerChunk)
        throw PBException("HFMetric frame block size cannot be smaller than "
                          "trace block size!");

    sandwichTolerance_ = sandwichTolerance;
    frameRate_ = frameRate;
    realtimeActivityLabels_ = realtimeActivityLabels;
    Data::BatchDimensions dims;
    dims.framesPerBatch = framesPerChunk;
    dims.lanesPerBatch = lanesPerBatch;
    dims.laneWidth = laneSize;
    lanesPerBatch_ = lanesPerBatch;
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
    dims.framesPerBatch = framesPerChunk_;
    dims.lanesPerBatch = lanesPerBatch_;
    dims.laneWidth = laneSize;

    SyncDirection syncDir = hostExecution ? SyncDirection::HostWriteDeviceRead
                                          : SyncDirection::HostReadDeviceWrite;
    metricsFactory_ = std::make_unique<Data::BasecallingMetricsFactory<laneSize>>(
            dims, syncDir, true);
}

void HFMetricsFilter::DestroyAllocationPools()
{
    metricsFactory_.release();
}

void HostHFMetricsFilter::FinalizeBlock()
{
    for (size_t l = 0; l < lanesPerBatch_; ++l)
    {
        metrics_[l].FinalizeMetrics(realtimeActivityLabels_, frameRate_);
    }
}

HostHFMetricsFilter::~HostHFMetricsFilter() = default;

void HostHFMetricsFilter::AddPulses(const Data::PulseBatch& pulseBatch)
{
    const auto& pulses = pulseBatch.Pulses();
    for (size_t l = 0; l < lanesPerBatch_; l++)
    {
        const auto& laneCalls = pulses.LaneView(l);
        metrics_[l].Count(laneCalls, pulseBatch.Dims().framesPerBatch);
        metrics_[l].AddPulseDetectionMetrics(pulseBatch.PdMetrics().GetHostView()[l]);
    }
}

void HostHFMetricsFilter::AddModels(const ModelsT& modelsBatch)
{
    for (size_t l = 0; l < lanesPerBatch_; l++)
    {
        const auto& models = modelsBatch.GetHostView()[l];
        metrics_[l].AddModels(models);
    }
}

void HostHFMetricsFilter::AddBaselinerStats(const BaselinerStatsT& baselineStats)
{
    for (size_t l = 0; l < lanesPerBatch_; l++)
    {
        const auto& laneStats = baselineStats.GetHostView()[l];
        metrics_[l].AddBaselinerStats(laneStats);
    }
}

std::unique_ptr<HostHFMetricsFilter::BasecallingMetricsBatchT>
HostHFMetricsFilter::Process(
        const PulseBatchT& pulseBatch,
        const BaselinerStatsT& baselinerStats,
        const ModelsT& models)
{
    if (framesSeen_ == 0)
    {
        for (size_t l = 0; l < lanesPerBatch_; ++l)
        {
            metrics_[l].Reset();
        }
    }
    AddBaselinerStats(baselinerStats);
    AddPulses(pulseBatch);
    AddModels(models);
    framesSeen_ += pulseBatch.Dims().framesPerBatch;

    if (framesSeen_ >= framesPerHFMetricBlock_)
    {
        FinalizeBlock();
        framesSeen_ = 0;
        auto ret = metricsFactory_->NewBatch();
        for (size_t l = 0; l < lanesPerBatch_; ++l)
        {
            metrics_[l].PopulateBasecallingMetrics(
                    ret->GetHostView()[l]);
        }
        return ret;
    }
    return std::unique_ptr<BasecallingMetricsBatchT>();
}


NoHFMetricsFilter::~NoHFMetricsFilter() = default;



}}} // PacBio::Mongo::Basecaller
