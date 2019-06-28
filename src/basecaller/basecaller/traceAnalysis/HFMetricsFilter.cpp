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
uint32_t HFMetricsFilter::zmwsPerBatch_;

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
    zmwsPerBatch_ = dims.zmwsPerBatch();
}

// TODO (mds 20190628) share a metricsPool with BasecallBatchFactory?
HFMetricsFilter::HFMetricsFilter(uint32_t poolId)
    : poolId_(poolId)
    , framesSeen_(0)
    , metrics_(zmwsPerBatch_, Cuda::Memory::SyncDirection::HostWriteDeviceRead, true)
    , metricsPool_(std::make_shared<Cuda::Memory::DualAllocationPools>(
        zmwsPerBatch_*sizeof(Data::BasecallingMetrics), true))
{}

void HFMetricsFilter::AddBatch(ElementTypeIn& batch)
{
    auto& basecalls = batch.Basecalls();
    for (uint32_t l = 0; l < batch.Dims().lanesPerBatch; l++)
    {
        auto laneCalls = basecalls.LaneView(l);
        for (uint32_t z = 0; z < laneSize; ++z)
        {
            for (uint32_t b = 0; b < laneCalls.size(z); ++b)
            {
                metrics_.GetHostView()[l * laneSize + z].Count(laneCalls(z, b));
            }
        }
    }
}

void HFMetricsFilter::Finalize()
{
    // Add HQR blocklabel, etc
}

void HFMetricsFilter::NewMetrics()
{
    metrics_ = Cuda::Memory::UnifiedCudaArray<Data::BasecallingMetrics>(
            zmwsPerBatch_, Cuda::Memory::SyncDirection::HostWriteDeviceRead, true, metricsPool_);
}

void HFMetricsFilter::operator()(ElementTypeIn& batch)
{
    if (framesSeen_ == 0)
    {
        NewMetrics();
    }

    AddBatch(batch);
    framesSeen_ += batch.Dims().framesPerBatch;

    if (framesSeen_ >= framesPerHFMetricBlock_)
    {
        Finalize();
        batch.Metrics(std::move(metrics_));
        framesSeen_ = 0;
    }
}


}}} // PacBio::Mongo::Basecaller
