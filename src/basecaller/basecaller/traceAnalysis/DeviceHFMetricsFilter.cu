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
/// \file   DeviceHFMetricsFilter.cpp
/// \brief  A filter for computing or aggregating trace- and pulse-metrics
///         on a time scale equal to or greater than the standard block size.

#include "DeviceHFMetricsFilter.h"
#include <dataTypes/BatchData.cuh>
#include <dataTypes/BatchVectors.cuh>

namespace PacBio {
namespace Mongo {
namespace Basecaller {

size_t threadsPerBlock_ = 32;

__global__ void InitializeMetrics(
        Cuda::Memory::DeviceView<DeviceHFMetricsFilter::BasecallingMetricsAccumulatorT> metrics)
{
    // TODO
}

__global__ void FinalizeMetrics(
        bool realtimeActivityLabels,
        float frameRate,
        Cuda::Memory::DeviceView<DeviceHFMetricsFilter::BasecallingMetricsAccumulatorT> metrics)
{
    // TODO
}

__global__ void ProcessChunk(
        Cuda::Memory::DeviceView<const DeviceHFMetricsFilter::BaselinerStatsT> baselinerStats,
        Cuda::Memory::DeviceView<const DeviceHFMetricsFilter::ModelsT> models,
        Data::GpuBatchVectors<Data::Pulse> pulses,
        Cuda::Memory::DeviceView<const Data::PulseDetectionMetrics> pdMetrics,
        Cuda::Memory::DeviceView<DeviceHFMetricsFilter::BasecallingMetricsAccumulatorT> metrics)
{
    // TODO
}

__global__ void PopulateBasecallingMetrics(
        Cuda::Memory::DeviceView<DeviceHFMetricsFilter::BasecallingMetricsAccumulatorT> metrics,
        Cuda::Memory::DeviceView<DeviceHFMetricsFilter::BasecallingMetricsT> outMetrics)
{
    // TODO
}

void DeviceHFMetricsFilter::FinalizeBlock()
{
    FinalizeMetrics<<<lanesPerBatch_, threadsPerBlock_>>>(realtimeActivityLabels_, frameRate_, metrics_.GetDeviceView());
}

DeviceHFMetricsFilter::~DeviceHFMetricsFilter() = default;

std::unique_ptr<DeviceHFMetricsFilter::BasecallingMetricsBatchT>
DeviceHFMetricsFilter::Process(
        const PulseBatchT& pulseBatch,
        const BaselinerStatsBatchT& baselinerStats,
        const ModelsBatchT& models)
{
    if (framesSeen_ == 0)
    {
        InitializeMetrics<<<lanesPerBatch_,
                            threadsPerBlock_>>>(metrics_.GetDeviceView());
    }
    ProcessChunk<<<lanesPerBatch_,
                    threadsPerBlock_>>>(baselinerStats.GetDeviceHandle(),
                                        models.GetDeviceHandle(),
                                        pulseBatch.Pulses(),
                                        pulseBatch.PdMetrics().GetDeviceHandle(),
                                        metrics_.GetDeviceView());
    framesSeen_ += pulseBatch.Dims().framesPerBatch;

    if (framesSeen_ >= framesPerHFMetricBlock_)
    {
        FinalizeBlock();
        framesSeen_ = 0;
        auto ret = metricsFactory_->NewBatch();
        PopulateBasecallingMetrics<<<lanesPerBatch_,
                                     threadsPerBlock_>>>(metrics_.GetDeviceView(),
                                                         ret->GetDeviceHandle());
        return ret;
    }
    return std::unique_ptr<BasecallingMetricsBatchT>();
}



}}} // PacBio::Mongo::Basecaller
