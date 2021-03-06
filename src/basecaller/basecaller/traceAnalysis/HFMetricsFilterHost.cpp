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
/// \file   HFMetricsFilterHost.cpp
/// \brief  A filter for computing or aggregating trace- and pulse-metrics
///         on a time scale equal to or greater than the standard block size.

#include "HFMetricsFilterHost.h"

#include <tbb/task_arena.h>
#include <tbb/parallel_for.h>

namespace PacBio {
namespace Mongo {
namespace Basecaller {

void HFMetricsFilterHost::FinalizeBlock()
{
    for (auto& metric : metrics_)
    {
        metric.FinalizeMetrics(realtimeActivityLabels_, static_cast<float>(frameRate_));
    }
}

HFMetricsFilterHost::~HFMetricsFilterHost() = default;

void HFMetricsFilterHost::AddPulses(const Data::PulseBatch& pulseBatch)
{
    const auto& pulses = pulseBatch.Pulses();
    tbb::task_arena().execute([&] {
        tbb::parallel_for(uint32_t{0}, pulseBatch.Dims().lanesPerBatch, [&](size_t l) {
            const auto& laneCalls = pulses.LaneView(l);
            metrics_[l].Count(laneCalls, pulseBatch.Dims().framesPerBatch);
        });
    });
}

void HFMetricsFilterHost::AddModels(const ModelsBatchT& modelsBatch)
{
    tbb::task_arena().execute([&] {
        tbb::parallel_for(size_t{0}, modelsBatch.Size(), [&](size_t l) {
            const auto& models = modelsBatch.GetHostView()[l];
            metrics_[l].AddModels(models);
        });
    });
}

void HFMetricsFilterHost::AddMetrics(
        const Data::BaselinerMetrics& baselinerMetrics,
        const Data::FrameLabelerMetrics& frameLabelerMetrics,
        const Data::PulseDetectorMetrics& pdMetrics)
{
    tbb::task_arena().execute([&] {
        tbb::parallel_for(size_t{0}, metrics_.size(), [&](size_t l) {
            metrics_[l].AddBatchMetrics(
                    baselinerMetrics.baselinerStats.GetHostView()[l],
                    frameLabelerMetrics.viterbiScore.GetHostView()[l],
                    pdMetrics.baselineStats.GetHostView()[l]);
        });
    });
}

std::unique_ptr<HFMetricsFilterHost::BasecallingMetricsBatchT>
HFMetricsFilterHost::Process(
        const PulseBatchT& pulseBatch,
        const Data::BaselinerMetrics& baselinerMetrics,
        const ModelsBatchT& models,
        const Data::FrameLabelerMetrics& frameLabelerMetrics,
        const Data::PulseDetectorMetrics& pdMetrics)
{
    if (framesSeen_ == 0)
    {
        for (auto& metric : metrics_) metric.Reset();
    }
    AddPulses(pulseBatch);
    AddModels(models);
    AddMetrics(baselinerMetrics, frameLabelerMetrics, pdMetrics);
    framesSeen_ += pulseBatch.Dims().framesPerBatch;

    if (pulseBatch.GetMeta().LastFrame() % framesPerHFMetricBlock_ < pulseBatch.Dims().framesPerBatch)
    {
        FinalizeBlock();
        framesSeen_ = 0;
        auto ret = metricsFactory_->NewBatch(pulseBatch.Dims().lanesPerBatch);
        tbb::task_arena().execute([&] {
            tbb::parallel_for(size_t{0}, metrics_.size(), [&](size_t l) {
                metrics_[l].PopulateBasecallingMetrics(
                        ret->GetHostView()[l]);
            });
        });
        return ret;
    }
    return std::unique_ptr<BasecallingMetricsBatchT>();
}



}}} // PacBio::Mongo::Basecaller
