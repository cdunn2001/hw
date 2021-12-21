// Copyright (c) 2019,2020 Pacific Biosciences of California, Inc.
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

#include "DetectionModelEstimator.h"

#include <basecaller/analyzer/AlgoFactory.h>
#include <basecaller/traceAnalysis/CoreDMEstimator.h>
#include <basecaller/traceAnalysis/BaselineStatsAggregator.h>
#include <basecaller/traceAnalysis/TraceHistogramAccumulator.h>
#include <dataTypes/configs/BasecallerDmeConfig.h>

namespace PacBio {
namespace Mongo {
namespace Basecaller {

// static
uint32_t DetectionModelEstimator::numFramesPreAccumStats_ = 0;
uint32_t DetectionModelEstimator::minFramesForEstimate_ = 0;

// static
void DetectionModelEstimator::Configure(const Data::BasecallerDmeConfig& dmeConfig)
{
    numFramesPreAccumStats_ = dmeConfig.NumFramesPreAccumStats;
    PBLOG_INFO << "DetectionModelEstimator: NumFramesPreAccumStats = "
               << numFramesPreAccumStats_ << '.';

    minFramesForEstimate_ = dmeConfig.MinFramesForEstimate;
    PBLOG_INFO << "DetectionModelEstimator: MinFramesForEstimate= "
               << minFramesForEstimate_ << '.';

    if (minFramesForEstimate_ < numFramesPreAccumStats_)
        PBLOG_WARN << "minFramesForEstimate_ less than numFramesPreAccumStats_.  "
                   << "Only the first histogram will respect numFramesPreAccumStats_";
}


// Need destructor definition here since the header file only has forward
// declaration for things like the CoreDMEstimator.
DetectionModelEstimator::~DetectionModelEstimator() = default;

DetectionModelEstimator::DetectionModelEstimator(uint32_t poolId,
                                                 const Data::BatchDimensions& dims,
                                                 Cuda::Memory::StashableAllocRegistrar& registrar,
                                                 const AlgoFactory& algoFac)
    : framesPerBatch_(dims.framesPerBatch)
    , traceAccumulator_(algoFac.CreateTraceHistAccumulator(poolId, dims, registrar))
    , baselineAggregator_(algoFac.CreateBaselineStatsAggregator(poolId, dims, registrar))
    , coreEstimator_(algoFac.CreateCoreDMEstimator(poolId, dims, registrar))
{
    framesRemaining_ = NumFramesPreAccumStats();
}

// Successive calls to this function will move this class through
// a state machine that handles the progression of various initialization
// phases leading up towards actual estimatins.  Namely:
// * Accumulate baseline statistics until we have a more robust baseline estimate
//   Once this stage finishes we will
//    * Use the aggregated stats to compute hist bounds and initialize the histograms
//    * Reset the baseline stats
// * In the next phase we accumulate both the next round of baseline stats as well
//   as histograms of the data.
//    * With each new piece of data we re-initialize the models using the latest
//      baseline information (but no actual estimation occurs)
//    * Aggregation continues until we have enough histogram data for a full model estimation
//    * Upon full model estimation the baseline stats are used to re-init the histograms, and
//      the baseline stats are reset
// * Until program termination, we continue mostly as in the last phase, with the main
//   difference being we don't update the models between estimation attempts any longer.
//    * Future implementations likely will have a more lightweight update step between
//      estimations
bool DetectionModelEstimator::AddBatch(const Data::TraceBatch<int16_t>& traces,
                                       const Data::BaselinerMetrics& metrics,
                                       PoolDetModel* models,
                                       AnalysisProfiler& profiler)
{
    assert(models);
    
    auto aggProf = profiler.CreateScopedProfiler(AnalysisStages::AggregateStats);
    (void)aggProf;
    baselineAggregator_->AddMetrics(metrics);

    auto histProf = profiler.CreateScopedProfiler(AnalysisStages::Histogram);
    (void)histProf;
    if (poolStatus_ != PoolStatus::STARTUP_HIST_INIT)
    {
        traceAccumulator_->AddBatch(traces, *models);
    }

    if (poolStatus_ != PoolStatus::SEQUENCING)
    {
        const auto runningStats = baselineAggregator_->TraceStats();
        *models = coreEstimator_->InitDetectionModels(runningStats.baselinerStats);
    }

    bool ranEstimation = false;

    framesRemaining_ -= traces.NumFrames();
    if (framesRemaining_ <= 0)
    {
        switch(poolStatus_)
        {
        case PoolStatus::STARTUP_DME_INIT:
            {
                poolStatus_ = PoolStatus::SEQUENCING;
                // Intentional fallthrough, we want an estimate now
            }
        case PoolStatus::SEQUENCING:
            {
                ranEstimation = true;
                const auto& hists = traceAccumulator_->Histogram();
                const auto runningStats = baselineAggregator_->TraceStats();

                auto dmeProf = profiler.CreateScopedProfiler(AnalysisStages::DME);
                (void)dmeProf;

                coreEstimator_->Estimate(hists, runningStats, models);
                // Intentional fallthrough, we want to reset our
                // histograms and baseline stats
            }
        case PoolStatus::STARTUP_HIST_INIT:
            {
                const auto runningStats = baselineAggregator_->TraceStats();
                traceAccumulator_->Reset(runningStats);
                baselineAggregator_->Reset();
                break;
            }
        default:
            throw PBException("Unexpected DetectionModelEstimator PoolStatus");
        }

        if (poolStatus_ == PoolStatus::STARTUP_HIST_INIT)
            poolStatus_ = PoolStatus::STARTUP_DME_INIT;

        framesRemaining_ = MinFramesForEstimate();
    }

    return ranEstimation;
}

uint32_t DetectionModelEstimator::StartupLatency() const
{
    auto RoundToChunk = [&](size_t frames) { return (frames + framesPerBatch_ - 1) / framesPerBatch_ * framesPerBatch_; };
    return RoundToChunk(NumFramesPreAccumStats())
         + RoundToChunk(MinFramesForEstimate());
}

}}}     // namespace PacBio::Mongo::Basecaller
