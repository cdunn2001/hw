
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
//  Defines members of class BatchAnalyzer.

#include "BatchAnalyzer.h"

#include <algorithm>

#include <pacbio/PBAssert.h>
#include <pacbio/logging/Logger.h>
#include <pacbio/dev/profile/ScopedProfilerChain.h>

#include <basecaller/traceAnalysis/Baseliner.h>
#include <basecaller/traceAnalysis/FrameLabeler.h>
#include <basecaller/traceAnalysis/PulseAccumulator.h>
#include <basecaller/traceAnalysis/DetectionModelEstimator.h>
#include <basecaller/traceAnalysis/TraceHistogramAccumulator.h>
#include <basecaller/traceAnalysis/HFMetricsFilter.h>

#include <common/cuda/PBCudaRuntime.h>

#include <dataTypes/LaneDetectionModel.h>
#include <dataTypes/PoolHistogram.h>
#include <dataTypes/TraceBatch.h>
#include <dataTypes/configs/MovieConfig.h>
#include <dataTypes/configs/StaticDetModelConfig.h>

#include "AlgoFactory.h"

using namespace PacBio::Mongo::Data;

namespace PacBio {
namespace Mongo {
namespace Basecaller {

SMART_ENUM(
    FilterStages,
    Upload,
    Download,
    Baseline,
    FrameLabeling,
    PulseAccumulating,
    Metrics
);

using Profiler = PacBio::Dev::Profile::ScopedProfilerChain<FilterStages>;

void BatchAnalyzer::ReportPerformance()
{
    Profiler::FinalReport();
}

// These are required here even though defaulted, as this class
// has unique_ptrs to types that are incomplete in the header
// file
BatchAnalyzer& BatchAnalyzer::operator=(BatchAnalyzer&&) = default;
BatchAnalyzer::~BatchAnalyzer() = default;

BatchAnalyzer::BatchAnalyzer(uint32_t poolId,
                             const Data::BatchDimensions& dims,
                             const AlgoFactory& algoFac)
    : models_(dims.lanesPerBatch ,
              Cuda::Memory::SyncDirection::HostWriteDeviceRead,
              SOURCE_MARKER())
    , poolId_ (poolId)
{
    baseliner_ = algoFac.CreateBaseliner(poolId, dims);
    traceHistAccum_ = algoFac.CreateTraceHistAccumulator(poolId, dims);
    dme_ = algoFac.CreateDetectionModelEstimator(poolId, dims);
    frameLabeler_ = algoFac.CreateFrameLabeler(poolId, dims);
    pulseAccumulator_ = algoFac.CreatePulseAccumulator(poolId, dims);
    hfMetrics_ = algoFac.CreateHFMetricsFilter(poolId, dims);
}

SingleEstimateBatchAnalyzer::SingleEstimateBatchAnalyzer(uint32_t poolId,
                                                         const Data::BatchDimensions& dims,
                                                         const AlgoFactory& algoFac)
    : BatchAnalyzer(poolId, dims, algoFac)
{
}

DynamicEstimateBatchAnalyzer::DynamicEstimateBatchAnalyzer(uint32_t poolId,
                                                           uint32_t maxPoolId,
                                                           const Data::BatchDimensions& dims,
                                                           const Data::BasecallerDmeConfig& dmeConfig,
                                                           const AlgoFactory& algoFac)
    : BatchAnalyzer(poolId, dims, algoFac)
{
    // Set up a staggering pattern, so that we always estimate at the requested
    // interval, but estimates for individual batches are spread out evenly
    // between chunks.  For instance if it takes 4 chunks to get enough data
    // for an estimate, then once all startup latencies are finally finished,
    // we'll estimate the first quarter of batches during one chunk, the next
    // quarter during the next chunk, and so on.
    const auto framesPerChunk = dims.framesPerBatch;
    const auto chunksPerEstimate = (dmeConfig.MinFramesForEstimate + framesPerChunk - 1)
                                 / framesPerChunk;
    const auto fraction = static_cast<float>(poolId) / (maxPoolId+1);
    poolDmeDelayFrames_ = static_cast<uint32_t>(fraction * chunksPerEstimate)
                  * dims.framesPerBatch;
}

FixedModelBatchAnalyzer::FixedModelBatchAnalyzer(uint32_t poolId,
                                                 const Data::BatchDimensions& dims,
                                                 const Data::StaticDetModelConfig& staticDetModelConfig,
                                                 const Data::MovieConfig& movieConfig,
                                                 const AlgoFactory& algoFac)
    : BatchAnalyzer(poolId, dims, algoFac)
{
    // Not running DME, need to fake our model
    Data::LaneModelParameters<Cuda::PBHalf, laneSize> model;

    model.BaselineMode().SetAllMeans(staticDetModelConfig.baselineMean);
    model.BaselineMode().SetAllVars(staticDetModelConfig.baselineVariance);

    auto analogs = staticDetModelConfig.SetupAnalogs(movieConfig);
    for (size_t i = 0; i < analogs.size(); i++)
    {
        model.AnalogMode(i).SetAllMeans(analogs[i].mean);
        model.AnalogMode(i).SetAllVars(analogs[i].var);
    }

    auto view = models_.GetHostView();
    for (size_t i = 0; i < view.Size(); ++i)
    {
        view[i] = model;
    }
}


BatchAnalyzer::OutputType BatchAnalyzer::operator()(const TraceBatch<int16_t>& tbatch)
{
    PBAssert(tbatch.Metadata().PoolId() == poolId_, "Bad pool ID.");
    PBAssert(tbatch.Metadata().FirstFrame() == nextFrameId_, "Bad frame ID.");

    auto ret = AnalyzeImpl(tbatch);

    if (Cuda::StreamErrorCount() > 0)
        throw PBException("Unexpected stream synchronization issues were detected");

    nextFrameId_ = tbatch.Metadata().LastFrame();

    return ret;
}

BatchAnalyzer::OutputType FixedModelBatchAnalyzer::AnalyzeImpl(const TraceBatch<int16_t>& tbatch)
{
    auto mode = Profiler::Mode::REPORT;
    if (tbatch.Metadata().FirstFrame() < 1281) mode = Profiler::Mode::OBSERVE;
    if (tbatch.Metadata().FirstFrame() < 257) mode = Profiler::Mode::IGNORE;
    Profiler profiler(mode, 3.0, 100.0);

    auto upload = profiler.CreateScopedProfiler(FilterStages::Upload);
    (void)upload;
    tbatch.CopyToDevice();
    Cuda::CudaSynchronizeDefaultStream();

    auto baselineProfile = profiler.CreateScopedProfiler(FilterStages::Baseline);
    (void)baselineProfile;
    auto baselinedTracesAndMetrics = (*baseliner_)(tbatch);
    auto baselinedTraces = std::move(baselinedTracesAndMetrics.first);
    auto baselinerMetrics = std::move(baselinedTracesAndMetrics.second);

    auto frameProfile = profiler.CreateScopedProfiler(FilterStages::FrameLabeling);
    (void)frameProfile;
    auto labelsAndMetrics = (*frameLabeler_)(std::move(baselinedTraces), models_);
    auto labels = std::move(labelsAndMetrics.first);
    auto frameLabelerMetrics = std::move(labelsAndMetrics.second);

    auto pulseProfile = profiler.CreateScopedProfiler(FilterStages::PulseAccumulating);
    (void)pulseProfile;
    auto pulsesAndMetrics = (*pulseAccumulator_)(std::move(labels));
    auto pulses = std::move(pulsesAndMetrics.first);
    auto pulseDetectorMetrics = std::move(pulsesAndMetrics.second);

    auto metricsProfile = profiler.CreateScopedProfiler(FilterStages::Metrics);
    (void)metricsProfile;

    auto basecallingMetrics = (*hfMetrics_)(
            pulses, baselinerMetrics, models_, frameLabelerMetrics, pulseDetectorMetrics);

    auto download = profiler.CreateScopedProfiler(FilterStages::Download);
    (void)download;
    pulses.Pulses().LaneView(0);

    return BatchResult(std::move(pulses), std::move(basecallingMetrics));
}

BatchAnalyzer::OutputType SingleEstimateBatchAnalyzer::AnalyzeImpl(const TraceBatch<int16_t>& tbatch)
{
    // Baseline estimation and subtraction.
    // Includes computing baseline moments.
    assert(baseliner_);
    auto baselinedTracesAndMetrics = (*baseliner_)(tbatch);
    auto baselinedTraces = std::move(baselinedTracesAndMetrics.first);
    auto baselinerMetrics = std::move(baselinedTracesAndMetrics.second);

    if (!isModelInitialized_)
    {
        // TODO: Factor model initialization and estimation operations.

        // Accumulate histogram of baseline-subtracted trace data.
        // This operation also accumulates baseliner statistics.
        assert(traceHistAccum_);
        traceHistAccum_->AddBatch(baselinedTraces,
                                  baselinerMetrics.baselinerStats);

        // When sufficient trace data have been histogrammed,
        // estimate detection model.
        const auto minFramesForDme = DetectionModelEstimator::MinFramesForEstimate();
        if (traceHistAccum_->HistogramFrameCount() >= minFramesForDme)
        {
            // Initialize the detection model from baseliner statistics.
            models_ = dme_->InitDetectionModels(traceHistAccum_->TraceStats());
            isModelInitialized_ = true;

            // Estimate model parameters from histogram.
            assert(dme_);
            dme_->Estimate(traceHistAccum_->Histogram(), &models_);
        }
    }

    auto pulsesAndMetrics = [&baselinedTraces, this]() {
        // When detection model is available, ...
        if (isModelInitialized_)
        {
            // Classify frames.
            assert(frameLabeler_);
            auto labelsAndMetrics = (*frameLabeler_)(std::move(baselinedTraces),
                                                     models_);
            auto labels = std::move(labelsAndMetrics.first);
            auto frameLabelerMetrics = std::move(labelsAndMetrics.second);

            // Generate pulses with metrics.
            assert(pulseAccumulator_);
            auto pulsesAndMetrics = (*pulseAccumulator_)(std::move(labels));
            auto pulses = std::move(pulsesAndMetrics.first);
            auto pulseDetectorMetrics = std::move(pulsesAndMetrics.second);

            return std::make_tuple(std::move(pulses),
                                   std::move(frameLabelerMetrics),
                                   std::move(pulseDetectorMetrics));
        }
        else
        {
            auto frameLabelerMetrics = frameLabeler_->EmptyMetrics(baselinedTraces.StorageDims());

            auto pulsesAndMetrics = pulseAccumulator_->EmptyPulseBatch(baselinedTraces.Metadata(),
                                                                       baselinedTraces.StorageDims());
            auto pulses = std::move(pulsesAndMetrics.first);
            auto pulseDetectorMetrics = std::move(pulsesAndMetrics.second);
            return std::make_tuple(std::move(pulses),
                                   std::move(frameLabelerMetrics),
                                   std::move(pulseDetectorMetrics));
        }
    }();
    auto pulses = std::move(std::get<0>(pulsesAndMetrics));
    auto frameLabelerMetrics = std::move(std::get<1>(pulsesAndMetrics));
    auto pulseDetectorMetrics = std::move(std::get<2>(pulsesAndMetrics));

    auto basecallingMetrics = (*hfMetrics_)(
            pulses, baselinerMetrics, models_, frameLabelerMetrics,
            pulseDetectorMetrics);

    return BatchResult(std::move(pulses), std::move(basecallingMetrics));
}


// WiP: Prototype for analysis that supports slowly varying detection
// model parameters.
BatchAnalyzer::OutputType
DynamicEstimateBatchAnalyzer::AnalyzeImpl(const Data::TraceBatch<int16_t>& tbatch)
{
    // This constant depends on the baseliner implementation and configuration.
    // It should be initialized by a call to a Baseliner member.
    static const unsigned int nFramesBaselinerStartUp = 100;

    // Minimum number of frames needed for estimating the detection model.
    static const auto minFramesForDme = DetectionModelEstimator::MinFramesForEstimate();

    // Baseline estimation and subtraction.
    // Includes computing baseline moments.
    assert(baseliner_);
    auto baselinedTracesAndMetrics = (*baseliner_)(tbatch);
    auto baselinedTraces = std::move(baselinedTracesAndMetrics.first);
    auto baselinerMetrics = std::move(baselinedTracesAndMetrics.second);

    // Wait for the baseliner startup transients to pass.
    // Then wait a while more to stagger DME executions of different pools.
    // Then reset the trace histograms and start accumulating data for the first DME execution.
    if (poolStatus_ == PoolStatus::STARTUP_DME_DELAY
            && baselinedTraces.GetMeta().FirstFrame() >= nFramesBaselinerStartUp + poolDmeDelayFrames_)
    {
        traceHistAccum_->FullReset();
        poolStatus_ = PoolStatus::STARTUP_DME_INIT;
    }

    // Accumulate histogram of baseline-subtracted trace data.
    // This operation also accumulates baseliner statistics.
    assert(traceHistAccum_);
    traceHistAccum_->AddBatch(baselinedTraces,
                              baselinerMetrics.baselinerStats);

    // Don't bother trying DME during the initial startup phase.
    const bool doDme = poolStatus_ != PoolStatus::STARTUP_DME_DELAY
            && traceHistAccum_->HistogramFrameCount() >= minFramesForDme;

    if (doDme && poolStatus_ == PoolStatus::STARTUP_DME_INIT)
    {
        // Initialize the detection model from baseliner statistics.
        models_ = dme_->InitDetectionModels(traceHistAccum_->TraceStats());
        poolStatus_ = PoolStatus::SEQUENCING;
    }

    // When sufficient trace data have been histogrammed,
    // estimate detection model.
    if (doDme)
    {
        // Estimate/update model parameters from histogram.
        assert(dme_);
        dme_->Estimate(traceHistAccum_->Histogram(), &models_);
        traceHistAccum_->Reset();

    }

    // This part simply mimics the StaticModelPipeline.
    auto labelsAndMetrics = (*frameLabeler_)(std::move(baselinedTraces), models_);
    auto labels = std::move(labelsAndMetrics.first);
    auto frameLabelerMetrics = std::move(labelsAndMetrics.second);

    auto pulsesAndMetrics = (*pulseAccumulator_)(std::move(labels));
    auto pulses = std::move(pulsesAndMetrics.first);
    auto pulseDetectorMetrics = std::move(pulsesAndMetrics.second);

    auto basecallingMetrics = (*hfMetrics_)(pulses, baselinerMetrics, models_,
                                            frameLabelerMetrics, pulseDetectorMetrics);

    // TODO: What is the "schedule" pattern of producing metrics.
    // Do not need to be aligned over pools (or lanes).

    // TODO: When metrics are produced, use them to update detection models.

    if (poolStatus_ != PoolStatus::SEQUENCING)
    {
        auto pulsesAndMetrics = pulseAccumulator_->EmptyPulseBatch(baselinedTraces.Metadata(),
                                                                   baselinedTraces.StorageDims());
        return BatchResult(std::move(pulsesAndMetrics.first), nullptr);
    } else
    {
        return BatchResult(std::move(pulses), std::move(basecallingMetrics));
    }
}

}}}     // namespace PacBio::Mongo::Basecaller
