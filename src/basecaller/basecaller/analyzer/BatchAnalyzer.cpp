
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
using namespace PacBio::Cuda::Memory;

namespace PacBio {
namespace Mongo {
namespace Basecaller {

SMART_ENUM(
    FilterStages,
    Upload,
    Download,
    Baseline,
    Histogram,
    DME,
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
                             const AlgoFactory& algoFac,
                             DeviceAllocationStash& stash)
    : models_(dims.lanesPerBatch ,
              Cuda::Memory::SyncDirection::HostWriteDeviceRead,
              SOURCE_MARKER())
    , poolId_ (poolId)
{
    Cuda::Memory::StashableAllocRegistrar registrar(poolId, stash);

    baseliner_ = algoFac.CreateBaseliner(poolId, dims, registrar);
    traceHistAccum_ = algoFac.CreateTraceHistAccumulator(poolId, dims, registrar);
    dme_ = algoFac.CreateDetectionModelEstimator(poolId, dims, registrar);
    frameLabeler_ = algoFac.CreateFrameLabeler(poolId, dims, registrar);
    pulseAccumulator_ = algoFac.CreatePulseAccumulator(poolId, dims, registrar);
    hfMetrics_ = algoFac.CreateHFMetricsFilter(poolId, dims, registrar);
}

SingleEstimateBatchAnalyzer::SingleEstimateBatchAnalyzer(uint32_t poolId,
                                                         const Data::BatchDimensions& dims,
                                                         const AlgoFactory& algoFac,
                                                         DeviceAllocationStash& stash)
    : BatchAnalyzer(poolId, dims, algoFac, stash)
{
}

DynamicEstimateBatchAnalyzer::DynamicEstimateBatchAnalyzer(uint32_t poolId,
                                                           uint32_t maxPoolId,
                                                           const Data::BatchDimensions& dims,
                                                           const Data::BasecallerDmeConfig& dmeConfig,
                                                           const AlgoFactory& algoFac,
                                                           DeviceAllocationStash& stash)
    : BatchAnalyzer(poolId, dims, algoFac, stash)
{
    // Set up a stagger pattern, such that we always estimate at the requested
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
                                                 const AlgoFactory& algoFac,
                                                 DeviceAllocationStash& stash)
    : BatchAnalyzer(poolId, dims, algoFac, stash)
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

    if (Cuda::StreamErrorCount() > 0)
        throw PBException("Unexpected stream synchronization issues were detected before analyze");

    auto ret = AnalyzeImpl(tbatch);

    if (Cuda::StreamErrorCount() > 0)
        throw PBException("Unexpected stream synchronization issues were detected after analyze");

    nextFrameId_ = tbatch.Metadata().LastFrame();

    return ret;
}

BatchAnalyzer::OutputType FixedModelBatchAnalyzer::AnalyzeImpl(const TraceBatch<int16_t>& tbatch)
{
    auto mode = Profiler::Mode::REPORT;
    if (tbatch.Metadata().FirstFrame() < tbatch.NumFrames()*10+1) mode = Profiler::Mode::OBSERVE;
    if (tbatch.Metadata().FirstFrame() < tbatch.NumFrames()*2+1)  mode = Profiler::Mode::IGNORE;
    Profiler profiler(mode, 3.0, 100.0);

    // TODO baseliner should have a virtual "PrepareData" function or something.
    //      Having an explicitly measurable upload step makes the profiles
    //      more meaningful, but these explicit gpu calls means we can't run
    //      on a system without a gpu even if we're using purely host modules.
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
    return BatchResult(std::move(pulses), std::move(basecallingMetrics));
}

BatchAnalyzer::OutputType SingleEstimateBatchAnalyzer::AnalyzeImpl(const TraceBatch<int16_t>& tbatch)
{
    auto mode = Profiler::Mode::IGNORE;
    if (isModelInitialized_)
    {
        auto framesSince = tbatch.Metadata().FirstFrame() - firstFrameWithEstimates_;
        if (framesSince > tbatch.NumFrames()*2)  mode = Profiler::Mode::OBSERVE;
        if (framesSince > tbatch.NumFrames()*10) mode = Profiler::Mode::REPORT;
    }
    Profiler profiler(mode, 3.0, 100.0);

    // TODO baseliner should have a virtual "PrepareData" function or something.
    //      Having an explicitly measurable upload step makes the profiles
    //      more meaningful, but these explicit gpu calls means we can't run
    //      on a system without a gpu even if we're using purely host modules.
    auto upload = profiler.CreateScopedProfiler(FilterStages::Upload);
    (void)upload;
    tbatch.CopyToDevice();
    Cuda::CudaSynchronizeDefaultStream();

    // Baseline estimation and subtraction.
    // Includes computing baseline moments.
    assert(baseliner_);
    auto baselineProfile = profiler.CreateScopedProfiler(FilterStages::Baseline);
    (void)baselineProfile;
    auto baselinedTracesAndMetrics = (*baseliner_)(tbatch);
    auto baselinedTraces = std::move(baselinedTracesAndMetrics.first);
    auto baselinerMetrics = std::move(baselinedTracesAndMetrics.second);

    if (!isModelInitialized_ && tbatch.GetMeta().FirstFrame() > baseliner_->StartupLatency())
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
            firstFrameWithEstimates_ = tbatch.Metadata().FirstFrame();

            // Estimate model parameters from histogram.
            assert(dme_);
            dme_->Estimate(traceHistAccum_->Histogram(), &models_);
        }
    }

    auto pulsesAndMetrics = [&baselinedTraces, &profiler, this]() {
        // When detection model is available, ...
        if (isModelInitialized_)
        {
            // Classify frames.
            assert(frameLabeler_);
            auto frameProfile = profiler.CreateScopedProfiler(FilterStages::FrameLabeling);
            (void)frameProfile;
            auto labelsAndMetrics = (*frameLabeler_)(std::move(baselinedTraces),
                                                     models_);
            auto labels = std::move(labelsAndMetrics.first);
            auto frameLabelerMetrics = std::move(labelsAndMetrics.second);

            // Generate pulses with metrics.
            auto pulseProfile = profiler.CreateScopedProfiler(FilterStages::PulseAccumulating);
            (void)pulseProfile;
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
    auto& pulses = std::get<0>(pulsesAndMetrics);
    const auto& frameLabelerMetrics = std::get<1>(pulsesAndMetrics);
    const auto& pulseDetectorMetrics = std::get<2>(pulsesAndMetrics);

    auto metricsProfile = profiler.CreateScopedProfiler(FilterStages::Metrics);
    (void)metricsProfile;
    auto basecallingMetrics = (*hfMetrics_)(
            pulses, baselinerMetrics, models_, frameLabelerMetrics,
            pulseDetectorMetrics);

    auto download = profiler.CreateScopedProfiler(FilterStages::Download);
    (void)download;
    return BatchResult(std::move(pulses), std::move(basecallingMetrics));
}


// WiP: Prototype for analysis that supports slowly varying detection
// model parameters.
BatchAnalyzer::OutputType
DynamicEstimateBatchAnalyzer::AnalyzeImpl(const Data::TraceBatch<int16_t>& tbatch)
{
    assert(baseliner_);
    assert(traceHistAccum_);
    static const unsigned int nFramesBaselinerStartUp = baseliner_->StartupLatency();

    // Minimum number of frames needed for estimating the detection model.
    static const auto minFramesForDme = DetectionModelEstimator::MinFramesForEstimate();

    auto roundToChunkMultiple = [&](size_t val)
    {
        auto numFrames = tbatch.NumFrames();
        return (val + numFrames - 1) / numFrames * numFrames;
    };
    // We want to avoid intial transient phases before profiling.
    // First the baseline filter needs to flush latent transients,
    // then the histogram needs to get a reliable baseline statistics,
    // and then finally we need to gather enough data for the DME to
    // run.  Since DME execution is staggered, we really need to wait
    // 2x that number of frames, in order for everything to be steady
    // state
    const auto startupLatency =
          roundToChunkMultiple(nFramesBaselinerStartUp)
        + roundToChunkMultiple(traceHistAccum_->NumFramesPreAccumStats())
        + 2* roundToChunkMultiple(minFramesForDme);
    auto mode = Profiler::Mode::IGNORE;
    if (tbatch.Metadata().FirstFrame() > startupLatency)
    {
        auto framesSince = tbatch.Metadata().FirstFrame() - startupLatency;
        if (framesSince > 2*minFramesForDme)  mode = Profiler::Mode::OBSERVE;
        if (framesSince > 10*minFramesForDme) mode = Profiler::Mode::REPORT;
    }
    Profiler profiler(mode, 3.0, 100.0);

    // TODO baseliner should have a virtual "PrepareData" function or something.
    //      Having an explicitly measurable upload step makes the profiles
    //      more meaningful, but these explicit gpu calls means we can't run
    //      on a system without a gpu even if we're using purely host modules.
    auto upload = profiler.CreateScopedProfiler(FilterStages::Upload);
    (void)upload;
    tbatch.CopyToDevice();
    Cuda::CudaSynchronizeDefaultStream();

    // Baseline estimation and subtraction.
    // Includes computing baseline moments.
    auto baselineProfile = profiler.CreateScopedProfiler(FilterStages::Baseline);
    (void)baselineProfile;
    auto baselinedTracesAndMetrics = (*baseliner_)(tbatch);
    auto baselinedTraces = std::move(baselinedTracesAndMetrics.first);
    auto baselinerMetrics = std::move(baselinedTracesAndMetrics.second);

    // Wait for the baseliner startup transients to pass.
    // Then wait a while more to stagger DME executions of different pools.
    // Then reset the trace histograms and start accumulating data for the first DME execution.
    if (poolStatus_ == PoolStatus::STARTUP_DME_DELAY
            && baselinedTraces.GetMeta().FirstFrame() >= nFramesBaselinerStartUp + poolDmeDelayFrames_)
    {
        traceHistAccum_->Reset();
        poolStatus_ = PoolStatus::STARTUP_DME_INIT;
    }

    // Accumulate histogram of baseline-subtracted trace data.
    // This operation also accumulates baseliner statistics.
    auto histProfile = profiler.CreateScopedProfiler(FilterStages::Histogram);
    (void)histProfile;
    assert(traceHistAccum_);
    traceHistAccum_->AddBatch(baselinedTraces,
                              baselinerMetrics.baselinerStats);

    // Don't bother trying DME during the initial startup phase.
    const bool doDme = poolStatus_ != PoolStatus::STARTUP_DME_DELAY
            && traceHistAccum_->HistogramFrameCount() >= minFramesForDme;

    // Keep our model at our best guess from baseline stats.  This smooths
    // the transition to a real model, when considering any latent data
    // that may be stored in downstream filters
    if (poolStatus_ == PoolStatus::STARTUP_DME_INIT)
    {
        models_ = dme_->InitDetectionModels(traceHistAccum_->TraceStats());
        if (doDme) poolStatus_ = PoolStatus::SEQUENCING;
    }

    // When sufficient trace data have been histogrammed,
    // estimate detection model.
    if (doDme)
    {
        auto dmeProfile = profiler.CreateScopedProfiler(FilterStages::DME);
        (void)dmeProfile;
        // Estimate/update model parameters from histogram.
        assert(dme_);
        dme_->Estimate(traceHistAccum_->Histogram(), &models_);
        traceHistAccum_->Clear();

    }

    // This part simply mimics the StaticModelPipeline.

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
    auto basecallingMetrics = (*hfMetrics_)(pulses, baselinerMetrics, models_,
                                            frameLabelerMetrics, pulseDetectorMetrics);

    // TODO: What is the "schedule" pattern of producing metrics.
    // Do not need to be aligned over pools (or lanes).

    // TODO: When metrics are produced, use them to update detection models.

    auto download = profiler.CreateScopedProfiler(FilterStages::Download);
    (void)download;
    if (poolStatus_ != PoolStatus::SEQUENCING)
    {
        auto emptyPulsesAndMetrics = pulseAccumulator_->EmptyPulseBatch(baselinedTraces.Metadata(),
                                                                   baselinedTraces.StorageDims());
        return BatchResult(std::move(emptyPulsesAndMetrics.first), nullptr);
    } else
    {
        return BatchResult(std::move(pulses), std::move(basecallingMetrics));
    }
}

}}}     // namespace PacBio::Mongo::Basecaller
