
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
#include <basecaller/traceAnalysis/AnalysisProfiler.h>

#include <basecaller/traceAnalysis/Baseliner.h>
#include <basecaller/traceAnalysis/FrameLabeler.h>
#include <basecaller/traceAnalysis/PulseAccumulator.h>
#include <basecaller/traceAnalysis/HFMetricsFilter.h>

#include <common/cuda/PBCudaRuntime.h>

#include <dataTypes/LaneDetectionModel.h>
#include <dataTypes/PoolHistogram.h>
#include <dataTypes/TraceBatch.h>
#include <dataTypes/configs/AnalysisConfig.h>
#include <dataTypes/configs/StaticDetModelConfig.h>

#include "AlgoFactory.h"

using namespace PacBio::Mongo::Data;
using namespace PacBio::Cuda::Memory;

namespace PacBio {
namespace Mongo {
namespace Basecaller {

void BatchAnalyzer::ReportPerformance()
{
    AnalysisProfiler::FinalReport();
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
    dme_ = std::make_unique<DetectionModelEstimator>(poolId, dims, registrar, algoFac);
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
    // we'll estimate one quarter of batches during one chunk, the another
    // quarter during the next chunk, and so on.  We'll also arrange things such
    // that we don't do dme estimation on consecutive batches (assuming batches are
    // processed in poolID order), so that the computational workload is a bit more
    // evenly spread out
    const auto framesPerChunk = dims.framesPerBatch;
    const auto chunksPerEstimate = (std::max(dmeConfig.MinFramesForEstimate,1u) + framesPerChunk - 1)
                                 / framesPerChunk;
    poolDmeDelayFrames_ = (poolId % chunksPerEstimate) * framesPerChunk;
}

FixedModelBatchAnalyzer::FixedModelBatchAnalyzer(uint32_t poolId,
                                                 const Data::BatchDimensions& dims,
                                                 const Data::StaticDetModelConfig& staticDetModelConfig,
                                                 const Data::AnalysisConfig& analysisConfig,
                                                 const AlgoFactory& algoFac,
                                                 DeviceAllocationStash& stash)
    : BatchAnalyzer(poolId, dims, algoFac, stash)
{
    // Not running DME, need to fake our model
    Data::LaneModelParameters<Cuda::PBHalf, laneSize> model;

    model.BaselineMode().SetAllMeans(staticDetModelConfig.baselineMean);
    model.BaselineMode().SetAllVars(staticDetModelConfig.baselineVariance);

    auto analogs = staticDetModelConfig.SetupAnalogs(analysisConfig);
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


BatchAnalyzer::OutputType BatchAnalyzer::operator()(const TraceBatchVariant& tbatch)
{
    PBAssert(tbatch.Metadata().PoolId() == poolId_, "Bad pool ID.");
    PBAssert(tbatch.Metadata().FirstFrame() == nextFrameId_, "Bad frame ID.");

    if (Cuda::StreamErrorCount() > 0)
    {
            throw PBException("Unexpected stream synchronization issue(s) was detected before TraceBatch analysis. StreamErrorCount="
            + std::to_string(Cuda::StreamErrorCount()));
    }

    auto ret = AnalyzeImpl(tbatch);

    if (Cuda::StreamErrorCount() > 0)
    {
        throw PBException("Unexpected stream synchronization issue(s) was detected after TraceBatch analysis. StreamErrorCount="
            + std::to_string(Cuda::StreamErrorCount()));
    }
    nextFrameId_ = tbatch.Metadata().LastFrame();

    return ret;
}

BatchAnalyzer::OutputType FixedModelBatchAnalyzer::AnalyzeImpl(const TraceBatchVariant& tbatch)
{
    auto mode = AnalysisProfiler::Mode::REPORT;
    if (tbatch.Metadata().FirstFrame() < static_cast<int32_t>(tbatch.NumFrames())*10+1) mode = AnalysisProfiler::Mode::OBSERVE;
    if (tbatch.Metadata().FirstFrame() < static_cast<int32_t>(tbatch.NumFrames())*2+1)  mode = AnalysisProfiler::Mode::IGNORE;
    AnalysisProfiler profiler(mode, 3.0, 100.0);

    auto baselineProfile = profiler.CreateScopedProfiler(AnalysisStages::Baseline);
    (void)baselineProfile;
    auto baselinedTracesAndMetrics = (*baseliner_)(tbatch);
    auto baselinedTraces = std::move(baselinedTracesAndMetrics.first);
    auto baselinerMetrics = std::move(baselinedTracesAndMetrics.second);

    auto frameProfile = profiler.CreateScopedProfiler(AnalysisStages::FrameLabeling);
    (void)frameProfile;
    auto labelsAndMetrics = (*frameLabeler_)(std::move(baselinedTraces), models_);
    auto labels = std::move(labelsAndMetrics.first);
    auto frameLabelerMetrics = std::move(labelsAndMetrics.second);

    auto pulseProfile = profiler.CreateScopedProfiler(AnalysisStages::PulseAccumulating);
    (void)pulseProfile;
    auto pulsesAndMetrics = (*pulseAccumulator_)(std::move(labels), models_);
    auto pulses = std::move(pulsesAndMetrics.first);
    auto pulseDetectorMetrics = std::move(pulsesAndMetrics.second);

    auto metricsProfile = profiler.CreateScopedProfiler(AnalysisStages::Metrics);
    (void)metricsProfile;

    auto basecallingMetrics = (*hfMetrics_)(
            pulses, baselinerMetrics, models_, frameLabelerMetrics, pulseDetectorMetrics);

    return BatchResult(std::move(pulses), std::move(basecallingMetrics));
}

BatchAnalyzer::OutputType SingleEstimateBatchAnalyzer::AnalyzeImpl(const TraceBatchVariant& tbatch)
{
    auto mode = AnalysisProfiler::Mode::IGNORE;
    if (isModelInitialized_)
    {
        auto framesSince = tbatch.Metadata().FirstFrame() - firstFrameWithEstimates_;
        if (framesSince > tbatch.NumFrames()*2)  mode = AnalysisProfiler::Mode::OBSERVE;
        if (framesSince > tbatch.NumFrames()*10) mode = AnalysisProfiler::Mode::REPORT;
    }
    AnalysisProfiler profiler(mode, 3.0, 100.0);

    // Baseline estimation and subtraction.
    // Includes computing baseline moments.
    assert(baseliner_);
    auto baselineProfile = profiler.CreateScopedProfiler(AnalysisStages::Baseline);
    (void)baselineProfile;
    auto baselinedTracesAndMetrics = (*baseliner_)(tbatch);
    auto baselinedTraces = std::move(baselinedTracesAndMetrics.first);
    auto baselinerMetrics = std::move(baselinedTracesAndMetrics.second);

    if (!isModelInitialized_ && tbatch.Metadata().FirstFrame() > static_cast<int32_t>(baseliner_->StartupLatency()))
    {
        // Run data through the DME until we get our first real estimate, at which point we
        // stop using the DME and just keep that model forever.
        isModelInitialized_ = dme_->AddBatch(baselinedTraces, baselinerMetrics, &models_, profiler);
    }

    auto pulsesAndMetrics = [&baselinedTraces, &profiler, this]() {
        // When detection model is available, ...
        if (isModelInitialized_)
        {
            // Classify frames.
            assert(frameLabeler_);
            auto frameProfile = profiler.CreateScopedProfiler(AnalysisStages::FrameLabeling);
            (void)frameProfile;
            auto labelsAndMetrics = (*frameLabeler_)(std::move(baselinedTraces),
                                                     models_);
            auto labels = std::move(labelsAndMetrics.first);
            auto frameLabelerMetrics = std::move(labelsAndMetrics.second);

            // Generate pulses with metrics.
            auto pulseProfile = profiler.CreateScopedProfiler(AnalysisStages::PulseAccumulating);
            (void)pulseProfile;
            assert(pulseAccumulator_);
            auto pulsesAndMetrics = (*pulseAccumulator_)(std::move(labels), models_);
            auto pulses = std::move(pulsesAndMetrics.first);
            auto pulseDetectorMetrics = std::move(pulsesAndMetrics.second);

            return std::make_tuple(std::move(pulses),
                                   std::move(frameLabelerMetrics),
                                   std::move(pulseDetectorMetrics));
        }
        else
        {
            auto frameLabelerMetrics = frameLabeler_->EmptyMetrics(baselinedTraces.StorageDims());
            auto labels = frameLabeler_->EmptyLabelsBatch(std::move(baselinedTraces));
            auto pulsesAndMetrics = pulseAccumulator_->EmptyPulseBatch(labels.Metadata(),
                                                                       labels.StorageDims());
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

    auto metricsProfile = profiler.CreateScopedProfiler(AnalysisStages::Metrics);
    (void)metricsProfile;
    auto basecallingMetrics = (*hfMetrics_)(
            pulses, baselinerMetrics, models_, frameLabelerMetrics,
            pulseDetectorMetrics);

    return BatchResult(std::move(pulses), std::move(basecallingMetrics));
}


// WiP: Prototype for analysis that supports slowly varying detection
// model parameters.
BatchAnalyzer::OutputType
DynamicEstimateBatchAnalyzer::AnalyzeImpl(const Data::TraceBatchVariant& tbatch)
{
    assert(baseliner_);
    static const unsigned int nFramesBaselinerStartUp = baseliner_->StartupLatency();
    static const unsigned int nFramesDmeStartUp = dme_->StartupLatency();

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
    // run.
    const auto startupLatency =
          roundToChunkMultiple(nFramesBaselinerStartUp)
        + roundToChunkMultiple(nFramesDmeStartUp);
    auto mode = AnalysisProfiler::Mode::IGNORE;
    if (tbatch.Metadata().FirstFrame() > static_cast<int32_t>(startupLatency))
    {
        auto framesSince = tbatch.Metadata().FirstFrame() - startupLatency;
        if (framesSince > 2*minFramesForDme)  mode = AnalysisProfiler::Mode::OBSERVE;
        if (framesSince > 10*minFramesForDme) mode = AnalysisProfiler::Mode::REPORT;
    }
    AnalysisProfiler profiler(mode, 3.0, 100.0);

    // Baseline estimation and subtraction.
    // Includes computing baseline moments.
    auto baselineProfile = profiler.CreateScopedProfiler(AnalysisStages::Baseline);
    (void)baselineProfile;
    auto baselinedTracesAndMetrics = (*baseliner_)(tbatch);
    auto baselinedTraces = std::move(baselinedTracesAndMetrics.first);
    auto baselinerMetrics = std::move(baselinedTracesAndMetrics.second);

    // We need an early return after baselining.  We don't want to feed the early
    // baseline info into the DME so that it's not tainted with transient startup
    // innacuracies, and we shouldn't run the downstream filters either because
    // the models aren't even sensibly initialized until the first time we hand
    // data to the DME
    if (baselinedTraces.GetMeta().FirstFrame() < static_cast<int32_t>(nFramesBaselinerStartUp + poolDmeDelayFrames_))
    {
        auto emptyLabels = frameLabeler_->EmptyLabelsBatch(std::move(baselinedTraces));
        auto emptyPulsesAndMetrics = pulseAccumulator_->EmptyPulseBatch(emptyLabels.Metadata(),
                                                                        emptyLabels.StorageDims());
        return BatchResult(std::move(emptyPulsesAndMetrics.first), nullptr);
    }
    bool fullEstimation = dme_->AddBatch(baselinedTraces, baselinerMetrics, &models_, profiler);
    if (fullEstimation) fullEstimationOccured_ = true;

    auto frameProfile = profiler.CreateScopedProfiler(AnalysisStages::FrameLabeling);
    (void)frameProfile;
    auto labelsAndMetrics = (*frameLabeler_)(std::move(baselinedTraces), models_);
    auto labels = std::move(labelsAndMetrics.first);
    auto frameLabelerMetrics = std::move(labelsAndMetrics.second);

    auto pulseProfile = profiler.CreateScopedProfiler(AnalysisStages::PulseAccumulating);
    (void)pulseProfile;
    auto pulsesAndMetrics = (*pulseAccumulator_)(std::move(labels), models_);
    auto pulses = std::move(pulsesAndMetrics.first);
    auto pulseDetectorMetrics = std::move(pulsesAndMetrics.second);

    // TODO: What is the "schedule" pattern of producing metrics.
    // Do not need to be aligned over pools (or lanes).

    // TODO: When metrics are produced, use them to update detection models.
    if (!fullEstimationOccured_)
    {
        auto emptyLabels = frameLabeler_->EmptyLabelsBatch(std::move(baselinedTraces));
        auto emptyPulsesAndMetrics = pulseAccumulator_->EmptyPulseBatch(emptyLabels.Metadata(),
                                                                        emptyLabels.StorageDims());
        return BatchResult(std::move(emptyPulsesAndMetrics.first), nullptr);
    } else
    {
        auto metricsProfile = profiler.CreateScopedProfiler(AnalysisStages::Metrics);
        (void)metricsProfile;
        auto basecallingMetrics = (*hfMetrics_)(pulses, baselinerMetrics, models_,
                                                frameLabelerMetrics, pulseDetectorMetrics);

        return BatchResult(std::move(pulses), std::move(basecallingMetrics));
    }
}

}}}     // namespace PacBio::Mongo::Basecaller
