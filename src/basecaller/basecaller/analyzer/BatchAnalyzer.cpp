
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
#include <dataTypes/BasecallerConfig.h>

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

BatchAnalyzer::~BatchAnalyzer() = default;
BatchAnalyzer::BatchAnalyzer(BatchAnalyzer&&) = default;


BatchAnalyzer::BatchAnalyzer(uint32_t poolId, const AlgoFactory& algoFac)
    : poolId_ (poolId)
    , models_(PrimaryConfig().lanesPerPool, Cuda::Memory::SyncDirection::Symmetric, SOURCE_MARKER())
{
    baseliner_ = algoFac.CreateBaseliner(poolId);
    traceHistAccum_ = algoFac.CreateTraceHistAccumulator(poolId);
    dme_ = algoFac.CreateDetectionModelEstimator(poolId);
    frameLabeler_ = algoFac.CreateFrameLabeler(poolId);
    pulseAccumulator_ = algoFac.CreatePulseAccumulator(poolId);
    hfMetrics_ = algoFac.CreateHFMetricsFilter(poolId);
    // TODO: Create other algorithm components.
}

void BatchAnalyzer::SetupStaticModel(const PacBio::Mongo::Data::StaticDetModelConfig& staticDetModelConfig,
                                     const PacBio::Mongo::Data::MovieConfig& movieConfig)
{
    staticAnalysis_ = true;

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


BatchAnalyzer::OutputType BatchAnalyzer::operator()(TraceBatch<int16_t> tbatch)
{
    auto ret = [&]() {
        if(staticAnalysis_)
        {
            return StaticModelPipeline(std::move(tbatch));
        } else {
            return StandardPipeline(std::move(tbatch));
        }
    }();
    if (Cuda::StreamErrorCount() > 0)
        throw PBException("Unexpected stream synchronization issues were detected");
    return ret;
}

BatchAnalyzer::OutputType BatchAnalyzer::StaticModelPipeline(TraceBatch<int16_t> tbatch)
{
    PBAssert(tbatch.Metadata().PoolId() == poolId_, "Bad pool ID.");
    PBAssert(tbatch.Metadata().FirstFrame() == nextFrameId_, "Bad frame ID.");

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
    auto baselinedTracesAndMetrics = (*baseliner_)(std::move(tbatch));
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

    nextFrameId_ = tbatch.Metadata().LastFrame();

    return BatchResult(std::move(pulses), std::move(basecallingMetrics));
}

BatchAnalyzer::OutputType BatchAnalyzer::StandardPipeline(TraceBatch<int16_t> tbatch)
{
    PBAssert(tbatch.Metadata().PoolId() == poolId_, "Bad pool ID.");
    PBAssert(tbatch.Metadata().FirstFrame() == nextFrameId_, "Bad frame ID.");

    // Baseline estimation and subtraction.
    // Includes computing baseline moments.
    assert(baseliner_);
    auto baselinedTracesAndMetrics = (*baseliner_)(std::move(tbatch));
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

            auto pulsesAndMetrics = pulseAccumulator_->EmptyPulseBatch(baselinedTraces.Metadata());
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

    nextFrameId_ = tbatch.Metadata().LastFrame();

    return BatchResult(std::move(pulses), std::move(basecallingMetrics));
}

}}}     // namespace PacBio::Mongo::Basecaller
