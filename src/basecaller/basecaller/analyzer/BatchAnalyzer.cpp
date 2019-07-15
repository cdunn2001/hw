
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

#include <basecaller/traceAnalysis/Baseliner.h>
#include <basecaller/traceAnalysis/FrameLabeler.h>
#include <basecaller/traceAnalysis/PulseAccumulator.h>
#include <basecaller/traceAnalysis/DetectionModelEstimator.h>
#include <basecaller/traceAnalysis/TraceHistogramAccumulator.h>

#include <dataTypes/BasecallBatch.h>
#include <dataTypes/CameraTraceBatch.h>
#include <dataTypes/LaneDetectionModel.h>
#include <dataTypes/PoolHistogram.h>
#include <dataTypes/TraceBatch.h>
#include <dataTypes/BasecallerConfig.h>

#include "AlgoFactory.h"

using namespace PacBio::Mongo::Data;

namespace PacBio {
namespace Mongo {
namespace Basecaller {

std::unique_ptr<Data::BasecallBatchFactory> BatchAnalyzer::batchFactory_;
uint16_t BatchAnalyzer::maxCallsPerZmwChunk_;

BatchAnalyzer::~BatchAnalyzer() = default;
BatchAnalyzer::BatchAnalyzer(BatchAnalyzer&&) = default;


// static
void BatchAnalyzer::Configure(const Data::BasecallerAlgorithmConfig& bcConfig,
                              const Data::MovieConfig& movConfig)
{
    BatchDimensions dims;
    dims.framesPerBatch = GetPrimaryConfig().framesPerChunk;
    dims.laneWidth = laneSize;
    dims.lanesPerBatch = GetPrimaryConfig().lanesPerPool;

    batchFactory_ = std::make_unique<BasecallBatchFactory>(
        bcConfig.pulseAccumConfig.maxCallsPerZmw,
        dims,
        Cuda::Memory::SyncDirection::HostWriteDeviceRead,
        true);

    maxCallsPerZmwChunk_ = bcConfig.pulseAccumConfig.maxCallsPerZmw;
}


BatchAnalyzer::BatchAnalyzer(uint32_t poolId, const AlgoFactory& algoFac, bool staticAnalysis)
    : poolId_ (poolId)
    , models_(PrimaryConfig().lanesPerPool, Cuda::Memory::SyncDirection::Symmetric, true)
    , staticAnalysis_(staticAnalysis)
{
    baseliner_ = algoFac.CreateBaseliner(poolId);
    traceHistAccum_ = algoFac.CreateTraceHistAccumulator(poolId);
    dme_ = algoFac.CreateDetectionModelEstimator(poolId);
    frameLabeler_ = algoFac.CreateFrameLabeler(poolId);
    pulseAccumulator_ = algoFac.CreateAccumulator(poolId);
    // TODO: Create other algorithm components.

    // Not running DME, need to fake our model
    if (staticAnalysis_)
    {
        Data::LaneModelParameters<Cuda::PBHalf, laneSize> model;
        model.AnalogMode(0).SetAllMeans(227.13);
        model.AnalogMode(1).SetAllMeans(154.45);
        model.AnalogMode(2).SetAllMeans(97.67);
        model.AnalogMode(3).SetAllMeans(61.32);

        model.AnalogMode(0).SetAllVars(776);
        model.AnalogMode(1).SetAllVars(426);
        model.AnalogMode(2).SetAllVars(226);
        model.AnalogMode(3).SetAllVars(132);

        // Need a new trace file to target, these values come from a file with
        // zero baseline mean
        model.BaselineMode().SetAllMeans(0);
        model.BaselineMode().SetAllVars(33);

        auto view = models_.GetHostView();
        for (size_t i = 0; i < view.Size(); ++i)
        {
            view[i] = model;
        }
    }
}


BasecallBatch BatchAnalyzer::operator()(TraceBatch<int16_t> tbatch)
{
    if(staticAnalysis_)
    {
        return StaticModelPipeline(std::move(tbatch));
    } else {
        return StandardPipeline(std::move(tbatch));
    }
}

namespace {

// Temporary helper function for converting from the mongo Pulse data type to the old Sequel
// Base data type.  Will eventually be replaed by a proper pulse-to-base stage, (and presumably
// Conversion will be done to a new mongo-specific Base type
void ConvertPulsesToBases(const Data::PulseBatch& pulses, Data::BasecallBatch& bases)
{
    auto LabelConv = [](Data::Pulse::NucleotideLabel label) {
        switch (label)
        {
        case Data::Pulse::NucleotideLabel::A:
            return SmrtData::NucleotideLabel::A;
        case Data::Pulse::NucleotideLabel::C:
            return SmrtData::NucleotideLabel::C;
        case Data::Pulse::NucleotideLabel::G:
            return SmrtData::NucleotideLabel::G;
        case Data::Pulse::NucleotideLabel::T:
            return SmrtData::NucleotideLabel::T;
        case Data::Pulse::NucleotideLabel::N:
            return SmrtData::NucleotideLabel::N;
        default:
            assert(label == Data::Pulse::NucleotideLabel::NONE);
            return SmrtData::NucleotideLabel::NONE;
        }
    };
    for (size_t lane = 0; lane < pulses.Dims().lanesPerBatch; ++lane)
    {
        auto baseView = bases.Basecalls().LaneView(lane);
        auto pulsesView = pulses.Pulses().LaneView(lane);

        baseView.Reset();
        for (size_t zmw = 0; zmw < laneSize; ++zmw)
        {
            auto numBases = pulsesView.size(zmw);
            for (size_t b = 0; b < numBases; ++b)
            {
                const auto& pulse = pulsesView(zmw, b);
                PacBio::SmrtData::Basecall bc;

                static constexpr int8_t qvDefault_ = 0;

                auto label = LabelConv(pulse.Label());

                // Populate pulse data
                bc.GetPulse().Start(pulse.Start()).Width(pulse.Width());
                bc.GetPulse().MeanSignal(pulse.MeanSignal()).MidSignal(pulse.MidSignal()).MaxSignal(pulse.MaxSignal());
                bc.GetPulse().Label(label).LabelQV(qvDefault_);
                bc.GetPulse().AltLabel(label).AltLabelQV(qvDefault_);
                bc.GetPulse().MergeQV(qvDefault_);

                // Populate base data.
                bc.Base(label).InsertionQV(qvDefault_);
                bc.DeletionTag(SmrtData::NucleotideLabel::N).DeletionQV(qvDefault_);
                bc.SubstitutionTag(SmrtData::NucleotideLabel::N).SubstitutionQV(qvDefault_);

                baseView.push_back(zmw, bc);
            }
        }
    }

}

}

BasecallBatch BatchAnalyzer::StaticModelPipeline(TraceBatch<int16_t> tbatch)
{
    PBAssert(tbatch.Metadata().PoolId() == poolId_, "Bad pool ID.");
    PBAssert(tbatch.Metadata().FirstFrame() == nextFrameId_, "Bad frame ID.");

    auto ctb = (*baseliner_)(std::move(tbatch));
    auto labels = (*frameLabeler_)(std::move(ctb), models_);
    auto pulses = (*pulseAccumulator_)(std::move(labels));

    auto bases = batchFactory_->NewBatch(tbatch.Metadata());
    ConvertPulsesToBases(pulses, bases);

    nextFrameId_ = tbatch.Metadata().LastFrame();

    return bases;
}


BasecallBatch BatchAnalyzer::StandardPipeline(TraceBatch<int16_t> tbatch)
{
    PBAssert(tbatch.Metadata().PoolId() == poolId_, "Bad pool ID.");
    PBAssert(tbatch.Metadata().FirstFrame() == nextFrameId_, "Bad frame ID.");

    // TODO: Develop error handling logic.

    // Baseline estimation and subtraction.
    // Includes computing baseline moments.
    CameraTraceBatch ctb = (*baseliner_)(std::move(tbatch));

    if (!isModelInitialized_)
    {
        // TODO: Factor model initialization and estimation operations.

        // Accumulate histogram of baseline-subtracted trace data.
        // This operation also accumulates baseliner statistics.
        traceHistAccum_->AddBatch(ctb);

        // When sufficient trace data have been histogrammed,
        // estimate detection model.
        const auto minFramesForDme = DetectionModelEstimator::MinFramesForEstimate();
        if (traceHistAccum_->HistogramFrameCount() >= minFramesForDme)
        {
            // Initialize the detection model from baseliner statistics.
            models_ = dme_->InitDetectionModels(traceHistAccum_->TraceStats());
            isModelInitialized_ = true;

            // Estimate model parameters from histogram.
            dme_->Estimate(traceHistAccum_->Histogram(), &models_);
        }
    }

    // TODO: When detection model is available, classify frames.

    // TODO: When frames are classified, generate pulses with metrics.

    // TODO: When pulses are generated, call bases.

    nextFrameId_ = tbatch.Metadata().LastFrame();

    auto basecallsBatch = batchFactory_->NewBatch(tbatch.Metadata());

    using NucleotideLabel = PacBio::SmrtData::NucleotideLabel;

    // Repeating sequence of ACGT.
    const NucleotideLabel labels[] = { NucleotideLabel::A, NucleotideLabel::C,
                                       NucleotideLabel::G, NucleotideLabel::T };

    // Associated values
    const std::array<float, 4> meanSignals { { 20.0f, 10.0f, 16.0f, 8.0f } };
    const std::array<float, 4> midSignals { { 21.0f, 11.0f, 17.0f, 9.0f } };
    const std::array<float, 4> maxSignals { { 21.0f, 11.0f, 17.0f, 9.0f } };

    static constexpr int8_t qvDefault_ = 0;

    auto& basecalls = basecallsBatch.Basecalls();
    for (uint32_t lane = 0; lane < basecallsBatch.Dims().lanesPerBatch; lane++)
    {
        auto laneCalls = basecalls.LaneView(lane);
        for (uint32_t zmw = 0; zmw < laneSize; ++zmw)
        {
            for (uint16_t b = 0; b < maxCallsPerZmwChunk_; b++)
            {
                BasecallBatch::Basecall bc;
                auto& pulse = bc.GetPulse();

                size_t iL = b % 4;
                size_t iA = (b + 1) % 4;

                auto label = labels[iL];
                auto altLabel = labels[iA];

                // Populate pulse data
                pulse.Start(1).Width(3);
                pulse.MeanSignal(meanSignals[iL]).MidSignal(midSignals[iL]).MaxSignal(maxSignals[iL]);
                pulse.Label(label).LabelQV(qvDefault_);
                pulse.AltLabel(altLabel).AltLabelQV(qvDefault_);
                pulse.MergeQV(qvDefault_);

                // Populate base data.
                bc.Base(label).InsertionQV(qvDefault_);
                bc.DeletionTag(NucleotideLabel::N).DeletionQV(qvDefault_);
                bc.SubstitutionTag(NucleotideLabel::N).SubstitutionQV(qvDefault_);

                laneCalls.push_back(zmw, bc);
            }
        }
    }

    return basecallsBatch;
}

}}}     // namespace PacBio::Mongo::Basecaller
