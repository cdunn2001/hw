
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
#include <basecaller/traceAnalysis/DetectionModelEstimator.h>
#include <basecaller/traceAnalysis/TraceHistogramAccumulator.h>

#include <dataTypes/BasecallBatch.h>
#include <dataTypes/CameraTraceBatch.h>
#include <dataTypes/PoolDetectionModel.h>
#include <dataTypes/PoolHistogram.h>
#include <dataTypes/TraceBatch.h>
#include <dataTypes/BasecallerConfig.h>

#include "AlgoFactory.h"

using namespace PacBio::Mongo::Data;

namespace PacBio {
namespace Mongo {
namespace Basecaller {

BatchAnalyzer::~BatchAnalyzer() = default;
BatchAnalyzer::BatchAnalyzer(BatchAnalyzer&&) = default;


// static
void BatchAnalyzer::Configure(const Data::BasecallerAlgorithmConfig& bcConfig,
                              const Data::MovieConfig& movConfig)
{ }


BatchAnalyzer::BatchAnalyzer(uint32_t poolId, const AlgoFactory& algoFac, bool staticAnalysis)
    : poolId_ (poolId)
    , models_(PrimaryConfig().lanesPerPool, Cuda::Memory::SyncDirection::Symmetric, true)
    , staticAnalysis_(staticAnalysis)
{
    baseliner_ = algoFac.CreateBaseliner(poolId);
    traceHistAccum_ = algoFac.CreateTraceHistAccumulator(poolId);
    dme_ = algoFac.CreateDetectionModelEstimator(poolId);
    frameLabeler_ = algoFac.CreateFrameLabeler(poolId);
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


BasecallBatch BatchAnalyzer::StaticModelPipeline(TraceBatch<int16_t> tbatch)
{
    PBAssert(tbatch.Metadata().PoolId() == poolId_, "Bad pool ID.");
    PBAssert(tbatch.Metadata().FirstFrame() == nextFrameId_, "Bad frame ID.");

    // TODO: Define this so that it scales properly with chunk size, frame rate,
    // and max polymerization rate.
    const uint16_t maxCallsPerZmwChunk = 96;

    // TODO: Develop error handling logic.

    // Baseline estimation and subtraction.
    // Includes computing baseline moments.
    auto ctb = (*baseliner_)(std::move(tbatch));
    auto labels = (*frameLabeler_)(std::move(ctb), models_);
    labels.DeactivateGpuMem();

    nextFrameId_ = tbatch.Metadata().LastFrame();

    return BasecallBatch(maxCallsPerZmwChunk, tbatch.Dimensions(), tbatch.Metadata());
}


BasecallBatch BatchAnalyzer::StandardPipeline(TraceBatch<int16_t> tbatch)
{
    PBAssert(tbatch.Metadata().PoolId() == poolId_, "Bad pool ID.");
    PBAssert(tbatch.Metadata().FirstFrame() == nextFrameId_, "Bad frame ID.");

    // TODO: Define this so that it scales properly with chunk size, frame rate,
    // and max polymerization rate.
    const uint16_t maxCallsPerZmwChunk = 96;

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
        // TODO: Make this configurable.
        const unsigned int minFramesForDme = 4000u;
        if (traceHistAccum_->FramesAdded() >= minFramesForDme)
        {
            auto detModel = (*dme_)(traceHistAccum_->Histogram(),
                                    traceHistAccum_->TraceStats());
            models_ = std::move(detModel.laneModels);
            isModelInitialized_ = true;
        }
    }

    // TODO: When detection model is available, classify frames.

    // TODO: When frames are classified, generate pulses with metrics.

    // TODO: When pulses are generated, call bases.

    nextFrameId_ = tbatch.Metadata().LastFrame();

    auto basecalls = BasecallBatch(maxCallsPerZmwChunk, tbatch.Dimensions(), tbatch.Metadata());

    using NucleotideLabel = PacBio::SmrtData::NucleotideLabel;

    // Repeating sequence of ACGT.
    const NucleotideLabel labels[] = { NucleotideLabel::A, NucleotideLabel::C,
                                       NucleotideLabel::G, NucleotideLabel::T };

    // Associated values
    const std::array<float, 4> meanSignals { { 20.0f, 10.0f, 16.0f, 8.0f } };
    const std::array<float, 4> midSignals { { 21.0f, 11.0f, 17.0f, 9.0f } };
    const std::array<float, 4> maxSignals { { 21.0f, 11.0f, 17.0f, 9.0f } };

    static constexpr int8_t qvDefault_ = 0;

    for (uint32_t z = 0; z < basecalls.Dims().ZmwsPerBatch(); z++)
    {
        for (uint16_t b = 0; b < maxCallsPerZmwChunk; b++)
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

            basecalls.PushBack(z, bc);
        }
    }

    return basecalls;
}

}}}     // namespace PacBio::Mongo::Basecaller
