// Copyright (c) 2021, Pacific Biosciences of California, Inc.
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
//  Defines members of class HostMultiScaleBaseliner.

#include "HostMultiScaleBaseliner.h"

#include <cmath>
#include <sstream>

#include <tbb/task_arena.h>
#include <tbb/parallel_for.h>

#include <dataTypes/BasicTypes.h>
#include <dataTypes/BaselinerStatAccumulator.h>
#include <dataTypes/configs/MovieConfig.h>


namespace PacBio {
namespace Mongo {
namespace Basecaller {

// static
float HostMultiScaleBaseliner::sigmaEmaAlpha_ = 0.0f;

void HostMultiScaleBaseliner::Configure(const Data::BasecallerBaselinerConfig& bbc,
                                        const Data::MovieConfig& movConfig)
{
    const auto hostExecution = true;
    InitFactory(hostExecution, movConfig);

    {
        // Validation has already been handled in the configuration framework.
        // This just asserts that the configuration is indeed valid.
        assert(bbc.Validate());
        const float sigmaEmaScale = bbc.SigmaEmaScaleStrides;
        std::ostringstream msg;
        msg << "SigmaEmaScaleStrides = " << sigmaEmaScale << '.';
        // TODO: Use a scoped logger.
        PBLOG_INFO << msg.str();
        sigmaEmaAlpha_ = std::exp2(-1.0f / sigmaEmaScale);
        assert(0.0f <= sigmaEmaAlpha_ && sigmaEmaAlpha_ <= 1.0f);
    }
}

void HostMultiScaleBaseliner::Finalize() {}

std::pair<Data::TraceBatch<HostMultiScaleBaseliner::ElementTypeOut>,
          Data::BaselinerMetrics>
HostMultiScaleBaseliner::FilterBaseline(const Data::TraceBatchVariant& batch)
{
    return std::visit([&](const auto& rawTrace)
    {
        assert(rawTrace.LanesPerBatch() <= baselinerByLane_.size());

        auto out = batchFactory_->NewBatch(rawTrace.GetMeta(), rawTrace.StorageDims());

        // TODO: We don't need to allocate these large buffers, we only need 2 BlockView<T> buffers which can be reused.
        Data::BatchData<ElementTypeOut> lowerBuffer(rawTrace.StorageDims(),
                                                   Cuda::Memory::SyncDirection::HostWriteDeviceRead, SOURCE_MARKER());
        Data::BatchData<ElementTypeOut> upperBuffer(rawTrace.StorageDims(),
                                                   Cuda::Memory::SyncDirection::HostWriteDeviceRead, SOURCE_MARKER());

        auto statsView = out.second.baselinerStats.GetHostView();
        tbb::task_arena().execute([&] {
            tbb::parallel_for(size_t{0}, rawTrace.LanesPerBatch(), [&](size_t laneIdx) {
                auto baselinerStats = baselinerByLane_[laneIdx].EstimateBaseline(
                        rawTrace.GetBlockView(laneIdx),
                        lowerBuffer.GetBlockView(laneIdx),
                        upperBuffer.GetBlockView(laneIdx),
                        out.first.GetBlockView(laneIdx));

                statsView[laneIdx] = baselinerStats.GetState();
            });
        });

        return out;
    }, batch.Data());
}

template <typename T>
Data::BaselinerStatAccumulator<HostMultiScaleBaseliner::ElementTypeOut>
HostMultiScaleBaseliner::MultiScaleBaseliner::EstimateBaseline(const Data::BlockView<const T>& traceData,
                                                               Data::BlockView<ElementTypeOut> lowerBuffer,
                                                               Data::BlockView<ElementTypeOut> upperBuffer,
                                                               Data::BlockView<ElementTypeOut> baselineSubtractedData)
{
    assert(traceData.NumFrames() == lowerBuffer.NumFrames());
    assert(traceData.NumFrames() == upperBuffer.NumFrames());
    auto inItr = traceData.CBegin();
    auto lowItr = lowerBuffer.Begin();
    auto upItr = upperBuffer.Begin();
    for ( ; inItr != traceData.CEnd(); ++inItr, ++lowItr, ++upItr)
    {
        auto dat = inItr.Extract();
        lowItr.Store(dat);
        upItr.Store(dat);
    }
    // Run lower and upper filters, results are strided out.
    const auto& lower = msLowerOpen_(&lowerBuffer);
    const auto& upper = msUpperOpen_(&upperBuffer);

    // Compute and subtract baseline while tabulating the stats.
    auto trIt = traceData.CBegin();
    auto blsIt = baselineSubtractedData.Begin();
    auto loIt = lower->CBegin(), upIt = upper->CBegin();

    const auto sbInv = 1.0f / cSigmaBias_;
    const size_t inputCount = traceData.NumFrames() / Stride();

    auto baselinerStats = Data::BaselinerStatAccumulator<ElementTypeOut>{};
    for (size_t i = 0; i < inputCount; i++, upIt++, loIt++)
    {
        auto upperVal = upIt.Extract() - pedestal_;
        auto lowerVal = loIt.Extract() - pedestal_;
        const auto& bias = (upperVal + lowerVal) * 0.5f;
        const auto& framebkgndSigma = (upperVal - lowerVal) * sbInv;
        const auto& smoothedBkgndSigma = GetSmoothedSigma(framebkgndSigma);
        FloatArray baselineEstimate((bias + cMeanBias_ * smoothedBkgndSigma) * scaler_);

        // Estimates are scattered on stride intervals.
        for (size_t j = 0; j < Stride(); j++, trIt++, blsIt++)
        {
            // Data scaled first
            auto rawSignal = (trIt.Extract() - pedestal_) * scaler_;
            LaneArray blSubtractedFrame(rawSignal - baselineEstimate);
            // ... then added to statistics
            blsIt.Store(blSubtractedFrame);
            AddToBaselineStats(rawSignal, blSubtractedFrame, baselinerStats);
        }
    }

    return baselinerStats;
}

void HostMultiScaleBaseliner::MultiScaleBaseliner::AddToBaselineStats(const LaneArray& traceData,
                                                                      const LaneArray& baselineSubtractedFrame,
                                                                      Data::BaselinerStatAccumulator<ElementTypeOut>& baselinerStats)
{
    // Thresholds below are specified as floats whereas incoming frame data are shorts.

    // Compute the high mask at the plus-1 position (this) for variance
    const auto& maskHp1 = baselineSubtractedFrame < thrHigh_ * scaler_;

    // Compute the full mask to use for the single-frame latent variance
    // Minus-1[High] & Pos-0[Low] & Plus-1[High]
    const auto& mask = latHMask1_ & latLMask_ & maskHp1;

    // Push the plus-1 frame masks
    latLMask_ = baselineSubtractedFrame < thrLow_ * scaler_;
    latHMask2_ = latHMask1_;
    latHMask1_ = maskHp1;

    baselinerStats.AddSample(latRawData_, latData_, mask);

    // Set latent data.
    latRawData_ = traceData;
    latData_ = baselineSubtractedFrame;
}

HostMultiScaleBaseliner::FloatArray
HostMultiScaleBaseliner::MultiScaleBaseliner::GetSmoothedSigma(const FloatArray& sigma)
{
    // Fixed thresholds for variance computation.
    // TODO - Make these tunable parameters.
    const float alphaFactor {SigmaEmaAlpha()};
    const float minSigma { sqrt(1.0f/12.0f) };

    bgSigma_ = (1.0f - alphaFactor) * bgSigma_
                + alphaFactor * max(sigma, FloatArray{minSigma});

    // Update thresholds for classifying baseline frames.
    constexpr float sigmaThrL { 4.5f };
    thrLow_ = FloatArray{sigmaThrL} * bgSigma_;

    constexpr float sigmaThrH { 4.5f };
    thrHigh_ = FloatArray{sigmaThrH} * bgSigma_;

    return bgSigma_;
}

}}}      // namespace PacBio::Mongo::Basecaller
