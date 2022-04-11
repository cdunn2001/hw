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
#include <dataTypes/configs/AnalysisConfig.h>


namespace PacBio {
namespace Mongo {
namespace Basecaller {

void HostMultiScaleBaseliner::Configure(const Data::BasecallerBaselinerConfig& bbc,
                                        const Data::AnalysisConfig& analysisConfig)
{
    const auto hostExecution = true;
    InitFactory(hostExecution, analysisConfig);

    MultiScaleBaseliner::Configure(bbc, analysisConfig);
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

        const auto& tracemd = out.first.Metadata();
        out.second.frameInterval = {tracemd.FirstFrame(), tracemd.LastFrame()};

        return out;
    }, batch.Data());
}

template <typename T>
Data::BaselinerStatAccumulator<HostMultiScaleBaseliner::ElementTypeOut>
HostMultiScaleBaseliner::LaneBaseliner::EstimateBaseline(const Data::BlockView<const T>& traceData,
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
    const size_t fdSize = traceData.NumFrames() / Stride();
    auto baselinerStats = Data::BaselinerStatAccumulator<ElementTypeOut>{};
    for (size_t i = 0; i < fdSize; i++, upIt++, loIt++)
    {
        auto lowerVal = loIt.Extract() - pedestal_;
        auto upperVal = upIt.Extract() - pedestal_;
        FloatArray blEst = GetSmoothedBlEstimate(lowerVal, upperVal);

        // Estimates are scattered on stride intervals.
        for (size_t j = 0; j < Stride(); j++, trIt++, blsIt++)
        {
            // Data shifted and scaled
            auto rawSignal = trIt.Extract() - pedestal_;
            LaneArray blSubtractedFrame((rawSignal - blEst) * scaler_);
            // ... stored as output traces
            blsIt.Store(blSubtractedFrame);
            // ... and added to statistics
            AddToBaselineStats(rawSignal * scaler_, blSubtractedFrame, baselinerStats);
        }
    }

    return baselinerStats;
}

void HostMultiScaleBaseliner::LaneBaseliner::AddToBaselineStats(const LaneArray& traceData,
                                                                      const LaneArray& baselineSubtractedFrame,
                                                                      Data::BaselinerStatAccumulator<ElementTypeOut>& baselinerStats)
{
    // Thresholds below are specified as floats whereas incoming frame data are shorts.
    constexpr float sigmaThrL = 4.5f;
    FloatArray thrLow = FloatArray(sigmaThrL) * blSigmaEma_;

    constexpr float sigmaThrH = 4.5f;
    FloatArray thrHigh = FloatArray(sigmaThrH) * blSigmaEma_;

    // Compute the high mask at the plus-1 position (this) for variance
    const auto& maskHp1 = baselineSubtractedFrame < thrHigh * scaler_;

    // Compute the full mask to use for the single-frame latent variance
    // Minus-1[High] & Pos-0[Low] & Plus-1[High]
    const auto& mask = latHMask1_ & latLMask_ & maskHp1;

    // Push the plus-1 frame masks
    latLMask_ = baselineSubtractedFrame < thrLow * scaler_;
    latHMask2_ = latHMask1_;
    latHMask1_ = maskHp1;

    baselinerStats.AddSample(latRawData_, latData_, mask);

    // Set latent data.
    latRawData_ = traceData;
    latData_ = baselineSubtractedFrame;
}

HostMultiScaleBaseliner::FloatArray
HostMultiScaleBaseliner::LaneBaseliner::GetSmoothedBlEstimate(const LaneArray& lower, const LaneArray& upper)
{
    static constexpr float minSigma = .288675135f; // sqrt(1.0f/12.0f);
    const float sigmaEmaAlpha = SigmaEmaAlpha();

    auto sigma = max((upper - lower) / cSigmaBias_, minSigma);
    auto newSigmaEma = sigmaEmaAlpha * blSigmaEma_ + (1.0f - sigmaEmaAlpha) * sigma;

    // Calculate the new single-stride estimate of baseline mean.
    const FloatArray blEst = 0.5f * (upper + lower) + cMeanBias_ * newSigmaEma;

    // We presume that large jumps represent pathological enzyme-analog
    // binding events.
    // After the first estimate, don't update exponential moving
    // averages if blEst exceeds previous baseline EMA by more than
    // jump tolerance.
    // Notice the asymmetry--only positive jumps are suppressed.
    
    // TODO: Enable masking for jumpTolCoeff_
    // const auto mask = ((blMeanUemaWeight_ == 0.0f)
    //     | (blEst - blMeanUemaSum_ / blMeanUemaWeight_ < jumpTolCoeff_* blSigmaEma_));
    const BoolArray mask = true;

    // Conditionally update EMAs of baseline mean and sigma.
    const FloatArray newWeight = meanEmaAlpha_ * blMeanUemaWeight_ + (1.0f - meanEmaAlpha_);
    const FloatArray newSum    = meanEmaAlpha_ * blMeanUemaSum_    + (1.0f - meanEmaAlpha_) * blEst;
    blMeanUemaWeight_ = Blend(mask, newWeight, blMeanUemaWeight_);
    blMeanUemaSum_    = Blend(mask, newSum, blMeanUemaSum_);
    blSigmaEma_       = Blend(mask, newSigmaEma, blSigmaEma_);

    assert(all(blMeanUemaWeight_ > 0.0f));

    return blMeanUemaSum_ / blMeanUemaWeight_;
}

}}}      // namespace PacBio::Mongo::Basecaller
