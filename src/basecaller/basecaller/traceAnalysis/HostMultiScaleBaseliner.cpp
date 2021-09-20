//  Copyright (c) 2019, Pacific Biosciences of California, Inc.
//
//  All rights reserved.
//
//  Redistribution and use in source and binary forms, with or without
//  modification, are permitted provided that the following conditions are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of Pacific Biosciences nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.
//
//  NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
//  THIS LICENSE.  THIS SOFTWARE IS PROVIDED BY PACIFIC BIOSCIENCES AND ITS
//  CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
//  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
//  PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL PACIFIC BIOSCIENCES OR
//  ITS CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
//  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
//  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
//  BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
//  IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
//  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
//  POSSIBILITY OF SUCH DAMAGE.

#include "HostMultiScaleBaseliner.h"
#include <tbb/task_arena.h>
#include <tbb/parallel_for.h>

#include <dataTypes/BasicTypes.h>
#include <dataTypes/BaselinerStatAccumulator.h>


namespace PacBio {
namespace Mongo {
namespace Basecaller {

void HostMultiScaleBaseliner::Configure(const Data::BasecallerBaselinerConfig&,
                                        const Data::MovieConfig&)
{
    const auto hostExecution = true;
    Baseliner::InitFactory(hostExecution);
}

void HostMultiScaleBaseliner::Finalize() {}

std::pair<Data::TraceBatch<HostMultiScaleBaseliner::ElementTypeOut>,
          Data::BaselinerMetrics>
HostMultiScaleBaseliner::FilterBaseline_(const Data::TraceBatch<ElementTypeIn>& rawTrace)
{
    assert(rawTrace.LanesPerBatch() <= baselinerByLane_.size());

    auto out = batchFactory_->NewBatch(rawTrace.GetMeta(), rawTrace.StorageDims());

    // TODO: We don't need to allocate these large buffers, we only need 2 BlockView<T> buffers which can be reused.
    Data::BatchData<ElementTypeIn> lowerBuffer(rawTrace.StorageDims(),
                                               Cuda::Memory::SyncDirection::HostWriteDeviceRead, SOURCE_MARKER());
    Data::BatchData<ElementTypeIn> upperBuffer(rawTrace.StorageDims(),
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
}

Data::BaselinerStatAccumulator<HostMultiScaleBaseliner::ElementTypeOut>
HostMultiScaleBaseliner::MultiScaleBaseliner::EstimateBaseline(const Data::BlockView<const ElementTypeIn>& traceData,
                                                               Data::BlockView<ElementTypeIn> lowerBuffer,
                                                               Data::BlockView<ElementTypeIn> upperBuffer,
                                                               Data::BlockView<ElementTypeOut> baselineSubtractedData)
{
    // Run lower and upper filters, results are strided out.
    std::memcpy(lowerBuffer.Data(), traceData.Data(), traceData.Size()*sizeof(ElementTypeIn));
    std::memcpy(upperBuffer.Data(), traceData.Data(), traceData.Size()*sizeof(ElementTypeIn));
    const auto& lower = msLowerOpen_(&lowerBuffer);
    const auto& upper = msUpperOpen_(&upperBuffer);

    // Compute and subtract baseline while tabulating the stats.
    auto trIt = traceData.CBegin();
    auto blsIt = baselineSubtractedData.Begin();
    auto loIt = lower->CBegin(), upIt = upper->CBegin();

    auto sbInv = 1.0f / cSigmaBias_;

    size_t inputCount = traceData.NumFrames() / Stride();
    auto baselinerStats = Data::BaselinerStatAccumulator<ElementTypeOut>{};
    for (size_t i = 0; i < inputCount; i++, upIt++, loIt++)
    {
        auto upperVal = upIt.Extract(), lowerVal = loIt.Extract();
        const auto& bias = (upperVal + lowerVal) * 0.5f;
        const auto& framebkgndSigma = (upperVal - lowerVal) * sbInv;
        const auto& smoothedBkgndSigma = GetSmoothedSigma(framebkgndSigma);
        FloatArray baselineEstimate((bias + cMeanBias_ * smoothedBkgndSigma) * scaler_);

        // Estimates are scattered on stride intervals.
        for (size_t j = 0; j < Stride(); j++, trIt++, blsIt++)
        {
            // Data scaled first
            auto rawSignal = trIt.Extract() * scaler_;
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
    const FloatArray alphaFactor{0.7f};
    const float minSigma { sqrt(1.0f/12.0f) };

    bgSigma_ = ((FloatArray{1.0f} - alphaFactor) * bgSigma_)
                             + (alphaFactor * max(sigma, FloatArray{minSigma}));

    // Update thresholds for classifying baseline frames.
    constexpr float sigmaThrL { 4.5f };
    thrLow_ = FloatArray{sigmaThrL} * bgSigma_;

    constexpr float sigmaThrH { 4.5f };
    thrHigh_ = FloatArray{sigmaThrH} * bgSigma_;

    return bgSigma_;
}

}}}      // namespace PacBio::Mongo::Basecaller
