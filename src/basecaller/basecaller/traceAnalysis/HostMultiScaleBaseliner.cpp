#include "HostMultiScaleBaseliner.h"

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
HostMultiScaleBaseliner::Process(const Data::TraceBatch<ElementTypeIn>& rawTrace)
{
    assert(rawTrace.LanesPerBatch() <= baselinerByLane_.size());

    auto out = batchFactory_->NewBatch(rawTrace.GetMeta(), rawTrace.StorageDims());

    // TODO: We don't need to allocate these large buffers, we only need 2 BlockView<T> buffers which can be reused.
    Data::BatchData<ElementTypeIn> lowerBuffer(rawTrace.StorageDims(),
                                               Cuda::Memory::SyncDirection::HostWriteDeviceRead, SOURCE_MARKER());
    Data::BatchData<ElementTypeIn> upperBuffer(rawTrace.StorageDims(),
                                               Cuda::Memory::SyncDirection::HostWriteDeviceRead, SOURCE_MARKER());

    auto statsView = out.second.baselinerStats.GetHostView();
    tbb::parallel_for(size_t{0}, rawTrace.LanesPerBatch(), [&](size_t laneIdx)
    {
        const auto& traceData = rawTrace.GetBlockView(laneIdx);
        auto baselineSubtracted = out.first.GetBlockView(laneIdx);
        auto& baseliner = baselinerByLane_[laneIdx];

        auto baselinerStats = baseliner.EstimateBaseline(traceData,
                                                         lowerBuffer.GetBlockView(laneIdx),
                                                         upperBuffer.GetBlockView(laneIdx),
                                                         baselineSubtracted);

        statsView[laneIdx] = baselinerStats.GetState();
    });

    return out;
}

Data::BaselinerStatAccumulator<HostMultiScaleBaseliner::ElementTypeOut>
HostMultiScaleBaseliner::MultiScaleBaseliner::EstimateBaseline(const Data::BlockView<const ElementTypeIn>& traceData,
                                                               Data::BlockView<ElementTypeIn> lowerBuffer,
                                                               Data::BlockView<ElementTypeIn> upperBuffer,
                                                               Data::BlockView<ElementTypeOut> baselineSubtractedData)
{
    // Run lower filter, results are strided out.
    std::memcpy(lowerBuffer.Data(), traceData.Data(), traceData.Size()*sizeof(ElementTypeIn));
    const auto& lower = msLowerOpen_(&lowerBuffer);

    // Run upper filter, results are strided out.
    std::memcpy(upperBuffer.Data(), traceData.Data(), traceData.Size()*sizeof(ElementTypeIn));
    const auto& upper = msUpperOpen_(&upperBuffer);

    // Compute and subtract baseline while tabulating the stats.
    auto baselinerStats = Data::BaselinerStatAccumulator<ElementTypeOut>{};
    auto trIter = traceData.CBegin();
    auto blsIter = baselineSubtractedData.Begin();
    auto lowerIter = lower->CBegin();
    auto upperIter = upper->CBegin();

    using ValueType = typename decltype(trIter)::ValueType;
    for ( ; blsIter != baselineSubtractedData.End() && lowerIter != lower->CEnd() && upperIter != upper->CEnd();
            blsIter += Stride(), lowerIter += Stride(), upperIter += Stride())
    {
        auto upperVal = upperIter.Extract();
        auto lowerVal = lowerIter.Extract();
        const auto& bias = (upperVal + lowerVal) / 2.0f;
        const auto& framebkgndSigma = (upperVal - lowerVal) / cSigmaBias_;
        const auto& smoothedBkgndSigma = GetSmoothedSigma(framebkgndSigma * FloatArray{Scale()});
        const auto& frameBiasEstimate = cMeanBias_ * smoothedBkgndSigma;

        // Estimates are scattered on stride intervals.
        auto& strideIter = blsIter;
        for (size_t i = 0; i < Stride() && strideIter != baselineSubtractedData.End() && trIter != traceData.CEnd();
             i++, strideIter++, trIter++)
        {
            const auto& rawSignal = trIter.Extract();

            // NOTE: We need to scale the trace data (from DN to e-) and
            // end up converting the baseline subtracted data to float in order
            // to perform the conversion and then end up converting it back.
            auto out = AsShort((rawSignal - bias - frameBiasEstimate) * Scale());
            strideIter.Store(out);

            AddToBaselineStats(rawSignal, out, baselinerStats);

            firstFrame_ = false;
        }
    }

    return baselinerStats;
}

void HostMultiScaleBaseliner::MultiScaleBaseliner::AddToBaselineStats(const LaneArray& traceData,
                                                                      const LaneArray& baselineSubtractedFrame,
                                                                      Data::BaselinerStatAccumulator<ElementTypeOut>& baselinerStats)
{
    if (!firstFrame_)
    {
        // NOTE: Thresholds below are specified as floats whereas
        // incoming frame data are shorts.

        // Compute the high mask at the plus-1 position (this) for variance
        const auto& maskHp1 = baselineSubtractedFrame < thrHigh_;

        // Compute the full mask to use for the single-frame latent variance
        // Minus-1[High] & Pos-0[Low] & Plus-1[High]
        const auto& mask = latHMask_.front() & latLMask_ & maskHp1;

        // Push the plus-1 frame masks
        latLMask_ = baselineSubtractedFrame < thrLow_;
        latHMask_.push_back(maskHp1);

        baselinerStats.AddSample(latRawData_, latData_, mask);
    }

    // Set latent data.
    latRawData_ = traceData;
    latData_ = baselineSubtractedFrame;
}

HostMultiScaleBaseliner::FloatArray
HostMultiScaleBaseliner::MultiScaleBaseliner::GetSmoothedSigma(const FloatArray& sigma)
{
    // Fixed thresholds for variance computation.
    // TODO - Make these tunable parameters.
    const float sigmaThrL { 4.5 };
    const float sigmaThrH { 4.5 };

    if (firstFrame_)
    {
        // NOTE: We initialize with the first sigma value
        // but a different strategy might be better here i.e
        // taking the average of the first few sigmas before
        // applying the smoothing.
        prevSigma_ = sigma;
    }

    // TODO: Make configurable.
    const FloatArray alphaFactor{0.7f};

    const auto& newSigma = ((FloatArray{1.0f} - alphaFactor) * prevSigma_)
                             + (alphaFactor * sigma);

    // Update thresholds for classifying baseline frames.
    thrLow_ = FloatArray{sigmaThrL} * newSigma;
    thrHigh_ = FloatArray{sigmaThrH} * newSigma;

    prevSigma_ = newSigma;

    return newSigma;
}

}}}      // namespace PacBio::Mongo::Basecaller
