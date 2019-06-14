#include "HostMultiScaleBaseliner.h"

#include <dataTypes/BasicTypes.h>
#include <dataTypes/BaselinerStatAccumulator.h>

namespace PacBio {
namespace Mongo {
namespace Basecaller {

void HostMultiScaleBaseliner::Configure(const Data::BasecallerBaselinerConfig &baselinerConfig,
                                  const Data::MovieConfig &movConfig)
{
    const auto hostExecution = false;
    Baseliner::InitAllocationPools(hostExecution);
}

void HostMultiScaleBaseliner::Finalize()
{
    Baseliner::DestroyAllocationPools();
}

Data::CameraTraceBatch HostMultiScaleBaseliner::Process(Data::TraceBatch <ElementTypeIn> rawTrace)
{
    auto out = batchFactory_->NewBatch(rawTrace.GetMeta(), rawTrace.Dimensions());
    auto pools = rawTrace.GetAllocationPools();

    // TODO: We don't need to allocate these large buffers, we only need 2 BlockView<T> buffers which can be reused.
    Data::BatchData<ElementTypeIn> lowerBuffer(rawTrace.Dimensions(),
                                               Cuda::Memory::SyncDirection::HostReadDeviceWrite, pools, true);
    Data::BatchData<ElementTypeIn> upperBuffer(rawTrace.Dimensions(),
                                               Cuda::Memory::SyncDirection::HostReadDeviceWrite, pools, true);

    for (size_t laneIdx = 0; laneIdx < rawTrace.LanesPerBatch(); ++laneIdx)
    {
        auto traceData = rawTrace.GetBlockView(laneIdx);
        auto baseline = out.GetBlockView(laneIdx);
        auto& baseliner = baselinerByLane_[laneIdx];

        auto baselinerStats = baseliner.EstimateBaseline(traceData,
                                                         lowerBuffer.GetBlockView(laneIdx),
                                                         upperBuffer.GetBlockView(laneIdx),
                                                         baseline);

        out.Stats(laneIdx) = baselinerStats.ToBaselineStats();
    }

    return std::move(out);
}

Data::BaselinerStatAccumulator<HostMultiScaleBaseliner::ElementTypeOut>
HostMultiScaleBaseliner::MultiScaleBaseliner::EstimateBaseline(const Data::BlockView<ElementTypeIn>& traceData,
                                                               Data::BlockView<ElementTypeIn> lowerBuffer,
                                                               Data::BlockView<ElementTypeIn> upperBuffer,
                                                               Data::BlockView<ElementTypeOut> baselineEst)
{
    // Run lower filter, results are strided out.
    std::memcpy(lowerBuffer.Data(), traceData.Data(), traceData.Size()*sizeof(ElementTypeIn));
    auto lower = msLowerOpen_(&lowerBuffer);

    // Run upper filter, results are strided out.
    std::memcpy(upperBuffer.Data(), traceData.Data(), traceData.Size()*sizeof(ElementTypeIn));
    auto upper = msUpperOpen_(&upperBuffer);

    // Compute and subtract baseline while tabulating the stats.
    auto baselinerStats = Data::BaselinerStatAccumulator<ElementTypeOut>{};
    auto trIter = traceData.cbegin();
    auto blIter = baselineEst.begin();
    auto lowerIter = lower->cbegin();
    auto upperIter = upper->cbegin();
    for ( ; blIter != baselineEst.end() && lowerIter != lower->end() && upperIter != upper->end();
            blIter += Stride(), lowerIter += Stride(), upperIter += Stride())
    {
        // NOTE: We are storing the results as int16_t so this division will bias the results.
        auto bias = (*upperIter + *lowerIter) / LaneArray{2};
        auto framebkgndSigma = (*upperIter - *lowerIter).AsFloat() / cSigmaBias_;
        auto smoothedBkgndSigma = GetSmoothedSigma(framebkgndSigma * FloatArray{Scale()});
        auto frameBiasEstimate = cMeanBias_ * smoothedBkgndSigma;

        // Estimates are scattered on stride intervals.
        for (auto strideIter = blIter; strideIter != baselineEst.end() && trIter != traceData.cend();
             strideIter++, trIter++)
        {
            auto rawSignal = *trIter;

            // NOTE: We convert the bias estimate to short.
            *strideIter = rawSignal - bias - frameBiasEstimate.AsShort();

            // NOTE: We need to scale the trace data (from DN to e-) and
            // end up converting the baseline subtracted data to float in order
            // to perform the conversion and then end up converting it back.
            *strideIter = ((*strideIter).AsFloat() * FloatArray{Scale()}).AsShort();

            AddToBaselineStats(rawSignal, *strideIter, baselinerStats);

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
        const auto maskHp1 = baselineSubtractedFrame.AsFloat() < thrHigh_;

        // Compute the full mask to use for the single-frame latent variance
        // Minus-1[High] & Pos-0[Low] & Plus-1[High]
        const Mask mask = latHMask_.front() & latLMask_ & maskHp1;

        // Push the plus-1 frame masks
        latLMask_ = baselineSubtractedFrame.AsFloat() < thrLow_;
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

    FloatArray newSigma = ((FloatArray{1.0f} - alphaFactor) * prevSigma_)
                          + (alphaFactor * sigma);

    // Update thresholds for classifying baseline frames.
    thrLow_ = FloatArray{sigmaThrL} * newSigma;
    thrHigh_ = FloatArray{sigmaThrH} * newSigma;

    prevSigma_ = sigma;

    return newSigma;
}

}}}      // namespace PacBio::Mongo::Basecaller