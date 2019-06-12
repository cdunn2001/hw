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

    Data::BatchData<ElementTypeIn> lowerBuffer(rawTrace.Dimensions(), Cuda::Memory::SyncDirection::HostReadDeviceWrite, pools, true);
    Data::BatchData<ElementTypeIn> upperBuffer(rawTrace.Dimensions(), Cuda::Memory::SyncDirection::HostReadDeviceWrite, pools, true);

    for (size_t laneIdx = 0; laneIdx < rawTrace.LanesPerBatch(); ++laneIdx)
    {
        auto numEstimates = EstimateBaseline(rawTrace.GetBlockView(laneIdx),
                                             lowerBuffer.GetBlockView(laneIdx),
                                             upperBuffer.GetBlockView(laneIdx),
                                             out.GetBlockView(laneIdx));


        // TODO: Perform baseline subtraction and baseline stats.
        auto baselinerStats = Data::BaselinerStatAccumulator<Data::BaselinedTraceElement>{};

        out.Stats(laneIdx) = baselinerStats.ToBaselineStats();
    }

    return std::move(out);
}

template <typename T>
size_t HostMultiScaleBaseliner::EstimateBaseline(const Data::BlockView<T>& traceData,
                                                 Data::BlockView<T> lowerBuffer,
                                                 Data::BlockView<T> upperBuffer,
                                                 Data::BlockView<T> baselineEst)
{
    // Run lower filter.
    std::memcpy(lowerBuffer.Data(), traceData.Data(), traceData.Size()*sizeof(ElementTypeIn));
    auto lower = msLowerOpen_(&lowerBuffer);

    // Run upper filter.
    std::memcpy(upperBuffer.Data(), traceData.Data(), traceData.Size()*sizeof(ElementTypeIn));
    auto upper = msUpperOpen_(&upperBuffer);

    // Compute baseline, results should be strided-out in the lower and upper block views.
    auto blIter = baselineEst.begin();
    auto lowerIter = lower->cbegin();
    auto upperIter = upper->cbegin();
    size_t nE = 0;
    LaneArray backgroundSum;
    LaneArray upperlowerGapSum;
    while (lowerIter != lower->cend() || upperIter != upper->cend())
    {
        *blIter = (*upperIter + *lowerIter) / LaneArray{2};

        backgroundSum += *blIter;
        upperlowerGapSum += (*upperIter - *lowerIter);

        blIter += Stride();
        lowerIter += Stride();
        upperIter += Stride();
        nE++;
    }

    FloatArray numEstimates{static_cast<float>(nE)};
    FloatArray backgroundMean;
    FloatArray backgroundSigma;
    FloatArray biasEstimate;
    if (nE > 0)
    {
        backgroundMean = backgroundSum.AsFloat() / numEstimates;
        backgroundSigma = upperlowerGapSum.AsFloat() / (cSigmaBias_ * numEstimates);
        biasEstimate = cMeanBias_ * backgroundSigma;
    }

    return nE;
}

HostMultiScaleBaseliner::~HostMultiScaleBaseliner() = default;

}}}      // namespace PacBio::Mongo::Basecaller