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
        auto frameData = rawTrace.GetBlockView(laneIdx);
        auto baselinerStats = Data::BaselinerStatAccumulator<Data::BaselinedTraceElement>{};

        // Run lower filter.
        auto lowBuf = lowerBuffer.GetBlockView(laneIdx);
        std::memcpy(lowBuf.Data(), frameData.Data(), frameData.Size()*sizeof(ElementTypeIn));
        auto lower = msLowerOpen_(&lowBuf);

        // Run upper filter.
        auto upperBuf = upperBuffer.GetBlockView(laneIdx);
        std::memcpy(upperBuf.Data(), frameData.Data(), frameData.Size()*sizeof(ElementTypeIn));
        auto upper = msUpperOpen_(&upperBuf);

        // Compute baseline, results should be strided-out in the lower and upper block views.
        auto baselineEst = out.GetBlockView(laneIdx);
        auto blIter = baselineEst.begin();
        auto lowerIter = lower->cbegin();
        auto upperIter = upper->cbegin();
        while (lowerIter != lower->cend() || upperIter != upper->cend())
        {
            *blIter = (*upperIter + *lowerIter) / LaneArray{2};
            blIter += Stride();
            lowerIter += Stride();
            upperIter += Stride();
        }

        // TODO: Perform baseline subtraction and baseline stats.

        out.Stats(laneIdx) = baselinerStats.ToBaselineStats();
    }

    return std::move(out);
}

HostMultiScaleBaseliner::~HostMultiScaleBaseliner() = default;

}}}      // namespace PacBio::Mongo::Basecaller