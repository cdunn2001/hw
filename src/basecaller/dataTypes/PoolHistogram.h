#ifndef mongo_dataTypes_PoolHistogram_H_
#define mongo_dataTypes_PoolHistogram_H_

#include <common/MongoConstants.h>
#include <common/cuda/utility/CudaArray.h>
#include <common/cuda/memory/UnifiedCudaArray.h>

namespace PacBio {
namespace Mongo {
namespace Data {

/// A pure data type that represents a histogram for each ZMW in a lane.
/// For a particular ZMW, the bin size is constant.
/// The number of bins is the same for each ZMW.
template <typename DataT, typename CountT>
struct LaneHistogram
{
    template <typename T>
    using Array = Cuda::Utility::CudaArray<T, laneSize>;

    using DataType = DataT;
    using CountType = CountT;

    // This constant must be large enough to accomodate high SNR data.
    // Ideally, it would be a function of BinSizeCoeff and the SNR and excess-
    // noise CV of the brightest analog.
    // nBins(snr, bsc, xsn) = (snr + 4(1 + sqrt(1 + snr + xsn*snr^2))) / bsc,
    // where bsc = binSizeCoeff, and xsn = excess noise CV.
    // In practice, the value will probably be determined somewhat empirically.
    static constexpr unsigned int numBins = 300;

    /// The lower bound of the lowest bin.
    Array<DataT> lowBound;

    /// The size of all bins for each ZMW.
    Array<DataT> binSize;

    /// The number of data less than lowBound.
    Array<CountT> outlierCountLow;

    /// The number of data >= the high bound = lowBound + numBins*binSize.
    Array<CountT> outlierCountHigh;

    /// The number of data in each bin.
    Cuda::Utility::CudaArray<Array<CountT>, numBins> binCount;
};


/// A collection of trace histograms for a pool of ZMWs.
template <typename DataT, typename CountT>
struct PoolHistogram
{
    static constexpr auto cudaSyncMode = Cuda::Memory::SyncDirection::Symmetric;

    // TODO: Where will this come from?
    static std::shared_ptr<Cuda::Memory::DualAllocationPools> poolHistPool;

    Cuda::Memory::UnifiedCudaArray<LaneHistogram<DataT, CountT>> data;
    uint32_t poolId;

    PoolHistogram(uint32_t aPoolId, unsigned int numLanes, bool pinnedAlloc)
        : data (numLanes, cudaSyncMode, pinnedAlloc)
        , poolId (aPoolId)
    { }
};

template <typename DataT, typename CountT>
std::shared_ptr<Cuda::Memory::DualAllocationPools>
PoolHistogram<DataT, CountT>::poolHistPool;

}}}     // namespace PacBio::Mongo::Data

#endif // mongo_dataTypes_PoolHistogram_H_
