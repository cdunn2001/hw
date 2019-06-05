#include "Baseliner.h"

#include <dataTypes/BasecallerConfig.h>
#include <dataTypes/MovieConfig.h>

namespace PacBio {
namespace Mongo {
namespace Basecaller {

std::unique_ptr<Data::CameraBatchFactory> Baseliner::batchFactory_;

// static
void Baseliner::Configure(const Data::BasecallerBaselinerConfig& baselinerConfig,
                          const Data::MovieConfig& movConfig)
{
    const auto hostExecution = true;
    InitAllocationPools(hostExecution);
}

void Baseliner::InitAllocationPools(bool hostExecution)
{
    using Cuda::Memory::SyncDirection;

    const auto framesPerChunk = PacBio::Mongo::Data::GetPrimaryConfig().framesPerChunk;
    const auto lanesPerPool = PacBio::Mongo::Data::GetPrimaryConfig().lanesPerPool;
    SyncDirection syncDir = hostExecution ? SyncDirection::HostWriteDeviceRead : SyncDirection::HostReadDeviceWrite;
    batchFactory_ = std::make_unique<Data::CameraBatchFactory>(framesPerChunk, lanesPerPool, syncDir, true);
}

void Baseliner::DestroyAllocationPools()
{
    batchFactory_.release();
}

void Baseliner::Finalize()
{
    DestroyAllocationPools();
}

Baseliner::Baseliner(uint32_t poolId)
    : poolId_ (poolId)
{

}

Data::CameraTraceBatch
Baseliner::process(Data::TraceBatch<ElementTypeIn> rawTrace)
{
    // TODO
    return batchFactory_->NewBatch(rawTrace.GetMeta(), rawTrace.Dimensions());
}

}}}     // namespace PacBio::Mongo::Basecaller
