#include "Baseliner.h"

#include <dataTypes/BasecallerConfig.h>
#include <dataTypes/MovieConfig.h>

namespace PacBio {
namespace Mongo {
namespace Basecaller {

std::unique_ptr<Data::CameraBatchFactory> Baseliner::batchFactory_;

// static
void Baseliner::Configure(const Data::BasecallerBaselinerConfig&,
                          const Data::MovieConfig&)
{
    const auto hostExecution = true;
    InitAllocationPools(hostExecution);
}

void Baseliner::InitAllocationPools(bool hostExecution)
{
    using Cuda::Memory::SyncDirection;

    const auto framesPerChunk = Data::GetPrimaryConfig().framesPerChunk;
    const auto lanesPerPool = Data::GetPrimaryConfig().lanesPerPool;
    SyncDirection syncDir = hostExecution ? SyncDirection::HostWriteDeviceRead : SyncDirection::HostReadDeviceWrite;
    batchFactory_ = std::make_unique<Data::CameraBatchFactory>(framesPerChunk, lanesPerPool, syncDir);
}

void Baseliner::DestroyAllocationPools()
{
    batchFactory_.reset();
}

void Baseliner::Finalize()
{
    DestroyAllocationPools();
}

}}}     // namespace PacBio::Mongo::Basecaller
