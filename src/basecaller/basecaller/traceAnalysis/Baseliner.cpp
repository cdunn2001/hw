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
    InitFactory(hostExecution);
}

void Baseliner::InitFactory(bool hostExecution)
{
    using Cuda::Memory::SyncDirection;

    SyncDirection syncDir = hostExecution ? SyncDirection::HostWriteDeviceRead : SyncDirection::HostReadDeviceWrite;
    batchFactory_ = std::make_unique<Data::CameraBatchFactory>(syncDir);
}

void Baseliner::Finalize() {}

}}}     // namespace PacBio::Mongo::Basecaller
