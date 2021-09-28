#include "Baseliner.h"

#include <dataTypes/configs/BasecallerBaselinerConfig.h>
#include <dataTypes/configs/MovieConfig.h>

namespace PacBio {
namespace Mongo {
namespace Basecaller {

std::unique_ptr<Data::CameraBatchFactory> Baseliner::batchFactory_;
float Baseliner::movieScaler_ = 1.0f;

// static
void Baseliner::Configure(const Data::BasecallerBaselinerConfig&,
                          const Data::MovieConfig& movConfig)
{
    const auto hostExecution = true;
    InitFactory(hostExecution, movConfig.photoelectronSensitivity);
}

void Baseliner::InitFactory(bool hostExecution, float movieScaler)
{
    using Cuda::Memory::SyncDirection;

    SyncDirection syncDir = hostExecution ? SyncDirection::HostWriteDeviceRead : SyncDirection::HostReadDeviceWrite;
    batchFactory_ = std::make_unique<Data::CameraBatchFactory>(syncDir);

    movieScaler_ = movieScaler;
}

void Baseliner::Finalize() {}

}}}     // namespace PacBio::Mongo::Basecaller
