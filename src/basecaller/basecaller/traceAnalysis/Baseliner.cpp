#include "Baseliner.h"

#include <dataTypes/configs/BasecallerBaselinerConfig.h>
#include <dataTypes/configs/MovieConfig.h>

namespace PacBio {
namespace Mongo {
namespace Basecaller {

std::unique_ptr<Data::CameraBatchFactory> Baseliner::batchFactory_;
float Baseliner::movieScaler_ = 1.0f;
int16_t Baseliner::pedestal_ = 0;
DataSource::PacketLayout::EncodingFormat Baseliner::expectedEncoding_ = DataSource::PacketLayout::INT16;

// static
void Baseliner::Configure(const Data::BasecallerBaselinerConfig&,
                          const Data::MovieConfig& movConfig)
{
    const auto hostExecution = true;
    InitFactory(hostExecution, movConfig);
}

void Baseliner::InitFactory(bool hostExecution, const Data::MovieConfig& movConfig)
{
    using Cuda::Memory::SyncDirection;

    SyncDirection syncDir = hostExecution ? SyncDirection::HostWriteDeviceRead : SyncDirection::HostReadDeviceWrite;
    batchFactory_ = std::make_unique<Data::CameraBatchFactory>(syncDir);

    movieScaler_ = movConfig.photoelectronSensitivity;
    pedestal_ = movConfig.pedestal;
    expectedEncoding_ = movConfig.encoding;
}

void Baseliner::Finalize() {}

}}}     // namespace PacBio::Mongo::Basecaller
