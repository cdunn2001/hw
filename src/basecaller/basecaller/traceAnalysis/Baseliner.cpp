#include "Baseliner.h"

#include <dataTypes/configs/BasecallerBaselinerConfig.h>
#include <dataTypes/configs/AnalysisConfig.h>

namespace PacBio {
namespace Mongo {
namespace Basecaller {

std::unique_ptr<Data::CameraBatchFactory> Baseliner::batchFactory_;
float Baseliner::movieScaler_ = 1.0f;
int16_t Baseliner::pedestal_ = 0;
DataSource::PacketLayout::EncodingFormat Baseliner::expectedEncoding_ = DataSource::PacketLayout::INT16;

// static
void Baseliner::Configure(const Data::BasecallerBaselinerConfig&,
                          const Data::AnalysisConfig& analysisConfig)
{
    const auto hostExecution = true;
    InitFactory(hostExecution, analysisConfig);
}

void Baseliner::InitFactory(bool hostExecution, const Data::AnalysisConfig& analysisConfig)
{
    using Cuda::Memory::SyncDirection;

    SyncDirection syncDir = hostExecution ? SyncDirection::HostWriteDeviceRead : SyncDirection::HostReadDeviceWrite;
    batchFactory_ = std::make_unique<Data::CameraBatchFactory>(syncDir);

    movieScaler_ = analysisConfig.movieInfo.photoelectronSensitivity;
    pedestal_ = analysisConfig.pedestal;
    expectedEncoding_ = analysisConfig.encoding;
}

void Baseliner::Finalize() {}

}}}     // namespace PacBio::Mongo::Basecaller
