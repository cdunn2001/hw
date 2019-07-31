#include "HostNoOpFrameLabeler.h"

namespace PacBio {
namespace Mongo {
namespace Basecaller {


void HostNoOpFrameLabeler::Configure(int lanesPerPool, int framesPerChunk)
{
    const auto hostExecution = true;
    InitAllocationPools(hostExecution, 0);
}

void HostNoOpFrameLabeler::Finalize()
{
    DestroyAllocationPools();
}

HostNoOpFrameLabeler::HostNoOpFrameLabeler(uint32_t poolId)
    : FrameLabeler(poolId)
{}

HostNoOpFrameLabeler::~HostNoOpFrameLabeler() = default;

Data::LabelsBatch
HostNoOpFrameLabeler::Process(Data::CameraTraceBatch trace,
                              const PoolModelParameters& models)
{
    auto ret = batchFactory_->NewBatch(std::move(trace));
    for (size_t laneIdx = 0; laneIdx < ret.LanesPerBatch(); laneIdx++)
    {
        std::memset(ret.GetBlockView(laneIdx).Data(), 0,
                    ret.GetBlockView(laneIdx).Size() * sizeof(Data::LabelsBatch::ElementType));
    }
    return ret;
}

}}} // PacBio::Mongo::Basecaller




