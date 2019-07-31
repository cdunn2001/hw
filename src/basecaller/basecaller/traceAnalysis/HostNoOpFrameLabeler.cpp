#include "HostNoOpFrameLabeler.h"

namespace PacBio {
namespace Mongo {
namespace Basecaller {


void HostNoOpFrameLabeler::Configure(int lanesPerPool, int framesPerChunk)
{
    const auto hostExecution = true;
    // FIXME: Viterbi lookback of 16 frames which should
    // not be hard-coded and eventually pulled from a single
    // source.
    InitAllocationPools(hostExecution, 16u);
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
    return ret;
}

}}} // PacBio::Mongo::Basecaller




