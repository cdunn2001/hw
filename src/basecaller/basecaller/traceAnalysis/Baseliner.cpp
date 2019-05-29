#include "Baseliner.h"

#include <dataTypes/BasecallerConfig.h>
#include <dataTypes/MovieConfig.h>

namespace PacBio {
namespace Mongo {
namespace Basecaller {

// static
void Baseliner::Configure(const Data::BasecallerBaselinerConfig& baselinerConfig,
                          const Data::MovieConfig& movConfig)
{
    // TODO
}

Baseliner::Baseliner(uint32_t poolId)
    : poolId_ (poolId)
{

}

Data::CameraTraceBatch
Baseliner::process(Data::TraceBatch<ElementTypeIn> rawTrace)
{
    // TODO
    return Data::CameraTraceBatch(std::move(rawTrace));
}

}}}     // namespace PacBio::Mongo::Basecaller
