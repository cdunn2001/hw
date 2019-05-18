#include "Baseliner.h"

#include <dataTypes/BasecallerConfig.h>
#include <dataTypes/MovieConfig.h>

namespace PacBio {
namespace Mongo {
namespace Basecaller {

Baseliner::Baseliner(uint32_t poolId,
                     const Data::BasecallerBaselinerConfig& baselinerConfig,
                     const Data::MovieConfig& movConfig)
    : poolId_ (poolId)
{

}

}}}     // namespace PacBio::Mongo::Basecaller
