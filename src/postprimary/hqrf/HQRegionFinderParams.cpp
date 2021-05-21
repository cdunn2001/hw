
#include <json/reader.h>

#include <pacbio/text/String.h>
#include <pacbio/logging/Logger.h>
#include <pacbio/PBException.h>

#include "HQRegionFinderParams.h"
#include "SpiderCrfParams.h"
#include "ZOffsetCrfParams.h"
#include "SequelCrfParams.h"

namespace PacBio {
namespace Primary {
namespace Postprimary {

HQRFMethod CoeffLookup(const std::shared_ptr<PpaAlgoConfig>& ppaAlgoConfig)
{
    return GetPrivateHQRFMethod(ppaAlgoConfig->hqrf.method);
}

}}}
