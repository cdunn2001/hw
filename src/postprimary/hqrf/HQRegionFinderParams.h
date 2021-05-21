
#pragma once

#include <utility>
#include <vector>

#include <bazio/FileHeader.h>
#include <pacbio/primary/HQRFMethod.h>

#include <postprimary/application/PpaAlgoConfig.h>

#include "HQRegionFinderModels.h"

namespace PacBio {
namespace Primary {
namespace Postprimary {

HQRFMethod CoeffLookup(const std::shared_ptr<PpaAlgoConfig>& ppaAlgoConfig);

}}} // ::PacBio::Primary::Postprimary


