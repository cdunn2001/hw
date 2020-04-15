//
// Created by mlakata on 4/14/20.
//

#ifndef PA_MONGO_SOURCECONFIG_H
#define PA_MONGO_SOURCECONFIG_H

#include <array>

#include <pacbio/configuration/PBConfig.h>
#include <pacbio/utilities/SmartEnum.h>

namespace PacBio {
namespace Mongo {
namespace Data {

SMART_ENUM(Source_t,
        UNKNOWN,
        TRACE_FILE,
        WX2,
        OTHER_TBD // could add USB3, FRAME_GRABBER, PCIE
);

struct SourceConfig  : public Configuration::PBConfig<SourceConfig>
{
    PB_CONFIG(SourceConfig);

    PB_CONFIG_PARAM(Source_t,source, Source_t::TRACE_FILE);
};

}}}     // namespace PacBio::Mongo::Data

#endif //PA_MONGO_SOURCECONFIG_H
