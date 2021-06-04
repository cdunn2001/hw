#ifndef KES_CAL_CONFIG_H
#define KES_CAL_CONFIG_H

#include <stdint.h>
#include <stdlib.h>

#include <array>

// pa-common includes
#include <pacbio/configuration/PBConfig.h>
#include <pacbio/sensor/Platform.h>

// local includes

namespace PacBio {
namespace Primary {
namespace Calibration {

using PacBio::Configuration::DefaultFunc;

struct PaCalConfig : PacBio::Configuration::PBConfig<PaCalConfig>
{
    PB_CONFIG(PaCalConfig);

    /// This is a convenient ctor for use in unit tests where one wants to hardcode the platform type
    /// using strict enum types, rather than feed in a JSON object.
    PaCalConfig(PacBio::Sensor::Platform p) 
       : PaCalConfig( [p](){ 
           Json::Value x;
           x["platform"]=p.toString();
           return x;
         }() ) 
    {}
    PaCalConfig(PacBio::Sensor::Platform::RawEnum p) 
        : PaCalConfig(PacBio::Sensor::Platform(p))
    {}  

    PB_CONFIG_PARAM(PacBio::Sensor::Platform, platform, PacBio::Sensor::Platform::UNKNOWN);

    PB_CONFIG_PARAM(double, pollingInternalSeconds, 1.0); ///< Interval between housekeeping events
};

void FactoryConfig(PaCalConfig* c);
void Spider2Config(PaCalConfig* c);
void MongoConfig(PaCalConfig* c);
void KestrelConfig(PaCalConfig* c);


}}} // namespace


#endif // WXDAEMON_CONFIG_H
