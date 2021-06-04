#include "PaCalConfig.h"
#include <assert.h>

namespace PacBio {
namespace Primary {
namespace Calibration {

using namespace PacBio::Sensor;

// static size_t GiB = 0x4000'0000ULL;

void Sequel2DefaultConfig(PaCalConfig* config)
{
    assert(config);
}

void MongoConfig(PaCalConfig* config)
{
    assert(config);
}

void KestrelConfig(PaCalConfig* config)
{
    assert(config);
}

void FactoryConfig(PaCalConfig* config)
{
    assert(config);
    switch(config->platform)
    {
        case Platform::Sequel2Lvl1: 
        case Platform::Sequel2Lvl2: 
            Sequel2DefaultConfig(config); 
            break;
        case Platform::Mongo: 
            MongoConfig(config); 
            break;
        case Platform::Kestrel: 
            KestrelConfig(config); 
            break;
        default:
        PBLOG_WARN << "Can't do a factory reset for platform:" << config->platform.toString();
    }
}

}}} // namespace
