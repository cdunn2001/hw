// Copyright (c) 2021, Pacific Biosciences of California, Inc.
//
// All rights reserved.
//
// THIS SOFTWARE CONSTITUTES AND EMBODIES PACIFIC BIOSCIENCES' CONFIDENTIAL
// AND PROPRIETARY INFORMATION.
//
// Disclosure, redistribution and use of this software is subject to the
// terms and conditions of the applicable written agreement(s) between you
// and Pacific Biosciences, where "you" refers to you or your company or
// organization, as applicable.  Any other disclosure, redistribution or
// use is prohibited.
//
// THIS SOFTWARE IS PROVIDED BY PACIFIC BIOSCIENCES AND ITS CONTRIBUTORS "AS
// IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
// THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL PACIFIC BIOSCIENCES OR ITS
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
// OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
// WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
// OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
// ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// File Description:
///  \brief Definitions of the pa-ws application configuration tree
//
// Programmer: Mark Lakata

#ifndef PA_WS_CONFIG_H
#define PA_WS_CONFIG_H

#include <stdint.h>
#include <stdlib.h>

#include <array>

// pa-common includes
#include <pacbio/configuration/PBConfig.h>
#include <pacbio/sensor/Platform.h>

// local includes
#include "PaWsConstants.h"

namespace PacBio {
namespace Primary {
namespace PaWs {

using PacBio::Configuration::DefaultFunc;

struct PaWsDebugConfig : PacBio::Configuration::PBConfig<PaWsDebugConfig>
{
    PB_CONFIG(PaWsDebugConfig);

    // 0 = no simulation
    // 1 = use "mockup" fake JSON data to test REST API
    // 2 = run apps in simulation mode
    PB_CONFIG_PARAM(uint32_t, simulationLevel, 0);
};


struct PaWsConfig : PacBio::Configuration::PBConfig<PaWsConfig>
{
    PB_CONFIG(PaWsConfig);

    /// This is a convenient ctor for use in unit tests where one wants to hardcode the platform type
    /// using strict enum types, rather than feed in a JSON object.
    PaWsConfig(PacBio::Sensor::Platform p) 
       : PaWsConfig( [p](){ 
           Json::Value x;
           x["platform"]=p.toString();
           return x;
         }() ) 
    {}
    PaWsConfig(PacBio::Sensor::Platform::RawEnum p) 
        : PaWsConfig(PacBio::Sensor::Platform(p))
    {}  

    PB_CONFIG_PARAM(PacBio::Sensor::Platform, platform, PacBio::Sensor::Platform::UNKNOWN);

    PB_CONFIG_PARAM(uint32_t, numSRAs, 0); ///< Number of threads to launch to service the HT Units representing SRAs
    PB_CONFIG_PARAM(double, pollingInternalSeconds, 1.0); ///< Interval between housekeeping events
    PB_CONFIG_PARAM(bool, logHttpGets, false); ///< Whether or not incoming http GETs are logged (could be very noisy)
    PB_CONFIG_PARAM(uint16_t, port, PORT_REST_PAWS); ///< The dedicated REST port.
    PB_CONFIG_PARAM(uint32_t, numThreads, 5); ///< Number http threads

    PB_CONFIG_OBJECT(PaWsDebugConfig, debug);
};

/// A general method for overwriting members of the PaWsConfig struct
/// with the defaults specific to the Platform.
/// \param config An in/out pointer whose ->platform member will be used to
/// call one of the platform specific initializers, such as Spider2Config,
/// KestrelConfig, etc.
void FactoryConfig(PaWsConfig* config);

/// \param config An out pointer to a PB Config struct that will be 
/// written with default values suitable for Spider, aka Sequel2
void Spider2Config(PaWsConfig* config);

/// \param config An out pointer to a PB Config struct that will be 
/// written with default values suitable for Mongo, aka four parallel Sequel2
/// instances.
void MongoConfig(PaWsConfig* config);

/// \param config An out pointer to a PB Config struct that will be 
/// written with default values suitable for Kestrel.
void KestrelConfig(PaWsConfig* config);


}}} // namespace


#endif // WXDAEMON_CONFIG_H
