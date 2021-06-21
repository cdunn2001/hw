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
///  \brief Defines the calibration parameters for the pa-cal application


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
    explicit PaCalConfig(PacBio::Sensor::Platform p) 
       : PaCalConfig( [p](){ 
           Json::Value x;
           x["platform"]=p.toString();
           return x;
         }() ) 
    {}
    explicit PaCalConfig(PacBio::Sensor::Platform::RawEnum p) 
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
