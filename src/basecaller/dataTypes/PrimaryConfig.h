#ifndef mongo_dataTypes_PrimaryConfig_H_
#define mongo_dataTypes_PrimaryConfig_H_

// Copyright (c) 2016-2019, Pacific Biosciences of California, Inc.
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
//  Description:
/// \brief  Global configuration for the Primary realtime pipeline. These values
///         may be changed at run time.

#include <pacbio/utilities/Finally.h>
#include <pacbio/ipc/PoolFactory.h>
#include <pacbio/process/ConfigurationBase.h>
#include <pacbio/utilities/CpuInfo.h>

//#include <pacbio/primary/SequelDefinitions.h>
//#include <pacbio/primary/ipc_config.h>
//#include <pacbio/primary/ChipClass.h>

namespace PacBio {
namespace Mongo {
namespace Data {

class PrimaryConfig :  public PacBio::Process::ConfigurationObject
{
    ADD_PARAMETER(uint32_t, zmwsPerLane, 64);
    ADD_PARAMETER(uint32_t, lanesPerPool, 4096);
    ADD_PARAMETER(uint32_t, framesPerChunk, 128);

    // Rate at which frames are read from the sensor.
    ADD_PARAMETER(double, sensorFrameRate, 100.0f);

    // Maxmimum polymerization rate supported.
    ADD_PARAMETER(float, maxPolRate, 1.5f);

    // Metrics/BazWriter
    ADD_PARAMETER(uint32_t, framesPerHFMetricBlock, 4096);
    ADD_PARAMETER(bool, realtimeActivityLabels, true);

public:
    void StreamOut(std::ostream& os)
    {
        os << "    zmwsPerLane = " << zmwsPerLane << '\n'
           << "    lanesPerPool = " << lanesPerPool << '\n'
           << "    framesPerChunk = " << framesPerChunk << '\n'
           << "    sensorFrameRate = " << sensorFrameRate << '\n'
           << "    maxPolRate, = " << maxPolRate << '\n'
           << std::flush;
    }
};


inline std::ostream& operator<<(std::ostream& o, PrimaryConfig& config)
{
    config.StreamOut(o);
    return o;
}

inline PrimaryConfig& GetPrimaryConfig()
{
   static PrimaryConfig config;
   return config;
}

#define primaryConfig GetPrimaryConfig()

}}}     // namespace PacBio::Mongo::Data

#endif //mongo_dataTypes_PrimaryConfig_H_
