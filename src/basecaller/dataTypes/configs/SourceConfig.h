// Copyright (c) 2019-2020, Pacific Biosciences of California, Inc.
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

#ifndef mongo_dataTypes_configs_sourceConfig_H
#define mongo_dataTypes_configs_sourceConfig_H

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

struct WX2SourceConfig  : public Configuration::PBConfig<WX2SourceConfig>
{
    PB_CONFIG(WX2SourceConfig);
    PB_CONFIG_PARAM(std::string, dataPath, "Normal"); // FIXME I'm using a string here because it is portable at the moment. Not sure how DataPath_t will be ported.
    PB_CONFIG_PARAM(std::string, platform, "Spider"); // FIXME I'm using a string here because it is portable at the moment. The Platform.h header has not been ported yet.
    PB_CONFIG_PARAM(double, simulatedFrameRate, 100.0);
};

struct SourceConfig  : public Configuration::PBConfig<SourceConfig>
{
    PB_CONFIG(SourceConfig);

    PB_CONFIG_PARAM(Source_t, sourceType, Source_t::TRACE_FILE);
    PB_CONFIG_OBJECT(WX2SourceConfig,wx2SourceConfig);
};

}}}     // namespace PacBio::Mongo::Data

#endif //mongo_dataTypes_configs_sourceConfig_H
