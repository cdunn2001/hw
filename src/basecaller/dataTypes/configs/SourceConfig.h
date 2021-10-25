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
#include <pacbio/configuration/Validation.h>
#include <pacbio/utilities/SmartEnum.h>
#include <pacbio/sensor/Platform.h>

namespace PacBio::Mongo::Data {

struct WX2LayoutConfig  : public Configuration::PBConfig<WX2LayoutConfig>
{
    PB_CONFIG(WX2LayoutConfig);
    PB_CONFIG_PARAM(uint32_t, lanesPerPacket, 2);
    PB_CONFIG_PARAM(uint32_t, framesPerPacket, 512); // PacBio::Mongo::DataSource::Tile::NumFrames);
    PB_CONFIG_PARAM(uint32_t, zmwsPerLane, 32); // PacBio::Mongo::DataSource::Tile::NumPixels);
};

// WARNING
// This is a shadow struct. Also modify WXDataSourceConfig.h and adjust the constructor in std::make_unique<WXDataSource> in SmrtBasecaller.h
// TODO: remove the shadow structs.
struct WX2SourceConfig  : public Configuration::PBConfig<WX2SourceConfig>
{
    PB_CONFIG(WX2SourceConfig);
    PB_CONFIG_PARAM(std::string, dataPath, "Normal"); // FIXME I'm using a string here because it is portable at the moment. Not sure how DataPath_t will be ported.
    PB_CONFIG_PARAM(std::string, simulatedInputFile, ""); // TODO deprecate this
    PB_CONFIG_PARAM(double, simulatedFrameRate, 100.0);  // TODO deprecate this
    PB_CONFIG_PARAM(double, sleepDebug, 0.0);
    PB_CONFIG_PARAM(uint32_t, maxPopLoops, 10);
    PB_CONFIG_PARAM(double, tilePoolFactor, 3.0);  // TODO deprecate this
    PB_CONFIG_OBJECT(WX2LayoutConfig, wxlayout);
};

struct TraceReanalysis : public Configuration::PBConfig<TraceReanalysis>
{
    PB_CONFIG(TraceReanalysis);

    PB_CONFIG_PARAM(std::string, traceFile, "");
};

SMART_ENUM(TraceInputType, Natural, INT16, UINT8);
struct TraceReplication : public Configuration::PBConfig<TraceReplication>
{
    PB_CONFIG(TraceReplication)

    PB_CONFIG_PARAM(std::string, traceFile, "");

    PB_CONFIG_PARAM(size_t, numFrames, 10000);
    PB_CONFIG_PARAM(size_t, numZmwLanes, 131072);
    // Cache the full trace file in memory before producing traces
    PB_CONFIG_PARAM(bool, cache, false);
    // Generates some number of chunks before producing traces.
    // Potentially useful if the trace file is too large to cache,
    // but you still want to have some data to measure throughput
    // without being limited by disk read space
    PB_CONFIG_PARAM(size_t, preloadChunks, 0);
    // The max number of chunks the data source will get ahead of
    // the actual analysis without self throttling
    PB_CONFIG_PARAM(size_t, maxQueueSize, 0);

    PB_CONFIG_PARAM(TraceInputType, inputType, TraceInputType::Natural);
};

}     // namespace PacBio::Mongo::Data

namespace PacBio::Configuration {

template <>
inline void ValidateConfig(const Mongo::Data::TraceReplication& cfg, ValidationResults* results)
{
    if (cfg.traceFile.empty()) results->AddError("Must Specify a trace file");
}

template <>
inline void ValidateConfig(const Mongo::Data::TraceReanalysis& cfg, ValidationResults* results)
{
    if (cfg.traceFile.empty()) results->AddError("Must Specify a trace file");
}

}

#endif //mongo_dataTypes_configs_sourceConfig_H
