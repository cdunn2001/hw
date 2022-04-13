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

#ifndef mongo_dataTypes_configs_traceSaverConfig_H
#define mongo_dataTypes_configs_traceSaverConfig_H

#include <array>

#include <pacbio/configuration/PBConfig.h>

namespace PacBio {
namespace Mongo {
namespace Data {

struct TraceSaverConfig  : public Configuration::PBConfig<TraceSaverConfig>
{
    PB_CONFIG(TraceSaverConfig);

    PB_CONFIG_PARAM(std::vector<std::vector<int>>, roi, std::vector<std::vector<int>>());

    // "Natural" means just match whatever data type the rest
    // of the pipeline is configured to produce.
    SMART_ENUM(OutFormat, Natural, INT16, UINT8);
    PB_CONFIG_PARAM(OutFormat, outFormat, OutFormat::Natural);

    // Defines the chunk size that the HDF5 library will use for the trace file.
    // This is the granularity at which the library will write to disk, regardless
    // of what size data is handed to the public read/write routines.  This means that
    // if chunking is small then throughput will suffer because even large swaths
    // of data will be broken up into small individual writes.
    //
    // That said, the default cache size for HDF5 is 1MiB.  If the chunk size is
    // larger than the cache size, then presumably we can't keep even a single
    // chunk around between write operations, which means that if we try to provide
    // HDF5 with buffers to write that are smaller than a chunk, we'll have a lot
    // of unecessary cache thrashing and disk operations.
    //
    // These parameters were chosen to match the default cache size.  This lets
    // us perpetually keep one active chunk in cache, which well suits our
    // streaming write patterns.
    // TODO: If we were to update our own HDF5 wrappers to allow configuration
    //       of the cache size then there is room for marginal improvements via larger
    //       writes, but there is no time for that at the moment
    PB_CONFIG_PARAM(uint32_t, frameChunking,
                    Configuration::DefaultFunc([](uint32_t framesPerChunk)
                                               {
                                                   return framesPerChunk;
                                               },
                                               {"layout.framesPerChunk"}));
    PB_CONFIG_PARAM(uint32_t, zmwChunking,
                    Configuration::DefaultFunc([](uint32_t framesPerChunk)
                                               {
                                                   return (1 << 20) / framesPerChunk;
                                               },
                                               {"layout.framesPerChunk"}));
};

}}}     // namespace PacBio::Mongo::Data

#endif //mongo_dataTypes_configs_traceSaverConfig_H
