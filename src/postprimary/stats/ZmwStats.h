// Copyright (c) 2014-2018, Pacific Biosciences of California, Inc.
//
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted (subject to the limitations in the
// disclaimer below) provided that the following conditions are met:
//
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//
//  * Redistributions in binary form must reproduce the above
//    copyright notice, this list of conditions and the following
//    disclaimer in the documentation and/or other materials provided
//    with the distribution.
//
//  * Neither the name of Pacific Biosciences nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE
// GRANTED BY THIS LICENSE. THIS SOFTWARE IS PROVIDED BY PACIFIC
// BIOSCIENCES AND ITS CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
// OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL PACIFIC BIOSCIENCES OR ITS
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
// USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
// OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
// SUCH DAMAGE.

#pragma once

#include <bazio/file/FileHeader.h>
#include <bazio/RegionLabel.h>
#include <pacbio/primary/ZmwStatsFileData.h>

#include <postprimary/bam/EventData.h>
#include <postprimary/bam/Platform.h>
#include <postprimary/stats/ZmwMetrics.h>

namespace PacBio {
namespace Primary {
namespace Postprimary {

using namespace PacBio::Primary;

class ZmwStats
{
public:
    using FileHeader = BazIO::FileHeader;
public: // stateless
    static void FillPerZmwStats(const RegionLabel& hqRegion,
                                const ZmwMetrics& zmwMetrics,
                                const EventData& events,
                                const BlockLevelMetrics& blockMetrics,
                                const bool control,
                                const bool addDiagnostics,
                                PacBio::Primary::ZmwStats& zmwStats);

public: // tors
    // Default constructor
    ZmwStats() = delete;
    // Move constructor
    ZmwStats(ZmwStats&&) = delete;
    // Copy constructor
    ZmwStats(const ZmwStats&) = delete;
    // Move assignment operator
    ZmwStats& operator=(ZmwStats&& rhs) noexcept = delete;
    // Copy assignment operator
    ZmwStats& operator=(const ZmwStats&) = delete;
};

}}}
