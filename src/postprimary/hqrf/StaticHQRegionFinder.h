// Copyright (c) 2018, Pacific Biosciences of California, Inc.
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

#include <string>
#include <utility>
#include <vector>

#include "HQRegionFinder.h"

namespace PacBio {
namespace Primary {
namespace Postprimary {


// Implementation of HQRegionFinder, which handles both processing a fakeHQFile
// as well as marking the entire read as HQ.
class StaticHQRegionFinder final : public HQRegionFinder
{
public: // structors
    /// Primary constructor
    /// If fakeHQFile is empty, the entirety of each read will be marked as HQ.
    /// If fakeHQFile is not empty, it must contain an entry for each zmw in
    /// the baz file, else an exception will be thrown.
    explicit StaticHQRegionFinder(const std::string& fakeHQFile);

    // No copy/move
    StaticHQRegionFinder(StaticHQRegionFinder&&) = delete;
    StaticHQRegionFinder(const StaticHQRegionFinder&) = delete;
    StaticHQRegionFinder& operator=(StaticHQRegionFinder&&) = delete;
    StaticHQRegionFinder& operator=(const StaticHQRegionFinder&) = delete;
    
    ~StaticHQRegionFinder() override = default;

    /// Finds the HQR as a range in pulse coordinates.
    std::pair<size_t, size_t> FindHQRegion(
        const BlockLevelMetrics& metrics,
        const EventData& zmw) const override;

private: // Members
    std::vector<std::pair<int,int>> fakeHQRegions_;
};

}}}// ::PacBio::Primary::Postprimary
