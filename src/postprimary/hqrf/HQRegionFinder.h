// Copyright (c) 2015-2018, Pacific Biosciences of California, Inc.
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

#include <memory>
#include <utility>
#include <vector>

#include <bazio/RegionLabel.h>

#include <postprimary/application/UserParameters.h>
#include <postprimary/application/PpaAlgoConfig.h>
#include <postprimary/bam/EventData.h>

namespace PacBio {
namespace Primary {
namespace Postprimary {

// Virtual base class, used to hide some of the complexities of the different
// HQRF modes supported.
class HQRegionFinder
{
public: // structors
    HQRegionFinder() = default;

    // No copy/move
    HQRegionFinder(HQRegionFinder&&) = delete;
    HQRegionFinder(const HQRegionFinder&) = delete;
    HQRegionFinder& operator=(HQRegionFinder&&) = delete;
    HQRegionFinder& operator=(const HQRegionFinder&) = delete;
    
    virtual ~HQRegionFinder() = default;

public: /// Client API

    /// Finds the HQ region and returns a RegionLable for it.
    RegionLabel FindAndAnnotateHQRegion(const BlockLevelMetrics& metrics, const EventData& zmw) const;

private:
    /// Finds the HQR as a range in pulses coordinates.
    virtual std::pair<size_t, size_t> FindHQRegion(const BlockLevelMetrics& metrics, const EventData& zmw) const = 0;
};

// Helper class to return a concrete implementation of HQRegionFinder
std::unique_ptr<HQRegionFinder> HQRegionFinderFactory(
        const UserParameters& user,
        const std::shared_ptr<PpaAlgoConfig>& ppaAlgoConfig,
        double frameRateHz);

}}} // ::PacBio::Primary::Postprimary

