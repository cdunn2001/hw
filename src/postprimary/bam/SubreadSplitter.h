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

#include <unordered_map>
#include <pacbio/logging/Logger.h>

#include <bazio/PacketFieldName.h>
#include <bazio/RegionLabelType.h>

#include <postprimary/stats/ProductivityMetrics.h>

#include "EventData.h"
#include "ResultPacket.h"
#include "SubreadContext.h"

namespace PacBio {
namespace Primary {
namespace Postprimary {

/// Given a StitchedZmw and boundaries, create a subread BamRecord encapsulated
/// in a ResultPacket.
class SubreadSplitter
{
public: // non-modifying methods
    /// Converts a part of a ZMW, given the SubreadContext, to a ResultPacket
    static ResultPacket PartialReadToResultPacket(
        const ProductivityInfo& pinfo,
        const EventData& events,
        SubreadContext&& lc,
        bool nobam,
        const std::string& movieName,
        const std::unordered_map<size_t, size_t>& base2frame)
    {
        // const SubreadContext local = std::move(lc);
        return PartialReadToResultPacket(pinfo, events, lc, nobam, movieName, base2frame);
    }

    /// Converts a part of a ZMW, given the SubreadContext, to a ResultPacket
    static ResultPacket PartialReadToResultPacket(
        const ProductivityInfo& pinfo,
        const EventData& events,
        const SubreadContext& lc,
        bool nobam,
        const std::string& movieName,
        const std::unordered_map<size_t, size_t>& base2frame);

    /// Converts a part of a ZMW, given begin, end, and label, to a ResultPacket
    static ResultPacket PartialReadToResultPacket(
        const ProductivityInfo& pinfo,
        const EventData& events,
        const size_t begin, const size_t end,
        const RegionLabelType label,
        bool nobam,
        const std::string& movieName,
        const std::unordered_map<size_t, size_t>& base2frame);

public: // structors
    // Default constructor
    SubreadSplitter() = delete;
    // Move constructor
    SubreadSplitter(SubreadSplitter&&) = delete;
    // Copy constructor
    SubreadSplitter(const SubreadSplitter&) = delete;
    // Move assignment operator
    SubreadSplitter& operator=(SubreadSplitter&&) = delete;
    // Copy assignment operator
    SubreadSplitter& operator=(const SubreadSplitter&) = delete;
    // Destructor
    ~SubreadSplitter() = delete;

};

}}} // ::PacBio::Primary::Postprimary


