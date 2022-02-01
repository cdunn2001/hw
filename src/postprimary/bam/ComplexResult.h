// Copyright (c) 2014-2016, Pacific Biosciences of California, Inc.
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

// Programmer: Armin Töpfer

#pragma once

#include <numeric>

#include <postprimary/stats/ZmwMetrics.h>

#include "ResultPacket.h"

namespace PacBio {
namespace Primary {
namespace Postprimary {


/// Encapsulates a vector of ResultPackets and stats maps
struct ComplexResult
{
public:
    ComplexResult(std::vector<ResultPacket>&& resultPacketsArg,
                  ZmwMetrics&& metrics)
        : resultPackets(std::forward<std::vector<ResultPacket>>(resultPacketsArg))
        , zmwMetrics(std::move(metrics))
    {
        // Remove subread specific tags from filtered subreads that will
        // be saved in scraps
        for (auto& result : resultPackets)
            if (result.label == RegionLabelType::FILTERED || result.control)
                StripTags(&result);
    }

    ComplexResult(ComplexResult&&) = default;
    // Copy constructor
    ComplexResult(const ComplexResult&) = delete;
    // Move assignment operator
    ComplexResult& operator=(ComplexResult&&) = default;
    // Copy assignment operator
    ComplexResult& operator=(const ComplexResult&) = delete;
    // Destructor
    ~ComplexResult() = default;

public:
    std::vector<ResultPacket> resultPackets;
    ZmwMetrics zmwMetrics;

    size_t EstimatedBytesUsed() const
    {
        return zmwMetrics.EstimatedBytesUsed() +
               std::accumulate(resultPackets.cbegin(), resultPackets.cend(), size_t(0),
                               [](size_t v,
                                  const PacBio::Primary::Postprimary::ResultPacket& r)
                                  { return v + r.EstimatedBytesUsed(); });
    }

    bool control;

private: // modifying methods
    void StripTags(ResultPacket* result)
    {
        if (result->bamRecord.Impl().HasTag("cx"))
            result->bamRecord.Impl().RemoveTag("cx");            
        if (result->bamRecord.Impl().HasTag("bc"))
            result->bamRecord.Impl().RemoveTag("bc");            
        if (result->bamRecord.Impl().HasTag("bq"))
            result->bamRecord.Impl().RemoveTag("bq");  
    }
};

}}} // ::PacBio::Primary::Postprimary

