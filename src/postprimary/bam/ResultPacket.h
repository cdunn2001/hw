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

#include <string>

#include <pbbam/BamRecord.h>
#include <pbbam/TagCollection.h>

#include <bazio/RegionLabelType.h>

#include "SubreadContext.h"

namespace PacBio {
namespace Primary {
namespace Postprimary {

/// Encapsulates a BamRecord subread and necessary information to save it
struct ResultPacket
{
public:
    ResultPacket() = default;

    ResultPacket(ResultPacket&&) = default;
    // Copy constructor
    ResultPacket(const ResultPacket&) = delete;
    // Move assignment operator
    ResultPacket& operator=(ResultPacket&&) = default;
    // Copy assignment operator
    ResultPacket& operator=(const ResultPacket&) = delete;
    // Destructor
    ~ResultPacket() = default;

public:
    bool HasTag(const std::string& key)
    { return tags.find(key) != tags.cend(); }

public:
    void AddTagsToRecord()
    {
        for (const auto& tag : tags)
            bamRecord.Impl().AddTag(tag.first, tag.second);
        tags.clear();
        // bamRecord.Impl().Tags(tags);
    }

    size_t EstimatedBytesUsed() const
    {
        return sizeof(ResultPacket) +
               (sizeof(char) * name.capacity()) +
               (sizeof(char) * bases.capacity()) +
               (sizeof(char) * overallQv.capacity()) +
               bamRecord.EstimatedBytesUsed() +
               tags.EstimatedBytesUsed();
    }

public:
    std::string        name;
    std::string        bases;
    std::string        overallQv;
    size_t             length = 0;
    int                zmwNum;
    int                startPos;
    int                endPos;
    int                cx;
    int                cxBitsToPreserve = 0;
    BAM::BamRecord     bamRecord;
    RegionLabelType    label = RegionLabelType::NOT_SET;
    SubreadContext     context;
    bool               control = false;
    int                cxTagIdx = -1;
    bool               skipExistingCxTag = false;
    BAM::TagCollection tags;
    bool               rejected = false;
};

}}} // ::PacBio::Primary::Postprimary


