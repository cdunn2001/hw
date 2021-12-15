// Copyright (c) 2016, Pacific Biosciences of California, Inc.
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

#include <map>
#include <string>
#include <stdexcept>
#include <vector>

#include <json/json.h>

namespace PacBio {
namespace Primary {

/// Stores all BAZ file footer information and provides logic 
/// to parse a JSON footer.
class FileFooter
{
public:
    FileFooter() = default;

    FileFooter(char* header, const size_t length)
    { Init(header, length); }
    // Move constructor
    FileFooter(FileFooter&&) = default;
    // Copy constructor
    FileFooter(const FileFooter&) = delete;
    // Move assignment operator
    FileFooter& operator=(FileFooter&&) = default;
    // Copy assignment operator
    FileFooter& operator=(const FileFooter&) = delete;
    // Destructor
    ~FileFooter() = default;

public:
    bool IsZmwTruncated(uint32_t zmwId) const
    { return truncationMap_.find(zmwId) != truncationMap_.cend(); }

    const std::vector<uint32_t>& TruncatedSuperchunks(uint32_t zmwId) const
    { return truncationMap_.at(zmwId); }

    const std::map<uint32_t, std::vector<uint32_t>>& TruncationMap() const
    { return truncationMap_; }

private:
    std::map<uint32_t, std::vector<uint32_t>> truncationMap_;

private:
    void Init(char* header, const size_t length)
    {
        // 'rootValue' will contain the root value after parsing.
        Json::Value rootValue; 
        // Possible parsing error
        std::string jsonError;

        const char* end = header + length;
        // Parse json from char string
        Json::CharReaderBuilder builder;
        builder.settings_["collectComments"] = false;
        auto x = std::unique_ptr<Json::CharReader>(builder.newCharReader());
        x->parse(header, end, &rootValue, &jsonError);

        if (!rootValue.isMember("TYPE"))        
            throw std::runtime_error("Missing TYPE!");
        if (rootValue["TYPE"].asString().compare("BAZ") != 0)
            throw std::runtime_error("JSON TYPE is not BAZ! " + rootValue["TYPE"].asString());

        Json::Value& footerValue = rootValue["FOOTER"];

        if (footerValue.isMember("TRUNCATION_MAP") 
            && !footerValue["TRUNCATION_MAP"].isNull())
        {
            Json::Value& truncationMap = footerValue["TRUNCATION_MAP"];
            for (size_t i = 0; i < truncationMap.size(); ++i)
            {
                Json::Value& pair = truncationMap[static_cast<int>(i)];
                const uint32_t zmwId = pair[0].asUInt();

                std::vector<uint32_t> superchunks;
                for (size_t j = 0; j < pair[1].size(); ++j)
                    superchunks.push_back(pair[1][static_cast<int>(j)].asUInt());

                truncationMap_.insert(std::make_pair(zmwId, superchunks));
            }
        }        
    }
};

}}
