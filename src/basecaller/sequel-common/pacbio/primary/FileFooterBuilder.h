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

#include <map>
#include <vector>

#include <json/json.h>
#include <json/reader.h>

namespace PacBio {
namespace Primary {

/// Provides methods to create a BAZ FileFooter and export it to JSON.
class FileFooterBuilder
{
public:
    FileFooterBuilder() = default;
    // Move constructor
    FileFooterBuilder(FileFooterBuilder&&) = default;
    // Copy constructor
    FileFooterBuilder(const FileFooterBuilder&) = default;
    // Move assignment operator
    FileFooterBuilder& operator=(FileFooterBuilder&&) = default;
    // Copy assignment operator
    FileFooterBuilder& operator=(const FileFooterBuilder&) = delete;
    // Destructor
    ~FileFooterBuilder() = default;

public: 
    void AddTruncation(uint32_t zmwId, uint32_t superchunkId)
    { 
        if (truncationMap_.find(zmwId) == truncationMap_.cend())
            truncationMap_.insert(std::make_pair(zmwId, std::vector<uint32_t>())); 
            
        truncationMap_.at(zmwId).push_back(superchunkId); 
    }

    std::string CreateJSON()
    {
        Json::Value file;
        file["TYPE"] = "BAZ";
        Json::Value& footer = file["FOOTER"];

        Json::Value& truncation = footer["TRUNCATION_MAP"];
        for (const auto& kv : truncationMap_)
        {
            Json::Value field;
            field.append(kv.first); // zmwId
            Json::Value array;
            for (const auto& v : kv.second) // superchunkIds
                array.append(v); 
            field.append(array);
            truncation.append(field);
        }

        std::stringstream ss;
        ss << file;
        return ss.str();
    }

    std::vector<char> CreateJSONCharVector()
    {
        std::string jsonStream = CreateJSON();
        return std::vector<char>(jsonStream.begin(), jsonStream.end());
    }

private:
    std::map<uint32_t, std::vector<uint32_t>> truncationMap_;
};

}}
