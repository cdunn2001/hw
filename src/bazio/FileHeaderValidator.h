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

# pragma once

#include <json/json.h>

#include <bazio/SimulateConfigs.h>

namespace PacBio {
namespace Primary {

inline bool validateExperimentMetadata(Json::Value metadata)
{
    bool valid = true;
    if (!metadata.isMember("ChipInfo"))
    {
        PBLOG_WARN << "Missing ChipInfo in BAZ header";
        valid = false;
    }
    else if (!metadata["ChipInfo"].isMember("LayoutName"))
    {
        PBLOG_WARN << "Missing ChipInfo/LayoutName in BAZ header";
        valid = false;
    }
    if (!metadata.isMember("DyeSet"))
    {
        PBLOG_WARN << "Missing DyeSet info in BAZ header";
        valid = false;
    }
    else if (!metadata["DyeSet"].isMember("RelativeAmp"))
    {
        PBLOG_WARN << "Missing DyeSet/RelativeAmp (relative amplitudes) "
                      "in BAZ header";
        valid = false;
    }
    return valid;
}

inline bool validateExperimentMetadata(std::string metadata)
{
    Json::Value experimentMetadata;
    bool parsable = Json::Reader{}.parse(metadata, experimentMetadata);
    if (!parsable)
    {
        PBLOG_WARN << "Unparsable EXPERIMENT_METADATA in BAZ header";
        return false;
    }
    bool valid = validateExperimentMetadata(experimentMetadata);
    if (!valid)
        PBLOG_WARN << "Incomplete EXPERIMENT_METADATA in BAZ header";
    return (parsable & valid);
}

inline Json::Value failToDefault(std::string cause)
{
    PBLOG_WARN << cause + " EXPERIMENT_METADATA in BAZ header, default "
                  "Sequel values will be used that are almost "
                  "certainly WRONG";
    Json::Value experimentMetadata;
    // TODO: make this numChannel appropriate:
    Json::Reader{}.parse(generateExperimentMetadata(),
        experimentMetadata);
    return experimentMetadata;
}

inline Json::Value parseExperimentMetadata(std::string metadata)
{
    Json::Value experimentMetadata;

    if (metadata == "") return failToDefault("Missing");

    bool parseable = Json::Reader{}.parse(metadata, experimentMetadata);
    if (!parseable) return failToDefault("Unparsable");

    bool valid = validateExperimentMetadata(experimentMetadata);
    if (!valid) return failToDefault("Incomplete");

    return experimentMetadata;
}


}}
