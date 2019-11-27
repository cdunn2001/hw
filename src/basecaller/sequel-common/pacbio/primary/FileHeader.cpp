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

#include <assert.h>
#include <iostream>
#include <memory>
#include <string>
#include <stdexcept>

#include <pacbio/logging/Logger.h>

#include <pacbio/primary/FileHeaderValidator.h>
#include <pacbio/primary/PacketField.h>
#include <pacbio/primary/PacketFieldMap.h>
#include <pacbio/primary/MetricField.h>
#include <pacbio/primary/MetricFieldMap.h>
#include <json/reader.h>
#include <pacbio/primary/SmartMemory.h>

#include <pacbio/primary/FileHeader.h>

namespace PacBio {
namespace Primary {

void FileHeader::Init(const char* header, const size_t length)
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

    Json::Value& headerValue = rootValue["HEADER"];

    if (!headerValue.isMember("BAZ_MAJOR_VERSION"))
        throw std::runtime_error("Missing BAZ_MAJOR_VERSION in BAZ header");
    if (!headerValue.isMember("BAZ_MINOR_VERSION"))
        throw std::runtime_error("Missing BAZ_MINOR_VERSION in BAZ header");
    if (!headerValue.isMember("BAZ_PATCH_VERSION"))
        throw std::runtime_error("Missing BAZ_PATCH_VERSION in BAZ header");
    if (!headerValue.isMember("PACKET"))
        throw std::runtime_error("Missing PACKET in BAZ header");
    if (!headerValue.isMember("SLICE_LENGTH_FRAMES"))
        throw std::runtime_error("Missing SLICE_LENGTH_FRAMES in BAZ header");
    if (!headerValue.isMember("FRAME_RATE_HZ"))
        throw std::runtime_error("Missing FRAME_RATE_HZ in BAZ header");
    if (!headerValue.isMember("BASE_CALLER_VERSION"))
        throw std::runtime_error("Missing BASE_CALLER_VERSION in BAZ header");
    if (!headerValue.isMember("BAZWRITER_VERSION"))
        throw std::runtime_error("Missing BAZWRITER_VERSION in BAZ header");
    if (!headerValue.isMember("MOVIE_NAME"))
        throw std::runtime_error("Missing MOVIE_NAME in BAZ header");
    if (!headerValue.isMember("COMPLETE"))
        throw std::runtime_error("Missing COMPLETE in BAZ header");

    // Store headerValue fields
    bazMajorVersion_ = headerValue["BAZ_MAJOR_VERSION"].asUInt();
    bazMinorVersion_ = headerValue["BAZ_MINOR_VERSION"].asUInt();
    bazPatchVersion_ = headerValue["BAZ_PATCH_VERSION"].asUInt();
    sliceLengthFrames_ = headerValue["SLICE_LENGTH_FRAMES"].asUInt();
    frameRateHz_ = headerValue["FRAME_RATE_HZ"].asDouble();
    basecallerVersion_ = headerValue["BASE_CALLER_VERSION"].asString();
    bazWriterVersion_ = headerValue["BAZWRITER_VERSION"].asString();
    movieName_ = headerValue["MOVIE_NAME"].asString();
    complete_ = headerValue["COMPLETE"].asUInt();

    // Parse and check experiment metadata:
    experimentMetadata_ = parseExperimentMetadata(
            headerValue.get("EXPERIMENT_METADATA", Json::Value{""}).asString());

    chipLayoutName_ = experimentMetadata_["ChipInfo"]["LayoutName"].asString();

    if (!headerValue.isMember("BASECALLER_CONFIG"))
    {
        PBLOG_WARN << "Missing BASECALLER_CONFIG in BAZ header";
        basecallerConfig_ = "{}";
    }
    else
    {
        basecallerConfig_ = headerValue["BASECALLER_CONFIG"].asString();
    }


    if (headerValue.isMember("TRUNCATED"))
        truncated_ = headerValue["TRUNCATED"].asUInt();

    if (!headerValue.isMember("MOVIE_LENGTH_FRAMES"))
    {
        PBLOG_WARN << "Missing MOVIE_LENGTH_FRAMES in BAZ header. Using fallback of 3 hours";
        movieLengthFrames_ = static_cast<uint32_t>(frameRateHz_ * 60 * 60 * 3);
    }
    else
    {
        movieLengthFrames_ = headerValue["MOVIE_LENGTH_FRAMES"].asUInt();
    }

    if (headerValue.isMember("FILE_FOOTER_OFFSET"))
    {
        const auto& ffo = headerValue["FILE_FOOTER_OFFSET"];
        if (!ffo.isNull())
        {
            if (ffo.isString())
                offsetFileFooter_ = 0;
            else if (ffo.isUInt64())
                offsetFileFooter_ = ffo.asUInt64();
        }
    }

    if (headerValue.isMember("NUM_SUPER_CHUNKS"))
    {
        const auto& nsc = headerValue["NUM_SUPER_CHUNKS"];
        if (!nsc.isNull())
        {
            if (nsc.isString())
                numSuperChunks_ = 0;
            else if (nsc.isUInt())
                numSuperChunks_ = nsc.asUInt();
        }
    }

    // Parse packets and store them as vector of structs
    ParsePackets(headerValue, packetFields_, packetByteSize_);

    // Parse metric nodes
    if (headerValue.isMember("HF_METRIC"))
        ParseMetrics(headerValue["HF_METRIC"], hFMetricFields_, hFMetricByteSize_, hFMetricFrames_);

    if (headerValue.isMember("MF_METRIC"))
        ParseMetrics(headerValue["MF_METRIC"], mFMetricFields_, mFMetricByteSize_, mFMetricFrames_);

    if (headerValue.isMember("LF_METRIC"))
        ParseMetrics(headerValue["LF_METRIC"], lFMetricFields_, lFMetricByteSize_, lFMetricFrames_);

    // Check if medium- and low-frequency rate is multiple of high-frequency
    if (hFMetricFrames_ > 0 && mFMetricFrames_ % hFMetricFrames_ != 0)
        throw std::runtime_error("Medium-frequency is not a multiple of high-frequency.");
    if (hFMetricFrames_ > 0 && lFMetricFrames_ % hFMetricFrames_ != 0)
        throw std::runtime_error("Low-frequency is not a multiple of high-frequency.");

    packetByteSize_ /= 8;
    if (hFMetricByteSize_ > 0)
        hFMetricByteSize_ /= 8;
    if (mFMetricByteSize_ > 0)
        mFMetricByteSize_ /= 8;
    if (lFMetricByteSize_ > 0)
        lFMetricByteSize_ /= 8;

    // Compute ratio of hf to mf and hf to lf
    if (hFMetricFrames_ > 0)
    {
        hFbyMFRatio_ = mFMetricFrames_ / hFMetricFrames_;
        hFbyLFRatio_ = lFMetricFrames_ / hFMetricFrames_;
        numHFMBsPerSlice_ = sliceLengthFrames_ / hFMetricFrames_;
    }

    // Parse ZMW ID to NUMBER LUT
    if (headerValue.isMember("ZMW_NUMBER_LUT")
        && headerValue["ZMW_NUMBER_LUT"].isArray())
    {
        Json::Value zmwNumbersJson = headerValue["ZMW_NUMBER_LUT"];
        for (size_t i = 0; i < zmwNumbersJson.size(); ++i)
        {
            Json::Value singleZmw = zmwNumbersJson[static_cast<int>(i)];
            uint32_t start = std::stoul(singleZmw[0].asString(), nullptr, 16);
            uint32_t runLength = singleZmw[1].asUInt();
            for (uint32_t j = 0; j < runLength; ++j)
            {
                zmwIdToNumber_.emplace_back(start + j);
                zmwNumbersToId_.insert(std::make_pair(start + j, zmwIdToNumber_.size() - 1));
            }
        }
    }

    // Parse masked ZMW NUMBERs
    if (headerValue.isMember("ZMW_NUMBER_REJECTS_LUT")
        && headerValue["ZMW_NUMBER_REJECTS_LUT"].isArray())
    {
        Json::Value zmwNumberMaskJson = headerValue["ZMW_NUMBER_REJECTS_LUT"];
        for (size_t i = 0; i < zmwNumberMaskJson.size(); ++i)
        {
            Json::Value singleZmw = zmwNumberMaskJson[static_cast<int>(i)];
            uint32_t start = std::stoul(singleZmw[0].asString(), nullptr, 16);
            uint32_t runLength = singleZmw[1].asUInt();
            for (uint32_t j = 0; j < runLength; ++j)
                zmwNumberRejects_.emplace_back(start + j);
        }
    }

    // Parse masked ZMW NUMBERs
    if (headerValue.isMember("ZMW_UNIT_FEATURES_LUT")
        && headerValue["ZMW_UNIT_FEATURES_LUT"].isArray())
    {
        Json::Value zmwUnitFeaturesJson = headerValue["ZMW_UNIT_FEATURES_LUT"];
        for (size_t i = 0; i < zmwUnitFeaturesJson.size(); ++i)
        {
            Json::Value singleZmw = zmwUnitFeaturesJson[static_cast<int>(i)];
            uint32_t code = std::stoul(singleZmw[0].asString(), nullptr, 10);
            uint32_t runLength = singleZmw[1].asUInt();
            for (uint32_t j = 0; j < runLength; ++j)
                zmwUnitFeatures_.emplace_back(code);
        }
    }
    // std::cout << "\nPretty print:\n" << rootValue.toStyledString() << std::endl;
}

void FileHeader::ParseMetrics(Json::Value& metricNode,
                              std::vector<MetricField>& metricFields,
                              uint32_t& metricByteSize,
                              uint32_t& metricFrames)
{
    if (metricNode.empty()) return;

    if (metricNode.isMember("FRAMES"))
        metricFrames = metricNode["FRAMES"].asUInt();

    if (metricNode.isMember("FIELDS"))
    {
        Json::Value metricArray = metricNode["FIELDS"];
        if (!metricArray.isArray()) return;

        // Iterate over all METRIC fields
        for (unsigned int i = 0; i < metricArray.size(); ++i)
        {
            // Get current METRIC field
            const auto metricElemt = metricArray[i];
            MetricField field;

            // Name
            try
            {
                field.fieldName = MetricFieldName::fromString(metricElemt[0].asString());
            }
            catch (const MetricFieldName::Exception&)
            {
                field.fieldName = MetricFieldName::GAP;
            }

            // Size in bits
            field.fieldBitSize = static_cast<uint8_t>(metricElemt[1].asUInt());
            metricByteSize += field.fieldBitSize;

            // Signed or not
            field.fieldSigned = metricElemt[2].asBool();

            // ScalingFactor
            field.fieldScalingFactor = static_cast<uint16_t>(metricElemt[3].asUInt());

            // Print to console
            // field.Print();

            metricFields.push_back(std::move(field));
        }
    }
}

void FileHeader::ParsePackets(Json::Value& root,
                              std::vector<PacketField>& packetFields,
                              uint32_t& packetByteSize)
{
    // Get PACKET node that contains the individual fields
    const auto packetArray = root["PACKET"];

    // Are fields still smaller than a byte?
    // Once a full or a multiple of a byte has been defined,
    // sub byte definitions are not allowed anymore
    bool subByteFields = true;

    // Used to compute the bit-mask and bit-shift
    int8_t currentBit = 7;

    // Iterate over all PACKET fields
    for (unsigned int i = 0; i < packetArray.size(); ++i)
    {
        // Get current PACKET field
        const auto packetElemt = packetArray[i];
        PacketField field;

        // Name
        try
        {
            if (packetElemt[0].asString() == "")
            {
                field.fieldName = PacketFieldName::GAP;
            }
            else
            {
                field.fieldName = PacketFieldName::fromString(packetElemt[0].asString());
            }
        }
        catch (const PacketFieldName::Exception&)
        {
            field.fieldName = PacketFieldName::GAP;
        }
        // Size in bits
        field.fieldBitSize = static_cast<uint8_t>(packetElemt[1].asUInt());

        // Simple field
        if (packetElemt.size() == 2)
        {
            packetByteSize += field.fieldBitSize;
            // Size is sub byte
            if (field.fieldBitSize % 8 != 0)
            {
                // Sub bytes have to be consecutive
                assert(subByteFields);
                // Init bit-mask
                uint8_t bitmask = 0;
                // For each bit shift bit-mask correctly
                for (int ii = 0; ii < field.fieldBitSize; ++ii)
                {
                    // Bit-mask has to be intra byte.
                    // Cross byte field is not allowed!
                    assert(currentBit >= 0);
                    bitmask |= static_cast<uint8_t>(1 << currentBit--);
                }
                // Store
                field.fieldBitMask = bitmask;
                field.fieldBitShift = static_cast<uint8_t>(currentBit + 1);
                // If current byte is filled, switch to new mask
                if (currentBit < 0)
                    currentBit = 7;
            }
            else // Size is multiple of a byte
            {
                if (subByteFields)
                    subByteFields = false;
            }
            // Print to console
            // field.Print();
        }
            // Complex field structure with extensions
        else if (packetElemt.size() == 5)
        {
            // Store fields
            field.fieldEscapeValue = static_cast<uint8_t>(packetElemt[2].asUInt());
            field.hasFieldEscape = true;
            field.extensionName = PacketFieldName::fromString(packetElemt[3].asString());
            field.extensionBitSize = static_cast<uint8_t>(packetElemt[4].asUInt());
            // field.Print();
        }
        // Move PacketField to global vector
        packetFields.push_back(std::move(field));
    }
}

}}
