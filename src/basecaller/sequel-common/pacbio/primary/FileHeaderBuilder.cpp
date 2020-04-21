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

// Programmer: Armin Töpfer


#include <assert.h>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <stdexcept>
#include <vector>
#include <utility>
#include <json/json.h>
#include <json/reader.h>
#include <pacbio/smrtdata/MetricsVerbosity.h>
#include <pacbio/smrtdata/Readout.h>
#include <pacbio/logging/Logger.h>
#include <pacbio/primary/SmartMemory.h>
#include <pacbio/primary/FileHeaderValidator.h>
#include <pacbio/primary/MetricField.h>
#include <pacbio/primary/MetricFieldMap.h>
#include <pacbio/primary/MetricFieldName.h>
#include <pacbio/primary/MetricFrequency.h>
#include <pacbio/primary/PacketField.h>
#include <pacbio/primary/PacketFieldMap.h>
#include <pacbio/primary/PacketFieldName.h>
#include <pacbio/primary/Sanity.h>
#include <pacbio/primary/SuperChunkMeta.h>
#include <pacbio/primary/ZmwSliceHeader.h>
#include <pacbio/primary/FileHeaderBuilder.h>
#include <BazVersion.h>


using namespace PacBio::SmrtData;

namespace PacBio {
namespace Primary {

FileHeaderBuilder::FileHeaderBuilder(const std::string& movieName,
                                     const float frameRateHz,
                                     const uint32_t movieLengthFrames,
                                     const Readout readout,
                                     const MetricsVerbosity metricsVerbosity,
                                     const std::string& experimentMetadata,
                                     const std::string& basecallerConfig,
                                     const std::vector<uint32_t> zmwNumbers,
                                     const std::vector<uint32_t> zmwUnitFeatures,
                                     const uint32_t hFMetricFrames,
                                     const uint32_t mFMetricFrames,
                                     const uint32_t sliceLengthFrames,
                                     const bool spiderOnSequel,
                                     const bool newBazFormat,
                                     const bool useHalfFloat,
                                     const bool realtimeActivityLabels)
        : movieName_(movieName)
          , readout_(readout)
          , metricsVerbosity_(metricsVerbosity)
          , experimentMetadata_(experimentMetadata)
          , basecallerConfig_(basecallerConfig)
          , zmwNumbers_(zmwNumbers)
          , zmwUnitFeatures_(zmwUnitFeatures)
          , sliceLengthFrames_(sliceLengthFrames)
          , frameRateHz_(frameRateHz)
          , hFMetricFrames_(hFMetricFrames)
          , mFMetricFrames_(mFMetricFrames)
          , lFMetricFrames_(sliceLengthFrames)
          , movieLengthFrames_(movieLengthFrames)
          , useHalfFloat_(useHalfFloat)
          , realtimeActivityLabels_(realtimeActivityLabels)
{
    (void) spiderOnSequel;

    Default();

    if (!SanityCheckMetricBlockSizes())
    {
        throw PBException("Bad metric block sizes configuration");
    }

    switch (readout)
    {
        case Readout::BASES:
        case Readout::BASES_WITHOUT_QVS:
            DefaultBases();
            break;
        case Readout::PULSES:
            DefaultPulses();
            break;
        default:
            std::runtime_error("Unknown Readout mode");
    }

    if (newBazFormat)
    {
        DefaultMetricsSpider();
    }
    else
    {
        switch (metricsVerbosity)
        {
            case MetricsVerbosity::MINIMAL:
                DefaultMetricsMinimal();
                break;
            case MetricsVerbosity::HIGH:
                DefaultMetricsHigh();
                break;
            case MetricsVerbosity::NONE:
                break;
            default:
                throw PBException("Unknown MetricsVerbosity mode");
        }
    }
}

bool FileHeaderBuilder::SanityCheckMetricBlockSizes()
{
    return (hFMetricFrames_ <= mFMetricFrames_) &&
           (mFMetricFrames_ <= lFMetricFrames_) &&
           (lFMetricFrames_ % mFMetricFrames_ == 0) &&
           (mFMetricFrames_ % hFMetricFrames_ == 0);
}

std::string FileHeaderBuilder::CreateJSON()
{
    Json::Value file;
    file["TYPE"] = "BAZ";
    Json::Value& header = file["HEADER"];
    header["COMPLETE"] = complete_ ? 1 : 0;
    header["TRUNCATED"] = truncated_ ? 1 : 0;
    header["MOVIE_NAME"] = movieName_;
    header["BAZ_MAJOR_VERSION"] = bazMajorVersion_;
    header["BAZ_MINOR_VERSION"] = bazMinorVersion_;
    header["BAZ_PATCH_VERSION"] = bazPatchVersion_;
    header["P4_VERSION"] = p4Version_;
    header["BASE_CALLER_VERSION"] = basecallerVersion_;
    header["BAZWRITER_VERSION"] = bazWriterVersion_;
    header["FRAME_RATE_HZ"] = frameRateHz_;
    header["SLICE_LENGTH_FRAMES"] = sliceLengthFrames_;
    header["MOVIE_LENGTH_FRAMES"] = movieLengthFrames_;
    validateExperimentMetadata(experimentMetadata_);
    header["EXPERIMENT_METADATA"] = experimentMetadata_;
    header["BASECALLER_CONFIG"] = basecallerConfig_;

    if (fileFooterOffset_ == 0)
        header["FILE_FOOTER_OFFSET"] = "0000000000000000000000000000000000000000000000000000000000000000";
    else
        header["FILE_FOOTER_OFFSET"] = (Json::UInt64) fileFooterOffset_;

    if (numSuperChunks_ == 0)
        header["NUM_SUPER_CHUNKS"] = "00000000000000000000000000000000";
    else
        header["NUM_SUPER_CHUNKS"] = numSuperChunks_;

    AddMetricsToJson(header, lFMetricFields_, lFMetricFrames_, MetricFrequency::LOW);
    AddMetricsToJson(header, mFMetricFields_, mFMetricFrames_, MetricFrequency::MEDIUM);
    AddMetricsToJson(header, hFMetricFields_, hFMetricFrames_, MetricFrequency::HIGH);
    AddPacketsToJson(header, packetFields_);

    if (!zmwNumbers_.empty())
        header["ZMW_NUMBER_LUT"] = RunLengthEncLUTHexJson(RunLengthEncLUT(zmwNumbers_));

    if (!zmwNumberRejects_.empty())
        header["ZMW_NUMBER_REJECTS_LUT"] = RunLengthEncLUTHexJson(RunLengthEncLUT(zmwNumberRejects_));

    if (zmwUnitFeatures_.empty())
        for (size_t i = 0; i < zmwNumbers_.size(); ++i)
            zmwUnitFeatures_.push_back(i);
    header["ZMW_UNIT_FEATURES_LUT"] = RunLengthEncLUTJson(RunLengthEncLUT(zmwUnitFeatures_));

    std::stringstream ss;
    ss << file;
    return ss.str();
}

std::vector<char> FileHeaderBuilder::CreateJSONCharVector()
{
    std::string jsonStream = CreateJSON();
    return std::vector<char>(jsonStream.begin(), jsonStream.end());
}

Json::Value FileHeaderBuilder::RunLengthEncLUTJson(
        const std::vector<std::pair<uint32_t, uint32_t>>& input)
{
    Json::Value lut;
    for (const auto& p : input)
    {
        Json::Value singleLut;
        singleLut.append(p.first);
        singleLut.append(p.second);
        lut.append(std::move(singleLut));
    }
    return lut;
}

Json::Value FileHeaderBuilder::RunLengthEncLUTHexJson(
        const std::vector<std::pair<uint32_t, uint32_t>>& input)
{
    Json::Value lut;
    for (const auto& p : input)
    {
        Json::Value singleLut;
        std::stringstream ss;
        ss << "0x" << std::hex << p.first << std::dec;
        singleLut.append(ss.str());
        singleLut.append(p.second);
        lut.append(std::move(singleLut));
    }
    return lut;
}

std::vector<std::pair<uint32_t, uint32_t>> FileHeaderBuilder::RunLengthEncLUT(const std::vector<uint32_t>& input)
{
    std::vector<std::pair<uint32_t, uint32_t>> rleLut;
    if (!input.empty())
    {
        uint32_t startNumber = input[0];
        uint32_t currentNumber = input[0];
        uint32_t currentCount = 1;
        for (size_t i = 1; i < input.size(); ++i)
        {
            if (input[i] == currentNumber + 1)
            {
                currentNumber = input[i];
                ++currentCount;
            }
            else
            {
                rleLut.emplace_back(startNumber, currentCount);
                startNumber = input[i];
                currentNumber = input[i];
                currentCount = 1;
            }
        }
        rleLut.emplace_back(startNumber, currentCount);
    }

    return rleLut;
}

void FileHeaderBuilder::ClearMetricFields(const MetricFrequency& frequency)
{
    switch (frequency)
    {
        case MetricFrequency::LOW:
            lFMetricFields_.clear();
            break;
        case MetricFrequency::MEDIUM:
            mFMetricFields_.clear();
            break;
        case MetricFrequency::HIGH:
            hFMetricFields_.clear();
            break;
    }
}

void FileHeaderBuilder::AddMetricField(const MetricFrequency& frequency,
                                       const MetricFieldName fieldName,
                                       const uint8_t fieldBitSize,
                                       const bool fieldSigned,
                                       const uint16_t fieldScalingFactor)
{
    if (fieldBitSize == 8 && fieldScalingFactor == 0)
    {
        throw PBException("Cannot add 8-bit floating point!");
    }

    // Use scale factor of 0 to indicate usage of half float.
    const auto scaleFactor = (useHalfFloat_ && fieldScalingFactor != 1) ? 0 : fieldScalingFactor;

    switch (frequency)
    {
        case MetricFrequency::LOW:
            lFMetricFields_.emplace_back(fieldName, fieldBitSize, fieldSigned, scaleFactor);
            break;
        case MetricFrequency::MEDIUM:
            mFMetricFields_.emplace_back(fieldName, fieldBitSize, fieldSigned, scaleFactor);
            break;
        case MetricFrequency::HIGH:
            hFMetricFields_.emplace_back(fieldName, fieldBitSize, fieldSigned, scaleFactor);
            break;
    }
}

void FileHeaderBuilder::Default()
{
    ClearMetricFields(MetricFrequency::LOW);
    ClearMetricFields(MetricFrequency::MEDIUM);
    ClearMetricFields(MetricFrequency::HIGH);
    ClearPacketFields();

    basecallerVersion_ = BAZ_BASECALLER_ALGO_VERSION;
    bazWriterVersion_ = BAZIO_VERSION;
    bazMajorVersion_ = BAZ_MAJOR_VERSION;
    bazMinorVersion_ = BAZ_MINOR_VERSION;
    bazPatchVersion_ = BAZ_PATCH_VERSION;
    p4Version_ = BAZIO_P4VERSION;
}

void FileHeaderBuilder::DefaultBases()
{
    AddPacketField(PacketFieldName::READOUT, 2);
    AddPacketField(PacketFieldName::OVERALL_QV, 4);
    AddPacketField(PacketFieldName::GAP, 2);
    AddPacketField(PacketFieldName::IPD_V1, 8);
    AddPacketField(PacketFieldName::PW_V1, 8);
}

void FileHeaderBuilder::DefaultPulses()
{
    AddPacketField(PacketFieldName::READOUT, 2);
    AddPacketField(PacketFieldName::DEL_TAG, 3);
    AddPacketField(PacketFieldName::SUB_TAG, 3);
    AddPacketField(PacketFieldName::DEL_QV, 4);
    AddPacketField(PacketFieldName::SUB_QV, 4);
    AddPacketField(PacketFieldName::INS_QV, 4);
    AddPacketField(PacketFieldName::MRG_QV, 4);
    AddPacketField(PacketFieldName::ALT_QV, 4);
    AddPacketField(PacketFieldName::LAB_QV, 4);
    AddPacketField(PacketFieldName::IS_BASE, 1);
    AddPacketField(PacketFieldName::IS_PULSE, 1);
    AddPacketField(PacketFieldName::ALT_LABEL, 3);
    AddPacketField(PacketFieldName::LABEL, 3);
    AddPacketField(PacketFieldName::IPD_LL, 8, 255, PacketFieldName::IPD32_LL, 32);
    AddPacketField(PacketFieldName::PW_LL, 8, 255, PacketFieldName::PW32_LL, 32);
    AddPacketField(PacketFieldName::PKMEAN_LL, 8, 255, PacketFieldName::PKMEAN16_LL, 16);
    AddPacketField(PacketFieldName::PKMID_LL, 8, 255, PacketFieldName::PKMID16_LL, 16);
    // AddPacketField(PacketFieldName::PKMEAN2_LL,8, 255, PacketFieldName::PKMEAN216_LL,16);
    // AddPacketField(PacketFieldName::PKMID2_LL, 8, 255, PacketFieldName::PKMID216_LL, 16);

}

void FileHeaderBuilder::DefaultMetricsMinimal()
{
    AddMetricField(MetricFrequency::HIGH, MetricFieldName::NUM_BASES, 16, false, 1);
    AddMetricField(MetricFrequency::MEDIUM, MetricFieldName::NUM_BASES, 16, false, 1);
    AddMetricField(MetricFrequency::LOW, MetricFieldName::NUM_BASES, 16, false, 1);

    AddMetricField(MetricFrequency::HIGH, MetricFieldName::NUM_PULSES, 16, false, 1);
    AddMetricField(MetricFrequency::MEDIUM, MetricFieldName::NUM_PULSES, 16, false, 1);
    AddMetricField(MetricFrequency::LOW, MetricFieldName::NUM_PULSES, 16, false, 1);

    AddMetricField(MetricFrequency::HIGH, MetricFieldName::NUM_FRAMES, 16, false, 1);

#ifndef PPA_ONLY_HF_METRICS

    AddMetricField(MetricFrequency::MEDIUM, MetricFieldName::NUM_FRAMES, 16, false, 1);
    AddMetricField(MetricFrequency::MEDIUM, MetricFieldName::NUM_HALF_SANDWICHES, 16, false, 1);
    AddMetricField(MetricFrequency::MEDIUM, MetricFieldName::NUM_SANDWICHES, 16, false, 1);
    AddMetricField(MetricFrequency::MEDIUM, MetricFieldName::NUM_PULSE_LABEL_STUTTERS, 16, false, 1);
    AddMetricField(MetricFrequency::MEDIUM, MetricFieldName::PULSE_WIDTH, 16, false, 1);
    AddMetricField(MetricFrequency::MEDIUM, MetricFieldName::NUM_PULSES_A, 16, false, 1);
    AddMetricField(MetricFrequency::MEDIUM, MetricFieldName::NUM_PULSES_C, 16, false, 1);
    AddMetricField(MetricFrequency::MEDIUM, MetricFieldName::NUM_PULSES_G, 16, false, 1);
    AddMetricField(MetricFrequency::MEDIUM, MetricFieldName::NUM_PULSES_T, 16, false, 1);
    AddMetricField(MetricFrequency::MEDIUM, MetricFieldName::PKMAX_A, 16, true, 10);
    AddMetricField(MetricFrequency::MEDIUM, MetricFieldName::PKMAX_C, 16, true, 10);
    AddMetricField(MetricFrequency::MEDIUM, MetricFieldName::PKMAX_G, 16, true, 10);
    AddMetricField(MetricFrequency::MEDIUM, MetricFieldName::PKMAX_T, 16, true, 10);
    AddMetricField(MetricFrequency::MEDIUM, MetricFieldName::PULSE_DETECTION_SCORE, 16, true, 100);
    AddMetricField(MetricFrequency::MEDIUM, MetricFieldName::TRACE_AUTOCORR, 16, true, 10000);

    AddMetricField(MetricFrequency::LOW, MetricFieldName::NUM_FRAMES, 16, false, 1);
    AddMetricField(MetricFrequency::LOW, MetricFieldName::PKMID_A, 16, true, 10);
    AddMetricField(MetricFrequency::LOW, MetricFieldName::PKMID_C, 16, true, 10);
    AddMetricField(MetricFrequency::LOW, MetricFieldName::PKMID_G, 16, true, 10);
    AddMetricField(MetricFrequency::LOW, MetricFieldName::PKMID_T, 16, true, 10);
    AddMetricField(MetricFrequency::LOW, MetricFieldName::PKMID_FRAMES_A, 16, false, 1);
    AddMetricField(MetricFrequency::LOW, MetricFieldName::PKMID_FRAMES_C, 16, false, 1);
    AddMetricField(MetricFrequency::LOW, MetricFieldName::PKMID_FRAMES_G, 16, false, 1);
    AddMetricField(MetricFrequency::LOW, MetricFieldName::PKMID_FRAMES_T, 16, false, 1);
    AddMetricField(MetricFrequency::LOW, MetricFieldName::BASELINE_RED_SD, 16, true, 10);
    AddMetricField(MetricFrequency::LOW, MetricFieldName::BASELINE_GREEN_SD, 16, true, 10);
    AddMetricField(MetricFrequency::LOW, MetricFieldName::BASELINE_RED_MEAN, 16, true, 10);
    AddMetricField(MetricFrequency::LOW, MetricFieldName::BASELINE_GREEN_MEAN, 16, true, 10);
    AddMetricField(MetricFrequency::LOW, MetricFieldName::PULSE_WIDTH, 16, false, 1);
    AddMetricField(MetricFrequency::LOW, MetricFieldName::BASE_WIDTH, 16, false, 1);
    AddMetricField(MetricFrequency::LOW, MetricFieldName::PIXEL_CHECKSUM, 16, true, 1);
    AddMetricField(MetricFrequency::LOW, MetricFieldName::ANGLE_A, 16, true, 100);
    AddMetricField(MetricFrequency::LOW, MetricFieldName::ANGLE_C, 16, true, 100);
    AddMetricField(MetricFrequency::LOW, MetricFieldName::ANGLE_G, 16, true, 100);
    AddMetricField(MetricFrequency::LOW, MetricFieldName::ANGLE_T, 16, true, 100);
    AddMetricField(MetricFrequency::LOW, MetricFieldName::BPZVAR_A, 16, false, 10000);
    AddMetricField(MetricFrequency::LOW, MetricFieldName::BPZVAR_C, 16, false, 10000);
    AddMetricField(MetricFrequency::LOW, MetricFieldName::BPZVAR_G, 16, false, 10000);
    AddMetricField(MetricFrequency::LOW, MetricFieldName::BPZVAR_T, 16, false, 10000);
    AddMetricField(MetricFrequency::LOW, MetricFieldName::PKZVAR_A, 16, false, 1000);
    AddMetricField(MetricFrequency::LOW, MetricFieldName::PKZVAR_C, 16, false, 1000);
    AddMetricField(MetricFrequency::LOW, MetricFieldName::PKZVAR_G, 16, false, 1000);
    AddMetricField(MetricFrequency::LOW, MetricFieldName::PKZVAR_T, 16, false, 1000);

#else
#warning "COMPILING W/ HF_METRICS ONLY"
#endif
}

void FileHeaderBuilder::DefaultMetricsHigh()
{
    AddMetricField(MetricFrequency::HIGH, MetricFieldName::NUM_BASES, 16, false, 1);
    AddMetricField(MetricFrequency::MEDIUM, MetricFieldName::NUM_BASES, 16, false, 1);
    AddMetricField(MetricFrequency::LOW, MetricFieldName::NUM_BASES, 16, false, 1);

    AddMetricField(MetricFrequency::HIGH, MetricFieldName::NUM_PULSES, 16, false, 1);
    AddMetricField(MetricFrequency::MEDIUM, MetricFieldName::NUM_PULSES, 16, false, 1);
    AddMetricField(MetricFrequency::LOW, MetricFieldName::NUM_PULSES, 16, false, 1);

    AddMetricField(MetricFrequency::HIGH, MetricFieldName::NUM_FRAMES, 16, false, 1);

#ifndef PPA_ONLY_HF_METRICS

    AddMetricField(MetricFrequency::MEDIUM, MetricFieldName::NUM_FRAMES, 16, false, 1);
    AddMetricField(MetricFrequency::MEDIUM, MetricFieldName::NUM_HALF_SANDWICHES, 16, false, 1);
    AddMetricField(MetricFrequency::MEDIUM, MetricFieldName::NUM_SANDWICHES, 16, false, 1);
    AddMetricField(MetricFrequency::MEDIUM, MetricFieldName::NUM_PULSE_LABEL_STUTTERS, 16, false, 1);
    AddMetricField(MetricFrequency::MEDIUM, MetricFieldName::PKMID_A, 16, true, 10);
    AddMetricField(MetricFrequency::MEDIUM, MetricFieldName::PKMID_C, 16, true, 10);
    AddMetricField(MetricFrequency::MEDIUM, MetricFieldName::PKMID_G, 16, true, 10);
    AddMetricField(MetricFrequency::MEDIUM, MetricFieldName::PKMID_T, 16, true, 10);
    AddMetricField(MetricFrequency::MEDIUM, MetricFieldName::PKMID_FRAMES_A, 16, false, 1);
    AddMetricField(MetricFrequency::MEDIUM, MetricFieldName::PKMID_FRAMES_C, 16, false, 1);
    AddMetricField(MetricFrequency::MEDIUM, MetricFieldName::PKMID_FRAMES_G, 16, false, 1);
    AddMetricField(MetricFrequency::MEDIUM, MetricFieldName::PKMID_FRAMES_T, 16, false, 1);
    AddMetricField(MetricFrequency::MEDIUM, MetricFieldName::PULSE_WIDTH, 16, false, 1);
    AddMetricField(MetricFrequency::MEDIUM, MetricFieldName::BASELINE_RED_SD, 16, true, 10);
    AddMetricField(MetricFrequency::MEDIUM, MetricFieldName::BASELINE_GREEN_SD, 16, true, 10);
    AddMetricField(MetricFrequency::MEDIUM, MetricFieldName::BASELINE_RED_MEAN, 16, true, 10);
    AddMetricField(MetricFrequency::MEDIUM, MetricFieldName::BASELINE_GREEN_MEAN, 16, true, 10);
    AddMetricField(MetricFrequency::MEDIUM, MetricFieldName::NUM_PULSES_A, 16, false, 1);
    AddMetricField(MetricFrequency::MEDIUM, MetricFieldName::NUM_PULSES_C, 16, false, 1);
    AddMetricField(MetricFrequency::MEDIUM, MetricFieldName::NUM_PULSES_G, 16, false, 1);
    AddMetricField(MetricFrequency::MEDIUM, MetricFieldName::NUM_PULSES_T, 16, false, 1);
    AddMetricField(MetricFrequency::MEDIUM, MetricFieldName::ANGLE_A, 16, true, 100);
    AddMetricField(MetricFrequency::MEDIUM, MetricFieldName::ANGLE_C, 16, true, 100);
    AddMetricField(MetricFrequency::MEDIUM, MetricFieldName::ANGLE_G, 16, true, 100);
    AddMetricField(MetricFrequency::MEDIUM, MetricFieldName::ANGLE_T, 16, true, 100);
    AddMetricField(MetricFrequency::MEDIUM, MetricFieldName::PKMAX_A, 16, true, 10);
    AddMetricField(MetricFrequency::MEDIUM, MetricFieldName::PKMAX_C, 16, true, 10);
    AddMetricField(MetricFrequency::MEDIUM, MetricFieldName::PKMAX_G, 16, true, 10);
    AddMetricField(MetricFrequency::MEDIUM, MetricFieldName::PKMAX_T, 16, true, 10);
    AddMetricField(MetricFrequency::MEDIUM, MetricFieldName::PULSE_DETECTION_SCORE, 16, true, 100);
    AddMetricField(MetricFrequency::MEDIUM, MetricFieldName::TRACE_AUTOCORR, 16, true, 10000);
    AddMetricField(MetricFrequency::MEDIUM, MetricFieldName::BPZVAR_A, 16, false, 10000);
    AddMetricField(MetricFrequency::MEDIUM, MetricFieldName::BPZVAR_C, 16, false, 10000);
    AddMetricField(MetricFrequency::MEDIUM, MetricFieldName::BPZVAR_G, 16, false, 10000);
    AddMetricField(MetricFrequency::MEDIUM, MetricFieldName::BPZVAR_T, 16, false, 10000);
    AddMetricField(MetricFrequency::MEDIUM, MetricFieldName::PKZVAR_A, 16, false, 1000);
    AddMetricField(MetricFrequency::MEDIUM, MetricFieldName::PKZVAR_C, 16, false, 1000);
    AddMetricField(MetricFrequency::MEDIUM, MetricFieldName::PKZVAR_G, 16, false, 1000);
    AddMetricField(MetricFrequency::MEDIUM, MetricFieldName::PKZVAR_T, 16, false, 1000);


    AddMetricField(MetricFrequency::LOW, MetricFieldName::NUM_FRAMES, 16, false, 1);
    AddMetricField(MetricFrequency::LOW, MetricFieldName::PULSE_WIDTH, 16, false, 1);
    AddMetricField(MetricFrequency::LOW, MetricFieldName::BASE_WIDTH, 16, false, 1);
    AddMetricField(MetricFrequency::LOW, MetricFieldName::PIXEL_CHECKSUM, 16, true, 1);

#else
#warning "COMPILING W/ HF_METRICS ONLY"
#endif
}

void FileHeaderBuilder::DefaultMetricsSequel(MetricFrequency frequency)
{
    AddMetricField(frequency, MetricFieldName::NUM_FRAMES, 16, false, 1);
    AddMetricField(frequency, MetricFieldName::NUM_BASES_A, 16, false, 1);
    AddMetricField(frequency, MetricFieldName::NUM_BASES_C, 16, false, 1);
    AddMetricField(frequency, MetricFieldName::NUM_BASES_G, 16, false, 1);
    AddMetricField(frequency, MetricFieldName::NUM_BASES_T, 16, false, 1);
    AddMetricField(frequency, MetricFieldName::NUM_HALF_SANDWICHES, 8, false, 1);
    AddMetricField(frequency, MetricFieldName::NUM_SANDWICHES, 8, false, 1);
    AddMetricField(frequency, MetricFieldName::NUM_PULSE_LABEL_STUTTERS, 8, false, 1);
    AddMetricField(frequency, MetricFieldName::PULSE_WIDTH, 16, false, 1);
    AddMetricField(frequency, MetricFieldName::BASE_WIDTH, 16, false, 1);
    AddMetricField(frequency, MetricFieldName::NUM_PULSES, 16, false, 1);
    AddMetricField(frequency, MetricFieldName::PKMAX_A, 16, true, 10);
    AddMetricField(frequency, MetricFieldName::PKMAX_C, 16, true, 10);
    AddMetricField(frequency, MetricFieldName::PKMAX_G, 16, true, 10);
    AddMetricField(frequency, MetricFieldName::PKMAX_T, 16, true, 10);
    AddMetricField(frequency, MetricFieldName::PULSE_DETECTION_SCORE, 16, true, 100);
    AddMetricField(frequency, MetricFieldName::TRACE_AUTOCORR, 16, true, 10000);
    AddMetricField(frequency, MetricFieldName::PKMID_A, 16, true, 10);
    AddMetricField(frequency, MetricFieldName::PKMID_C, 16, true, 10);
    AddMetricField(frequency, MetricFieldName::PKMID_G, 16, true, 10);
    AddMetricField(frequency, MetricFieldName::PKMID_T, 16, true, 10);
    AddMetricField(frequency, MetricFieldName::PKMID_FRAMES_A, 16, false, 1);
    AddMetricField(frequency, MetricFieldName::PKMID_FRAMES_C, 16, false, 1);
    AddMetricField(frequency, MetricFieldName::PKMID_FRAMES_G, 16, false, 1);
    AddMetricField(frequency, MetricFieldName::PKMID_FRAMES_T, 16, false, 1);
    AddMetricField(frequency, MetricFieldName::BASELINE_RED_SD, 16, true, 10);
    AddMetricField(frequency, MetricFieldName::BASELINE_GREEN_SD, 16, true, 10);
    AddMetricField(frequency, MetricFieldName::BASELINE_RED_MEAN, 16, true, 10);
    AddMetricField(frequency, MetricFieldName::BASELINE_GREEN_MEAN, 16, true, 10);
    AddMetricField(frequency, MetricFieldName::PIXEL_CHECKSUM, 16, true, 1);
    AddMetricField(frequency, MetricFieldName::ANGLE_GREEN, 16, true, 100);
    AddMetricField(frequency, MetricFieldName::ANGLE_RED, 16, true, 100);
    AddMetricField(frequency, MetricFieldName::DME_STATUS, 8, false, 1);
    AddMetricField(frequency, MetricFieldName::BPZVAR_A, 16, false, 10000);
    AddMetricField(frequency, MetricFieldName::BPZVAR_C, 16, false, 10000);
    AddMetricField(frequency, MetricFieldName::BPZVAR_G, 16, false, 10000);
    AddMetricField(frequency, MetricFieldName::BPZVAR_T, 16, false, 10000);
    AddMetricField(frequency, MetricFieldName::PKZVAR_A, 16, false, 1000);
    AddMetricField(frequency, MetricFieldName::PKZVAR_C, 16, false, 1000);
    AddMetricField(frequency, MetricFieldName::PKZVAR_G, 16, false, 1000);
    AddMetricField(frequency, MetricFieldName::PKZVAR_T, 16, false, 1000);
}

void FileHeaderBuilder::DefaultMetricsSpider(MetricFrequency frequency)
{
    AddMetricField(frequency, MetricFieldName::NUM_FRAMES, 16, false, 1);
    AddMetricField(frequency, MetricFieldName::NUM_BASES_A, 16, false, 1);
    AddMetricField(frequency, MetricFieldName::NUM_BASES_C, 16, false, 1);
    AddMetricField(frequency, MetricFieldName::NUM_BASES_G, 16, false, 1);
    AddMetricField(frequency, MetricFieldName::NUM_BASES_T, 16, false, 1);
    AddMetricField(frequency, MetricFieldName::NUM_HALF_SANDWICHES, 8, false, 1);
    AddMetricField(frequency, MetricFieldName::NUM_SANDWICHES, 8, false, 1);
    AddMetricField(frequency, MetricFieldName::NUM_PULSE_LABEL_STUTTERS, 8, false, 1);
    AddMetricField(frequency, MetricFieldName::PULSE_WIDTH, 16, false, 1);
    AddMetricField(frequency, MetricFieldName::BASE_WIDTH, 16, false, 1);
    AddMetricField(frequency, MetricFieldName::NUM_PULSES, 16, false, 1);
    AddMetricField(frequency, MetricFieldName::PKMAX_A, 16, true, 10);
    AddMetricField(frequency, MetricFieldName::PKMAX_C, 16, true, 10);
    AddMetricField(frequency, MetricFieldName::PKMAX_G, 16, true, 10);
    AddMetricField(frequency, MetricFieldName::PKMAX_T, 16, true, 10);
    AddMetricField(frequency, MetricFieldName::PULSE_DETECTION_SCORE, 16, true, 100);
    AddMetricField(frequency, MetricFieldName::TRACE_AUTOCORR, 16, true, 10000);
    AddMetricField(frequency, MetricFieldName::PKMID_A, 16, true, 10);
    AddMetricField(frequency, MetricFieldName::PKMID_C, 16, true, 10);
    AddMetricField(frequency, MetricFieldName::PKMID_G, 16, true, 10);
    AddMetricField(frequency, MetricFieldName::PKMID_T, 16, true, 10);
    AddMetricField(frequency, MetricFieldName::PKMID_FRAMES_A, 16, false, 1);
    AddMetricField(frequency, MetricFieldName::PKMID_FRAMES_C, 16, false, 1);
    AddMetricField(frequency, MetricFieldName::PKMID_FRAMES_G, 16, false, 1);
    AddMetricField(frequency, MetricFieldName::PKMID_FRAMES_T, 16, false, 1);
    AddMetricField(frequency, MetricFieldName::BASELINE_SD, 16, true, 10);
    AddMetricField(frequency, MetricFieldName::BASELINE_MEAN, 16, true, 10);
    AddMetricField(frequency, MetricFieldName::PIXEL_CHECKSUM, 16, true, 1);
    AddMetricField(frequency, MetricFieldName::DME_STATUS, 8, false, 1);
    if (realtimeActivityLabels_)
    {
        AddMetricField(frequency, MetricFieldName::ACTIVITY_LABEL, 8, false, 1);
    }
    AddMetricField(frequency, MetricFieldName::BPZVAR_A, 16, false, 20000);
    AddMetricField(frequency, MetricFieldName::BPZVAR_C, 16, false, 20000);
    AddMetricField(frequency, MetricFieldName::BPZVAR_G, 16, false, 20000);
    AddMetricField(frequency, MetricFieldName::BPZVAR_T, 16, false, 20000);
    AddMetricField(frequency, MetricFieldName::PKZVAR_A, 16, false, 1000);
    AddMetricField(frequency, MetricFieldName::PKZVAR_C, 16, false, 1000);
    AddMetricField(frequency, MetricFieldName::PKZVAR_G, 16, false, 1000);
    AddMetricField(frequency, MetricFieldName::PKZVAR_T, 16, false, 1000);
}

void FileHeaderBuilder::AddMetricsToJson(Json::Value& header,
                                         const std::vector<MetricField>& metrics,
                                         const int frames,
                                         const MetricFrequency& frequency)
{
    std::string jsonField;
    switch (frequency)
    {
        case MetricFrequency::LOW:
            jsonField = "LF_METRIC";
            break;
        case MetricFrequency::MEDIUM:
            jsonField = "MF_METRIC";
            break;
        case MetricFrequency::HIGH:
            jsonField = "HF_METRIC";
            break;
    }
    Json::Value& metricRoot = header[jsonField];
    if (!metrics.empty())
    {
        Json::Value& metricFields = metricRoot["FIELDS"];
        for (const auto& x : metrics)
        {
            Json::Value field;
            field.append(x.fieldName.toString());
            field.append(x.fieldBitSize);
            field.append(x.fieldSigned);
            field.append(x.fieldScalingFactor);
            metricFields.append(field);
        }
    }
    metricRoot["FRAMES"] = frames;
}

void
FileHeaderBuilder::AddPacketsToJson(Json::Value& header,
                                    const std::vector<PacketField>& packets)
{
    if (packets.empty())
        throw std::runtime_error("Packets may not be empty!");

    std::string jsonField;
    Json::Value& packetFields = header["PACKET"];
    for (const auto& x : packets)
    {
        Json::Value field;
        field.append(x.fieldName.toString());
        field.append(x.fieldBitSize);
        if (x.hasFieldEscape)
        {
            field.append(x.fieldEscapeValue);
            field.append(x.extensionName.toString());
            field.append(x.extensionBitSize);
        }
        packetFields.append(field);
    }
}

/// Calculate division with round-up on integer values.
static size_t DivCeil(size_t numerator, size_t denominator)
{
    return (numerator + denominator - 1) / denominator;
}

/// Calculate the total size of the packet by summing the
/// bit contributions from the fields.
double FileHeaderBuilder::PacketSize() const
{
    size_t sbits = 0;
    for (const auto& p : packetFields_)
    {
        sbits += p.fieldBitSize;
    }
    size_t sbytes = DivCeil(sbits, 8);

    // Because of pseudo escape encoding of values greater than 255 in PULSES mode,
    // pulses widths and IPDs sometimes need a few extra bytes.
    // Empirically, it is about 50% ! overall.
    double overhead = 1.0;
    if (ReadoutConfig() == Readout::PULSES)
    {
        // measured on    /pbi/collections/326/3260001/r54002_20170309_210557/1_A01/m54002_170309_210631.baz
        overhead = 13.7 / 9.0;
        // /pbi/collections/326/3260002/r54075_20170322_174827/1_A01/m54075_170322_174924.baz
        // overhead = 12.33/9.0
    }

    return sbytes * overhead;
}

/// Calculate the metrics block size for a particular frequency (aka set of fields)
/// by summing the bit contributions from the fields.
size_t FileHeaderBuilder::MetricBlockSize(const std::vector<MetricField>& metricFields) const
{
    size_t sbits = 0;
    for (const auto& m : metricFields)
    {
        sbits += m.fieldBitSize;
    }
    size_t sbytes = DivCeil(sbits, 8);

    return sbytes;
}

size_t FileHeaderBuilder::HfMetricBlockSize() const
{
    return MetricBlockSize(hFMetricFields_);
}

size_t FileHeaderBuilder::MfMetricBlockSize() const
{
    return MetricBlockSize(mFMetricFields_);
}

size_t FileHeaderBuilder::LfMetricBlockSize() const
{
    return MetricBlockSize(lFMetricFields_);
}

// approximate the size of the metrics by assuming that the individual metrics blocks scale directly with
// the movie length. This is not strictly true, as there will likely be a few extra metrics blocks emitted
// at the end of the movie depending on the method to deal with latency.
size_t FileHeaderBuilder::MetricsByteSize() const
{
#if 0
    PBLOG_INFO << "HfMetricBlockSize " << HfMetricBlockSize();
    PBLOG_INFO << "movieLengthFrames_ "<< movieLengthFrames_;
    PBLOG_INFO << "hFMetricFrames_ "   << hFMetricFrames_;
    PBLOG_INFO << "MfMetricBlockSize " << MfMetricBlockSize();
    PBLOG_INFO << "mFMetricFrames_ "   << mFMetricFrames_;
    PBLOG_INFO << "LfMetricBlockSize " << LfMetricBlockSize();
    PBLOG_INFO << "lFMetricFrames_ "   << lFMetricFrames_;
#endif

    size_t metricsSize = zmwNumbers_.size() *
                         ((HfMetricBlockSize() * DivCeil(movieLengthFrames_, hFMetricFrames_)) +
                          (MfMetricBlockSize() * DivCeil(movieLengthFrames_, mFMetricFrames_)) +
                          (LfMetricBlockSize() * DivCeil(movieLengthFrames_, lFMetricFrames_)));
    PBLOG_DEBUG << " FileHeaderBuilder::MetricsByteSize() = " << metricsSize;
    return metricsSize;
};

double FileHeaderBuilder::MetricsByteRate() const
{
    double bytesPerFrame = zmwNumbers_.size() *
                           (
                                   (hFMetricFrames_ ? (1.0 * HfMetricBlockSize() / hFMetricFrames_) : 0) +
                                   (mFMetricFrames_ ? (1.0 * MfMetricBlockSize() / mFMetricFrames_) : 0) +
                                   (lFMetricFrames_ ? (1.0 * LfMetricBlockSize() / lFMetricFrames_) : 0)
                           );
    return bytesPerFrame * frameRateHz_;
}

double FileHeaderBuilder::EventsByteRate(double averageBaseRatePerSecond) const
{
    return zmwNumbers_.size() * averageBaseRatePerSecond * PacketSize();
}

size_t FileHeaderBuilder::EventsByteSize(double averageBaseRatePerSecond) const
{
    const double movieLengthSeconds = static_cast<double>(movieLengthFrames_) / frameRateHz_;

    const size_t eventsSize = static_cast<size_t>(movieLengthSeconds * EventsByteRate(averageBaseRatePerSecond));

    PBLOG_DEBUG << " FileHeaderBuilder::EventsByteSize() = " << eventsSize;

    return eventsSize;
}

size_t FileHeaderBuilder::HeaderSize() const
{
    size_t headerSize = zmwNumbers_.size() * .5 + 4400;
    return headerSize;
}

size_t FileHeaderBuilder::PaddingSize() const
{
    const size_t numSuperChunks = DivCeil(movieLengthFrames_, sliceLengthFrames_);
    size_t numPads = (zmwNumbers_.size() / 1000 + 1) * numSuperChunks;
    return numPads * (4000 * 0.5) + 5400; // assume average padding is a half block.
}

// additional overhead for BAZ file
size_t FileHeaderBuilder::OverheadSize() const
{
    const size_t numSuperChunks = DivCeil(movieLengthFrames_, sliceLengthFrames_);

    const size_t overhead = numSuperChunks * SuperChunkMeta::SizeOf() +
                            numSuperChunks * zmwNumbers_.size() * (ZmwSliceHeader::SizeOf() + Sanity::SizeOf());

    PBLOG_DEBUG << " FileHeaderBuilder::OverheadSize() = " << overhead;
    return overhead;
}

size_t FileHeaderBuilder::ExpectedFileByteSize(double averageBaseRatePerSecond) const
{
    return HeaderSize() + PaddingSize() + EventsByteSize(averageBaseRatePerSecond) + MetricsByteSize() +
           OverheadSize();
}

double FileHeaderBuilder::EstimatedBytesPerSecond(double averageBaseRatePerSecond) const
{
    return MetricsByteRate() + EventsByteRate(averageBaseRatePerSecond);
}

void FileHeaderBuilder::EstimatesSummary(std::ostream& os, double averageBaseRatePerSecond) const
{
    os << "Est Header/Footer Bytes:" << HeaderSize() << std::endl;
    os << "Est Padding Bytes      :" << PaddingSize() << std::endl;
    os << "Est OverheadBytes      :" << OverheadSize() << std::endl;
    os << "Est EventBytes         :" << EventsByteSize(averageBaseRatePerSecond) << std::endl;
    os << "Est MetricsBytes       :" << MetricsByteSize() << std::endl;
    os << "----------------------------------------" << std::endl;
    os << "Est Total BytesWritten :" << ExpectedFileByteSize(averageBaseRatePerSecond) << std::endl;
}

}}
