// Copyright (c) 2021, Pacific Biosciences of California, Inc.
//
// All rights reserved.
//
// THIS SOFTWARE CONSTITUTES AND EMBODIES PACIFIC BIOSCIENCES' CONFIDENTIAL
// AND PROPRIETARY INFORMATION.
//
// Disclosure, redistribution and use of this software is subject to the
// terms and conditions of the applicable written agreement(s) between you
// and Pacific Biosciences, where "you" refers to you or your company or
// organization, as applicable.  Any other disclosure, redistribution or
// use is prohibited.
//
// THIS SOFTWARE IS PROVIDED BY PACIFIC BIOSCIENCES AND ITS CONTRIBUTORS "AS
// IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
// THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL PACIFIC BIOSCIENCES OR ITS
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
// OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
// WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
// OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
// ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <iostream>
#include <string>
#include <stdexcept>
#include <vector>
#include <utility>
#include <json/json.h>
#include <json/reader.h>
#include <pacbio/smrtdata/MetricsVerbosity.h>
#include <pacbio/logging/Logger.h>

#include "FileHeaderBuilder.h"

#include <bazio/SmartMemory.h>
#include <bazio/FileHeaderValidator.h>
#include <bazio/MetricFieldName.h>
#include <bazio/MetricFrequency.h>

#include <BazVersion.h>


using namespace PacBio::SmrtData;

namespace PacBio {
namespace BazIO {

FileHeaderBuilder::FileHeaderBuilder(const std::string& movieName,
                                     const float frameRateHz,
                                     const uint32_t movieLengthFrames,
                                     const std::vector<GroupParams>& pulseGroups,
                                     const MetricsVerbosity metricsVerbosity,
                                     const std::string& experimentMetadata,
                                     const std::string& basecallerConfig,
                                     const std::vector<uint32_t> zmwNumbers,
                                     const std::vector<uint32_t> zmwUnitFeatures,
                                     const uint32_t hFMetricFrames,
                                     const uint32_t mFMetricFrames,
                                     const uint32_t sliceLengthFrames,
                                     const Flags& flags)
    : movieName_(movieName)
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
    , flags_(flags)
{
    Default();

    if (!SanityCheckMetricBlockSizes())
    {
        throw PBException("Bad metric block sizes configuration");
    }

    // Add members from group params.
    for (const auto& gp : pulseGroups)
    {
        encodeInfo_.push_back(gp);
    }

    if (flags_.RealTimeActivityLabels())
    {
        DefaultMetricsRTAL();
    }
    else
    {
        DefaultMetrics();
    }
}


bool FileHeaderBuilder::SanityCheckMetricBlockSizes()
{
    return (hFMetricFrames_ <= mFMetricFrames_) &&
           (mFMetricFrames_ <= lFMetricFrames_) &&
           (mFMetricFrames_ >0 && hFMetricFrames_ > 0) &&
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
    header["BASE_CALLER_VERSION"] = basecallerVersion_;
    header["BAZWRITER_VERSION"] = bazWriterVersion_;
    header["FRAME_RATE_HZ"] = frameRateHz_;
    header["MOVIE_LENGTH_FRAMES"] = movieLengthFrames_;
    Primary::validateExperimentMetadata(experimentMetadata_);
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
    AddPacketsToJson(header);

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

    if (fieldScalingFactor > 1 && !fieldSigned && fieldBitSize == 16)
    {
        PBLOG_WARN << "Unsigned short fixed-point scaling not supported for metric = " << fieldName.toString();
    }

    // Use scale factor of 0 to indicate usage of half float.
    const auto scaleFactor = (fieldScalingFactor != 1) ? 0 : fieldScalingFactor;

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

    basecallerVersion_ = BAZ_BASECALLER_ALGO_VERSION;
    bazWriterVersion_ = BAZIO_VERSION;
    bazMajorVersion_ = BAZ_MAJOR_VERSION;
    bazMinorVersion_ = BAZ_MINOR_VERSION;
    bazPatchVersion_ = BAZ_PATCH_VERSION;
}

void FileHeaderBuilder::DefaultMetrics(MetricFrequency frequency)
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
    AddMetricField(frequency, MetricFieldName::BPZVAR_A, 16, true, 20000);
    AddMetricField(frequency, MetricFieldName::BPZVAR_C, 16, true, 20000);
    AddMetricField(frequency, MetricFieldName::BPZVAR_G, 16, true, 20000);
    AddMetricField(frequency, MetricFieldName::BPZVAR_T, 16, true, 20000);
    AddMetricField(frequency, MetricFieldName::PKZVAR_A, 16, true, 1000);
    AddMetricField(frequency, MetricFieldName::PKZVAR_C, 16, true, 1000);
    AddMetricField(frequency, MetricFieldName::PKZVAR_G, 16, true, 1000);
    AddMetricField(frequency, MetricFieldName::PKZVAR_T, 16, true, 1000);
}

void FileHeaderBuilder::DefaultMetricsRTAL(MetricFrequency frequency)
{
    AddMetricField(frequency, MetricFieldName::NUM_BASES_A, 16, false, 1);
    AddMetricField(frequency, MetricFieldName::NUM_BASES_C, 16, false, 1);
    AddMetricField(frequency, MetricFieldName::NUM_BASES_G, 16, false, 1);
    AddMetricField(frequency, MetricFieldName::NUM_BASES_T, 16, false, 1);
    AddMetricField(frequency, MetricFieldName::PULSE_WIDTH, 16, false, 1);
    AddMetricField(frequency, MetricFieldName::BASE_WIDTH, 16, false, 1);
    AddMetricField(frequency, MetricFieldName::NUM_PULSES, 16, false, 1);
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
    AddMetricField(frequency, MetricFieldName::ACTIVITY_LABEL, 8, false, 1);
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

void FileHeaderBuilder::AddPacketsToJson(Json::Value& header)
{
    for (const auto& g : encodeInfo_)
    {
        header["PACKET"].append(g.Serialize());
    }
}

}} // PacBio::BazIO
