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

#include <string>
#include <stdexcept>
#include <vector>
#include <json/json.h>
#include <json/reader.h>
#include <pacbio/smrtdata/MetricsVerbosity.h>
#include <pacbio/logging/Logger.h>


#include "FileHeaderBuilder.h"
#include "RunLength.h"

#include <bazio/SmartMemory.h>
#include <bazio/file/FileHeader.h>
#include <bazio/MetricFieldName.h>

#include <BazVersion.h>


using namespace PacBio::SmrtData;

namespace PacBio {
namespace BazIO {

FileHeaderBuilder::FileHeaderBuilder(const std::string& movieName,
                                     const float frameRateHz,
                                     const uint32_t movieLengthFrames,
                                     const std::vector<GroupParams<PacketFieldName>>& pulseGroups,
                                     const std::string& experimentMetadata,
                                     const std::string& basecallerConfig,
                                     const ZmwInfo& zmwInfo,
                                     const uint32_t metricFrames,
                                     const Flags& flags)
    : movieName_(movieName)
    , experimentMetadata_(experimentMetadata)
    , basecallerConfig_(basecallerConfig)
    , zmwInfo_(zmwInfo)
    , frameRateHz_(frameRateHz)
    , metricFrames_(metricFrames)
    , movieLengthFrames_(movieLengthFrames)
    , flags_(flags)
{
    Default();
    
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
    if(FileHeader::ValidateExperimentMetadata(FileHeader::ParseExperimentMetadata(experimentMetadata_)))
        header["EXPERIMENT_METADATA"] = experimentMetadata_;
    else
        throw PBException("Error validating experiment metadata for creating JSON file header");
    header["BASECALLER_CONFIG"] = basecallerConfig_;

    if (fileFooterOffset_ == 0)
        header["FILE_FOOTER_OFFSET"] = "0000000000000000000000000000000000000000000000000000000000000000";
    else
        header["FILE_FOOTER_OFFSET"] = (Json::UInt64) fileFooterOffset_;

    if (numSuperChunks_ == 0)
        header["NUM_SUPER_CHUNKS"] = "00000000000000000000000000000000";
    else
        header["NUM_SUPER_CHUNKS"] = numSuperChunks_;

    AddMetricsToJson(header, metricFields_, metricFrames_);
    AddPacketsToJson(header);

    header[ZmwInfo::JsonKey::ZmwInfo] = zmwInfo_.ToJson();

    if (!zmwNumberRejects_.empty())
        header["ZMW_NUMBER_REJECTS_LUT"] = RunLengthEncLUTHexJson(RunLengthEncLUT(zmwNumberRejects_));

    std::stringstream ss;
    ss << file;
    return ss.str();
}

std::vector<char> FileHeaderBuilder::CreateJSONCharVector()
{
    std::string jsonStream = CreateJSON();
    return std::vector<char>(jsonStream.begin(), jsonStream.end());
}

void FileHeaderBuilder::AddMetricField(const MetricFieldName fieldName,
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

    metricFields_.emplace_back(fieldName, fieldBitSize, fieldSigned, scaleFactor);
}

void FileHeaderBuilder::Default()
{
    ClearMetricFields();
    
    basecallerVersion_ = BAZ_BASECALLER_ALGO_VERSION;
    bazWriterVersion_ = BAZIO_VERSION;
    bazMajorVersion_ = BAZ_MAJOR_VERSION;
    bazMinorVersion_ = BAZ_MINOR_VERSION;
    bazPatchVersion_ = BAZ_PATCH_VERSION;
}

void FileHeaderBuilder::DefaultMetrics()
{
    AddMetricField(MetricFieldName::NUM_FRAMES, 16, false, 1);
    AddMetricField(MetricFieldName::NUM_BASES_A, 16, false, 1);
    AddMetricField(MetricFieldName::NUM_BASES_C, 16, false, 1);
    AddMetricField(MetricFieldName::NUM_BASES_G, 16, false, 1);
    AddMetricField(MetricFieldName::NUM_BASES_T, 16, false, 1);
    AddMetricField(MetricFieldName::NUM_HALF_SANDWICHES, 8, false, 1);
    AddMetricField(MetricFieldName::NUM_SANDWICHES, 8, false, 1);
    AddMetricField(MetricFieldName::NUM_PULSE_LABEL_STUTTERS, 8, false, 1);
    AddMetricField(MetricFieldName::PULSE_WIDTH, 16, false, 1);
    AddMetricField(MetricFieldName::BASE_WIDTH, 16, false, 1);
    AddMetricField(MetricFieldName::NUM_PULSES, 16, false, 1);
    AddMetricField(MetricFieldName::PKMAX_A, 16, true, 10);
    AddMetricField(MetricFieldName::PKMAX_C, 16, true, 10);
    AddMetricField(MetricFieldName::PKMAX_G, 16, true, 10);
    AddMetricField(MetricFieldName::PKMAX_T, 16, true, 10);
    AddMetricField(MetricFieldName::PULSE_DETECTION_SCORE, 16, true, 100);
    AddMetricField(MetricFieldName::TRACE_AUTOCORR, 16, true, 10000);
    AddMetricField(MetricFieldName::PKMID_A, 16, true, 10);
    AddMetricField(MetricFieldName::PKMID_C, 16, true, 10);
    AddMetricField(MetricFieldName::PKMID_G, 16, true, 10);
    AddMetricField(MetricFieldName::PKMID_T, 16, true, 10);
    AddMetricField(MetricFieldName::PKMID_FRAMES_A, 16, false, 1);
    AddMetricField(MetricFieldName::PKMID_FRAMES_C, 16, false, 1);
    AddMetricField(MetricFieldName::PKMID_FRAMES_G, 16, false, 1);
    AddMetricField(MetricFieldName::PKMID_FRAMES_T, 16, false, 1);
    AddMetricField(MetricFieldName::BASELINE_SD, 16, true, 10);
    AddMetricField(MetricFieldName::BASELINE_MEAN, 16, true, 10);
    AddMetricField(MetricFieldName::PIXEL_CHECKSUM, 16, true, 1);
    AddMetricField(MetricFieldName::DME_STATUS, 8, false, 1);
    AddMetricField(MetricFieldName::BPZVAR_A, 16, true, 20000);
    AddMetricField(MetricFieldName::BPZVAR_C, 16, true, 20000);
    AddMetricField(MetricFieldName::BPZVAR_G, 16, true, 20000);
    AddMetricField(MetricFieldName::BPZVAR_T, 16, true, 20000);
    AddMetricField(MetricFieldName::PKZVAR_A, 16, true, 1000);
    AddMetricField(MetricFieldName::PKZVAR_C, 16, true, 1000);
    AddMetricField(MetricFieldName::PKZVAR_G, 16, true, 1000);
    AddMetricField(MetricFieldName::PKZVAR_T, 16, true, 1000);
    AddMetricField(MetricFieldName::ACTIVITY_LABEL, 8, false, 1);
}

void FileHeaderBuilder::DefaultMetricsRTAL()
{
    AddMetricField(MetricFieldName::NUM_BASES_A, 16, false, 1);
    AddMetricField(MetricFieldName::NUM_BASES_C, 16, false, 1);
    AddMetricField(MetricFieldName::NUM_BASES_G, 16, false, 1);
    AddMetricField(MetricFieldName::NUM_BASES_T, 16, false, 1);
    AddMetricField(MetricFieldName::PULSE_WIDTH, 16, false, 1);
    AddMetricField(MetricFieldName::BASE_WIDTH, 16, false, 1);
    AddMetricField(MetricFieldName::NUM_PULSES, 16, false, 1);
    AddMetricField(MetricFieldName::PKMID_A, 16, true, 10);
    AddMetricField(MetricFieldName::PKMID_C, 16, true, 10);
    AddMetricField(MetricFieldName::PKMID_G, 16, true, 10);
    AddMetricField(MetricFieldName::PKMID_T, 16, true, 10);
    AddMetricField(MetricFieldName::PKMID_FRAMES_A, 16, false, 1);
    AddMetricField(MetricFieldName::PKMID_FRAMES_C, 16, false, 1);
    AddMetricField(MetricFieldName::PKMID_FRAMES_G, 16, false, 1);
    AddMetricField(MetricFieldName::PKMID_FRAMES_T, 16, false, 1);
    AddMetricField(MetricFieldName::BASELINE_SD, 16, true, 10);
    AddMetricField(MetricFieldName::BASELINE_MEAN, 16, true, 10);
    AddMetricField(MetricFieldName::PIXEL_CHECKSUM, 16, true, 1);
    AddMetricField(MetricFieldName::DME_STATUS, 8, false, 1);
    AddMetricField(MetricFieldName::ACTIVITY_LABEL, 8, false, 1);
}

void FileHeaderBuilder::AddMetricsToJson(Json::Value& header,
                                         const std::vector<MetricField>& metrics,
                                         const uint32_t frames)
{
    Json::Value& metricRoot = header["METRIC"];
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
