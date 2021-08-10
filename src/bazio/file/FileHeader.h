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

#ifndef PACBIO_BAZIO_FILE_FILE_HEADER_H
#define PACBIO_BAZIO_FILE_FILE_HEADER_H

#include <map>
#include <string>
#include <stdexcept>
#include <vector>

#include <bazio/encoding/EncodingParams.h>

#include <bazio/MetricField.h>
#include <bazio/MetricFieldName.h>
#include <bazio/MetricFrequency.h>

#include <json/json.h>

namespace PacBio {
namespace BazIO {

class MetricFieldMap;

/// Stores all BAZ file header information and provides logic 
/// to parse a JSON header.
class FileHeader
{
public:
    using MetricField = PacBio::Primary::MetricField;
    using MetricFieldName = PacBio::Primary::MetricFieldName;
    using MetricFrequency = PacBio::Primary::MetricFrequency;
public:
    FileHeader() = default;

    FileHeader(const char* header, const size_t length)
    { Init(header, length); }

    // Allow for copy construction but not assignment so
    // that the FileHeader is constructed properly.
    FileHeader(FileHeader&&) = default;
    FileHeader(const FileHeader&) = default;
    FileHeader& operator=(FileHeader&&) = default;
    FileHeader& operator=(const FileHeader&) = delete;
    ~FileHeader() = default;

public:
    std::string BazVersion() const
    { 
        return std::to_string(bazMajorVersion_) + "." + 
               std::to_string(bazMinorVersion_) + "." + 
               std::to_string(bazPatchVersion_); 
     }

    uint64_t BazLongVersion() const
    { return 10000000 * bazMajorVersion_ 
        + 1000 * bazMinorVersion_ 
        + bazPatchVersion_; }

    uint64_t BazMajorVersion() const
    { return bazMajorVersion_; }

    uint64_t BazMinorVersion() const
    { return bazMinorVersion_; }

    uint64_t BazPatchVersion() const
    { return bazPatchVersion_; }

    uint32_t HFMetricByteSize() const
    { return hFMetricByteSize_; }

    uint32_t MFMetricByteSize() const
    { return mFMetricByteSize_; }

    uint32_t LFMetricByteSize() const
    { return lFMetricByteSize_; }

    uint32_t OutputLengthFrames() const
    { return outputLengthFrames_; }

    double FrameRateHz() const
    { return frameRateHz_; }

    uint32_t HFMetricFrames() const
    { return hFMetricFrames_; }

    uint32_t MFMetricFrames() const
    { return mFMetricFrames_; }

    uint32_t LFMetricFrames() const
    { return lFMetricFrames_; }

    uint32_t HFbyMFRatio() const
    { return hFbyMFRatio_; }

    uint32_t HFbyLFRatio() const
    { return hFbyLFRatio_; }

    uint64_t OffsetFirstChunk() const
    { return offsetFirstChunk_; }

    std::vector<FieldParams> PacketFields() const
    {
        std::vector<FieldParams> fp;
        for (const auto& g : encodeInfo_)
        {
            fp.insert(std::end(fp), g.members.begin(), g.members.end());
        }
        return fp;
    }

    const std::vector<GroupParams>& PacketGroups() const
    { return encodeInfo_; }

    const std::vector<MetricField>& HFMetricFields() const
    { return hFMetricFields_; }

    const std::vector<MetricField>& MFMetricFields() const
    { return mFMetricFields_; }

    const std::vector<MetricField>& LFMetricFields() const
    { return lFMetricFields_; }

    bool HasPacketField(PacketFieldName fieldName) const
    {
        return std::any_of(encodeInfo_.begin(), encodeInfo_.end(),
                           [&fieldName](const GroupParams& gp)
                           { return std::any_of(gp.members.begin(), gp.members.end(),
                                         [&fieldName](const FieldParams& fp) { return fp.name == fieldName; }); });
    }

    bool HasHFMField(MetricFieldName fieldName) const
    {
        for (const auto& field : hFMetricFields_)
        {
            if (field.fieldName == fieldName) return true;
        }
        return false;
    }

    bool HasMFMField(MetricFieldName fieldName) const
    {
        for (const auto& field : mFMetricFields_)
        {
            if (field.fieldName == fieldName) return true;
        }
        return false;
    }

    bool HasLFMField(MetricFieldName fieldName) const
    {
        for (const auto& field : lFMetricFields_)
        {
            if (field.fieldName == fieldName) return true;
        }
        return false;
    }

    int MetricFieldScaling(const MetricFieldName fieldName,
                           const MetricFrequency fieldFrequency) const
    {
        const std::vector<MetricField>* fields;
        switch (fieldFrequency)
        {
            case MetricFrequency::LOW:    fields = &lFMetricFields_; break;
            case MetricFrequency::MEDIUM: fields = &mFMetricFields_; break;
            case MetricFrequency::HIGH:   fields = &hFMetricFields_; break;
            default: throw std::runtime_error("Unknown fieldFrequency");
        }
        for (const auto& field : *fields)
            if (field.fieldName == fieldName)
                return field.fieldScalingFactor;
        return 0;
    }

    const std::string& BaseCallerVersion() const
    { return basecallerVersion_; }

    const std::string& BazWriterVersion() const
    { return bazWriterVersion_; }

    const std::string& BasecallerConfig() const
    { return basecallerConfig_; }

    const std::vector<float> RelativeAmplitudes() const
    {
        // Returned amplitudes are in the order given by BaseMap().
        std::vector<float> relamps;
        for (const auto& val : experimentMetadata_["DyeSet"]["RelativeAmp"])
        {
            relamps.push_back(val.asFloat());
        }
        return relamps;
    }

    const std::string BaseMap() const
    { return experimentMetadata_["DyeSet"]["BaseMap"].asString(); }

    const Json::Value ExperimentMetadata() const
    { return experimentMetadata_; }

    const std::string& MovieName() const
    { return movieName_; }

    uint32_t MovieLengthFrames() const
    { return movieLengthFrames_; }

    uint32_t ZmwIdToNumber(const uint32_t id) const
    { return zmwIdToNumber_.at(id); }

    uint32_t ZmwNumberToId(const uint32_t number) const
    { return zmwNumbersToId_.at(number); }

    const std::vector<uint32_t>& ZmwNumbers() const
    { return zmwIdToNumber_; }

    uint32_t MaxNumZMWs() const
    { return zmwIdToNumber_.size(); }

    bool Complete() const
    { return complete_; }

    bool Truncated() const
    { return truncated_; }

    float MovieTimeInHrs() const
    { return static_cast<float>(movieLengthFrames_ / frameRateHz_ / 3600.0); }

    bool IsZmwNumberRejected(uint32_t zmwNumber) const
    { return std::find(zmwNumberRejects_.cbegin(), zmwNumberRejects_.cend(), zmwNumber) != zmwNumberRejects_.cend(); }

    const std::vector<uint32_t>& ZmwNumberRejects() const
    { return zmwNumberRejects_; }

    uint32_t ZmwUnitFeatures(uint32_t zmwIndex) const
    { 
        if (zmwIndex < zmwUnitFeatures_.size())
            return zmwUnitFeatures_[zmwIndex];
        else if (zmwUnitFeatures_.size() == 0)
            return 0; // if there are no features loaded, then send out a dummy 0.
        else
            PBExceptionStream() << "zmwIndex out of range: " << zmwIndex <<" size:"  << zmwUnitFeatures_.size();
    }

    uint64_t FileFooterOffset() const
    { return offsetFileFooter_; }

    const std::vector<uint32_t>& ZmwUnitFeatures() const
    { return zmwUnitFeatures_; }

    uint32_t NumSuperChunks() const
    { return numSuperChunks_; }

    bool Internal() const
    { return internal_; }

public:
    void BazMajorVersion(int v)
    { bazMajorVersion_ = v; }

    void BazMinorVersion(int v)
    { bazMinorVersion_ = v; }

    void BazPatchVersion(int v)
    { bazPatchVersion_ = v; }

    void HFbyMFRatio(uint32_t arg)
    { hFbyMFRatio_ = arg; }

    void HFbyLFRatio(uint32_t arg)
    { hFbyLFRatio_ = arg; }

    void FrameRateHz(double arg)
    { frameRateHz_ = arg; }

    void BaseCallerVersion(const std::string& version)
    { basecallerVersion_ = version; }

    void BazWriterVersion(const std::string& version)
    { bazWriterVersion_ = version; }

    void BasecallerConfig(const std::string& config)
    { basecallerConfig_ = config; }

    void MovieLengthFrames(const uint32_t movieLengthFrames)
    { movieLengthFrames_ = movieLengthFrames; }

    void Internal(bool internal)
    { internal_ = internal; }

private:
    static const uint8_t MAGICNUMBER0 = 0x02;
    static const uint8_t MAGICNUMBER1 = 'B';
    static const uint8_t MAGICNUMBER2 = 'A';
    static const uint8_t MAGICNUMBER3 = 'Z';

private:
    std::vector<GroupParams> encodeInfo_;
    std::vector<MetricField> hFMetricFields_;
    std::vector<MetricField> mFMetricFields_;
    std::vector<MetricField> lFMetricFields_;

    std::vector<uint32_t> zmwIdToNumber_;
    std::map<uint32_t, uint32_t> zmwNumbersToId_;

    std::vector<uint32_t> zmwNumberRejects_;

    std::vector<uint32_t> zmwUnitFeatures_;

    uint64_t offsetFirstChunk_ = 0;
    uint64_t offsetFileFooter_ = 0;

    uint32_t bazMajorVersion_ = 0;
    uint32_t bazMinorVersion_ = 0;
    uint32_t bazPatchVersion_ = 0;

    std::string basecallerVersion_;
    std::string bazWriterVersion_;
    std::string basecallerConfig_;
    Json::Value experimentMetadata_;

    std::string movieName_;

    uint32_t packetByteSize_   = 0;
    uint32_t hFMetricByteSize_ = 0;
    uint32_t mFMetricByteSize_ = 0;
    uint32_t lFMetricByteSize_ = 0;

    uint32_t outputLengthFrames_ = 0;
    double frameRateHz_ = -1;

    uint32_t hFMetricFrames_ = 0;
    uint32_t mFMetricFrames_ = 0;
    uint32_t lFMetricFrames_ = 0;

    uint32_t hFbyMFRatio_ = 0;
    uint32_t hFbyLFRatio_ = 0;

    uint32_t movieLengthFrames_ = 0;

    bool complete_ = false;
    bool truncated_ = false;
    bool internal_ = false;

    uint32_t numSuperChunks_ = 0;

private:
    void Init(const char* header, const size_t length);

    void ParseMetrics(Json::Value& metricNode,
                      std::vector<MetricField>& metricFields,
                      uint32_t& metricByteSize,
                      uint32_t& metricFrames);

    void ParsePackets(Json::Value& root, 
                      uint32_t& packetByteSize);
};

}} // PacBio::BazIO

#endif // PACBIO_BAZIO_FILE_FILE_HEADER_H
