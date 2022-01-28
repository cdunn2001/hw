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

#ifndef BAZIO_FILE_FILE_HEADER_BUILDER_H
#define BAZIO_FILE_FILE_HEADER_BUILDER_H

#include <algorithm>

#include <json/json.h>

#include <bazio/encoding/EncodingParams.h>

#include <bazio/MetricFieldName.h>
#include <bazio/MetricField.h>

#include "ZmwInfo.h"

namespace PacBio {
namespace BazIO {

/// Provides methods to create and manipulate a BAZ FileHeader and
/// export it to JSON.
class FileHeaderBuilder
{
public:
    using MetricField = PacBio::Primary::MetricField;
    using MetricFieldName = PacBio::Primary::MetricFieldName;

public:
    class Flags
    {
    public:
        Flags()
        : realtimeActivityLabels_{false}
        { }

        Flags(Flags&&) = default;
        Flags(const Flags&) = default;
        Flags& operator=(Flags&&) = default;
        Flags& operator=(const Flags&) = default;
        ~Flags() = default;

    public:
        bool RealTimeActivityLabels() const
        { return realtimeActivityLabels_; }

    public:
        Flags& RealTimeActivityLabels(bool v)
        {
            realtimeActivityLabels_ = v;
            return *this;
        }

    private:
        bool realtimeActivityLabels_;
    };

public:
    FileHeaderBuilder(const std::string& movieName,
                      const float frameRateHz,
                      const uint32_t movieLengthFrames,
                      const std::vector<GroupParams<PacketFieldName>>& pulseGroups,
                      const std::string& experimentMetadata,
                      const std::string& basecallerConfig,
                      const ZmwInfo& zmwInfo,
                      const uint32_t metricsFrames,
                      const Flags& flags=Flags());

    FileHeaderBuilder(FileHeaderBuilder&&) = default;
    FileHeaderBuilder(const FileHeaderBuilder&) = default;
    FileHeaderBuilder& operator=(FileHeaderBuilder&&) = default;
    FileHeaderBuilder& operator=(const FileHeaderBuilder&) = delete;
    ~FileHeaderBuilder() = default;

public:
    inline std::vector<FieldParams<PacketFieldName>> PacketFields() const
    {
        std::vector<FieldParams<PacketFieldName>> fps;
        for (const auto& g : encodeInfo_)
            for (const auto& f : g.members)
                fps.push_back(f);
        return fps;
    }

    inline std::vector<MetricField> MetricFields() const
    { return metricFields_; }

    inline std::string MovieName() const
    { return movieName_; }

    inline std::string ExperimentMetadata() const
    { return experimentMetadata_; }

    inline std::string BaseCallerVersion() const
    { return basecallerVersion_; }

    inline std::string BazWriterVersion() const
    { return bazWriterVersion_; }

    inline uint32_t BazMajorVersion() const
    { return bazMajorVersion_; }

    inline uint32_t BazMinorVersion() const
    { return bazMinorVersion_; }

    inline uint32_t BazPatchVersion() const
    { return bazPatchVersion_; }

    inline double FrameRateHz() const
    { return frameRateHz_; }

    inline uint32_t MetricFrames() const
    { return metricFrames_; }

    inline uint32_t MaxNumZmws() const
    { return zmwInfo_.NumZmws(); }

    inline bool Complete() const
    { return complete_; }

    inline bool Truncated() const
    { return truncated_; }

    inline uint32_t NumSuperChunks() const
    { return numSuperChunks_; }

    inline uint32_t MovieLengthFrames() const
    { return movieLengthFrames_; }

    inline const std::vector<uint32_t>& ZmwNumbers() const
    { return zmwInfo_.HoleNumbers(); }

    inline uint64_t FileFooterOffset() const
    { return fileFooterOffset_; }

    inline double MovieLengthSeconds() const
    { return MovieLengthFrames()/ FrameRateHz(); }

public:
    std::string CreateJSON();

    std::vector<char> CreateJSONCharVector();

    inline void ZmwNumberRejects(const std::vector<uint32_t>& rejects)
    { zmwNumberRejects_ = rejects; }

    void ClearMetricFields()
    { metricFields_.clear(); }

    void AddMetricField(const MetricFieldName fieldName,
                        const uint8_t fieldBitSize,
                        const bool fieldSigned,
                        const uint16_t fieldScalingFactor);

    inline void MetricFields(const std::vector<MetricField>& metricFields)
    { metricFields_ = metricFields; }

    inline void MovieName(const std::string& movieName)
    { movieName_ = movieName; }

    inline void BaseCallerVersion(const std::string& baseCallerVersion)
    { basecallerVersion_ = baseCallerVersion; }

    inline void BazWriterVersion(const std::string& bazWriterVersion)
    { bazWriterVersion_ = bazWriterVersion; }

    inline void BazMajorVersion(const uint32_t bazMajorVersion)
    { bazMajorVersion_ = bazMajorVersion; }

    inline void BazMinorVersion(const uint32_t bazMinorVersion)
    { bazMinorVersion_ = bazMinorVersion; }

    inline void BazPatchVersion(const uint32_t bazPatchVersion)
    { bazPatchVersion_ = bazPatchVersion; }

    inline void FrameRateHz(const double frameRateHz)
    { frameRateHz_ = frameRateHz; }

    inline void MetricFrames(const uint32_t metricFrames)
    { metricFrames_ = metricFrames; }

    inline void Done()
    { complete_ = true; }

    inline void Truncated()
    { truncated_ = true; }

    inline void IncSuperChunks()
    { ++numSuperChunks_; }

    inline void MovieLengthFrames(const uint32_t movieLengthFrames)
    { movieLengthFrames_ = movieLengthFrames; }

    inline void FileFooterOffset(const uint64_t fileFooterOffset)
    { fileFooterOffset_ = fileFooterOffset; }

private:
    std::vector<GroupParams<PacketFieldName>> encodeInfo_;
    std::vector<MetricField> metricFields_;

    std::string movieName_;
    std::string basecallerVersion_;
    std::string bazWriterVersion_;

    std::string experimentMetadata_;
    std::string basecallerConfig_;

    std::vector<uint32_t> zmwNumberRejects_;
    ZmwInfo zmwInfo_;

    uint32_t bazMajorVersion_;
    uint32_t bazMinorVersion_;
    uint32_t bazPatchVersion_;
    double frameRateHz_;
    uint32_t metricFrames_;

    bool complete_ = false;
    bool truncated_ = false;
    uint32_t numSuperChunks_ = 0;
    uint32_t movieLengthFrames_ = 0;

    uint64_t fileFooterOffset_ = 0;

    Flags flags_;

private:
    void Default();

    bool SanityCheckMetricBlockSizes();

    void DefaultMetrics();
    void DefaultMetricsRTAL();

    void AddMetricsToJson(Json::Value& header,
                          const std::vector<MetricField>& metrics,
                          const uint32_t frames);

    void AddPacketsToJson(Json::Value& header);
};

}} // PacBio::BazIO

#endif // BAZIO_FILE_FILE_HEADER_BUILDER_H
