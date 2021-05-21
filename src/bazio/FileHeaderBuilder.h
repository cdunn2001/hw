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

#include <algorithm>

#include <json/json.h>
#include <pacbio/smrtdata/Readout.h>
#include <pacbio/smrtdata/MetricsVerbosity.h>

#include "PacketFieldName.h"
#include "MetricFieldName.h"
#include "MetricField.h"
#include "MetricFrequency.h"
#include "PacketField.h"

namespace PacBio {
namespace Primary {

/// Provides methods to create and manipulate a BAZ FileHeader and
/// export it to JSON.
class FileHeaderBuilder
{
public: // static
    static Json::Value RunLengthEncLUTJson(
        const std::vector<std::pair<uint32_t, uint32_t>>& input);
    
    static Json::Value RunLengthEncLUTHexJson(
        const std::vector<std::pair<uint32_t, uint32_t>>& input);

    static std::vector<std::pair<uint32_t, uint32_t>> RunLengthEncLUT(
        const std::vector<uint32_t>& input);

    //static FileHeaderBuilder* MakeDummyFileHeaderBuilder();

    //static FileHeaderBuilder* MakeDummyFileHeaderBuilder(const Acquisition::Setup& setup);

public:
    class Flags
    {
    public:
        Flags()
        : spiderOnSequel_{false}
          , newBazFormat_{true}
          , useHalfFloat_{true}
          , realtimeActivityLabels_{false}
          , useFullBaz_{false}
        {}

        // Move constructor
        Flags(Flags&&) = default;

        // Copy constructor
        Flags(const Flags&) = default;

        // Move assignment operator
        Flags& operator=(Flags&&) = default;

        // Copy assignment operator
        Flags& operator=(const Flags&) = default;

        // Destructor
        ~Flags() = default;

    public:
        bool SpiderOnSequel() const
        { return spiderOnSequel_; }

        bool NewBazFormat() const
        { return newBazFormat_; }

        bool UseHalfFloat() const
        { return useHalfFloat_; }

        bool RealTimeActivityLabels() const
        { return realtimeActivityLabels_; }

        bool UseFullBaz() const
        { return useFullBaz_; }

    public:
        Flags& SpiderOnSequel(bool v)
        {
            spiderOnSequel_ = v;
            return *this;
        }

        Flags& NewBazFormat(bool v)
        {
            newBazFormat_ = v;
            return *this;
        }

        Flags& UseHalfFloat(bool v)
        {
            useHalfFloat_ = v;
            return *this;
        }

        Flags& RealTimeActivityLabels(bool v)
        {
            realtimeActivityLabels_ = v;
            return *this;
        }

        Flags& UseFullBaz(bool v)
        {
            useFullBaz_ = v;
            return *this;
        }

    private:
        bool spiderOnSequel_;
        bool newBazFormat_;
        bool useHalfFloat_;
        bool realtimeActivityLabels_;
        bool useFullBaz_;
    };

public:
    FileHeaderBuilder(const std::string& movieName,
                      const float frameRateHz,
                      const uint32_t movieLengthFrames,
                      const SmrtData::Readout readout ,
                      const SmrtData::MetricsVerbosity metricsVerbosity,
                      const std::string& experimentMetadata,
                      const std::string& basecallerConfig,
                      const std::vector<uint32_t> zmwNumbers,
                      const std::vector<uint32_t> zmwUnitFeatures,
                      const uint32_t hFMetricFrames,
                      const uint32_t mFMetricFrames,
                      const uint32_t sliceLengthFrames,
                      const Flags& flags=Flags());

    // Move constructor
    FileHeaderBuilder(FileHeaderBuilder&&) = default;
    // Copy constructor
    FileHeaderBuilder(const FileHeaderBuilder&) = default;
    // Move assignment operator
    FileHeaderBuilder& operator=(FileHeaderBuilder&&) = default;
    // Copy assignment operator
    FileHeaderBuilder& operator=(const FileHeaderBuilder&) = delete;
    // Destructor
    ~FileHeaderBuilder() = default;

public:
    inline std::vector<PacketField> PacketFields() const
    { return packetFields_; }

    inline std::vector<MetricField> HFMetricFields() const
    { return hFMetricFields_; }

    inline std::vector<MetricField> MFMetricFields() const
    { return mFMetricFields_; }

    inline std::vector<MetricField> LFMetricFields() const
    { return lFMetricFields_; }

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

    inline uint32_t SliceLengthFrames() const
    { return sliceLengthFrames_; }

    inline double FrameRateHz() const
    { return frameRateHz_; }

    inline uint32_t HFMetricFrames() const
    { return hFMetricFrames_; }

    inline uint32_t MFMetricFrames() const
    { return mFMetricFrames_; }

    inline uint32_t LFMetricFrames() const
    { return lFMetricFrames_; }

    inline std::string P4Version() const
    { return p4Version_; }

    inline SmrtData::Readout ReadoutConfig() const
    { return readout_; }

    inline SmrtData::MetricsVerbosity MetricsVerbosityConfig() const
    { return metricsVerbosity_; }

    inline uint32_t MaxNumZmws() const
    { return zmwNumbers_.size(); }

    inline bool Complete() const
    { return complete_; }

    inline bool Truncated() const
    { return truncated_; }

    inline uint32_t NumSuperChunks() const
    { return numSuperChunks_; }

    inline uint32_t MovieLengthFrames() const
    { return movieLengthFrames_; }

    inline const std::vector<uint32_t>& ZmwNumbers() const
    { return zmwNumbers_; }

    inline uint64_t FileFooterOffset() const
    { return fileFooterOffset_; }

    inline double MovieLengthSeconds() const
    { return MovieLengthFrames()/ FrameRateHz(); }
public:
    std::string CreateJSON();

    std::vector<char> CreateJSONCharVector();

    inline void AddZmwNumber(const uint32_t number)
    { zmwNumbers_.emplace_back(number); }

    inline void ZmwNumbers(const std::vector<uint32_t>& numbers)
    { zmwNumbers_ = numbers; }

    inline void ZmwUnitFeatures(const std::vector<uint32_t>& unitFeatures)
    { zmwUnitFeatures_ = unitFeatures; }

    inline void ZmwNumberRejects(const std::vector<uint32_t>& rejects)
    { zmwNumberRejects_ = rejects; }

    inline void ClearPacketFields()
    { packetFields_.clear(); }

    void ClearAllMetricFields()
    {
        lFMetricFields_.clear();
        mFMetricFields_.clear();
        hFMetricFields_.clear();
    }
    void ClearMetricFields(const MetricFrequency& frequency);

    inline void AddPacketField(const PacketFieldName fieldName,
                               const uint8_t fieldBitSize)
    { packetFields_.emplace_back(fieldName, fieldBitSize); }

    void AddPacketField(const PacketFieldName fieldName, 
                     const uint8_t fieldBitSize,
                     const uint8_t fieldEscapeValue,
                     const PacketFieldName extensionName,
                     const uint8_t extensionBitSize)
    { packetFields_.emplace_back(fieldName, 
                                 fieldBitSize,
                                 fieldEscapeValue,
                                 extensionName,
                                 extensionBitSize); }

    void AddMetricField(const MetricFrequency& frequency,
                     const MetricFieldName fieldName, 
                     const uint8_t fieldBitSize,
                     const bool fieldSigned,
                     const uint16_t fieldScalingFactor);

    inline void PacketFields(const std::vector<PacketField>& packetFields)
    { packetFields_ = packetFields; }

    inline void HFMetricFields(const std::vector<MetricField>& hFMetricFields)
    { hFMetricFields_ = hFMetricFields; }

    inline void MFMetricFields(const std::vector<MetricField>& mFMetricFields)
    { mFMetricFields_ = mFMetricFields; }

    inline void LFMetricFields(const std::vector<MetricField>& lFMetricFields)
    { lFMetricFields_ = lFMetricFields; }

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

    inline void P4Version(const std::string& p4Version)
    { p4Version_ = p4Version; }

    inline void SliceLengthFrames(const uint32_t sliceLengthFrames)
    { sliceLengthFrames_ = sliceLengthFrames; }

    inline void FrameRateHz(const double frameRateHz)
    { frameRateHz_ = frameRateHz; }

    inline void HFMetricFrames(const uint32_t hFMetricFrames)
    { hFMetricFrames_ = hFMetricFrames; }

    inline void LFMetricFrames(const uint32_t lFMetricFrames)
    { lFMetricFrames_ = lFMetricFrames; }

    inline void Done()
    { complete_ = true; }

    inline void Truncated()
    { truncated_ = true; }

#if 0
    // I think this should be deprecated. The method of setting the numSuperChunks_
    // is IncSuperChunks(), not this.
    inline void NumSuperChunks(const uint32_t numSuperChunks)
    { numSuperChunks_ = numSuperChunks; }
#endif

    inline void IncSuperChunks()
    { ++numSuperChunks_; }

    inline void MovieLengthFrames(const uint32_t movieLengthFrames)
    { movieLengthFrames_ = movieLengthFrames; }

    inline void FileFooterOffset(const uint64_t fileFooterOffset)
    { fileFooterOffset_ = fileFooterOffset; }

    /// Conservative limit on the BAZ file size.
    /// \param averageBaseRatePerSecond - a empirical whole-chip estimate of the number of bases produced per ZMW. This includes loading efficiency, readlength, chemistry, etc.
    /// \returns an upper limit on the BAZ file size in bytes
    size_t ExpectedFileByteSize(double averageBaseRatePerSecond) const;

    /// Conservative limit on the BAZ file size due to pulses/bases (aka events)
    /// \param averageBaseRatePerSecond - a empirical whole-chip estimate of the number of bases produced per ZMW. This includes loading efficiency, readlength, chemistry, etc.
    /// \returns an upper limit on the BAZ file size in bytes due to bases alone.
    size_t EventsByteSize(double averageBaseRatePerSecond) const;

    /// Accurate limit on the BAZ file size due to metrics
    /// \returns an accurate contribution on the BAZ file size in bytes due to metrics
    size_t MetricsByteSize() const;

    /// Estimate of the bytes/second consumed by the current BAZ write operation
    double EstimatedBytesPerSecond(double averageBaseRatePerSecond) const;

    /// Estimate of the bytes/second due to event (bases or pulses) only
    double EventsByteRate(double averageBaseRatePerSecond) const;

    /// Estimate of the bytes/second due to metrics only
    double MetricsByteRate() const;

    void EstimatesSummary(std::ostream& os, double averageBaseRatePerSecond) const;

    std::string EstimatesSummary(double averageBaseRatePerSecond) const
    {
        std::stringstream ss;
        EstimatesSummary(ss, averageBaseRatePerSecond);
        return ss.str();
    }


private:
    std::vector<PacketField> packetFields_;
    std::vector<MetricField> hFMetricFields_;
    std::vector<MetricField> mFMetricFields_;
    std::vector<MetricField> lFMetricFields_;

    std::string movieName_;
    std::string basecallerVersion_;
    std::string bazWriterVersion_;
    std::string p4Version_;

    SmrtData::Readout readout_;
    SmrtData::MetricsVerbosity metricsVerbosity_;
    std::string experimentMetadata_;
    std::string basecallerConfig_;

    std::vector<uint32_t> zmwNumbers_;
    std::vector<uint32_t> zmwNumberRejects_;
    std::vector<uint32_t> zmwUnitFeatures_;

    uint32_t bazMajorVersion_;
    uint32_t bazMinorVersion_;
    uint32_t bazPatchVersion_;
    uint32_t sliceLengthFrames_;
    double frameRateHz_;
    uint32_t hFMetricFrames_;
    uint32_t mFMetricFrames_;
    uint32_t lFMetricFrames_;

    bool complete_ = false;
    bool truncated_ = false;
    uint32_t numSuperChunks_ = 0;
    uint32_t movieLengthFrames_ = 0;

    uint64_t fileFooterOffset_ = 0;

    Flags flags_;

private:
    void Default();

    bool SanityCheckMetricBlockSizes();

    void DefaultBases();
    void DefaultPulses();

    void DefaultMetricsMinimal();
    void DefaultMetricsHigh();

    void DefaultMetricsSequel(MetricFrequency frequency = MetricFrequency::MEDIUM);
    void DefaultMetricsSpider(MetricFrequency frequency = MetricFrequency::MEDIUM);
    void DefaultMetricsSpiderRTAL(MetricFrequency frequency = MetricFrequency::MEDIUM);

    void AddMetricsToJson(Json::Value& header,
                          const std::vector<MetricField>& metrics,
                          const int frames,
                          const MetricFrequency& frequency);

    void AddPacketsToJson(Json::Value& header,
                          const std::vector<PacketField>& packets);
protected:
    double PacketSize() const;
    size_t MetricBlockSize(const std::vector<MetricField>& metricFields) const;
    size_t HfMetricBlockSize() const;
    size_t MfMetricBlockSize() const;
    size_t LfMetricBlockSize() const;
    size_t OverheadSize() const;
    size_t HeaderSize() const;
    size_t PaddingSize() const;
};

}}
