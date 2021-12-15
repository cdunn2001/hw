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

#include <functional>
#include <numeric>
#include <queue>

#include <bazio/file/FileHeader.h>

#include "DataParsing.h"
#include "ZmwSliceHeader.h"
#include "FileFooter.h"

namespace PacBio {
namespace Primary {

/// Reads BAZ file
class BazReader
{
public:
    struct FileHeaders
    {
    public:
        FileHeaders(std::vector<std::unique_ptr<BazIO::FileHeader>>&& fhs);

        // Default constructor
        FileHeaders() = delete;
        // Move constructor
        FileHeaders(FileHeaders&&) = delete;
        // Copy constructor
        FileHeaders(const FileHeaders&) = delete;
        // Move assignment operator
        FileHeaders& operator=(FileHeaders&&) = delete;
        // Copy assignment operator
        FileHeaders& operator=(const FileHeaders&) = delete;
        // Destructor
        ~FileHeaders() = default;

    public:
        const std::vector<size_t>& NumZmws() const
        { return numZmws_; }

        uint32_t MaxNumZMWs() const
        { return maxNumZmws_; }

        uint32_t NumSuperChunks() const
        { return fhs_.front()->NumSuperChunks(); }

        uint32_t MetricByteSize() const
        { return fhs_.front()->MetricByteSize();; }

        uint32_t ZmwNumberToIndex(const uint32_t holeNumber) const
        { return zmwNumbersToIndex_.at(holeNumber); }

        uint32_t ZmwIndexToNumber(const uint32_t index) const
        { return zmwNumbers_.at(index); }

        const std::vector<uint32_t>& ZmwNumbers() const
        { return zmwNumbers_; }

        uint32_t ZmwFeatures(uint32_t zmwIndex) const
        { return zmwFeatures_.at(zmwIndex); }

        const std::vector<MetricField>& MetricFields() const
        { return fhs_.front()->MFMetricFields(); }

        const std::vector<BazIO::FieldParams<BazIO::PacketFieldName>>& PacketFields() const
        { return fhs_.front()->PacketFields(); }

        const std::vector<BazIO::GroupParams<BazIO::PacketFieldName>>& PacketGroups() const
        { return fhs_.front()->PacketGroups(); }

        uint32_t MetricFrames() const
        { return fhs_.front()->HFMetricFrames(); }

        double FrameRateHz() const
        { return fhs_.front()->FrameRateHz(); }

        const std::vector<float>& RelativeAmplitudes() const
        { return fhs_.front()->RelativeAmplitudes(); }

        const std::string& BaseMap() const
        { return fhs_.front()->BaseMap(); }

        const std::string& MovieName() const
        { return movieName_; }

        const Json::Value ExperimentMetadata() const
        { return fhs_.front()->ExperimentMetadata(); }

        const BazIO::FileHeader& FileHeaderNo(const size_t fileNo) const
        { return *fhs_.at(fileNo); }

        size_t NumHeaders() const
        { return fhs_.size(); }

        std::string BaseCallerVersion() const
        { return fhs_.front()->BaseCallerVersion(); }

        std::string BazVersion() const
        { return fhs_.front()->BazVersion(); }

        std::string BazWriterVersion() const
        { return fhs_.front()->BazWriterVersion(); }

        float MovieTimeInHrs() const
        { return fhs_.front()->MovieTimeInHrs(); }

        bool Internal() const
        { return fhs_.front()->Internal(); }

        bool HasPacketField(BazIO::PacketFieldName fieldName) const
        { return fhs_.front()->HasPacketField(fieldName); }

        bool IsZmwNumberRejected(uint32_t zmwNumber) const
        {
            return std::any_of(fhs_.begin(), fhs_.end(),
                               [&](const auto& fh) { return fh->IsZmwNumberRejected(zmwNumber); });
        }

        float MovieLengthFrames() const
        { return fhs_.front()->MovieLengthFrames(); }

    private:
        std::vector<size_t> numZmws_;
        uint32_t maxNumZmws_ = 0;
        uint32_t numSuperChunks_ = 0;
        std::map<uint32_t,uint32_t> zmwNumbersToIndex_;
        std::vector<uint32_t> zmwNumbers_;
        std::vector<uint32_t> zmwFeatures_;
        std::string movieName_;
        std::vector<std::unique_ptr<BazIO::FileHeader>> fhs_;
    };

    class FileFooters
    {
    public:
        FileFooters(std::vector<std::unique_ptr<FileFooter>>&& ffs, const std::vector<size_t>& numZmws);

        // Default constructor
        FileFooters() = delete;
        // Move constructor
        FileFooters(FileFooters&&) = delete;
        // Copy constructor
        FileFooters(const FileFooters&) = delete;
        // Move assignment operator
        FileFooters& operator=(FileFooters&&) = delete;
        // Copy assignment operator
        FileFooters& operator=(const FileFooters&) = delete;
        // Destructor
        ~FileFooters() = default;
    public:
        bool IsZmwTruncated(uint32_t zmwId) const
        { return truncationMap_.find(zmwId) != truncationMap_.cend(); }
    private:
        std::map<uint32_t, std::vector<uint32_t>> truncationMap_;
        std::vector<std::unique_ptr<FileFooter>> ffs_;
    };

public: // structors
    /// Reads file header and super chunk header for indexed access.
    /// The callBackCheck is used to check if an abort has
    /// been sent as the parsing of the headers takes a substantial amount
    /// of time since the whole BAZ file must be accessed.
    BazReader(const std::vector<std::string>& fileNames, size_t zmwBatchMB, size_t zmwHeaderBatchMB, bool silent,
              const std::function<bool(void)>& callBackCheck=nullptr);
    /// Simple constructor, if you don't care about IO settings
    BazReader(const std::vector<std::string>& fileNames)
    : BazReader(fileNames, 1000, 1000, true) {}
    // Default constructor
    BazReader() = delete;
    // Move constructor
    BazReader(BazReader&&) = delete;
    // Copy constructor
    BazReader(const BazReader&) = delete;
    // Move assignment operator
    BazReader& operator=(BazReader&&) = delete;
    // Copy assignment operator
    BazReader& operator=(const BazReader&) = delete;
    // Destructor
    ~BazReader();

public:
    /// Returns true if more ZMWs are available for reading.
    bool HasNext();

    // Provides the ZMW ids of the next slice.
    std::vector<uint32_t> NextZmwIds();

    /// Provides the next slice of stitched ZMWs.
    std::vector<ZmwByteData> NextSlice(const std::function<bool(void)>& callBackCheck=nullptr);

    /// Parses and provides the file headers from the file streams
    void ReadFileHeaders();

    /// Parses and provides the file footers from the file streams
    void ReadFileFooters(const std::vector<size_t>& numZmws);

    // Pops next slice from the queue of ZMW slices.
    inline void SkipNextSlice()
    {
        if (!zmwSlices_.empty())
        {
            auto& slice = zmwSlices_.front();
            uint32_t start = std::get<0>(slice);
            uint32_t end = std::get<1>(slice);
            zmwSlices_.pop();

            assert(headerReader_.HasMoreHeaders());
            headerReader_.SkipCount(end - start);
        }
    }

public:
    /// Returns reference to current file header
    const FileHeaders& Fileheader() const
    {
        if (!fh_) throw PBException("Null FileHeader!");
        return *fh_;
    }

    /// Returns reference to current file footer
    const FileFooters& Filefooter() const
    {
        if (!ff_) throw PBException("Null FileFooter!");
        return *ff_;
    }

    /// Number of ZMWs
    uint32_t NumZMWs() const;

    /// Number of Superchunks
    uint32_t NumSuperchunks() const;

public: // Debugging
    std::vector<std::vector<ZmwSliceHeader>> SuperChunkToZmwHeaders() const;

private:   // data
    std::vector<std::string> fileNames_;
    std::vector<std::shared_ptr<std::FILE>> files_;
    uint32_t            numZMWs_ = 0;
    uint32_t            numSuperchunks_ = 0;
    std::unique_ptr<FileHeaders> fh_;
    std::unique_ptr<FileFooters> ff_;

    std::queue<std::tuple<uint32_t, uint32_t>> zmwSlices_;

    // I thought about pulling this out to a separate file, but it's intimately
    // tied to the bazreader (via metaPositions and file), so it seems best still
    // as a private type.
    class HeaderReader
    {
    public:
        HeaderReader(const std::vector<std::vector<size_t>> metaPositionsFiles, const std::vector<size_t>& numZmw, const std::vector<std::shared_ptr<std::FILE>>& files,
                     size_t batchSizeMB, bool silent)
            : idx_(0)
            , curFile_(0)
            , numZmw_(numZmw)
            , currentMaxZmw_(numZmw.front())
            , totalZmw_(std::accumulate(numZmw.begin(), numZmw.end(), 0))
            , batchSizeMB_(batchSizeMB)
            , silent_(silent)
            , files_(files)
            , metaPositionsFiles_(metaPositionsFiles)
            , zmwIdxToFileIdx_(totalZmw_)
        { }

        HeaderReader() = default;
        HeaderReader(const HeaderReader&) = delete;
        HeaderReader(HeaderReader&&) = default;
        HeaderReader& operator=(const HeaderReader&) = delete;
        HeaderReader& operator=(HeaderReader&&) = default;

        bool HasMoreHeaders() const { return idx_ < totalZmw_; }
        void SkipCount(size_t count)
        {
            while (idx_ + count > currentMaxZmw_ && curFile_ < files_.size())
            {
              NextFile();
            }
            idx_ = std::min(idx_ + count, totalZmw_);
        }
        const std::vector<ZmwSliceHeader>& NextHeaders(const std::function<bool(void)>& callBackCheck=nullptr);

        HeaderReader Clone() const
        {
            return HeaderReader(metaPositionsFiles_, numZmw_, files_, batchSizeMB_, silent_);
        }

        void NextFile()
        {
            zmwOffset_ = currentMaxZmw_;
            curFile_++;
            currentMaxZmw_ += numZmw_[curFile_];
        }

        size_t FileIdx(size_t zmwIndex) const
        {
            assert(zmwIndex < zmwIdxToFileIdx_.size());
            return zmwIdxToFileIdx_.at(zmwIndex);
        }
    private:
        void LoadNextBatch(const std::function<bool(void)>& callBackCheck=nullptr);

        size_t idx_;                    // Marker for current metadata
        size_t curFile_;                // Marker for current file
        std::vector<size_t> numZmw_;    // Number of zmws in each file
        size_t currentMaxZmw_;          // Maximum number of zmws based on current file being processed
        size_t totalZmw_;               // Total number of ZMWs for all files
        size_t batchSizeMB_;            // number of zmw metadata to load at once
        bool silent_;                   // controls stderr output
        std::vector<std::shared_ptr<std::FILE>> files_;       // Opened and closed by BazReader
        std::vector<std::vector<size_t>> metaPositionsFiles_;
        std::vector<size_t> zmwIdxToFileIdx_;   // Maps zmwIndex to fileIdx

        size_t firstLoaded_ = 0;        // Index of first zmw of current batch
        size_t zmwOffset_ = 0;
        std::vector<std::vector<ZmwSliceHeader>> loadedData_;
    };

    HeaderReader headerReader_;

    static constexpr size_t blocksize = 4000;
};

}} // namespace
