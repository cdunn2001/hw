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

#include <bazio/file/FileFooterSet.h>
#include <bazio/file/FileHeaderSet.h>

#include "DataParsing.h"
#include "ZmwSliceHeader.h"

namespace PacBio {
namespace Primary {

/// Reads BAZ file
class BazReader
{
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
    ~BazReader() = default;

public:
    // Provides the ZMW ids of the next slice.
    std::vector<uint32_t> NextZmwIds();

    /// Provides the next slice of stitched ZMWs.
    std::vector<ZmwByteData> NextSlice(const std::function<bool(void)>& callBackCheck=nullptr);

    // Pops next slice from the queue of ZMW slices.
    inline void SkipNextSlice()
    {
        if (!zmwSlices_.empty())
        {
            const auto& slice = zmwSlices_.front();
            zmwSlices_.pop();

            assert(headerReader_.HasMoreHeaders());
            headerReader_.SkipCount(slice.endZmw - slice.startZmw);
        }
    }

public:
    /// Returns true if more ZMWs are available for reading.
    bool HasNext() const
    { return !zmwSlices_.empty(); }

    /// Returns reference to current file header set
    const BazIO::FileHeaderSet& FileHeaderSet() const
    { return *fh_; }

    /// Returns reference to current file footer set
    const BazIO::FileFooterSet& FileFooterSet() const
    { return *ff_; }

    /// Number of Zmws
    uint32_t NumZmws() const
    { return numZmws_; }

public: // Debugging
    std::vector<std::vector<ZmwSliceHeader>> SuperChunkToZmwHeaders() const;

private:

    struct ZmwSliceInfo
    {
        size_t startZmw;
        size_t endZmw;
        uint32_t numSuperChunks;
        std::FILE* fp;
    };

    /// Parses and provides the file headers from the file streams
    void ReadFileHeaders();

    /// Parses and provides the file footers from the file streams
    void ReadFileFooters();

private:   // data
    std::vector<std::pair<std::string,std::unique_ptr<std::FILE,decltype(&std::fclose)>>> files_;
    uint32_t                numZmws_ = 0;
    std::vector<uint32_t>   zmwIndexByBazFile_;
    size_t                  currentBazIndex_ = 0;

    std::unique_ptr<BazIO::FileHeaderSet> fh_;
    std::unique_ptr<BazIO::FileFooterSet> ff_;

    std::queue<ZmwSliceInfo> zmwSlices_;

    // I thought about pulling this out to a separate file, but it's intimately
    // tied to the bazreader (via metaPositions and file), so it seems best still
    // as a private type.
    class HeaderReader
    {
    public:
        HeaderReader(const std::vector<std::vector<size_t>> metaPositionsFiles,
                     const std::vector<size_t>& maxNumZmws,
                     const std::vector<std::pair<std::string,std::FILE*>>& files,
                     size_t batchSizeMB, bool silent)
            : idx_(0)
            , curFile_(0)
            , maxNumZmws_(maxNumZmws)
            , currentMaxNumZmws_(maxNumZmws.front())
            , totalNumZmws_(std::accumulate(maxNumZmws.begin(), maxNumZmws.end(), 0))
            , batchSizeMB_(batchSizeMB)
            , silent_(silent)
            , files_(files)
            , metaPositionsFiles_(metaPositionsFiles)
        { }

        HeaderReader() = default;
        HeaderReader(const HeaderReader&) = delete;
        HeaderReader(HeaderReader&&) = default;
        HeaderReader& operator=(const HeaderReader&) = delete;
        HeaderReader& operator=(HeaderReader&&) = default;
    public:
        bool HasMoreHeaders() const { return idx_ < totalNumZmws_; }
        void SkipCount(size_t count)
        {
            idx_ = std::min(idx_ + count, totalNumZmws_);
            while (idx_ > currentMaxNumZmws_ && curFile_ < files_.size())
            {
              NextFile();
            }
        }
        const std::vector<ZmwSliceHeader>& NextHeaders(const std::function<bool(void)>& callBackCheck=nullptr);
        void NextFile()
        {
            if (curFile_ + 1 < maxNumZmws_.size())
            {
                zmwOffset_ = currentMaxNumZmws_;
                curFile_ = curFile_ + 1;
                currentMaxNumZmws_ += maxNumZmws_[curFile_];
            }
        }
    public:
        std::FILE* GetCurrentFilePointer()
        {   assert(curFile_ < files_.size());
            return files_[curFile_].second;
        }

        HeaderReader Clone() const
        {
            return HeaderReader(metaPositionsFiles_, maxNumZmws_, files_, batchSizeMB_, silent_);
        }
    private:
        void LoadNextBatch(const std::function<bool(void)>& callBackCheck=nullptr);

        size_t idx_;                        // Marker for current zmw metadata index returned
        size_t curFile_;                    // Marker for current file pointer
        std::vector<size_t> maxNumZmws_;    // Number of zmws in each file
        size_t currentMaxNumZmws_;          // Maximum number of zmws based on current file being processed
        size_t totalNumZmws_;               // Total number of ZMWs for all files
        size_t batchSizeMB_;                // Number of zmw metadata to load at once
        bool silent_;                       // controls stderr output

        std::vector<std::pair<std::string,std::FILE*>> files_;  // Opened and closed by BazReader
        std::vector<std::vector<size_t>> metaPositionsFiles_;   // Metadata locations for each file

        size_t firstLoaded_ = 0;        // Index of first zmw of current batch
        size_t zmwOffset_ = 0;          // Offset to compute absolute zmw index across all files
        std::vector<std::vector<ZmwSliceHeader>> loadedData_;
    };

    HeaderReader headerReader_;

    static constexpr size_t blocksize = 4000;
};

}} // namespace
