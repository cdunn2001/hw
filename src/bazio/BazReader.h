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
public: // structors
    /// Reads file header and super chunk header for indexed access.
    /// The callBackCheck is used to check if an abort has
    /// been sent as the parsing of the headers takes a substantial amount
    /// of time since the whole BAZ file must be accessed.
    BazReader(const std::string& fileName, size_t zmwBatchMB, size_t zmwHeaderBatchMB, bool silent,
              const std::function<bool(void)>& callBackCheck=nullptr);
    /// Simple constructor, if you don't care about IO settings
    BazReader(const std::string& fileName)
    : BazReader(fileName, 1000, 1000, true) {}
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

    /// Parses and provides the file header from the file stream
    std::unique_ptr<BazIO::FileHeader> ReadFileHeader();

    /// Parses and provides the file footer from the file stream
    std::unique_ptr<FileFooter> ReadFileFooter();

    // Pops next slice from the queue of ZMW slices.
    inline void SkipNextSlice()
    {
        if (!zmwSlices_.empty())
        {
            auto& slice = zmwSlices_.front();
            uint32_t start = slice.first;
            uint32_t end = slice.second;
            zmwSlices_.pop();

            assert(headerReader_.HasMoreHeaders());
            headerReader_.SkipCount(end - start);
        }
    }

public:
    /// Returns reference to current file header
    const BazIO::FileHeader& Fileheader();

    /// Returns reference to current file footer
    std::unique_ptr<FileFooter>& Filefooter();

    /// Number of ZMWs
    uint32_t NumZMWs() const;

    /// Number of Superchunks
    uint32_t NumSuperchunks() const;

public: // Debugging
    std::vector<std::vector<ZmwSliceHeader>> SuperChunkToZmwHeaders() const;

private:   // data
    FILE*          file_;
    uint32_t       numZMWs_ = 0;
    uint32_t       numSuperchunks_ = 0;
    std::unique_ptr<BazIO::FileHeader> fh_;
    std::unique_ptr<FileFooter> ff_;
    std::queue<std::pair<uint32_t, uint32_t>> zmwSlices_;

    // I thought about pulling this out to a separate file, but it's intimately
    // tied to the bazreader (via metaPositions and file), so it seems best still
    // as a private type.
    class HeaderReader
    {
    public:
        HeaderReader(const std::vector<size_t> metaPositions, size_t numZmws, FILE* file,
                     size_t batchSizeMB, bool silent)
            : idx_(0)
            , numZmw_(numZmws)
            , batchSizeMB_(batchSizeMB)
            , silent_(silent)
            , file_(file)
            , metaPositions_(metaPositions)
        {}

        HeaderReader() = default;
        HeaderReader(const HeaderReader&) = delete;
        HeaderReader(HeaderReader&&) = default;
        HeaderReader& operator=(const HeaderReader&) = delete;
        HeaderReader& operator=(HeaderReader&&) = default;

        bool HasMoreHeaders() const { return idx_ < numZmw_; }
        void SkipCount(size_t count) { idx_ = std::min(idx_ + count, numZmw_); }
        const std::vector<ZmwSliceHeader>& NextHeaders(const std::function<bool(void)>& callBackCheck=nullptr);

        HeaderReader Clone() const
        {
            return HeaderReader(metaPositions_, numZmw_, file_, batchSizeMB_, silent_);
        }
    private:
        void LoadNextBatch(const std::function<bool(void)>& callBackCheck=nullptr);

        size_t idx_;       // Marker for current metadata
        size_t numZmw_;    // Total number of zmw in file
        size_t batchSizeMB_; // number of zmw metadata to load at once
        bool silent_;      // controls stderr output
        FILE* file_;       // Opened and closed by BazReader
        std::vector<size_t> metaPositions_;

        size_t firstLoaded_ = 0;  // Index of first zmw of current batch
        std::vector<std::vector<ZmwSliceHeader>> loadedData_;
    };

    HeaderReader headerReader_;

    static constexpr size_t blocksize = 4000;
};

}} // namespace