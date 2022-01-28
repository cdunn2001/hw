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

#include <algorithm>
#include <cstring>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <stdlib.h>
#include <string>
#include <typeinfo>
#include <unordered_map>

#include <pacbio/logging/Logger.h>
#include <pacbio/text/String.h>

#include "BazCore.h"
#include "SmartMemory.h"
#include "Timing.h"
#include "BazReader.h"

namespace PacBio {
namespace Primary
{

/// Reads file header and super chunk header for indexed access.
BazReader::BazReader(const std::vector<std::string>& fileNames,
                     size_t zmwBatchMB,
                     size_t zmwHeaderBatchMB,
                     bool silent,
                     const std::function<bool(void)>& callBackCheck)
    : fileNames_(fileNames)
{
    if (callBackCheck && callBackCheck()) return;

    const size_t zmwBatchBytes = zmwBatchMB << 20;

    // Init
    for (size_t i = 0; i < fileNames.size(); ++i)
    {
        files_.emplace_back(std::shared_ptr<FILE>(std::fopen(fileNames[i].c_str(), "rb"), std::fclose));
        if (!files_[i]) throw std::runtime_error("Can't open " + fileNames[i] + " with BazReader");
    }

    // Timing
    // auto now = std::chrono::high_resolution_clock::now();

    // Read file headers from files
    if (callBackCheck && callBackCheck()) return;
    ReadFileHeaders();
    if (callBackCheck && callBackCheck()) return;
    ReadFileFooters(fh_->NumZmws());
    if (callBackCheck && callBackCheck()) return;

    numZMWs_ = fh_->MaxNumZMWs();
    numSuperchunks_ = fh_->NumSuperChunks();

    std::vector<std::vector<size_t>> zmwMetaLocationsFiles_;
    zmwMetaLocationsFiles_.reserve(files_.size());
    for (size_t i = 0; i < files_.size(); ++i)
    {
        // Position indicator (pi) to for next chunk
        auto pi = std::ftell(files_[i].get());

        // 4k alignment of binary data start
        if (pi % blocksize != 0)
            pi += blocksize - pi % blocksize;

        if (!silent)
        {
            std::cerr << "Finding Metadata locations\n";
            std::cerr.flush();
        }

        std::vector<size_t> zmwMetaLocations;
        zmwMetaLocations.reserve(NumSuperchunks());
        while (true)
        {
            if (callBackCheck && callBackCheck()) return;
            // Seek to next super-chunk
            fseek(files_[i].get(), pi, SEEK_SET);
            // Check sanity dword
            int counter = 0;
            while (!Sanity::ReadAndVerify(files_[i].get()))
            {
                // Seek one block further
                pi += blocksize - pi % blocksize;
                fseek(files_[i].get(), pi, SEEK_SET);
                // Only try three times and then fail.
                ++counter;
                if (counter == 3)
                {
                    PBLOG_ERROR << "Corrupt file. Cannot read SANITY DWORD before CHUNK_META";
                    exit(EXIT_FAILURE);
                }
                if (callBackCheck && callBackCheck()) return;
            }

            // Read super-chunk meta information
            SuperChunkMeta chunkMeta;
            // Read offset
            if (fread(&chunkMeta.offsetNextChunk, sizeof(SuperChunkMeta::offsetNextChunk), 1, files_[i].get()) != 1)
            {
                PBLOG_ERROR << "Corrupt file. Cannot read SUPER_CHUNK_META offset";
                exit(EXIT_FAILURE);
            }
            // Read numZmws
            if (fread(&chunkMeta.numZmws, sizeof(SuperChunkMeta::numZmws), 1, files_[i].get()) != 1)
            {
                PBLOG_ERROR << "Corrupt file. Cannot read SUPER_CHUNK_META numZmws";
                exit(EXIT_FAILURE);
            }

            // Check sanity dword
            if (!Sanity::ReadAndVerify(files_[i].get()))
            {
                PBLOG_ERROR << "Corrupt file. Cannot read SANITY DWORD after CHUNK_META";
                exit(EXIT_FAILURE);
            }

            // Set pi to next super-chunk offset
            pi = chunkMeta.offsetNextChunk;

            // If next offset is zero, stop.
            // This is used to mark EOF.
            if (pi == 0) break;

            assert(chunkMeta.numZmws == fh_->FileHeaderNo(i).MaxNumZMWs());
            zmwMetaLocations.emplace_back(ftell(files_[i].get()));
            if (callBackCheck && callBackCheck()) return;
        }
        zmwMetaLocationsFiles_.emplace_back(zmwMetaLocations);
    }

    for (const auto& a : zmwMetaLocationsFiles_)
    {
        (void)a;
        assert(a.size() == NumSuperchunks());
    }

    if (callBackCheck && callBackCheck()) return;

    // We're going to parse the header once upfront to figure out how large each batch
    // of reads will be.  Force silent mode to not pollute the log with extra header
    // reading messages
    headerReader_ = HeaderReader(zmwMetaLocationsFiles_, fh_->NumZmws(), files_, zmwHeaderBatchMB, true);

    // Store [first, last) zmw per slice
    uint32_t first = 0;
    size_t sliceBytes = 0;
    for (size_t zmw = 0; zmw < numZMWs_; zmw++)
    {
        if (callBackCheck && callBackCheck()) return;
        assert(headerReader_.HasMoreHeaders());
        const auto& headers = headerReader_.NextHeaders(callBackCheck);
        if (callBackCheck && callBackCheck()) return;
        size_t zmwBytes = 0;
        for (const auto& header : headers)
        {
            zmwBytes += header.packetsByteSize;
            zmwBytes += header.numMBs * fh_->MetricByteSize();
        }
        // If the current zmw fits within our quota, add it to the slice.
        // If it doesn't fit, but it's the first one, still add it to
        // the slice so we don't have an empty slice.  Need to still
        // limit our slice size to some sensible maximum, because we
        // have some vectors of vectors that scale with both movie length
        // and slice size.  They can add some extra memory pressure that slows
        // us down if the slice size is too large.
        static constexpr size_t maxSliceSize = 100000;
        const bool fitsInBatchBytes = sliceBytes + zmwBytes < zmwBatchBytes;
        const bool fitsInMaxSlice = (zmw - first) < maxSliceSize;
        if ((fitsInBatchBytes && fitsInMaxSlice) || zmw == first)
        {
            sliceBytes += zmwBytes;
        }
        else
        {
            // The current zmw does not fit in the current slice, so tie off the
            // current slice and start accumulating a new one
            assert(first != zmw);
            zmwSlices_.push({first, zmw});
            first = zmw;
            sliceBytes = zmwBytes;
        }
    }
    // Need to tie off the last slice.  We're guaranteed to have one in progress
    assert(sliceBytes != 0 || numZMWs_ == 0);
    zmwSlices_.push({first, numZMWs_});

    // Re-start the header reader, since we'll need the header information again while actually
    // reading in zmw data
    headerReader_ = HeaderReader(zmwMetaLocationsFiles_, fh_->NumZmws(), files_, zmwHeaderBatchMB, silent);
}

const std::vector<ZmwSliceHeader>& BazReader::HeaderReader::NextHeaders(const std::function<bool(void)>& callBackCheck)
{
    assert(idx_ >= firstLoaded_);
    if (idx_ - firstLoaded_ >= loadedData_.size() || loadedData_.empty())
        LoadNextBatch(callBackCheck);
    assert(idx_ - firstLoaded_ < loadedData_.size());
    const auto& ret = loadedData_[idx_ - firstLoaded_];
    idx_++;
    return ret;
}

void BazReader::HeaderReader::LoadNextBatch(const std::function<bool(void)>& callBackCheck)
{
    PBLOG_INFO << "Beginning read of next header batch";
    constexpr size_t sizeSingleHeader = ZmwSliceHeader::SizeOf();
    firstLoaded_ = idx_;
    // Make minimum 1 to avoid division by 0
    size_t zmwHeadersSize = std::max(sizeSingleHeader * metaPositionsFiles_[curFile_].size(), size_t(1));
    // Load as many zmw as fit, but make sure we load at least one.
    size_t numLoad = std::max((batchSizeMB_ << 20) / zmwHeadersSize,
                               size_t(1u));
    size_t end = idx_ + numLoad;

    // Load zmws up to the current file boundary.
    bool nextFile = false;
    if (end > currentMaxZmw_)
    {
        end = currentMaxZmw_;
        numLoad = end - firstLoaded_;
        nextFile = true;
    }
    assert(numLoad > 0);

    loadedData_.resize(0);
    for (size_t i = 0; i < numLoad; ++i)
    {
        loadedData_.emplace_back(metaPositionsFiles_[curFile_].size());
    }

    size_t headerId = 0;
    for (const auto& location : metaPositionsFiles_[curFile_])
    {
        if (callBackCheck && callBackCheck()) return;

        // Remap the zmw file offset.
        fseek(files_[curFile_].get(), location + (firstLoaded_ - zmwOffset_) * sizeSingleHeader, SEEK_SET);

        std::vector<uint8_t> chunkHeadersMemory(sizeSingleHeader * numLoad);
        if (!fread(chunkHeadersMemory.data(), chunkHeadersMemory.size(), 1, files_[curFile_].get()))
        {
            PBLOG_ERROR << "Corrupt file. Cannot read ZMW_SLICE_HEADER";
            exit(EXIT_FAILURE);
        }

        size_t bytesRead = 0;
        for (size_t i = 0; i < numLoad; ++i)
        {
            if (callBackCheck && callBackCheck()) return;

            ZmwSliceHeader& header = loadedData_[i][headerId];

            // Read package offset
            memcpy(&header.offsetPacket,
                   &chunkHeadersMemory[bytesRead],
                   sizeof(ZmwSliceHeader::offsetPacket));
            bytesRead += sizeof(ZmwSliceHeader::offsetPacket);
            // Read ZMW ID which will be sequential in each BAZ file starting at 0.
            // We will remap them so that they are sequential across BAZ files starting at 0.
            memcpy(&header.zmwIndex,
                   &chunkHeadersMemory[bytesRead],
                   sizeof(ZmwSliceHeader::zmwIndex));
            header.zmwIndex += zmwOffset_;
            assert(header.zmwIndex == i + firstLoaded_);
            zmwIdxToFileIdx_[header.zmwIndex] = curFile_;
            bytesRead += sizeof(ZmwSliceHeader::zmwIndex);

            // Read number of byte for current packet stream
            memcpy(&header.packetsByteSize,
                   &chunkHeadersMemory[bytesRead],
                   sizeof(ZmwSliceHeader::packetsByteSize));
            bytesRead += sizeof(ZmwSliceHeader::packetsByteSize);
            // Read number of events (bases/pulses)
            memcpy(&header.numEvents,
                   &chunkHeadersMemory[bytesRead],
                   sizeof(ZmwSliceHeader::numEvents));
            bytesRead += sizeof(ZmwSliceHeader::numEvents);
            // Read number of metric blocks
            memcpy(&header.numMBs,
                   &chunkHeadersMemory[bytesRead],
                   sizeof(ZmwSliceHeader::numMBs));
            bytesRead += sizeof(ZmwSliceHeader::numMBs);
        }

        ++headerId;
    }

    if (nextFile)
    {
        NextFile();
    }

    if (!silent_)
        std::cerr << "Read " << headerId << " SUPER_CHUNK_META" << std::endl;

    PBLOG_INFO << "Finished read of next header batch";
}

// Destructor
BazReader::~BazReader()
{ }

bool BazReader::HasNext()
{ return !zmwSlices_.empty(); }

std::vector<uint32_t> BazReader::NextZmwIds()
{
    // Get first ZMW for current slice
    size_t startZmw = std::get<0>(zmwSlices_.front());
    size_t end = std::get<1>(zmwSlices_.front());

    std::vector<uint32_t> list;
    list.reserve(end - startZmw);

    while (startZmw < end)
        list.push_back(startZmw++);

    std::sort(list.begin(), list.end());
    return list;
}

std::vector<ZmwByteData> BazReader::NextSlice(const std::function<bool(void)>& callBackCheck)
{
    if (callBackCheck && callBackCheck()) return std::vector<ZmwByteData>{};
    PBLOG_INFO << "Beginning read of next slice";
    // Timing
    // auto now3 = std::chrono::high_resolution_clock::now();

    // Get first ZMW for current slice
    size_t startZmw = std::get<0>(zmwSlices_.front());
    size_t end = std::get<1>(zmwSlices_.front());
    size_t sliceSize = end - startZmw;

    zmwSlices_.pop();
    // vector to store reads
    std::vector<ZmwDataCounts> dataSizes(sliceSize);

    std::vector<std::vector<ZmwSliceHeader>> chunkToHeaderSplit;
    chunkToHeaderSplit.reserve(NumSuperchunks());
    for (size_t i = 0; i < NumSuperchunks(); ++i)
    {
        chunkToHeaderSplit.emplace_back();
        chunkToHeaderSplit.back().reserve(sliceSize);
    }

    // Iterate over all chunks for ZMWs in current slice,
    // to save length of base calls and HQ regions.
    for (size_t i = startZmw; i < end; ++i)
    {
        assert(headerReader_.HasMoreHeaders());
        const auto& zmwHeaders = headerReader_.NextHeaders(callBackCheck);
        if (callBackCheck && callBackCheck()) return std::vector<ZmwByteData>{};
        for (size_t j = 0; j < NumSuperchunks(); ++j)
        {
            const auto& h = zmwHeaders[j];
            assert(i == h.zmwIndex);
            dataSizes[i - startZmw].packetsByteSize += h.packetsByteSize;
            dataSizes[i - startZmw].numMBs += h.numMBs;
            chunkToHeaderSplit[j].push_back(h);
        }
    }

    std::vector<ZmwByteData> batchByteData;
    batchByteData.reserve(sliceSize);
    for (size_t i = 0; i < dataSizes.size(); ++i)
        batchByteData.emplace_back(fh_->MetricByteSize(), dataSizes[i], i + startZmw);

    // Iterate over all chunks
    // super-chunk id -> ZmwSliceHeader
    for (const auto& singleChunk : chunkToHeaderSplit)
    {
        if (callBackCheck && callBackCheck()) return std::vector<ZmwByteData>{};

        // It could be that in this chunk, there are not ZMW stitched
        if (singleChunk.empty()) continue;

        // Current file index
        size_t currentFileIdx = headerReader_.FileIdx(singleChunk[0].zmwIndex);

        // Seek to first ZmwSlice
        auto seekResult = fseek(files_[currentFileIdx].get(), singleChunk[0].offsetPacket, SEEK_SET);
        if (seekResult != 0)
            throw std::runtime_error("(" + std::to_string(seekResult)
                                     + ")Cannot seek ahead to : " + std::to_string(singleChunk[0].offsetPacket));

        // Bytes written
        uint64_t bytesRead = 0;
        // Number of spacer bytes to jump
        int64_t jump = -1;
        // Position of the last ZMW offset 
        uint64_t oldPos = 0;
        // Iterate over all ZMWs of this slice in this chunk
        for (const auto& h : singleChunk)
        {
            if (headerReader_.FileIdx(h.zmwIndex) != currentFileIdx)
            {
                currentFileIdx = headerReader_.FileIdx(h.zmwIndex);
                // Seek to first ZmwSlice
                auto seekResult = fseek(files_[currentFileIdx].get(), h.offsetPacket, SEEK_SET);
                if (seekResult != 0)
                    throw std::runtime_error("(" + std::to_string(seekResult)
                                             + ")Cannot seek ahead to : " + std::to_string(h.offsetPacket));
            }

            // Calculate padding bytes to jump
            if (oldPos != 0)
            {
                jump = h.offsetPacket - oldPos - bytesRead;
                bytesRead = 0;
                if (jump > 0)
                {
                    switch (jump)
                    {
                        case 3:
                            std::fgetc(files_[currentFileIdx].get());
                        case 2:
                            std::fgetc(files_[currentFileIdx].get());
                        case 1:
                            std::fgetc(files_[currentFileIdx].get());
                            break;
                        default:
                            auto pos = fseek(files_[currentFileIdx].get(), h.offsetPacket, SEEK_SET);
                            if (pos != 0)
                                throw std::runtime_error("(" + std::to_string(pos)
                                                         + ")Cannot seek ahead to : " +
                                                         std::to_string(h.offsetPacket));
                            break;
                    }
                }
            }
            oldPos = h.offsetPacket;
            // Check sanity dword
            if (!Sanity::ReadAndVerify(files_[currentFileIdx].get()))
                throw std::runtime_error("Cannot find SANITY DWORD before ZMW chunk");
            bytesRead += 4;

            // We stitch ZMWs in a SingleZMW struct
            assert(h.zmwIndex >= startZmw);
            auto& s = batchByteData[h.zmwIndex - startZmw];

            if (h.numEvents > 0)
            {
                // Packets and pulse extensions are stored in consective memory
                auto* data = s.NextPacketBytes(h.packetsByteSize, h.numEvents);
                assert(data != nullptr);
                auto readResult = fread(data, 1, h.packetsByteSize, files_[currentFileIdx].get());
                if (readResult != h.packetsByteSize)
                    throw std::runtime_error("Error reading");
                bytesRead += h.packetsByteSize;
            }
            if (h.numMBs > 0)
            {
                auto* data = s.NextMetricBytes(h.numMBs * fh_->MetricByteSize());
                assert(data != nullptr);
                auto readResult = fread(data, 1, fh_->MetricByteSize() * h.numMBs, files_[currentFileIdx].get());
                if (readResult != fh_->MetricByteSize() * h.numMBs)
                    throw std::runtime_error("Error reading");
                bytesRead += fh_->MetricByteSize() * h.numMBs;
            }
        }
    }

    PBLOG_INFO << "Finished read of next slice";

    // TODO: This is necessary to handle some testing code that produces a baz
    // file with 0 superchunks.  It's arguable that the code path producing
    // such a file should really throw an exception and terminate instead.
    // May need to refactor the code to remove the `Error` function, and
    // update tests accordingly
    if (NumSuperchunks() == 0)
        return std::vector<ZmwByteData>{};

    return batchByteData;
}

void BazReader::ReadFileHeaders()
{
    std::vector<std::unique_ptr<BazIO::FileHeader>> fhs;
    for (size_t i = 0; i < files_.size(); ++i)
    {
        // Seek for SANITY block and last position of the file header
        uint64_t headerSize = Sanity::FindAndVerify(files_[i].get());

        // Wrap header into smrt pointer
        auto header = SmartMemory::AllocMemPtr<char>(headerSize + 1);

        // Set position indicator to beginning
        std::rewind(files_[i].get());

        // Read file header
        size_t result = std::fread(header.get(), 1, headerSize, files_[i].get());
        if (result != headerSize)
            throw std::runtime_error("Cannot read file header!");

        auto fh = std::unique_ptr<BazIO::FileHeader>(new BazIO::FileHeader(header.get(), headerSize));

        // Check if BAZ version is correct
        if (!(fh->BazMajorVersion() == 2 && fh->BazMinorVersion() == 0))
        {
            PBLOG_ERROR << "Incorrect BAZ version provided. Need version 2.0.x, provided version is "
                        << fh->BazVersion();
            exit(EXIT_FAILURE);
        }

        if (!fh->Complete())
        {
            PBLOG_ERROR << "Trying to read unfinished baz file " << fileNames_[i];
            exit(EXIT_FAILURE);
        }

        if (fh->Truncated())
            PBLOG_INFO << "Converting truncated file " << fileNames_[i];

        // Check sanity dword
        if (!Sanity::ReadAndVerify(files_[i].get()))
        {
            PBLOG_ERROR << "Corrupt file. Cannot read SANITY DWORD after FILE_HEADER";
            exit(EXIT_FAILURE);
        }

        fhs.push_back(std::move(fh));
    }

    if(!std::equal(fhs.begin() + 1, fhs.end(), fhs.begin(),
                   [](const std::unique_ptr<BazIO::FileHeader>& a, const std::unique_ptr<BazIO::FileHeader>& b)
                   { return a->IsConsistent(*b); }))
        throw PBException("Inconsistent file headers detected, check files!");

    fh_ = std::make_unique<FileHeaders>(std::move(fhs));
}

BazReader::FileHeaders::FileHeaders(std::vector<std::unique_ptr<BazIO::FileHeader>>&& fhs)
    : fhs_(std::move(fhs))
{
    for (size_t i = 0; i < fhs_.size(); ++i)
    {
        // Adjust index since each BAZ file stores it sequentially starting at 0.
        for (const auto& kv : fhs_[i]->ZmwInformation().ZmwNumbersToIndex())
            zmwNumbersToIndex_[kv.first] = maxNumZmws_ + kv.second;

        zmwNumbers_.insert(std::end(zmwNumbers_),
                           fhs_[i]->ZmwNumbers().begin(), fhs_[i]->ZmwNumbers().end());

        zmwFeatures_.insert(std::end(zmwFeatures_),
                            fhs_[i]->ZmwUnitFeatures().begin(), fhs_[i]->ZmwUnitFeatures().end());

        numZmws_.push_back(fhs_[i]->MaxNumZMWs());
        maxNumZmws_ += fhs_[i]->MaxNumZMWs();
    }

    movieName_ = PacBio::Text::String::Split(fhs_.front()->MovieName(), '.')[0];
}

void BazReader::ReadFileFooters(const std::vector<size_t>& numZmws)
{
    std::vector<std::unique_ptr<FileFooter>> ffs;
    for (size_t i = 0; i < fh_->NumHeaders(); ++i)
    {
        if (fh_->FileHeaderNo(i).FileFooterOffset() == 0)
        {
            ffs.emplace_back(std::unique_ptr<FileFooter>(new FileFooter()));
        }
        else
        {
            auto pi = std::ftell(files_[i].get());

            // Go to last Footer
            const auto fileFooterOffset = fh_->FileHeaderNo(i).FileFooterOffset();
            fseek(files_[i].get(), fileFooterOffset, SEEK_SET);

            // Seek for SANITY block and last position of the file header
            uint64_t footerEnd = Sanity::FindAndVerify(files_[i].get());
            int64_t footerSize = static_cast<int64_t>(footerEnd) - static_cast<int64_t>(fh_->FileHeaderNo(i).FileFooterOffset());

            // Wrap footer into smrt pointer
            auto footer = SmartMemory::AllocMemPtr<char>(footerSize + 1);

            // Set position indicator to beginning of footer
            fseek(files_[i].get(), fileFooterOffset, SEEK_SET);

            // Read file footer
            size_t result = std::fread(footer.get(), 1, footerSize, files_[i].get());
            if (result != static_cast<uint64_t>(footerSize))
                throw std::runtime_error("Cannot read file footer!");

            fseek(files_[i].get(), pi, SEEK_SET);

            // Let Filefooter parse JSON footer
            ffs.emplace_back(std::unique_ptr<FileFooter>(new FileFooter(footer.get(), footerSize)));
        }
    }
    ff_ = std::make_unique<FileFooters>(std::move(ffs), numZmws);
}

BazReader::FileFooters::FileFooters(std::vector<std::unique_ptr<FileFooter>>&& ffs,
                                    const std::vector<size_t>& numZmws)
    : ffs_(std::move(ffs))
{
    assert(ffs_.size() == numZmws.size());

    size_t zmwIndexOffset = 0;
    for (size_t i = 0; i < ffs_.size(); ++i)
    {
        for (const auto& kv : ffs_[i]->TruncationMap())
        {
            truncationMap_[zmwIndexOffset + kv.first].insert(std::end(truncationMap_[zmwIndexOffset + kv.first]),
                                            kv.second.begin(), kv.second.end());
        }
        zmwIndexOffset = numZmws[i];
    }
}

uint32_t BazReader::NumZMWs() const
{ return numZMWs_; }

uint32_t BazReader::NumSuperchunks() const
{ return numSuperchunks_; }

std::vector<std::vector<ZmwSliceHeader>> BazReader::SuperChunkToZmwHeaders() const
{
    std::vector<std::vector<ZmwSliceHeader>> ret(NumSuperchunks());
    for (size_t i = 0; i < NumSuperchunks(); ++i)
        ret[i].resize(NumZMWs());

    auto headers = headerReader_.Clone();

    size_t i = 0;
    while (headers.HasMoreHeaders())
    {
        assert(i < NumZMWs());
        const auto& zmwHeaders = headers.NextHeaders();
        for (size_t j = 0; j < NumSuperchunks(); ++j)
        {
            ret[j][i] = zmwHeaders[j];
        }
        i++;
    }
    return ret;
}

}} // namespace
