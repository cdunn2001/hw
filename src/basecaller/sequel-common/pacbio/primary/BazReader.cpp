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
#include <queue>
#include <sstream>
#include <stdexcept>
#include <stdlib.h>
#include <string>
#include <typeinfo>
#include <unordered_map>

#include <pacbio/logging/Logger.h>

#include <pacbio/primary/BazCore.h>
#include <pacbio/primary/SmartMemory.h>
#include <pacbio/primary/Timing.h>
#include <pacbio/primary/BazReader.h>

namespace PacBio {
namespace Primary
{

/// Reads file header and super chunk header for indexed access.
BazReader::BazReader(const std::string& fileName, size_t zmwBatchSize, size_t zmwHeaderBatchSize, bool silent)
        : sliceSize_(zmwBatchSize)
          , zmwHeaderBatchSize_(zmwHeaderBatchSize)
{
    // Init
    file_ = fopen(fileName.c_str(), "rb");
    if (!file_) throw std::runtime_error("Can't open " + fileName + " with BazReader");

    // Timing
    // auto now = std::chrono::high_resolution_clock::now();

    // Read file header from file
    fh_ = ReadFileHeader();
    ff_ = ReadFileFooter();

    // Check if BAZ version is correct
    if (!(fh_->BazMajorVersion() == 1 && fh_->BazMinorVersion() == 6))
    {
        PBLOG_ERROR << "Incorrect BAZ version provided. Need version 1.6.x, provided version is "
                    << fh_->BazVersion();
        exit(EXIT_FAILURE);
    }

    if (!fh_->Complete())
    {
        PBLOG_ERROR << "Trying to read unfinished baz file " << fileName;
        exit(EXIT_FAILURE);
    }

    if (fh_->Truncated())
        PBLOG_INFO << "Converting truncated file " << fileName;

    // Check sanity dword
    if (!Sanity::ReadAndVerify(file_))
    {
        PBLOG_ERROR << "Corrupt file. Cannot read SANITY DWORD after FILE_HEADER";
        exit(EXIT_FAILURE);
    }

    numZMWs_ = fh_->MaxNumZMWs();
    numSuperchunks_ = fh_->NumSuperChunks();

    // Position indicator (pi) to for next chunk
    auto pi = std::ftell(file_);

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
        // Seek to next super-chunk
        fseek(file_, pi, SEEK_SET);
        // Check sanity dword
        int counter = 0;
        while (!Sanity::ReadAndVerify(file_))
        {
            // Seek one block further
            pi += blocksize - pi % blocksize;
            fseek(file_, pi, SEEK_SET);
            // Only try three times and then fail.
            ++counter;
            if (counter == 3)
            {
                PBLOG_ERROR << "Corrupt file. Cannot read SANITY DWORD before CHUNK_META";
                exit(EXIT_FAILURE);
            }
        }

        // Read super-chunk meta information
        SuperChunkMeta chunkMeta;
        // Read offset
        if (fread(&chunkMeta.offsetNextChunk, sizeof(SuperChunkMeta::offsetNextChunk), 1, file_) != 1)
        {
            PBLOG_ERROR << "Corrupt file. Cannot read SUPER_CHUNK_META offset";
            exit(EXIT_FAILURE);
        }
        // Read numZmws
        if (fread(&chunkMeta.numZmws, sizeof(SuperChunkMeta::numZmws), 1, file_) != 1)
        {
            PBLOG_ERROR << "Corrupt file. Cannot read SUPER_CHUNK_META numZmws";
            exit(EXIT_FAILURE);
        }

        // Check sanity dword
        if (!Sanity::ReadAndVerify(file_))
        {
            PBLOG_ERROR << "Corrupt file. Cannot read SANITY DWORD after CHUNK_META";
            exit(EXIT_FAILURE);
        }


        // Set pi to next super-chunk offset
        pi = chunkMeta.offsetNextChunk;

        // If next offset is zero, stop.
        // This is used to mark EOF.
        if (pi == 0) break;

        assert(chunkMeta.numZmws == NumZMWs());
        zmwMetaLocations.emplace_back(ftell(file_));
    }
    assert(zmwMetaLocations.size() == NumSuperchunks());
    headerReader_ = HeaderReader(zmwMetaLocations, NumZMWs(), file_, zmwHeaderBatchSize_, silent);

    // Store first ZMW per slice
    for (size_t startZmw = 0; startZmw <= numZMWs_; startZmw += sliceSize_)
        zmwSlices_.push(startZmw);
}

const std::vector<ZmwSliceHeader>& BazReader::HeaderReader::NextHeaders()
{
    assert(idx_ >= firstLoaded_);
    if (idx_ - firstLoaded_ >= loadedData_.size() || loadedData_.empty())
        LoadNextBatch();
    assert(idx_ - firstLoaded_ < loadedData_.size());
    const auto& ret = loadedData_[idx_ - firstLoaded_];
    idx_++;
    return ret;
}

void BazReader::HeaderReader::LoadNextBatch()
{
    constexpr size_t sizeSingleHeader = ZmwSliceHeader::SizeOf();
    firstLoaded_ = idx_;
    size_t end = idx_ + batchSize_;
    if (end > numZmw_) end = numZmw_;
    size_t numLoad = end - firstLoaded_;

    loadedData_.resize(0);
    for (size_t i = 0; i < numLoad; ++i)
    {
        loadedData_.emplace_back(metaPositions_.size());
    }

    size_t headerId = 0;
    for (const auto& location : metaPositions_)
    {
        fseek(file_, location + firstLoaded_ * sizeSingleHeader, SEEK_SET);

        std::vector<uint8_t> chunkHeadersMemory(sizeSingleHeader * numLoad);
        if (!fread(chunkHeadersMemory.data(), chunkHeadersMemory.size(), 1, file_))
        {
            PBLOG_ERROR << "Corrupt file. Cannot read ZMW_SLICE_HEADER";
            exit(EXIT_FAILURE);
        }

        size_t bytesRead = 0;
        for (size_t i = 0; i < numLoad; ++i)
        {
            ZmwSliceHeader& header = loadedData_[i][headerId];
            // Read package offset
            memcpy(&header.offsetPacket,
                   &chunkHeadersMemory[bytesRead],
                   sizeof(ZmwSliceHeader::offsetPacket));
            bytesRead += sizeof(ZmwSliceHeader::offsetPacket);
            // Read ZMW ID
            memcpy(&header.zmwIndex,
                   &chunkHeadersMemory[bytesRead],
                   sizeof(ZmwSliceHeader::zmwIndex));
            assert(header.zmwIndex == i + firstLoaded_);
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
            // Read number of high-frequency metric blocks
            memcpy(&header.numHFMBs,
                   &chunkHeadersMemory[bytesRead],
                   sizeof(ZmwSliceHeader::numHFMBs));
            bytesRead += sizeof(ZmwSliceHeader::numHFMBs);
            // Read number of medium-frequency metric blocks
            memcpy(&header.numMFMBs,
                   &chunkHeadersMemory[bytesRead],
                   sizeof(ZmwSliceHeader::numMFMBs));
            bytesRead += sizeof(ZmwSliceHeader::numMFMBs);
            // Read number of low-frequency metric blocks
            memcpy(&header.numLFMBs,
                   &chunkHeadersMemory[bytesRead],
                   sizeof(ZmwSliceHeader::numLFMBs));
            bytesRead += sizeof(ZmwSliceHeader::numLFMBs);
        }

        ++headerId;
    }

    if (!silent_)
        std::cerr << "Read " << headerId << " SUPER_CHUNK_META" << std::endl;
}

// Destructor
BazReader::~BazReader()
{
    fclose(file_);
}

void BazReader::WhiteList(std::vector<uint32_t> whiteList)
{
    std::queue<uint32_t> intersection;
    while (zmwSlices_.empty())
    {
        size_t zmw = zmwSlices_.front();
        zmwSlices_.pop();
        if (std::find(whiteList.begin(), whiteList.end(), zmw) != whiteList.end())
            intersection.push(zmw);
    }
    zmwSlices_ = intersection;
}

bool BazReader::HasNext()
{ return !zmwSlices_.empty(); }

std::vector<uint32_t> BazReader::NextZmwIds()
{
    // Get first ZMW for current slice
    size_t startZmw = zmwSlices_.front();
    size_t end = startZmw + sliceSize_ > numZMWs_ ?
                 numZMWs_ + 1 :
                 startZmw + sliceSize_;

    std::vector<uint32_t> list;
    list.reserve(sliceSize_);

    while (startZmw < end)
        list.push_back(startZmw++);

    std::sort(list.begin(), list.end());
    return list;
}

std::vector<ZmwByteData> BazReader::NextSlice()
{
    // Timing
    // auto now3 = std::chrono::high_resolution_clock::now();

    // Get first ZMW for current slice
    size_t startZmw = zmwSlices_.front();
    size_t end = startZmw + sliceSize_ > numZMWs_ ?
                 numZMWs_ :
                 startZmw + sliceSize_;

    zmwSlices_.pop();
    // vector to store reads
    std::vector<ZmwDataCounts> dataSizes(end - startZmw);

    std::vector<std::vector<ZmwSliceHeader>> chunkToHeaderSplit;
    chunkToHeaderSplit.reserve(NumSuperchunks());
    for (size_t i = 0; i < NumSuperchunks(); ++i)
    {
        chunkToHeaderSplit.emplace_back();
        chunkToHeaderSplit.back().reserve(sliceSize_);
    }

    // Iterate over all chunks for ZMWs in current slice,
    // to save length of base calls and HQ regions.
    for (size_t i = startZmw; i < end; ++i)
    {
        assert(headerReader_.HasMoreHeaders());
        const auto& zmwHeaders = headerReader_.NextHeaders();
        for (size_t j = 0; j < NumSuperchunks(); ++j)
        {
            const auto& h = zmwHeaders[j];
            assert(i == h.zmwIndex);
            dataSizes[i - startZmw].packetsByteSize += h.packetsByteSize;
            dataSizes[i - startZmw].numHFMBs += h.numHFMBs;
            dataSizes[i - startZmw].numMFMBs += h.numMFMBs;
            dataSizes[i - startZmw].numLFMBs += h.numLFMBs;
            chunkToHeaderSplit[j].push_back(h);
        }
    }

    std::vector<ZmwByteData> batchByteData;
    batchByteData.reserve(sliceSize_);
    for (size_t i = 0; i < dataSizes.size(); ++i)
        batchByteData.emplace_back(*fh_, dataSizes[i], i + startZmw);

    // Iterate over all chunks
    // super-chunk id -> ZmwSliceHeader
    for (const auto& singleChunk : chunkToHeaderSplit)
    {
        // It could be that in this chunk, there are not ZMW stitched
        if (singleChunk.empty()) continue;

        // Seek to first ZmwSlice
        auto seekResult = fseek(file_, singleChunk[0].offsetPacket, SEEK_SET);
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
                            std::fgetc(file_);
                        case 2:
                            std::fgetc(file_);
                        case 1:
                            std::fgetc(file_);
                            break;
                        default:
                            auto pos = fseek(file_, h.offsetPacket, SEEK_SET);
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
            if (!Sanity::ReadAndVerify(file_))
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
                auto readResult = fread(data, 1, h.packetsByteSize, file_);
                if (readResult != h.packetsByteSize)
                    throw std::runtime_error("Error reading");
                bytesRead += h.packetsByteSize;
            }
            if (h.numHFMBs > 0)
            {
                // For high-frequency metric blocks
                auto* data = s.NextHFBytes(h.numHFMBs * fh_->HFMetricByteSize());
                assert(data != nullptr);
                auto readResult = fread(data, 1, fh_->HFMetricByteSize() * h.numHFMBs, file_);
                if (readResult != fh_->HFMetricByteSize() * h.numHFMBs)
                    throw std::runtime_error("Error reading");
                bytesRead += fh_->HFMetricByteSize() * h.numHFMBs;
            }
            if (h.numMFMBs > 0)
            {
                // For medium-frequency metric blocks
                auto* data = s.NextMFBytes(h.numMFMBs * fh_->MFMetricByteSize());
                assert(data != nullptr);
                auto readResult = fread(data, 1, fh_->MFMetricByteSize() * h.numMFMBs, file_);
                if (readResult != fh_->MFMetricByteSize() * h.numMFMBs)
                    throw std::runtime_error("Error reading");
                bytesRead += fh_->MFMetricByteSize() * h.numMFMBs;
            }
            if (h.numLFMBs > 0)
            {
                // For low-frequency metric blocks
                auto* data = s.NextLFBytes(h.numLFMBs * fh_->LFMetricByteSize());
                assert(data != nullptr);
                auto readResult = fread(data, 1, fh_->LFMetricByteSize() * h.numLFMBs, file_);
                if (readResult != fh_->LFMetricByteSize() * h.numLFMBs)
                    throw std::runtime_error("Error reading");
                bytesRead += fh_->LFMetricByteSize() * h.numLFMBs;
            }
        }
    }

    // TODO: This is necessary to handle some testing code that produces a baz
    // file with 0 superchunks.  It's arguable that the code path producing
    // such a file should really throw an exception and terminate instead.
    // May need to refactor the code to remove the `Error` function, and
    // update tests accordingly
    if (NumSuperchunks() == 0)
        return std::vector<ZmwByteData>{};

    return batchByteData;
}

std::unique_ptr<FileHeader> BazReader::ReadFileHeader()
{
    // Seek for SANITY block and last position of the file header
    uint64_t headerSize = Sanity::FindAndVerify(file_);

    // Wrap header into smrt pointer
    auto header = SmartMemory::AllocMemPtr<char>(headerSize + 1);

    // Set position indicator to beginning
    std::rewind(file_);

    // Read file header
    size_t result = std::fread(header.get(), 1, headerSize, file_);
    if (result != headerSize)
        throw std::runtime_error("Cannot read file header!");

    // Let FileHeader parse JSON header
    return std::unique_ptr<FileHeader>(new FileHeader(header.get(), headerSize));
}

std::unique_ptr<FileFooter> BazReader::ReadFileFooter()
{
    if (!fh_ || fh_->FileFooterOffset() == 0)
        return std::unique_ptr<FileFooter>(new FileFooter());

    auto pi = std::ftell(file_);

    // Go to last Footer
    const auto fileFooterOffset = fh_->FileFooterOffset();
    fseek(file_, fileFooterOffset, SEEK_SET);

    // Seek for SANITY block and last position of the file header
    uint64_t footerEnd = Sanity::FindAndVerify(file_);
    int64_t footerSize = static_cast<int64_t>(footerEnd) - static_cast<int64_t>(fh_->FileFooterOffset());

    // Wrap footer into smrt pointer
    auto footer = SmartMemory::AllocMemPtr<char>(footerSize + 1);

    // Set position indicator to beginning of footer
    fseek(file_, fileFooterOffset, SEEK_SET);

    // Read file footer
    size_t result = std::fread(footer.get(), 1, footerSize, file_);
    if (result != static_cast<uint64_t>(footerSize))
        throw std::runtime_error("Cannot read file footer!");

    fseek(file_, pi, SEEK_SET);

    // Let Filefooter parse JSON footer
    return std::unique_ptr<FileFooter>(new FileFooter(footer.get(), footerSize));
}

const FileHeader& BazReader::Fileheader()
{
    if (!fh_) throw PBException("null Fileheader");
    return *fh_;
}

std::unique_ptr<FileFooter>& BazReader::Filefooter()
{ return ff_; }

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
