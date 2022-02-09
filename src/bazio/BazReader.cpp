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
#include <stdexcept>
#include <stdlib.h>
#include <string>
#include <typeinfo>

#include <boost/range/combine.hpp>

#include <pacbio/logging/Logger.h>
#include <pacbio/text/String.h>

#include "BazCore.h"
#include "FileFooter.h"
#include "SmartMemory.h"
#include "Timing.h"


#include "BazReader.h"

namespace PacBio {
namespace Primary {

/// Reads file header and super chunk header for indexed access.
BazReader::BazReader(const std::vector<std::string>& fileNames,
                     size_t zmwBatchMB,
                     size_t zmwHeaderBatchMB,
                     bool silent,
                     const std::function<bool(void)>& callBackCheck)
{
    if (callBackCheck && callBackCheck()) return;

    const size_t zmwBatchBytes = zmwBatchMB << 20;

    // Init
    for (const auto& fileName : fileNames)
    {
        files_.emplace_back(fileName,
                            std::unique_ptr<std::FILE,decltype(&std::fclose)>(std::fopen(fileName.c_str(), "rb"),
                                                                              &std::fclose));
        if (!files_.back().second) throw std::runtime_error("Can't open " + fileName + " with BazReader");
    }

    // Timing
    // auto now = std::chrono::high_resolution_clock::now();

    // Read file headers from files
    if (callBackCheck && callBackCheck()) return;
    ReadFileHeaders();
    if (callBackCheck && callBackCheck()) return;
    ReadFileFooters();
    if (callBackCheck && callBackCheck()) return;

    numZmws_ = fh_->TotalNumZmws();
    const auto& maxNumZmws = fh_->MaxNumZmws();
    std::partial_sum(maxNumZmws.begin(), maxNumZmws.end(), std::back_inserter(zmwIndexByBazFile_));

    std::vector<std::vector<size_t>> zmwMetaLocationsFiles_;
    zmwMetaLocationsFiles_.reserve(files_.size());
    const auto& numSuperChunks = fh_->NumSuperChunks();
    std::vector<std::pair<std::string,std::FILE*>> filePtrs;
    for (size_t i = 0; i < files_.size(); ++i)
    {
        auto fn = files_[i].first;
        auto fp = files_[i].second.get();

        // Position indicator (pi) to for next chunk
        auto pi = std::ftell(fp);

        // 4k alignment of binary data start
        if (pi % blocksize != 0)
            pi += blocksize - pi % blocksize;

        if (!silent)
        {
            std::cerr << "Finding Metadata locations\n";
            std::cerr.flush();
        }

        std::vector<size_t> zmwMetaLocations;
        zmwMetaLocations.reserve(numSuperChunks[i]);
        while (true)
        {
            if (callBackCheck && callBackCheck()) return;
            // Seek to next super-chunk
            fseek(fp, pi, SEEK_SET);
            // Check sanity dword
            int counter = 0;
            while (!Sanity::ReadAndVerify(fp))
            {
                // Seek one block further
                pi += blocksize - pi % blocksize;
                fseek(fp, pi, SEEK_SET);
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
            if (fread(&chunkMeta.offsetNextChunk, sizeof(SuperChunkMeta::offsetNextChunk), 1, fp) != 1)
            {
                PBLOG_ERROR << "Corrupt file. Cannot read SUPER_CHUNK_META offset";
                exit(EXIT_FAILURE);
            }
            // Read numZmws
            if (fread(&chunkMeta.numZmws, sizeof(SuperChunkMeta::numZmws), 1, fp) != 1)
            {
                PBLOG_ERROR << "Corrupt file. Cannot read SUPER_CHUNK_META numZmws";
                exit(EXIT_FAILURE);
            }

            // Check sanity dword
            if (!Sanity::ReadAndVerify(fp))
            {
                PBLOG_ERROR << "Corrupt file. Cannot read SANITY DWORD after CHUNK_META";
                exit(EXIT_FAILURE);
            }

            // Set pi to next super-chunk offset
            pi = chunkMeta.offsetNextChunk;

            // If next offset is zero, stop.
            // This is used to mark EOF.
            if (pi == 0) break;

            if (!(chunkMeta.numZmws == maxNumZmws[i]))
            {
                std::ostringstream errMsg;
                errMsg << "Unexpected number of ZMWs in chunk metadata "
                       << "expected: " << maxNumZmws[i] << " but got: "
                       << chunkMeta.numZmws;
                PBLOG_ERROR << errMsg.str();
                throw PBException(errMsg.str());
            }
            zmwMetaLocations.emplace_back(ftell(fp));
            if (callBackCheck && callBackCheck()) return;
        }

        if (!(zmwMetaLocations.size() == numSuperChunks[i]))
        {
            std::ostringstream errMsg;
            errMsg << "Unexpected number of ZMW metadata locations"
                   << "expected: " << numSuperChunks[i] << " but got: "
                   << zmwMetaLocations.size();
            PBLOG_ERROR << errMsg.str();
            throw PBException(errMsg.str());
        }

        zmwMetaLocationsFiles_.emplace_back(std::move(zmwMetaLocations));
        filePtrs.emplace_back(fn, fp);
    }

    if (callBackCheck && callBackCheck()) return;

    // We're going to parse the header once upfront to figure out how large each batch
    // of reads will be.  Force silent mode to not pollute the log with extra header
    // reading messages
    headerReader_ = HeaderReader(zmwMetaLocationsFiles_, fh_->MaxNumZmws(), filePtrs, zmwHeaderBatchMB, true);

    // Store [first, last) zmw per slice
    uint32_t first = 0;
    size_t sliceBytes = 0;
    for (size_t zmw = 0; zmw < NumZmws(); zmw++)
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
        if ((fitsInBatchBytes && fitsInMaxSlice && zmw < zmwIndexByBazFile_[currentBazIndex_]) || zmw == first)
        {
            sliceBytes += zmwBytes;
        }
        else
        {
            // The current zmw does not fit in the current slice, so tie off the
            // current slice and start accumulating a new one.
            assert(first != zmw);
            zmwSlices_.push({first, zmw, numSuperChunks[currentBazIndex_], files_[currentBazIndex_].second.get()});
            first = zmw;
            sliceBytes = zmwBytes;
            if (zmw == zmwIndexByBazFile_[currentBazIndex_]) currentBazIndex_++;
        }
    }
    // Need to tie off the last slice.  We're guaranteed to have one in progress
    assert(sliceBytes != 0 || NumZmws() == 0);
    zmwSlices_.push({first, NumZmws(), numSuperChunks.back(), files_.back().second.get()});

    // Re-start the header reader, since we'll need the header information again while actually
    // reading in zmw data
    headerReader_ = HeaderReader(zmwMetaLocationsFiles_, fh_->MaxNumZmws(), filePtrs, zmwHeaderBatchMB, silent);
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

    // Load ZMWs for a batch only up to a find boundary as those
    // are guaranteed to have the same number of metadata locations
    // i.e. number of superchunks.
    bool nextFile = false;
    if (end > currentMaxNumZmws_)
    {
        end = currentMaxNumZmws_;
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

        auto fp = GetCurrentFilePointer();

        // Remap the zmw file offset.
        fseek(fp, location + (firstLoaded_ - zmwOffset_) * sizeSingleHeader, SEEK_SET);

        std::vector<uint8_t> chunkHeadersMemory(sizeSingleHeader * numLoad);
        if (!fread(chunkHeadersMemory.data(), chunkHeadersMemory.size(), 1, fp))
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
            if (!(header.zmwIndex == i + firstLoaded_))
            {
                std::ostringstream errMsg;
                errMsg << "Unexpected zmw index for header "
                       << "expected: " << i + firstLoaded_ << " but got: "
                       << header.zmwIndex;
                PBLOG_ERROR << errMsg.str();
                throw PBException(errMsg.str());
            }
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

std::vector<uint32_t> BazReader::NextZmwIds()
{
    // Get first ZMW for current slice
    const auto& slice = zmwSlices_.front();

    std::vector<uint32_t> list;
    list.reserve(slice.endZmw - slice.startZmw);

    size_t startZmw = slice.startZmw;
    while (startZmw < slice.endZmw)
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
    size_t startZmw = zmwSlices_.front().startZmw;
    size_t end = zmwSlices_.front().endZmw;
    uint32_t numSuperChunks = zmwSlices_.front().numSuperChunks;
    std::FILE* fp = zmwSlices_.front().fp;
    size_t sliceSize = end - startZmw;

    zmwSlices_.pop();
    // vector to store reads
    std::vector<ZmwDataCounts> dataSizes(sliceSize);

    std::vector<std::vector<ZmwSliceHeader>> chunkToHeaderSplit;
    chunkToHeaderSplit.reserve(numSuperChunks);
    for (size_t i = 0; i < numSuperChunks; ++i)
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
        assert(zmwHeaders.size() == numSuperChunks);
        for (size_t j = 0; j < numSuperChunks; ++j)
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

    // Iterate over all chunks which are guaranteed to be from the same file
    // super-chunk id -> ZmwSliceHeader
    for (const auto& singleChunk: chunkToHeaderSplit)
    {
        if (callBackCheck && callBackCheck()) return std::vector<ZmwByteData>{};

        // It could be that in this chunk, there are not ZMW stitched
        if (singleChunk.empty()) continue;

        // Seek to first ZmwSlice
        auto seekResult = fseek(fp, singleChunk[0].offsetPacket, SEEK_SET);
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
        for (const auto& h: singleChunk)
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
                            std::fgetc(fp);
                        case 2:
                            std::fgetc(fp);
                        case 1:
                            std::fgetc(fp);
                            break;
                        default:
                            auto pos = fseek(fp, h.offsetPacket, SEEK_SET);
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
            if (!Sanity::ReadAndVerify(fp))
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
                auto readResult = fread(data, 1, h.packetsByteSize, fp);
                if (readResult != h.packetsByteSize)
                    throw std::runtime_error("Error reading");
                bytesRead += h.packetsByteSize;
            }
            if (h.numMBs > 0)
            {
                auto* data = s.NextMetricBytes(h.numMBs * fh_->MetricByteSize());
                assert(data != nullptr);
                auto readResult = fread(data, 1, fh_->MetricByteSize() * h.numMBs, fp);
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
    if (numSuperChunks == 0)
        return std::vector<ZmwByteData>{};

    return batchByteData;
}

void BazReader::ReadFileHeaders()
{
    try
    {
        fh_ = std::make_unique<BazIO::FileHeaderSet>(files_);

        // Re-order files based on file header set ordering.
        std::vector<std::string> fns;
        std::for_each(fh_->FileHeaders().begin(), fh_->FileHeaders().end(),
                      [&fns](const auto& fh) { fns.push_back(fh.FileName()); });

        std::sort(files_.begin(), files_.end(),
                  [&fns](const auto& a, const auto& b)
                  {
                      const auto& aiter = std::find_if(fns.begin(), fns.end(),
                                                       [&a](const std::string& fn) { return fn == a.first; });
                      const auto& biter = std::find_if(fns.begin(), fns.end(),
                                                       [&b](const std::string& fn) { return fn == b.first; });
                      return std::distance(fns.begin(), aiter) < std::distance(fns.begin(), biter);
                  });
    }
    catch (const std::exception& ex)
    {
        PBLOG_ERROR << "Error reading BAZ file headers!";
        exit(EXIT_FAILURE);
    }
}

void BazReader::ReadFileFooters()
{
    PBLOG_WARN << "File footers currently not populated!";

    std::map<uint32_t, std::vector<uint32_t>> truncationMap;

    for (const auto& t : boost::combine(fh_->FileHeaders(), files_))
    {
        const auto& fh = t.get<0>();
        const auto& file = t.get<1>().second;

        if (!fh.FileFooterOffset())
        {
            auto pi = std::ftell(file.get());

            // Go to last Footer
            const auto fileFooterOffset = fh.FileFooterOffset();
            fseek(file.get(), fileFooterOffset, SEEK_SET);

            // Seek for SANITY block and last position of the file header
            uint64_t footerEnd = Sanity::FindAndVerify(file.get());
            int64_t footerSize = static_cast<int64_t>(footerEnd) - static_cast<int64_t>(fh.FileFooterOffset());

            // Wrap footer into smrt pointer
            auto footer = SmartMemory::AllocMemPtr<char>(footerSize + 1);

            // Set position indicator to beginning of footer
            fseek(file.get(), fileFooterOffset, SEEK_SET);

            // Read file footer
            size_t result = std::fread(footer.get(), 1, footerSize, file.get());
            if (result != static_cast<uint64_t>(footerSize))
                throw std::runtime_error("Cannot read file footer!");

            fseek(file.get(), pi, SEEK_SET);

            // Let Filefooter parse JSON footer
            FileFooter ff(footer.get(), footerSize);

            for (const auto& kv: ff.TruncationMap())
            {
                // Map to zmw number as that is unique.
                const auto zmwNumber = fh.ZmwInformation().ZmwIndexToNumber(kv.first);
                truncationMap[zmwNumber].insert(std::end(truncationMap[zmwNumber]),
                                                kv.second.begin(), kv.second.end());
            }
        }
    }

    ff_ = std::make_unique<BazIO::FileFooterSet>(std::move(truncationMap));
}

std::vector<std::vector<ZmwSliceHeader>> BazReader::SuperChunkToZmwHeaders() const
{
    // This method is only called by bazviewer so we assume that only a single BAZ file will be called with it.
    const auto& numSuperChunks = fh_->NumSuperChunks().front();
    const auto& numZmws = fh_->MaxNumZmws().front();
    std::vector<std::vector<ZmwSliceHeader>> ret(numSuperChunks);
    for (size_t i = 0; i < numSuperChunks; ++i)
        ret[i].resize(numZmws);

    auto headers = headerReader_.Clone();

    size_t i = 0;
    while (headers.HasMoreHeaders())
    {
        assert(i < numZmws);
        const auto& zmwHeaders = headers.NextHeaders();
        for (size_t j = 0; j < numSuperChunks; ++j)
        {
            ret[j][i] = zmwHeaders[j];
        }
        i++;
    }
    return ret;
}

}} // namespace PacBio::Primary
