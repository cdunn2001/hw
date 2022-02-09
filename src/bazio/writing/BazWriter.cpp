// Copyright (c) 2021, Pacific Biosciences of California, Inc.
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

#include "BazWriter.h"

#include <pacbio/POSIX.h>

#include <bazio/SuperChunkMeta.h>

namespace PacBio {
namespace BazIO {

BazWriter::BazWriter(const std::string& filePath,
                     FileHeaderBuilder& fileHeaderBuilder,
                     const Primary::BazIOConfig& ioConf,
                     std::shared_ptr<IOStatsAggregator> agg)
    : filePath_(filePath)
    , filePathTmp_(filePath + ".tmp")
    , fhb_(fileHeaderBuilder)
    , fileHandle_(new Primary::ManuallyBufferedFile(filePathTmp_, ioConf.BufferSizeMBytes, ioConf.MaxMBPerSecondWrite))
    , jsonFileHeader_(fileHeaderBuilder.CreateJSONCharVector())
    , dieGracefully_(false)
    , ioAggregator_(agg)
{
    if (!fileHeaderBuilder.MaxNumZmws())
        throw PBException("No ZmwNumber mapping provided in FileHeaderBuilder");
    Init(ioConf);
}


BazWriter::~BazWriter()
{
    try
    {
        if (!dieGracefully_)
        {
            PBLOG_ERROR << "ABORT";
            abort_ = true;
            fhb_.Truncated();
        }

        // Terminates writeThread_
        writeThreadContinue_ = false;
        if (writeThread_.joinable()) writeThread_.join();

        ioAggregator_->AggregateToFull(&ioStats_);
    }
    catch(const std::exception& ex)
    {
        PBLOG_ERROR << "Exception caught in ~BazWriter " << ex.what();
    }
    catch(...)
    {
        std::cerr << "Uncaught exception caught in ~BazWriter " << PacBio::GetBackTrace(5);
        PBLOG_FATAL << "Uncaught exception caught in ~BazWriter "  << PacBio::GetBackTrace(5);
        PacBio::Logging::PBLogger::Flush();
        std::terminate();
    }
}

void BazWriter::Init(const Primary::BazIOConfig& ioConfig)
{
    // Parse ASCII JSON to FileHeader instance,
    // includes sanity checks for validity
    fh_ = std::make_unique<FileHeader>(jsonFileHeader_.data(), jsonFileHeader_.size());

    // Start thread that saves BazBuffers to disk
    writeThreadContinue_ = true;
    writeThread_ = std::thread([this](){
        try
        {
            WriteToDiskLoop();
        }
        catch(std::exception& err)
        {
            // Forward
            PBLOG_ERROR << err.what();
            // Set truncated flag
            fhb_.Truncated();
            // Empty buffers
            while (!bazBufferQueue_.Empty())
                bazBufferQueue_.Pop();
            // Die gracefully
        }
    });
}

void BazWriter::Flush(std::unique_ptr<BazBuffer> buffer)
{
    if (abort_ || !writeThreadContinue_) return;

    bazBufferQueue_.Push(std::move(buffer));
}

/// Waits until all blocks have entered the writer thread, and then waits
/// for the writer thread to exit.

void BazWriter::WaitForTermination()
{
    dieGracefully_ = true;

    // wait for buffer to empty
    while (bazBufferQueue_.Size() > 0)
    {
        PacBio::POSIX::Sleep(0.010);
    }

    // wait for writing thread to exit
    writeThreadContinue_ = false;
    if (writeThread_.joinable()) writeThread_.join();
}

/// Writes zeros until the file pointer is aligned.
/// \param alignmentSize : the modulo to align to
/// \param accumulate : whether or not to keep track of the byte written.

void BazWriter::Align(size_t alignmentSize, bool accumulate)
{
    size_t ragged = Ftell() % alignmentSize;
    if (ragged)
    {
        size_t advance = alignmentSize - ragged;
        fileHandle_->Advance(advance);
        if (accumulate)
        {
            ioStats_.paddingBytes += advance;
            ioStats_.bytesWritten += advance;
        }
    }
}


void BazWriter::WriteJsonFileHeader(bool accumulate)
{
    /// Fwrite doesn't allow to write really big buffers. The write is broken into multiple smaller pieces.
    size_t bytesToWrite = jsonFileHeader_.size();
    const auto* ptr = jsonFileHeader_.data();
    while(bytesToWrite >0)
    {
        size_t bytesToWriteNow = (1UL << 20); // let's try 1 MiB each loop. 2GiB was too much.
        if (bytesToWriteNow > bytesToWrite) bytesToWriteNow = bytesToWrite;
        const auto justWrote = Fwrite(ptr,bytesToWriteNow, accumulate);
        if (accumulate) ioStats_.headerBytes += justWrote;
        bytesToWrite -= bytesToWriteNow;
        ptr += bytesToWriteNow;
    }
}

/// This is essentially the thread function that writes the entire baz file
/// from header to footer.
/// To end the thread, stop adding new baz buffers and then call WaitForTermination()

void BazWriter::WriteToDiskLoop()
{
    PBLOG_DEBUG << "BazWriter::WriteToDiskLoop() header, length: " << jsonFileHeader_.size();
    WriteJsonFileHeader(true /* accumulate bytes in statistics */);
    WriteSanity();

    Align(blocksize);

    std::vector<Primary::ZmwSliceHeader> headerBuffer;
    PBLOG_DEBUG << "BazWriter::WriteToDiskLoop() loop entered";
    size_t writtenSuperChunks = 0;
    ioAggregator_->AggregateToFull(&ioStats_);
    while (writeThreadContinue_ && !abort_)
    {
        std::unique_ptr<BazBuffer> bazBuffer;
        // Wait for new data, but only for 500ms. Otherwise restart the loop,
        // as it could have been the last item and the loop will be exited.

        if (bazBufferQueue_.Pop(bazBuffer, std::chrono::milliseconds(500)))
        {
            ioAggregator_->StartSegment(writtenSuperChunks);
            auto now = std::chrono::high_resolution_clock::now();

            PBLOG_DEBUG << "Starting writing superchunk " << writtenSuperChunks;
            auto currBytes = ioStats_.bytesWritten;

            // super chunk layout: SMSHHHHSPHMLSPHMLSPHMLSPHML
            uint32_t numZmws = bazBuffer->NumZmw();
            headerBuffer.resize(numZmws);

            PBLOG_DEBUG << "BazWriter::WriteToDiskLoop()  superchunk writing ... numZmws:" << numZmws;

            // keep track of where the file pointer, because we will rewind to this location
            // after the data is written.
            size_t superChunkStart = Ftell();

            // write temporary SuperChunkMeta, with zeros, in case file gets truncated
            WriteSuperChunkMeta(0, 0);

            // skip over headers
            FseekRelative(Primary::ZmwSliceHeader::SizeOf() * numZmws );

            // write actual data
            WriteChunkData(bazBuffer, headerBuffer);
            bazBuffer.reset();

            // superchunk is written, but we need to rewind, so keep track of the
            // tail of the file so we can resume on the next superchunk.
            size_t nextChunkStart = Ftell();

            // now jump back and rewrite the meta data and headers
            FseekAbsolute(superChunkStart);
            WriteSuperChunkMeta(nextChunkStart, numZmws, false); // don't count the second time
            WriteZmwSliceHeader(headerBuffer);

            // return to the end of the file
            FseekAbsolute(nextChunkStart);
            PBLOG_DEBUG << "nextChunkStart:" << std::hex << nextChunkStart << std::dec;

            fhb_.IncSuperChunks();

            fileHandle_->Flush();

            auto elapsed = std::chrono::duration<double, std::ratio<1>>(std::chrono::high_resolution_clock::now() - now);
            PBLOG_DEBUG << "Finished writing superchunk " << writtenSuperChunks << " to disk";
            PBLOG_DEBUG << "Wrote " << ioStats_.bytesWritten - currBytes << " to disk in "
                        << elapsed.count() << " seconds";
            if (bazBufferQueue_.Size() != 0)
            {
                PBLOG_WARN << "There are " << bazBufferQueue_.Size()
                           << " more baz buffers queued for writing!";
            }
            ioAggregator_->AggregateToSegment(writtenSuperChunks, &ioStats_);
            writtenSuperChunks++;
        }
    }
    PBLOG_DEBUG << "BazWriter::WriteToDiskLoop() loop exited";

    // write trailing metachunk, to indicate the file is done.
    WriteSuperChunkMeta(0, 0);

    // update the header to the position of the footer.
    size_t fileFooterOffset = Ftell();
    fhb_.FileFooterOffset(fileFooterOffset);
    fhb_.Done();

    // Write footer
    const auto jsonFileFooter_ = ffb_.CreateJSONCharVector();
    ioStats_.headerBytes += Fwrite(jsonFileFooter_.data(), jsonFileFooter_.size());
    WriteSanity();

    // Rewrite header
    FseekAbsolute(0);
    jsonFileHeader_ = fhb_.CreateJSONCharVector();
    WriteJsonFileHeader(false /* don't double count the Fwrite bytes*/);
    WriteSanity(false);
    Align(blocksize,false);

    fileHandle_.reset();

    PBLOG_DEBUG << "BazWriter::WriteToDiskLoop() file closed";

    auto renameResult = std::rename(filePathTmp_.c_str(), filePath_.c_str());
    if (renameResult)
        throw PBException("Cannot rename file from " + filePathTmp_ + " to " + filePath_);
}

/// write just the SuperChunkMeta data, and the surrounding Sanity words

void BazWriter::WriteSuperChunkMeta(const uint64_t nextPointer,
                                    uint32_t numZmws, bool accumulate)
{
    Primary::SuperChunkMeta chunkMeta;
    chunkMeta.offsetNextChunk = nextPointer;
    chunkMeta.numZmws         = numZmws;

    WriteSanity(accumulate);
    size_t written =  Fwrite(&chunkMeta, Primary::SuperChunkMeta::SizeOf(), accumulate);
    if (accumulate)
    {
        ioStats_.overheadBytes += written;
    }
    WriteSanity(accumulate);
}

/// write the ZmwSliceHeader

void BazWriter::WriteZmwSliceHeader(std::vector<Primary::ZmwSliceHeader>& headerBuffer)
{
    for (const auto& h : headerBuffer)
    {
        if (abort_) return;
        ioStats_.overheadBytes += Fwrite(&h, Primary::ZmwSliceHeader::SizeOf());
    }
}

/// Write the SuperChunk actual data (base calls and metrics)

void BazWriter::WriteChunkData(std::unique_ptr<BazBuffer>& bazBuffer,
                               std::vector<Primary::ZmwSliceHeader>& headerBuffer)
{
    size_t iHeader = 0;

    // Iterate over all ZmwSlices
    for (size_t zmwIdx = 0; zmwIdx < bazBuffer->NumZmw(); ++zmwIdx)
    {
        if (abort_) return;
        PBLOG_TRACE << " BazWriter::WriteChunkData " << iHeader;

        Primary::ZmwSliceHeader& h = headerBuffer[iHeader];

        Align(4);

        // Align every 1000th ZMW
        if (iHeader % 1000 == 0 && iHeader > 0)
        {
            Align(blocksize);
        }
        ++iHeader;

        auto slice          = bazBuffer->GetSlice(zmwIdx);
        h.offsetPacket      = Ftell();
        h.zmwIndex          = zmwIdx;
        h.packetsByteSize   = slice.packets.packetByteSize;
        h.numEvents         = slice.packets.numEvents;
        h.numMBs            = slice.metrics.size();

        WriteSanity();

        // Write bases/pulses
        if (slice.packets.packetByteSize != 0)
        {
            for (const auto& piece : slice.packets.pieces)
            {
                ioStats_.eventsBytes += Fwrite(piece.data, piece.count);
            }
        }

        // If there are no metrics, skip to new ZmwSlice
        if (h.numMBs == 0) continue;

        assert(fh_->MetricByteSize() != 0);
        // One buffer to rule them all. Minimize to one fwrite.
        // Buffer has the size for all three consecutive metric blocks

        uint64_t bufferSize = fh_->MetricByteSize() * h.numMBs;
        std::vector<uint8_t> buffer(bufferSize);
        size_t bufferCounter = 0;

        // Write metric blocks to buffer
        Write(fh_->MetricFields(), slice.metrics,
              buffer, bufferCounter);

        // Make sure that buffer has expected size
        assert(bufferCounter == bufferSize);
        ioStats_.metricsBytes +=  Fwrite(buffer.data(), bufferSize);
    }

    Align(blocksize);
}


void BazWriter::Write(const std::vector<Primary::MetricField>& metricFields,
                      const MemoryBufferView<TMetric>& metricBlocks,
                      std::vector<uint8_t>& buffer, size_t& c)
{
    // Iterate over all MBlocks
    for (size_t i = 0; i < metricBlocks.size(); ++i)
        metricBlocks[i].AppendToBaz(metricFields, buffer, c);
}

}}
