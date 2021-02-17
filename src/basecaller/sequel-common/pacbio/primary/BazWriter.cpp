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

#include <errno.h>
#include <cstdio>
#include <cstring>
#include <exception>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <thread>

#include <pacbio/primary/Timing.h>
#include <pacbio/primary/MetricFieldMap.h>
#include <pacbio/primary/BazWriter.h>

#include <pacbio/PBException.h>
#include <pacbio/primary/MemoryBuffer.h>
#include <pacbio/POSIX.h>

#include <pacbio/logging/Logger.h>

#ifndef PB_MIC_COPROCESSOR
#include <unistd.h>
#endif

namespace PacBio {
namespace Primary {

template <typename TMetric>
BazWriter<TMetric>::BazWriter(const std::string& filePath,
                              FileHeaderBuilder& fileHeaderBuilder,
                              const Primary::BazIOConfig& ioConf,
                              const size_t maxNumSamples,
                              bool silent)
    : maxNumSamples_(maxNumSamples)
    , success_(true)
    , filePath_(filePath)
    , filePathTmp_(filePath + ".tmp")
    , p2b_(std::unique_ptr<PrimaryToBaz<TMetric>>(new PrimaryToBaz<TMetric>(
            fileHeaderBuilder.MaxNumZmws(),
            fileHeaderBuilder.ReadoutConfig())))
    , maxZmwNumber_(fileHeaderBuilder.MaxNumZmws())
    , silent_(silent)
    , fhb_(fileHeaderBuilder)
    , metricsVerbosity_(fileHeaderBuilder.MetricsVerbosityConfig())
    , fileHandle_(new Primary::ManuallyBufferedFile(filePathTmp_, ioConf.BufferSizeMBytes, ioConf.MaxMBPerSecondWrite))
    , jsonFileHeader_(fileHeaderBuilder.CreateJSONCharVector())
    , dieGracefully_(false)
{
    if (!fileHeaderBuilder.MaxNumZmws())
        Error("No ZmwNumber mapping provided in FileHeaderBuilder");
    Init(ioConf);
}

template <typename TMetric>
BazWriter<TMetric>::~BazWriter() noexcept
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

#ifdef LOG_MISSING_ZMWS
        WriteSkippedZmws();
#endif
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

template <typename TMetric>
void BazWriter<TMetric>::WriteSkippedZmws()
{
    bool fileOpen = false;
    std::ofstream output;
    
    const auto end = zmwsAdded_.end();
    for (const auto x : fh_->ZmwNumbers())
    {
        if (zmwsAdded_.find(x) == end)
        {
            if (!fileOpen)
            {
                output.open(filePath_ + ".notAddedZmws.debug", std::ofstream::out);
                fileOpen = true;
            }
            output << x << std::endl;
        }
    }
    if (fileOpen) output.close();
}

template <typename TMetric>
void BazWriter<TMetric>::Init(const Primary::BazIOConfig& ioConfig)
{
    // Parse ASCII JSON to FileHeader instance,
    // includes sanity checks for validity
    fh_ = std::unique_ptr<FileHeader>(new FileHeader(jsonFileHeader_.data(), jsonFileHeader_.size()));

    // Compute ratio of hf to mf and hf to lf
    if (fh_->HFMetricFrames() == 0)
    {
        hFbyMFRatio_ = 0;
        hFbyLFRatio_ = 0;
    }
    else
    {
        hFbyMFRatio_ = static_cast<uint8_t>(fh_->MFMetricFrames()/fh_->HFMetricFrames());
        hFbyLFRatio_ = static_cast<uint8_t>(fh_->LFMetricFrames()/fh_->HFMetricFrames());
        PBLOG_DEBUG << "hFbyMFRatio_:" << hFbyMFRatio_ << "hFbyLFRatio_:" << hFbyLFRatio_;

        if (hFbyMFRatio_ == 0 || hFbyLFRatio_ == 0) Error("hFbyMFRatio_ == 0 || hFbyLFRatio_ == 0");
    }

    // Initialize the BazBuffer that will be written to.  We have to make a
    // guess as to how much memory it might need for pulses and metrics
    auto bytesPerZmwGuess = 500;
    uint32_t numBazBuffers = ioConfig.numBazBuffers;
    uint32_t numMetrics = ioConfig.numMetrics;
// #define SPI1595_METRICS
#if 1
#ifdef SPI1595_METRICS
    PBLOG_INFO << "SPI1595 numBazBuffers:" << numBazBuffers;
#endif
    for(uint32_t i=0;i<numBazBuffers;i++)
    {
        auto buffer = std::unique_ptr<BazBuffer<TMetric>>(new BazBuffer<TMetric>(
            maxZmwNumber_,
            numMetrics,
            maxZmwNumber_ * bytesPerZmwGuess,
            maxZmwNumber_ * bytesPerZmwGuess * 10));
        idleBuffers_.Push(std::move(buffer));
    }
    currentBazBuffer_ = idleBuffers_.Pop();
#else
    currentBazBuffer_ = std::unique_ptr<BazBuffer<TMetric>>(new BazBuffer<TMetric>(
            maxZmwNumber_,
            numMetrics,
            maxZmwNumber_ * bytesPerZmwGuess,
            maxZmwNumber_ * bytesPerZmwGuess * 10));
#endif


    // Start thread that saves BazBuffers to disk
    writeThreadContinue_ = true;
    writeThread_ = std::thread([this](){
        try
        {
            WriteToDiskLoop();
        }
        catch(std::exception& err)
        {
            errorMsg_ = err.what();
            // Forward
            PBLOG_ERROR << err.what();
            // Set internal state to error
            success_ = false;
            // Set truncated flag
            fhb_.Truncated();
            // Empty buffers
            while (!bazBufferQueue_.Empty())
                bazBufferQueue_.Pop();
            // Die gracefully
        }
    });
}

template <typename TMetric>
bool BazWriter<TMetric>::AddZmwSlice(const Basecall* basecall, const uint16_t numEvents,
                                     const std::vector<TMetric>& hfMetrics,
                                     const uint32_t zmwId)
{

    MemoryBufferView<TMetric> metrics{};

    if (!hfMetrics.empty())
    {
        metrics = currentBazBuffer_->MetricsBuffer().Copy(hfMetrics.data(), hfMetrics.size());
    }

    return AddZmwSlice(basecall, numEvents, metrics, zmwId);
}

template <typename TMetric>
bool BazWriter<TMetric>::AddZmwSlice(const Basecall* basecall, const uint16_t numEvents,
                                     const MemoryBufferView<TMetric>& metrics,
                                     const uint32_t zmwId)
{
#if 0
    if (zmwId < 100)
    {
        std::stringstream ss;
        ss << "zmwId: " << zmwId << " numEvents:" << numEvents << " metricsSize" << hfMetrics.size() << " ";
        for(uint32_t i=0;i<numEvents;i++)
        {
            ss << basecall[i] << " ";
        }
        PBLOG_INFO << ss.str();
        PBLOG_INFO << "success_:" << success_;
    }
    if (!success_) PBLOG_ERROR << "success_ flag was false, will skip zmwId:" << zmwId;

#endif
    if (!success_) return false;

    if (numEvents == 0 && metrics.empty()) return success_;

    if (zmwId >= maxZmwNumber_)
        Error("Provided zmw number " + std::to_string(zmwId)
              + ", which larger than the chip size of "
              + std::to_string(maxZmwNumber_));

    uint16_t numIncorporatedEvents = 0;
    uint32_t packetsByteSize = 0;

    // Create fake smrtptr as basecalls or pulses may be empty
    MemoryBufferView<uint8_t> packets{};

    // Only convert if there are bases/pulses
    if (basecall != nullptr && numEvents > 0)
    {
        packets = p2b_->SliceToPacket(basecall,
                                      numEvents,
                                      zmwId,
                                      currentBazBuffer_->PacketBuffer(),
                                      &packetsByteSize,
                                      &numIncorporatedEvents);
        // In production mode, it can happen that the provided Basecall*
        // only contains pulses.
        if (numIncorporatedEvents == 0 || packetsByteSize == 0)
        {
            numZmwsZeroIncorporatedEvents_++;
        }
    }

    // Because of the no bases in production possibility and if there are also
    // no metrics provided, don't try to add an empty ZMW.
    if (numIncorporatedEvents == 0 && metrics.empty())
    {
        numZmwsEmptyMetrics_++;
        return true;
    }

    auto& count = zmwsAdded_[fh_->ZmwIdToNumber(zmwId)];
    if (count != numSuperChunks_)
    {
        Error("Latency of ZMW id " + std::to_string(zmwId) + " detected."
            + " This is a serious problem. Aborting."
            + " Won't accept further incoming data.");
        return success_;
    }
    else
        count++;

    // If we made it until here, there are either bases/pulses and/or metrics
    ZmwSlice<TMetric> zs(packets, metrics);
    zs.NumEvents(numIncorporatedEvents);
    zs.PacketsByteSize(packetsByteSize);
    zs.ZmwIndex(zmwId);
    
    if (numEvents == maxNumSamples_)
        ffb_.AddTruncation(zmwId, numSuperChunks_);

    numEvents_ += numIncorporatedEvents;
    if (success_)
        currentBazBuffer_->AddZmwSlice(zs, hFbyMFRatio_, hFbyLFRatio_);
    return success_;
}

#ifdef SPI1595_METRICS
struct
{
    double totalIdleBlockingTime;
} spi1595 = {0.0};
#endif

template <typename TMetric>
bool BazWriter<TMetric>::Flush( )
{
    if (!success_ || abort_ || !writeThreadContinue_) return false;

    if (numZmwsZeroIncorporatedEvents_ > 0)
    {
        PBLOG_WARN << numZmwsZeroIncorporatedEvents_ << " ZMWs for superchunk:"
                   << numSuperChunks_ << " had numIncorporatedEvents:0 packetsByteSize:0";
        numZmwsZeroIncorporatedEvents_ = 0;
    }

    if (numZmwsEmptyMetrics_ > 0)
    {
        PBLOG_WARN << "Dropped " << numZmwsEmptyMetrics_ << " ZMWs for superchunk:"
                   << numSuperChunks_ << " because numIncorporatedEvents:0 and no metrics";
        numZmwsEmptyMetrics_ = 0;
    }

    if (currentBazBuffer_->Size() > 0)
    {
        assert(currentBazBuffer_->Size() == maxZmwNumber_);
        assert(currentBazBuffer_->SeenAllZmw());
        bazBufferQueue_.Push(std::move(currentBazBuffer_));

#ifdef SPI1595_METRICS
        double t0 = PacBio::Utilities::Time::GetMonotonicTime();
#endif

        currentBazBuffer_ = idleBuffers_.Pop();

#ifdef SPI1595_METRICS
        double t1 = PacBio::Utilities::Time::GetMonotonicTime();
        double elapsedTime = t1 - t0;
        spi1595.totalIdleBlockingTime += elapsedTime;
        PBLOG_INFO << "SPI1595 elapsedTime:" << elapsedTime <<
                    " totalIdleBlockingTime:" << spi1595.totalIdleBlockingTime;
#endif
        ++numSuperChunks_;
    }
    return success_;
}

/// Waits until all blocks have entered the writer thread, and then waits
/// for the writer thread to exit.
template <typename TMetric>
void BazWriter<TMetric>::WaitForTermination()
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
template <typename TMetric>
void BazWriter<TMetric>::Align(size_t alignmentSize, bool accumulate)
{
    size_t ragged = Ftell() % alignmentSize;
    if (ragged)
    {
        size_t advance = alignmentSize - ragged;
        fileHandle_->Advance(advance);
        if (accumulate)
        {
            paddingBytes_ += advance;
            bytesWritten_ += advance;
        }
    }
}

template <typename TMetric>
void BazWriter<TMetric>::WriteJsonFileHeader(bool accumulate)
{
    /// Fwrite doesn't allow to write really big buffers. The write is broken into multiple smaller pieces.
    size_t bytesToWrite = jsonFileHeader_.size();
    const auto* ptr = jsonFileHeader_.data();
    while(bytesToWrite >0)
    {
        size_t bytesToWriteNow = (1UL << 20); // let's try 1 MiB each loop. 2GiB was too much.
        if (bytesToWriteNow > bytesToWrite) bytesToWriteNow = bytesToWrite;
        const auto justWrote = Fwrite(ptr,bytesToWriteNow, accumulate);
        if (accumulate) headerBytes_ += justWrote;
        bytesToWrite -= bytesToWriteNow;
        ptr += bytesToWriteNow;
    }
}

/// This is essentially the thread function that writes the entire baz file
/// from header to footer.
/// To end the thread, stop adding new baz buffers and then call WaitForTermination()
template <typename TMetric>
void BazWriter<TMetric>::WriteToDiskLoop()
{
    PBLOG_DEBUG << "BazWriter::WriteToDiskLoop() header, length: " << jsonFileHeader_.size();
    WriteJsonFileHeader(true /* accumulate bytes in statistics */);
    WriteSanity();

    Align(blocksize);

    std::vector<ZmwSliceHeader> headerBuffer;
    PBLOG_DEBUG << "BazWriter::WriteToDiskLoop() loop entered";
    size_t writtenSuperChunks = 0;
    while (writeThreadContinue_ && !abort_)
    {
        std::unique_ptr<BazBuffer<TMetric>> bazBuffer;
        // Wait for new data, but only for 500ms. Otherwise restart the loop,
        // as it could have been the last item and the loop will be exited.

        if (bazBufferQueue_.Pop(bazBuffer, std::chrono::milliseconds(500)))
        {
            auto now = std::chrono::high_resolution_clock::now();

            PBLOG_INFO << "Starting superchunk " << writtenSuperChunks+1
                       << ", Events, MBlocks, Bytes: " << bazBuffer->NumEvents()
                       << ", " << bazBuffer->NumHFMBs() + bazBuffer->NumMFMBs() + bazBuffer->NumLFMBs()
                       << ", " << bazBuffer->MetricsDataSize() * sizeof(TMetric) + bazBuffer->PacketDataSize();

            // super chunk layout: SMSHHHHSPHMLSPHMLSPHMLSPHML
            uint32_t numZmws = bazBuffer->ZmwData().size();
            headerBuffer.resize(numZmws);

            PBLOG_DEBUG << "BazWriter::WriteToDiskLoop()  superchunk writing ... numZmws:" << numZmws;

            // keep track of where the file pointer, because we will rewind to this location
            // after the data is written.
            size_t superChunkStart = Ftell();

            // write temporary SuperChunkMeta, with zeros, in case file gets truncated
            WriteSuperChunkMeta(0, 0);

            // skip over headers
            FseekRelative(ZmwSliceHeader::SizeOf() * numZmws );

            // write actual data
            WriteChunkData(bazBuffer, headerBuffer);

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

            if (!silent_) Timing::PrintTime(now, "Buffer written");
            PBLOG_INFO << "Finished writing superchunk " << writtenSuperChunks << " to disk";
            writtenSuperChunks++;
            fileHandle_->Flush();

            bazBuffer->Reset();
            idleBuffers_.Push(std::move(bazBuffer));
        }
#ifdef FAKE_ERROR
        if (fakeError_)
            Error("Fake error");
#endif
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
    headerBytes_ += Fwrite(jsonFileFooter_.data(), jsonFileFooter_.size());
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
template <typename TMetric>
void BazWriter<TMetric>::WriteSuperChunkMeta(const uint64_t nextPointer,
                                             uint32_t numZmws, bool accumulate)
{
    SuperChunkMeta chunkMeta;
    chunkMeta.offsetNextChunk = nextPointer;
    chunkMeta.numZmws         = numZmws;

    WriteSanity(accumulate);
    size_t written =  Fwrite(&chunkMeta, SuperChunkMeta::SizeOf(), accumulate);
    if (accumulate)
    {
        overheadBytes_ += written;
    }
    WriteSanity(accumulate);
}

/// write the ZmwSliceHeader
template <typename TMetric>
void BazWriter<TMetric>::WriteZmwSliceHeader(std::vector<ZmwSliceHeader>& headerBuffer)
{
    for (const auto& h : headerBuffer)
    {
        if (abort_) return;
        overheadBytes_ += Fwrite(&h, ZmwSliceHeader::SizeOf());
    }
}

/// Write the SuperChunk actual data (base calls and metrics)
template <typename TMetric>
void BazWriter<TMetric>::WriteChunkData(std::unique_ptr<BazBuffer<TMetric>>& bazBuffer,
                                        std::vector<ZmwSliceHeader>& headerBuffer)
{
    size_t iHeader = 0;

    // Iterate over all ZmwSlices
    for (const auto& slice : bazBuffer->ZmwData())
    {
        if (abort_) return;
        PBLOG_TRACE << " BazWriter::WriteChunkData " << iHeader;

        ZmwSliceHeader& h = headerBuffer[iHeader];

        Align(4);

        // Align every 1000th ZMW
        if (iHeader % 1000 == 0 && iHeader > 0)
        {
            Align(blocksize);
        }
        ++iHeader;

        h.offsetPacket = Ftell();
        h.zmwIndex = slice.zmwIndex;
        h.packetsByteSize = slice.packetsByteSize;
        h.numEvents       = slice.numEvents;
        h.numHFMBs        = slice.hfmbs.size();
        h.numMFMBs        = (hFbyMFRatio_ > 0) ? ((slice.hfmbs.size() + hFbyMFRatio_ - 1) / hFbyMFRatio_) : 0;
        h.numLFMBs        = (hFbyLFRatio_ > 0) ? ((slice.hfmbs.size() + hFbyLFRatio_ - 1) / hFbyLFRatio_) : 0;

        WriteSanity();

#if 0
        if (h.zmwIndex < 100)
        {
            PBLOG_INFO << "Fwrite of zmwId:" << h.zmwIndex << " " << h.numEvents << " events, bytes:" << slice.packetsByteSize << " numhfmbs:" << h.numHFMBs;
        }
#endif
        // Write bases/pulses
        if (slice.packetsByteSize != 0)
            eventsBytes_ += Fwrite(&slice.packets[0], slice.packetsByteSize);

        if (h.numHFMBs == 0 && (fh_->HFMetricByteSize() + fh_->MFMetricByteSize() + fh_->LFMetricByteSize() != 0))
        {
            //PBLOG_WARN << fh_->HFMetricByteSize() << "," << fh_->MFMetricByteSize() << "," << fh_->LFMetricByteSize();
            //Error("kv.second->numHFMBs is zero but the metrics sizes are not zero!");
            continue;
        }
        if (h.numHFMBs != 0 && (fh_->HFMetricByteSize() + fh_->MFMetricByteSize() + fh_->LFMetricByteSize() == 0))
        {
            //PBLOG_WARN << kv.second->numHFMBs;
            //Error("kv.second->numHFMBs is not zero but the metrics sizes are zero!");
            continue;
        }


        // If there are no metrics, skip to new ZmwSlice
        if (h.numHFMBs == 0) continue;

        assert(fh_->HFMetricByteSize() + fh_->MFMetricByteSize() + fh_->LFMetricByteSize() != 0);
        // One buffer to rule them all. Minimize to one fwrite.
        // Buffer has the size for all three consecutive metric blocks

        uint64_t bufferSize =
                fh_->HFMetricByteSize() * h.numHFMBs +
                fh_->MFMetricByteSize() * h.numMFMBs +
                fh_->LFMetricByteSize() * h.numLFMBs;

        if (bufferSize > 0)
        {
            std::vector<uint8_t> buffer(bufferSize);
            size_t bufferCounter = 0;

            // Write high-frequency metric blocks to buffer
            if (h.numHFMBs > 0)
                Write(fh_->HFMetricFields(), slice.hfmbs,
                      buffer, bufferCounter);

            // Write medium-frequency metric blocks to buffer
            if (h.numMFMBs > 0)
                Write(fh_->MFMetricFields(), slice.hfmbs,
                      hFbyMFRatio_, buffer, bufferCounter);

            // Write low-frequency metric blocks to buffer
            if (h.numLFMBs > 0)
                Write(fh_->LFMetricFields(), slice.hfmbs,
                      hFbyLFRatio_, buffer, bufferCounter);

            // Make sure that buffer has expected size
            assert(bufferCounter == bufferSize);
            metricsBytes_ +=  Fwrite(buffer.data(), bufferSize);
        }
    }

    Align(blocksize);
}

template <typename TMetric>
void BazWriter<TMetric>::Write(const std::vector<MetricField>& hFMetricFields,
                               const MemoryBufferView<TMetric>& metricBlocks,
                               std::vector<uint8_t>& buffer, size_t& c)
{
    // Iterate over all HFMBlocks
    for (size_t i = 0; i < metricBlocks.size(); ++i)
        metricBlocks[i].AppendToBaz(hFMetricFields, buffer, c);
}

template <typename TMetric>
void BazWriter<TMetric>::Write(const std::vector<MetricField>& metricFields,
                               const MemoryBufferView<TMetric>& metricBlocks,
                               const size_t ratioToHFMetrics, std::vector<uint8_t>&  buffer, size_t& c)
{

    if (ratioToHFMetrics == 0)
    {
        throw PBException("ratioToHFMetrics is zero");
    }

    // Aggregate HFMBlocks, left over blocks are handled at the end
    for (size_t i = 0; i < metricBlocks.size(); i += ratioToHFMetrics)
    {
        TMetric newBlock(&metricBlocks[i],
                         i + ratioToHFMetrics < metricBlocks.size()
                         ? ratioToHFMetrics : metricBlocks.size() - i);
        newBlock.AppendToBaz(metricFields, buffer, c);
    }
}


/// handle non fatal errors. Fatal errors should be handled with an exception.
template <typename TMetric>
void BazWriter<TMetric>::Error(const std::string& errorMsg)
{
    errorMsg_ = errorMsg;
    PBLOG_ERROR << errorMsg_;
    success_ = false;
}

//
// Explicit instantiations
//

template class BazWriter<SequelMetricBlock>;
template class BazWriter<SpiderMetricBlock>;
  
}}
