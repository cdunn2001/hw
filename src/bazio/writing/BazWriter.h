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

#ifndef PACBIO_BAZIO_WRITING_BAZ_WRITER_H
#define PACBIO_BAZIO_WRITING_BAZ_WRITER_H

#include <bazio/file/FileHeader.h>
#include <bazio/file/FileHeaderBuilder.h>

#include <bazio/ManuallyBufferedFile.h>
#include <bazio/BazIOConfig.h>
#include <bazio/MetricBlock.h>
#include <bazio/FileFooterBuilder.h>
#include <bazio/FileHeader.h>
#include <bazio/Sanity.h>
#include <bazio/SuperChunkMeta.h>
#include <bazio/ZmwSliceHeader.h>

#include <bazio/writing/BazBuffer.h>

#include <pacbio/ipc/ThreadSafeQueue.h>

#include <pacbio/POSIX.h>



namespace PacBio {
namespace BazIO {

struct IOStats
{
    size_t bytesWritten = 0;
    size_t headerBytes = 0;
    size_t eventsBytes = 0;
    size_t metricsBytes = 0;
    size_t overheadBytes = 0;
    size_t paddingBytes = 0 ;
};

// Helper class, to coordinate io stat reporting when there are multiple
// baz files being simultaneously written.  It's a bit brittle and expects
// a certain usage pattern to work, so it's not very suitable for a
// broader usage in other contexts
class IOStatsAggregator
{
public:

    explicit IOStatsAggregator(size_t numParticipants)
        : numParticipants_(numParticipants)
    {}

    ~IOStatsAggregator()
    {
        if (!statsInProgress_.empty())
            PBLOG_ERROR << "Destroying IOStatsAggregator while there are active segments being written!";
    }

    // Indicates that a thread has started doing some io operations that we will want
    // included in a particular report.  Ideally only one "segment" is active at a
    // time, but the aggregate tries to take care and behave corretly even if this is
    // not the case
    void StartSegment(uint32_t segment)
    {
        auto lg = std::lock_guard<std::mutex>{m_};

        if (statsInProgress_.count(segment) == 0)
        {
            PBLOG_INFO << "Starting a round of Baz IO";
        }
        statsInProgress_[segment].numStarted++;
    }

    // Aggregates and resets the stats argument to the full io stats.  The data
    // supplied here will not be included in any of the "segment" reporting
    void AggregateToFull(IOStats* m)
    {
        auto lg = std::lock_guard<std::mutex>{m_};

        AggregateImpl(&fullStats_, *m);
        *m = IOStats{};
    }

    // Aggregates and resets the stats argument to a particular "segment".  Once
    // all threads for a given segment have checked in, a brief summary of progress
    // will be logged
    void AggregateToSegment(uint32_t segmentID, IOStats* m)
    {
        auto lg = std::lock_guard<std::mutex>{m_};

        auto itr = statsInProgress_.find(segmentID);

        if (itr == statsInProgress_.end())
            throw PBException("Calling AggregateToSegment without calling StartSegment!");

        auto& segment = itr->second;

        AggregateImpl(&segment.stats, *m);
        *m = IOStats{};

        segment.numFinished++;
        if (segment.numFinished > segment.numStarted)
            throw PBException("Call to AggregtaeToSegment without matching call to StartSegment");
        if (segment.numFinished == numParticipants_)
        {
            PBLOG_INFO << "Finished a round of Baz IO: "
                       << "Wrote " << PrettyPrint(segment.stats.bytesWritten)
                       << " in " << segment.timer.GetElapsedMilliseconds() / 1000.f << " seconds";
            AggregateImpl(&fullStats_, segment.stats);
            statsInProgress_.erase(itr);
        }
    }

    void Summarize(std::ostream& os) const
    {
        auto lg = std::lock_guard<std::mutex>{m_};

        if (statsInProgress_.size() != 0)
            PBLOG_WARN << "Calling Summarize while there are active and unaggregated IO operations\n";

        os << "Header/Footer Bytes : " << PrettyPrint(fullStats_.headerBytes) << std::endl;
        os << "Padding Bytes       : " << PrettyPrint(fullStats_.paddingBytes) << std::endl;
        os << "OverheadBytes       : " << PrettyPrint(fullStats_.overheadBytes) << std::endl;
        os << "EventBytes          : " << PrettyPrint(fullStats_.eventsBytes) << std::endl;
        os << "MetricsBytes        : " << PrettyPrint(fullStats_.metricsBytes) << std::endl;
        os << "----------------------------------------" << std::endl;
        os << "Total BytesWritten_ : " << PrettyPrint(fullStats_.bytesWritten) << std::endl;
    }

    // BazWriter expects these for testing purposes
    size_t BytesWritten1() const
    {
        auto lg = std::lock_guard<std::mutex>{m_};

        if (statsInProgress_.size() != 0)
            PBLOG_WARN << "Calling BytesWritten1 while there are active and unaggregated IO operations\n";

        return fullStats_.headerBytes + fullStats_.paddingBytes + fullStats_.overheadBytes + fullStats_.eventsBytes + fullStats_.metricsBytes;
    }
    size_t BytesWritten2() const
    {
        auto lg = std::lock_guard<std::mutex>{m_};

        if (statsInProgress_.size() != 0)
            PBLOG_WARN << "Calling BytesWritten2 while there are active and unaggregated IO operations\n";

        return fullStats_.bytesWritten;
    }

public: // Static functions

    static std::string PrettyPrint(size_t bytes)
    {
        std::stringstream stream;
        stream << std::setprecision(4);
        if (bytes > (1ull << 40))
            stream <<  bytes / static_cast<float>(1ull << 40) <<  " TiB";
        else if (bytes > (1ull << 30))
            stream <<  bytes / static_cast<float>(1ull << 30) <<  " GiB";
        else if (bytes > (1ull << 20))
            stream <<  bytes / static_cast<float>(1ull << 20) <<  " MiB";
        else if (bytes > (1ull << 10))
            stream <<  bytes / static_cast<float>(1ull << 10) <<  " KiB";
        else
            stream <<  bytes <<  " B";

        return stream.str();
    }

private:

    // Private function, because it needs the mutex to already
    // be held before entering
    static void AggregateImpl(IOStats* left, const IOStats& right)
    {
        left->bytesWritten += right.bytesWritten;
        left->headerBytes += right.headerBytes;
        left->eventsBytes += right.eventsBytes;
        left->metricsBytes += right.metricsBytes;
        left->overheadBytes += right.overheadBytes;
        left->paddingBytes += right.paddingBytes;
    }

    mutable std::mutex m_;

    struct Segment
    {
        Dev::QuietAutoTimer timer;
        size_t numStarted = 0;
        size_t numFinished = 0;
        IOStats stats;
    };

    std::map<uint32_t, Segment> statsInProgress_;
    IOStats fullStats_;

    uint32_t numParticipants_;
};

template <typename MetricBlockT,typename AggregatedMetricBlockT>
class BazWriter
{
public: // static
    using TMetric = Primary::SpiderMetricBlock;

public: // structors

    /// Creates a new BAZ file and writes file header.
    /// \param filePath          File name of the output BAZ file.
    /// \param fileHeaderBuilder JSON of file header from a builder.
    BazWriter(const std::string& filePath,
              FileHeaderBuilder& fileHeaderBuilder,
              const Primary::BazIOConfig& ioConf,
              std::shared_ptr<IOStatsAggregator> agg
                  = std::make_shared<IOStatsAggregator>(1))
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

    BazWriter(BazWriter&&) = delete;
    BazWriter(const BazWriter&) = delete;
    BazWriter& operator=(BazWriter&& rhs) = delete;
    BazWriter& operator=(const BazWriter&) = delete;


    /// Waits for writing thread to finish,
    /// before writing EOF chunk meta.
    ~BazWriter()
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

    static constexpr int blocksize = 4000;
public:

    /// Push internal BazBuffer to the queue of BazBuffers that are
    /// ready to be written to disk.
    ///
    /// \return bool if data was available to be flushed out
    void Flush(std::unique_ptr<BazBuffer<MetricBlockT,AggregatedMetricBlockT>> buffer)
    {
        if (abort_ || !writeThreadContinue_) return;
        bazBufferQueue_.Push(std::move(buffer));
    }

    /// Waits until internal buffers have been written. If this method is not
    /// being called, data may get lost.
    void WaitForTermination()
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

    // This is used for simulations to avoid flooding RAM.
    inline size_t BufferSize() const
    { return bazBufferQueue_.Size(); }

    /// Return the current file path of the BAZ file.
    const std::string& FilePath() const
    { return filePath_; }

    std::string Summary() const
    {
        std::stringstream ss;
        Summarize(ss);
        return ss.str();
    }

    void Summarize(std::ostream& os) const
    {
        ioAggregator_->Summarize(os);
    }

    std::shared_ptr<const IOStatsAggregator> GetAggregator() const { return ioAggregator_; }

    size_t BytesWritten() const
    {
        size_t bw = BytesWritten1();
        if (bw != BytesWritten2())
        {
            throw PBException("Internal miscount of byteswritten, " + std::to_string(BytesWritten1()) +
                              " " + std::to_string(BytesWritten2()));
        }
        return bw;
    }
    const FileHeaderBuilder& GetFileHeaderBuilder() const { return fhb_;}

protected:
    size_t BytesWritten1() const
    {
        return ioAggregator_->BytesWritten1();
    }

    size_t BytesWritten2() const {
        return ioAggregator_->BytesWritten2();
    }

private: // types
    using BazBufferT = BazBuffer<MetricBlockT,AggregatedMetricBlockT>;
    using BufferQueue = PacBio::ThreadSafeQueue<std::unique_ptr<BazBufferT>>;

private: // data

    std::atomic_bool writeThreadContinue_;

    std::string filePath_;
    std::string filePathTmp_;

    std::thread writeThread_;

    std::unique_ptr<FileHeader> fh_;

    BufferQueue bazBufferQueue_;
    FileHeaderBuilder fhb_;
    Primary::FileFooterBuilder ffb_;
    std::unique_ptr<Primary::ManuallyBufferedFile> fileHandle_;

    std::vector<char> jsonFileHeader_;

    bool abort_ = false;
    std::atomic_bool dieGracefully_;

    std::shared_ptr<IOStatsAggregator> ioAggregator_;
    IOStats ioStats_;

private: // modifying methods

    /// Is executed in its own thread. Writes BazBuffers to disk.
    void WriteToDiskLoop()
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
            std::unique_ptr<BazBufferT> bazBuffer;
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

private: // non-modifying methods

    void Init(const Primary::BazIOConfig& ioConfig)
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

    /// Writes meta information of super chunk.
    /// \param nextPointer offset to next meta
    /// \param numZmws     number of ZmwSlices in this super chunk
    /// \param accumulate  if true, accumulates bytes written statistics
    void WriteSuperChunkMeta(const uint64_t nextPointer, uint32_t numZmws, bool accumulate = true)
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


    /// Writes header information for each ZmwSlice for current super chunk.
    /// \param headerBuffer current header buffer to be written
    void WriteZmwSliceHeader(std::vector<Primary::ZmwSliceHeader>& headerBuffer)
    {
        for (const auto& h : headerBuffer)
        {
            if (abort_) return;
            ioStats_.overheadBytes += Fwrite(&h, Primary::ZmwSliceHeader::SizeOf());
        }
    }

    /// Writes base calls/pulses and metrics for each ZMW slice
    /// for current super chunk.
    /// \param bazBuffer    current buffer to be written
    /// \param headerBuffer corresponding header buffer for ZmwSlice offsets
    void WriteChunkData(std::unique_ptr<BazBufferT>& bazBuffer,
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

            auto slice = bazBuffer->GetSlice(zmwIdx);
            h.offsetPacket = Ftell();
            h.zmwIndex = zmwIdx;
            h.packetsByteSize = slice.packets.packetByteSize;
            h.numEvents       = slice.packets.numEvents;
            h.numHFMBs        = 0;
            h.numMFMBs        = slice.metrics.size();
            // h.numMFMBs        = (hFbyMFRatio_ > 0) ? ((slice.hfmbs.size() + hFbyMFRatio_ - 1) / hFbyMFRatio_) : 0;
            h.numLFMBs        = 0;//(hFbyLFRatio_ > 0) ? ((slice.hfmbs.size() + hFbyLFRatio_ - 1) / hFbyLFRatio_) : 0;

            WriteSanity();

            // Write bases/pulses
            if (slice.packets.packetByteSize != 0)
            {
                for (const auto& piece : slice.packets.pieces)
                {
                    ioStats_.eventsBytes += Fwrite(piece.data, piece.count);
                }
            }

            if (h.numHFMBs == 0 && (fh_->HFMetricByteSize() + fh_->MFMetricByteSize() + fh_->LFMetricByteSize() != 0))
            {
                //PBLOG_WARN << fh_->HFMetricByteSize() << "," << fh_->MFMetricByteSize() << "," << fh_->LFMetricByteSize();
                //Error("kv.second->numHFMBs is zero but the metrics sizes are not zero!");
                //continue;
            }
            if (h.numHFMBs != 0 && (fh_->HFMetricByteSize() + fh_->MFMetricByteSize() + fh_->LFMetricByteSize() == 0))
            {
                //PBLOG_WARN << kv.second->numHFMBs;
                //Error("kv.second->numHFMBs is not zero but the metrics sizes are zero!");
                continue;
            }


            // If there are no metrics, skip to new ZmwSlice
            if (h.numMFMBs == 0) continue;

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
                if (h.numMFMBs > 0)
                    Write(fh_->MFMetricFields(), slice.metrics,
                          buffer, bufferCounter);

                // Things are temporarily hacked to pieces.  We only
                // write at one frequency, which is not aggregated, and
                // isn't even guanteed consistent with the interval specified
                // in the header.  This should all change in relatively
                // short order once metrics handling is re-written

                //// Write medium-frequency metric blocks to buffer
                //if (h.numMFMBs > 0)
                //    Write(fh_->MFMetricFields(), slice.hfmbs,
                //          hFbyMFRatio_, buffer, bufferCounter);

                //// Write low-frequency metric blocks to buffer
                //if (h.numLFMBs > 0)
                //    Write(fh_->LFMetricFields(), slice.hfmbs,
                //          hFbyLFRatio_, buffer, bufferCounter);

                // Make sure that buffer has expected size
                assert(bufferCounter == bufferSize);
                ioStats_.metricsBytes +=  Fwrite(buffer.data(), bufferSize);
            }
        }

        Align(blocksize);
    }


    /// Write high-frequency metrics blocks to given buffer.
    /// \param hFMetricFields   User-defined high-frequency metric fields
    /// \param metricBlocks     HighFrequencyMetricBlock array
    /// \param numMetricBlocks  Number of HighFrequencyMetricBlocks
    /// \param buffer           Buffer to be appended
    /// \param c                Reference to buffer index
    void Write(const std::vector<Primary::MetricField>& hFMetricFields,
               const MemoryBufferView<TMetric>& metricBlocks,
               std::vector<uint8_t>& buffer, size_t& c)
    {
        // Iterate over all HFMBlocks
        for (size_t i = 0; i < metricBlocks.size(); ++i)
            metricBlocks[i].AppendToBaz(hFMetricFields, buffer, c);
    }

protected:

    void Align(size_t alignmentSize, bool accumulate = true)
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

    /// C++ wrapper to ftell
    /// \returns byte position from the start of the file
    size_t Ftell() const
    {
        return fileHandle_->Ftell();
    }

    /// C++ wrapper to fseek to an absolute file position
    /// \param position - position from start of file
    void FseekAbsolute(size_t position)
    {
        fileHandle_->FseekAbsolute(position);
    }

    /// C++ wrapper to fseek to a relative file position
    /// \param position - position from current file pointer
    void FseekRelative(size_t position)
    {
        fileHandle_->FseekRelative(position);
    }

    /// C++ wrapper to fwrite that throws an exception on error, keeps track of
    /// bytes written and will throw if the global abort_ flag is asserted.
    /// \param data - pointer to data to write
    /// \param size - number of bytes to write
    /// \param accumulate - if true, then aggregate the bytes written into the statistics
    size_t Fwrite(const void* data, size_t size, bool accumulate = true)
    {
        if (abort_) throw PBException("aborted");
        if (size > 0)
        {
            fileHandle_->Fwrite(data, size);
            if (accumulate) ioStats_.bytesWritten += size;
        }
        return size;
    }

    /// Writes the sanity word and keeps track of bytes written.
    void WriteSanity(bool accumulate = true)
    {
        std::array<uint8_t, 4> data;
        Primary::Sanity::Write(data);
        fileHandle_->Fwrite(data.data(), 4);
        if (accumulate)
        {
            ioStats_.bytesWritten += 4;
            ioStats_.overheadBytes += 4;
        }
    }

    /// Write the header to the fileHandle_. Because the header can be big, this method may loop
    /// over Fwrite as needed.
    /// \param accumulate - if true, then aggregate the bytes written into the statistics
    void WriteJsonFileHeader(bool accumulate)
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

};

}}

#endif //PACBIO_BAZIO_WRITING_BAZ_WRITER_H
