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

#include "bazio/ManuallyBufferedFile.h"
#include <bazio/BazIOConfig.h>
#include <bazio/MetricBlock.h>
#include <bazio/FileFooterBuilder.h>
#include <bazio/FileHeader.h>
#include <bazio/Sanity.h>
#include <bazio/writing/BazBuffer.h>
#include <bazio/ZmwSliceHeader.h>
#include <pacbio/ipc/ThreadSafeQueue.h>

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
                  = std::make_shared<IOStatsAggregator>(1));

    BazWriter(BazWriter&&) = delete;
    BazWriter(const BazWriter&) = delete;
    BazWriter& operator=(BazWriter&& rhs) = delete;
    BazWriter& operator=(const BazWriter&) = delete;


    /// Waits for writing thread to finish,
    /// before writing EOF chunk meta.
    ~BazWriter();

    static constexpr int blocksize = 4000;
public:

    /// Push internal BazBuffer to the queue of BazBuffers that are
    /// ready to be written to disk.
    ///
    /// \return bool if data was available to be flushed out
    void Flush(std::unique_ptr<BazBuffer> buffer);

    /// Waits until internal buffers have been written. If this method is not
    /// being called, data may get lost.
    void WaitForTermination();

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
    using BufferQueue = PacBio::ThreadSafeQueue<std::unique_ptr<BazBuffer>>;

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
    void WriteToDiskLoop();

private: // non-modifying methods

    void Init(const Primary::BazIOConfig& ioConfig);

    /// Writes meta information of super chunk.
    /// \param nextPointer offset to next meta
    /// \param numZmws     number of ZmwSlices in this super chunk
    /// \param accumulate  if true, accumulates bytes written statistics
    void WriteSuperChunkMeta(const uint64_t nextPointer, uint32_t numZmws, bool accumulate = true);

    /// Writes header information for each ZmwSlice for current super chunk.
    /// \param headerBuffer current header buffer to be written
    void WriteZmwSliceHeader(std::vector<Primary::ZmwSliceHeader>& headerBuffer);

    /// Writes base calls/pulses and metrics for each ZMW slice
    /// for current super chunk.
    /// \param bazBuffer    current buffer to be written
    /// \param headerBuffer corresponding header buffer for ZmwSlice offsets
    void WriteChunkData(std::unique_ptr<BazBuffer>& bazBuffer,
                        std::vector<Primary::ZmwSliceHeader>& headerBuffer);

    /// Write high-frequency metrics blocks to given buffer.
    /// \param hFMetricFields   User-defined high-frequency metric fields
    /// \param metricBlocks     HighFrequencyMetricBlock array
    /// \param numMetricBlocks  Number of HighFrequencyMetricBlocks
    /// \param buffer           Buffer to be appended
    /// \param c                Reference to buffer index
    void Write(const std::vector<Primary::MetricField>& hFMetricFields,
               const MemoryBufferView<TMetric>& metricBlocks,
               std::vector<uint8_t>& buffer, size_t& c);

protected:

    void Align(size_t alignmentSize, bool accumulate = true);

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
    void WriteJsonFileHeader(bool accumulate);

};

}}

#endif //PACBIO_BAZIO_WRITING_BAZ_WRITER_H
