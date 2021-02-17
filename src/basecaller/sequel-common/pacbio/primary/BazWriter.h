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

#define BYTETOBINARY(byte)  \
  (byte & 0x80 ? 1 : 0), \
  (byte & 0x40 ? 1 : 0), \
  (byte & 0x20 ? 1 : 0), \
  (byte & 0x10 ? 1 : 0), \
  (byte & 0x08 ? 1 : 0), \
  (byte & 0x04 ? 1 : 0), \
  (byte & 0x02 ? 1 : 0), \
  (byte & 0x01 ? 1 : 0)

#include <atomic>
#include <cstdio>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <string>
#include <thread>
#include <stdio.h>
#include <string.h>
#include <tuple>
#include <utility>
#include <vector>

#include <json/json.h>

#include <pacbio/ipc/ThreadSafeQueue.h>

#include <pacbio/primary/BazIOConfig.h>
#include <pacbio/primary/ManuallyBufferedFile.h>
#include <pacbio/smrtdata/Basecall.h>
#include <pacbio/smrtdata/Pulse.h>

#include <pacbio/primary/BazBuffer.h>
#include <pacbio/primary/PrimaryToBaz.h>
#include <pacbio/primary/BazCore.h>

using PacBio::SmrtData::MetricsVerbosity;
using PacBio::SmrtData::Readout;
using PacBio::SmrtData::Basecall;

namespace PacBio {
namespace Primary {

class IBazWriter
{
public:
    virtual bool Flush() = 0;
    virtual void WaitForTermination() = 0;
    virtual const std::string& FilePath() const = 0;
    virtual const std::string& ErrorMessage() const = 0;
    virtual ~IBazWriter() noexcept = default;
};

/// Writes BAZ file format
template <typename TMetric>
class BazWriter : public IBazWriter
{
public: // static
    /// Pre-check to get an approximate BAZ file size in Bytes for a given config.
    static size_t ExpectedFileByteSize(const uint64_t movieFrames,
                                       const double frameRateHz,
                                       const double basesPerSecond,
                                       const uint32_t numZmws,
                                       const Readout readout,
                                       const MetricsVerbosity metricsVerbosity);
    static constexpr int blocksize = 4000;
#ifdef FAKE_ERROR
public: // data
    bool fakeError_ = false;
#endif
    
public: // structors

    /// Creates a new BAZ file and writes file header.
    /// \param filePath          File name of the output BAZ file.
    /// \param fileHeaderBuilder JSON of file header from a builder.
    BazWriter(const std::string& filePath, 
              FileHeaderBuilder& fileHeaderBuilder,
              const Primary::BazIOConfig& ioConf,
              const size_t maxNumSamples = 100000,
              bool silent = true);
    // Default constructor
    BazWriter() = delete;
    // Move constructor
    BazWriter(BazWriter&&) = delete;
    // Copy constructor
    BazWriter(const BazWriter&) = delete;
    // Move assignment operator
    BazWriter& operator=(BazWriter&& rhs) noexcept = delete;
    // Copy assignment operator
    BazWriter& operator=(const BazWriter&) = delete;


    /// Waits for writing thread to finish,
    /// before writing EOF chunk meta.
    ~BazWriter() noexcept;

public: // modifying methods

    /// Given primary pulse/base calls and trace metrics
    /// convert and add to the BAZ.  MetricsConverter is used to convert/generate
    /// baz format metrics from an unspecified format, without doing either a
    /// memory copy, or injecting a new dependence on the other format
    /// (e.g. on the basecaller metrics format) into this class
    /// 
    /// MetricsConverter is expected to be a functor with the signature:
    /// void MetricsConverter(Primary::MemoryBufferView<MetricBlock>& metrics);
    ///
    /// \return bool if slice was successfully added. False can be cause by
    ///              basecall and hfMetrics are not present or basecall does
    ///              not contains bases in production mode.
    template <class MetricsConverter>
    bool AddZmwSlice(const Basecall* basecall, const uint16_t numEvents,
                     const MetricsConverter&& metricsConverter,
                     const size_t numMetrics,
                     const uint32_t zmwId)
    {

        Primary::MemoryBufferView<TMetric> metrics = currentBazBuffer_->MetricsBuffer().Allocate(numMetrics);
        metricsConverter(metrics);

        return AddZmwSlice(basecall, numEvents, metrics, zmwId);
    }

    /// Given primary pulse/base calls and trace metrics
    /// convert and add to the BAZ.  
    ///
    /// \return bool if slice was successfully added. False can be cause by
    ///              basecall and hfMetrics are not present or basecall does
    ///              not contains bases in production mode.
    bool AddZmwSlice(const Basecall* basecall, const uint16_t numEvents,
                     const std::vector<TMetric>& hfMetrics,
                     const uint32_t zmwId);

private:
    bool AddZmwSlice(const Basecall* basecall, const uint16_t numEvents,
                     const Primary::MemoryBufferView<TMetric>& metrics,
                     const uint32_t zmwId);

public:

    /// Push internal BazBuffer to the queue of BazBuffers that are
    /// ready to be written to disk.
    /// 
    /// \return bool if data was available to be flushed out
    bool Flush();

    /// Waits until internal buffers have been written. If this method is not
    /// being called, data may get lost.
    void WaitForTermination();

    // This is used for simulations to avoid flooding RAM.
    inline size_t BufferSize() const
    { return bazBufferQueue_.Size(); }

    /// Return the current file path of the BAZ file.
    const std::string& FilePath() const
    { return filePath_; }

    const std::string& ErrorMessage() const
    { return errorMsg_; }

    std::string Summary() const
    {
        std::stringstream ss;
        Summarize(ss);
        return ss.str();
    }

    void Summarize(std::ostream& os) const
    {
        os << "Header/Footer Bytes:" << headerBytes_ << std::endl;
        os << "Padding Bytes      :" << paddingBytes_ << std::endl;
        os << "OverheadBytes      :" << overheadBytes_ << std::endl;
        os << "EventBytes         :" << eventsBytes_ << std::endl;
        os << "MetricsBytes       :" << metricsBytes_ << std::endl;
        os << "----------------------------------------" << std::endl;
        os << "Total BytesWritten_:" << bytesWritten_ << std::endl;
    }

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

    size_t NumEvents() const { return numEvents_;}

protected:
    size_t BytesWritten1() const
    {
        return headerBytes_ + paddingBytes_ + overheadBytes_ + eventsBytes_ + metricsBytes_;
    }

    size_t BytesWritten2() const {
        return bytesWritten_;
    }

#if 0
    /// Compute the expected BAZ file size based on the user fileheader config,
    /// movie length and average bases/second.
    size_t ExpectedFileByteSize(const uint64_t movieFrames,
                                const double basesPerSecond);
#endif

public:
    static constexpr double packetSizeProdMin= 3; // These constants are ugly, but they are sanity checked with
    static constexpr double packetSizeProd   = 4; // static_asserts in PrimaryToBaz.
    static constexpr double packetSizeInt    = 9; //

private: // types
    using BufferQueue = PacBio::ThreadSafeQueue<std::unique_ptr<BazBuffer<TMetric>>>;
    
private: // static data
    static constexpr uint8_t      zero = '\0';

private: // data

    size_t maxNumSamples_;

    std::atomic_bool success_;
    std::atomic_bool writeThreadContinue_;

    std::string errorMsg_;
    std::string filePath_;
    std::string filePathTmp_;
    
    std::map<uint32_t, uint32_t> zmwsAdded_;

    std::thread writeThread_;

    std::unique_ptr<BazBuffer<TMetric>>    currentBazBuffer_;
    std::unique_ptr<FileHeader>   fh_;
    std::unique_ptr<PrimaryToBaz<TMetric>> p2b_;

    uint32_t maxZmwNumber_   = 0;
    uint32_t numSuperChunks_ = 0;

    bool silent_;

    uint8_t hFbyLFRatio_ = 0;
    uint8_t hFbyMFRatio_ = 0;

    BufferQueue bazBufferQueue_;
    BufferQueue idleBuffers_;
    FileHeaderBuilder fhb_;
    FileFooterBuilder ffb_;
    MetricsVerbosity metricsVerbosity_;
    std::unique_ptr<Primary::ManuallyBufferedFile> fileHandle_;

    std::vector<char> jsonFileHeader_;

    bool abort_ = false;
    std::atomic_bool dieGracefully_;

    size_t bytesWritten_ = 0;
    size_t headerBytes_ = 0;
    size_t eventsBytes_ = 0;
    size_t metricsBytes_ = 0;
    size_t overheadBytes_ = 0;
    size_t paddingBytes_= 0 ;

    size_t numEvents_ = 0;
    
    size_t numZmwsZeroIncorporatedEvents_ = 0;
    size_t numZmwsEmptyMetrics_ = 0;

private: // modifying methods

    /// Is executed in its own thread. Writes BazBuffers to disk.
    void WriteToDiskLoop();

    void Init(const BazIOConfig& );

private: // non-modifying methods

    /// Writes meta information of super chunk.
    /// \param nextPointer offset to next meta
    /// \param numZmws     number of ZmwSlices in this super chunk
    /// \param accumulate  if true, accumulates bytes written statistics
    void WriteSuperChunkMeta(const uint64_t nextPointer, uint32_t numZmws, bool accumulate = true);

    /// Writes header information for each ZmwSlice for current super chunk.
    /// \param headerBuffer current header buffer to be written
    void WriteZmwSliceHeader(std::vector<ZmwSliceHeader>& headerBuffer);

    /// Writes base calls/pulses and metrics for each ZMW slice
    /// for current super chunk.
    /// \param bazBuffer    current buffer to be written
    /// \param headerBuffer corresponding header buffer for ZmwSlice offsets
    void WriteChunkData(std::unique_ptr<BazBuffer<TMetric>>& bazBuffer,
                        std::vector<ZmwSliceHeader>& headerBuffer);

    /// Write high-frequency metrics blocks to given buffer.
    /// \param hFMetricFields   User-defined high-frequency metric fields
    /// \param metricBlocks     HighFrequencyMetricBlock array
    /// \param numMetricBlocks  Number of HighFrequencyMetricBlocks
    /// \param buffer           Buffer to be appended
    /// \param c                Reference to buffer index
    void Write(const std::vector<MetricField>& hFMetricFields,
               const Primary::MemoryBufferView<TMetric>& metricBlocks,
               std::vector<uint8_t>& buffer, size_t& c);

    /// Compute average of metrics wrt to ratioToHFMetrics and write it to
    /// given buffer.   
    /// \param hFMetricFields   User-defined high-frequency metric fields
    /// \param metricBlocks     HighFrequencyMetricBlock array
    /// \param numMetricBlocks  Number of HighFrequencyMetricBlocks
    /// \param ratioToHFMetrics Ratio HF/MF or HF/LF
    /// \param buffer           Buffer to be appended
    /// \param c                Reference to buffer index
    void Write(const std::vector<MetricField>& metricFields,
               const Primary::MemoryBufferView<TMetric>& metricBlocks,
               const size_t ratioToHFMetrics,
               std::vector<uint8_t>& buffer, size_t& c);

    void WriteSkippedZmws();

protected:
    virtual void Error(const std::string& errorMsg);

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
            if (accumulate) bytesWritten_ += size;
        }
        return size;
    }

    /// Writes the sanity word and keeps track of bytes written.
    void WriteSanity(bool accumulate = true)
    {
        std::array<uint8_t, 4> data;
        Sanity::Write(data);
        fileHandle_->Fwrite(data.data(), 4);
        if (accumulate)
        {
            bytesWritten_ += 4;
            overheadBytes_ += 4;
        }
    }

    /// Write the header to the fileHandle_. Because the header can be big, this method may loop
    /// over Fwrite as needed.
    /// \param accumulate - if true, then aggregate the bytes written into the statistics
    void WriteJsonFileHeader(bool accumulate);

};

}}
