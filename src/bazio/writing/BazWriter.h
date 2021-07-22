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
              const Primary::BazIOConfig& ioConf);

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

protected:
    size_t BytesWritten1() const
    {
        return headerBytes_ + paddingBytes_ + overheadBytes_ + eventsBytes_ + metricsBytes_;
    }

    size_t BytesWritten2() const {
        return bytesWritten_;
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

    size_t bytesWritten_ = 0;
    size_t headerBytes_ = 0;
    size_t eventsBytes_ = 0;
    size_t metricsBytes_ = 0;
    size_t overheadBytes_ = 0;
    size_t paddingBytes_= 0 ;

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
            if (accumulate) bytesWritten_ += size;
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

#endif //PACBIO_BAZIO_WRITING_BAZ_WRITER_H
