// Copyright (c) 2018, Pacific Biosciences of California, Inc.
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

#ifndef Sequel_Common_Pacbio_Primary_ManuallyBufferedFile_H_
#define Sequel_Common_Pacbio_Primary_ManuallyBufferedFile_H_

#include <cassert>
#include <cstring>
#include <fcntl.h>

#include <pacbio/dev/AutoTimer.h>
#include <pacbio/logging/Logger.h>

namespace PacBio {
namespace Primary {

// Class that manually handles file IO buffering as much as possible, to
// prevent the system from having to do so.  In general this is likely to
// *decrease* overall throughput, but in the context of realtime code it
// sometimes proves more valuable to have more reliable control on how much
// work the system is doing and when.
class ManuallyBufferedFile
{
public:
    ManuallyBufferedFile(const std::string& filePath,
                         size_t bufferSizeMBytes,
                         float maxMBPerSecond)
      : bufferSizeBytes_(bufferSizeMBytes * (1 << 20))
      , maxBytesPerUSecond_((1 << 20) * maxMBPerSecond / 1000000.0f)
      , dataIdx_(0)
      , filePos_(0)
      , data_(bufferSizeBytes_)
    {
        file_ = open(filePath.c_str(),
                     // Open a read-only file that is either created or truncated, depending on if it exists
                     O_CREAT | O_TRUNC | O_WRONLY ,
                     // User can read/write, group and others can only read
                     S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
	    // Check if file has been created
	    if (file_ == -1) throw PBExceptionErrno("Cannot create file");
    }

    ~ManuallyBufferedFile()
    {
        try
        {
            Flush();
            if (close(file_) == -1)
                throw PBExceptionErrno("Error closing baz file");
        }
        catch (const std::exception& e)
        {
            PBLOG_ERROR << "Error in destructor of ManuallyBufferedFile.  Swallowing exception to prevent termination:";
            PBLOG_ERROR << e.what();
        }
    }

    // Function to make sure we do not write to disk faster (on average) than
    // specified.  The overall time measured is between write completions,
    // so will include downtime spent in system calls actually doing the write,
    // as well as how long the code just spends doing other things in general
    void PauseAfterWrite(size_t bytesWritten)
    {
        auto elapsed = throttleTimer_.GetElapsedMilliseconds() * 1000;
        auto minWait = bytesWritten / maxBytesPerUSecond_;
        if (minWait > elapsed) usleep(static_cast<uint>(minWait - elapsed));

        throttleTimer_.Restart();
    }

    void Flush()
    {
        assert(dataIdx_ < bufferSizeBytes_);
        if (dataIdx_ == 0) return;

        WriteHelper(data_.data(), dataIdx_);
        dataIdx_ = 0;
    }

    size_t Ftell() const
    {
        assert(dataIdx_ < bufferSizeBytes_);
        return dataIdx_ + filePos_;
    }

    void Fwrite(const void* inData, size_t size)
    {
        assert(dataIdx_ < bufferSizeBytes_);
        if (dataIdx_ + size >= bufferSizeBytes_)
            Flush();

        // Must handle writes that are larger than our actual buffer size.
        // Could chunk it up so that we can let PauseAfterWrite keep our
        // average byte rate down, but so far there seems no need, especially
        // since the initial json header is the only thing that seems to trigger
        // this (for any reasonably sized bufferSize_).
        if (size > bufferSizeBytes_)
        {
            WriteHelper(inData, size);
        } else {
            memcpy(&data_[dataIdx_], inData, size);
            dataIdx_ += size;
            assert(dataIdx_ < bufferSizeBytes_);
        }
    }

    void FseekAbsolute(size_t position)
    {
        assert(dataIdx_ < bufferSizeBytes_);
        if (dataIdx_ == 0 && position == filePos_) return;
        if (dataIdx_ != 0) Flush();

        assert(dataIdx_ == 0);

        if (lseek(file_, position, SEEK_SET) == -1)
            throw PBExceptionErrno("fseek failed");
        filePos_ = position;
    }

    void FseekRelative(ssize_t position)
    {
        FseekAbsolute(filePos_ + dataIdx_ + position);
    }

    // Will fill the file with 0s for the specified distance
    void Advance(size_t dist)
    {
        assert(dataIdx_ < bufferSizeBytes_);
        assert(dist < bufferSizeBytes_);
        if (dataIdx_ + dist >= bufferSizeBytes_)
            Flush();

        memset(&data_[dataIdx_], 0, dist);
        dataIdx_ += dist;
    }

private:

    // Need to do everything in write/sync pairs, to make sure we
    // avoid/minimize any system buffering.  This seems necessary regardless,
    // but is especially true for an async NSF drive
    // Note: Using write() is *significantly* faster than using fwrite(),
    // which is why this class uses this lower level API.
    void WriteHelper(const void* data, size_t size)
    {
        if (write(file_, data, size) != static_cast<ssize_t>(size))
           throw PBExceptionErrno("Cannot write oversized packets");
// KNC does not implement this function, but we don't need it there anyway
#ifndef PB_KNC
// This library gets compiled in the bam2bam build for secondary that
// has an old glibc
#ifdef __GNU_LIBRARY__
#if ((__GLIBC__ == 2) && (__GLIBC_MINOR__ > 14)) || __GLIBC__ > 2
        if (syncfs(file_) == -1)
           throw PBExceptionErrno("Could not sync file");
#endif
#endif
#endif
        PauseAfterWrite(size);
        filePos_ += size;
    }

    size_t bufferSizeBytes_;
    float maxBytesPerUSecond_;
    size_t dataIdx_;
    size_t filePos_;
    int file_;
    std::vector<uint8_t> data_;
    Dev::QuietAutoTimer throttleTimer_;
};

}} // ::PacBio::Primary

#endif /* Sequel_Common_Pacbio_Primary_ManuallyBufferedFile_H_*/

