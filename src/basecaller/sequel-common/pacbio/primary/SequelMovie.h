// Copyright (c) 2014, Pacific Biosciences of California, Inc.
//
// All rights reserved.
//
// THIS SOFTWARE CONSTITUTES AND EMBODIES PACIFIC BIOSCIENCES' CONFIDENTIAL
// AND PROPRIETARY INFORMATION.
//
// Disclosure, redistribution and use of this software is subject to the
// terms and conditions of the applicable written agreement(s) between you
// and Pacific Biosciences, where "you" refers to you or your company or
// organization, as applicable.  Any other disclosure, redistribution or
// use is prohibited.
//
// THIS SOFTWARE IS PROVIDED BY PACIFIC BIOSCIENCES AND ITS CONTRIBUTORS "AS
// IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
// THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL PACIFIC BIOSCIENCES OR ITS
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
// OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
// WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
// OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
// ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// File Description:
//  C++ class wrapper to handle SequelMovieFiles and SequelMovieFrames.
//
// Programmer: Mark Lakata

#ifndef SEQUEL_MOVIE_H_
#define SEQUEL_MOVIE_H_

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <math.h>
#include <string.h>
#if defined(__SSE2__)
#include <emmintrin.h>
#endif
#ifndef WIN32
#include <unistd.h>
#endif

#include <atomic>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <memory>
#include <mutex>
#include <random>
#include <vector>
#include <list>

#include <boost/multi_array.hpp>

#include <pacbio/ipc/ThreadSafeQueue.h>
#include <pacbio/logging/Logger.h>
#include <pacbio/utilities/SmartEnum.h>
#include <pacbio/text/String.h>
#include <pacbio/Utilities.h>
#include <pacbio/utilities/ISO8601.h>
#include <pacbio/utilities/PerforceUtil.h>
#include <pacbio/HugePage.h>

#include <pacbio/primary/EventObject.h>
#include <pacbio/primary/Tile.h>
#include <pacbio/primary/SequelROI.h>
#include <pacbio/primary/SequelMovieFrame.h>
#include <pacbio/primary/HDF5cpp.h>
#include <pacbio/primary/WorkingFile.h>
#include <pacbio/primary/SequelHDF5.h>

namespace PacBio {
namespace Primary {

class EventObject;

/// ChunkBuffers are the elements that make up a SequelMovieBuffer. Each ChunkBuffer covers one Chunk's worth
/// of data for the output movie file. There are two derived classes from the abstract ChunkBuffer.
///  ChunkBufferFrameOriented - used for writing mov.h5 frames as well as CRC files.
///  ChunkBufferTraceOriented - used for writing trc.h5 traces.
/// ChunkBuffers must use preallocated memory, they don't construct pixel memory on their own. They do own
/// a chunk header tile, which they manage.
/// The ChunkBuffers are intended to be used as part of a container that wraps the ChunkBuffer pointers with
/// std:unique_ptr semantics.
class ChunkBuffer
{
public:
    ChunkBuffer(size_t bufferSize, uint8_t* buffer) :
            buffer_(buffer),
            header_(nullptr),
            bufferSize_(bufferSize),
            framesToWrite_(0)
    {

    }

    virtual ~ChunkBuffer() {}

    /// \returns true if the chunkbuffer has non-nullptr assigned to it with positive size.
    /// otherwise returns false.
    bool Ready() const
    {
        return buffer_ != nullptr && bufferSize_ > 0;
    }

    /// \returns a pointer to a copy of the headertile for this chunk
    const  PacBio::Primary::Tile* HeaderTile() const { return header_; }

    /// copies the header tile to the private copy
    void SetHeader(const Tile* chunkHeader)
    {
        if (!chunkHeader->IsHeader())
        {
            throw PBException("Invalid header tile, first word was " +
                              std::to_string(chunkHeader->HeaderWord()));
        }
        memcpy(header_, chunkHeader, sizeof(Tile));
    }

    /// \returns the size of the memory for this chunk buffer (starting at buffer_)
    size_t BufferByteSize() const
    {
        return bufferSize_;
    }

    /// The FramesToWrite are the frames to write from this chunk buffer. Typically, this is always 512 except
    /// for the last chunk that is written, which can be any things from 1 to 512 frames.
    void SetFramesToWrite(uint32_t nFrames)
    {
        framesToWrite_ = nFrames;
    }

    /// \returns The expected number of FramesToWrite for this chunk
    uint32_t FramesToWrite() const
    {
        return framesToWrite_;
    }

    /// Returns a pointer to data. Some derived classes use the itile parameter and some don't.
    /// If it uses the itile parameter, then the pointer returned is the start of the requested
    /// tile offset. If the derived class ignores the itile parameter, then the pointer returned is simply
    /// the start of the buffer for this ChunkBuffer.
    virtual const uint8_t* Data(int itile) const = 0;

    /// This is the same as the `Data()` method, except with read & write semantics (nonconst).
    virtual uint8_t* WritableData(int itile) = 0;

protected:
    /// Initializes the chunk buffer to defaults that are not random.
    void Initialize()
    {
        header_ = Tile::makeHeader();
        if (bufferSize_ > 0 && buffer_)
        {
            memset(buffer_, 0, bufferSize_);
        }
        else if (bufferSize_ == 0 && !buffer_)
        {
            // ok
        }
        else
        {
            throw PBException("ChunkBuffer constructor called with incompatible bufferSize and buffer arguments");
        }
    }

protected:
    uint8_t* buffer_;
    PacBio::Primary::Tile* header_;
    size_t bufferSize_;
    uint32_t framesToWrite_;
};

class ChunkBufferTraceOriented : public ChunkBuffer
{
public:
/// \param bufferSize - the size of the memory pointed at by buffer
/// \param buffer - a pointer to a preallocated memory. This is probably in a HugePage allocation. It can be nullptr,
/// but bufferSize must be zero in this case.
    ChunkBufferTraceOriented(size_t bufferSize, uint8_t* buffer) :
        ChunkBuffer(bufferSize,buffer),
            numTiles_(bufferSize/sizeof(Tile))
    {
        Initialize();
    }

    ChunkBufferTraceOriented(const ChunkBufferTraceOriented& a) = delete;

    ChunkBufferTraceOriented(ChunkBufferTraceOriented&& a) : ChunkBuffer(0,nullptr)
    {
        std::swap(buffer_,a.buffer_);
        std::swap(header_,a.header_);
        std::swap(bufferSize_,a.bufferSize_);
        std::swap(framesToWrite_,a.framesToWrite_);
        std::swap(numTiles_,a.numTiles_);
    }

    ChunkBufferTraceOriented& operator=(const ChunkBufferTraceOriented& a) = delete;

    ChunkBufferTraceOriented& operator=(ChunkBufferTraceOriented&& a)
    {
        Deallocate();
        std::swap(buffer_,a.buffer_);
        std::swap(header_,a.header_);
        std::swap(bufferSize_,a.bufferSize_);
        std::swap(framesToWrite_,a.framesToWrite_);
        std::swap(numTiles_,a.numTiles_);
        return *this;
    }

    ~ChunkBufferTraceOriented() override
    {
        Deallocate();
    }

    const uint8_t* Data(int itile) const override
    {
        if (itile <0 || itile >= static_cast<int>(numTiles_)) throw PBException("itile " + std::to_string(itile) +
                                                  " out of range 0:" + std::to_string(numTiles_));
        return buffer_+ sizeof(Tile)*itile;
    }
    uint8_t* WritableData(int itile)  override
    {
        //Scott Meyer's recommended way of not duplicating code:
        return const_cast<uint8_t*>(const_cast<ChunkBufferTraceOriented&>(*this).Data(itile));
    }


private:
    void Deallocate()
    {
        if (buffer_)
        {
            buffer_= nullptr;
        }
        if (header_)
        {
            delete header_;
            header_ = nullptr;
        }
        bufferSize_ = 0;
        framesToWrite_ = 0;
        numTiles_ = 0;
    }

private:
    uint32_t numTiles_;
};

/// A container class that manages a big buffer allocated in HugeMemory as well as a header tile.
/// The instances are designed to adhere to std::move semantics.
class ChunkBufferFrameOriented : public ChunkBuffer
{
public:
/// \param bufferSize - the size of the memory pointed at by buffer
/// \param buffer - a pointer to a preallocated memory. This is probably in a HugePage allocation. It can be nullptr,
/// but bufferSize must be zero in this case.
    ChunkBufferFrameOriented(size_t bufferSize, uint8_t* buffer) :
            ChunkBuffer(bufferSize,buffer)
    {
        Initialize();
    }

    ChunkBufferFrameOriented(const ChunkBufferFrameOriented& a) = delete;

    ChunkBufferFrameOriented(ChunkBufferFrameOriented&& a) : ChunkBuffer(0,nullptr)
    {
        std::swap(buffer_,a.buffer_);
        std::swap(header_,a.header_);
        std::swap(bufferSize_,a.bufferSize_);
        std::swap(framesToWrite_,a.framesToWrite_);
    }

    ChunkBufferFrameOriented& operator=(const ChunkBufferFrameOriented& a) = delete;

    ChunkBufferFrameOriented& operator=(ChunkBufferFrameOriented&& a)
    {
        Deallocate();
        std::swap(buffer_,a.buffer_);
        std::swap(header_,a.header_);
        std::swap(bufferSize_,a.bufferSize_);
        std::swap(framesToWrite_,a.framesToWrite_);
        return *this;
    }

    ~ChunkBufferFrameOriented() override
    {
        Deallocate();
    }

    const uint8_t* Data(int itile ) const override
    {
        assert(itile == 0);
        (void) itile;
        return buffer_;
    }
    uint8_t* WritableData(int itile)  override
    {
      // Scott Meyer's const to non const trick
        return const_cast<uint8_t*>(const_cast<ChunkBufferFrameOriented&>(*this).Data(itile));
    }


private:
    void Deallocate()
    {
        if (buffer_)
        {
            buffer_= nullptr;
        }
        if (header_)
        {
            delete header_;
            header_ = nullptr;
        }
        bufferSize_ = 0;
        framesToWrite_ = 0;
    }
private:
};


/// Manages ChunkBuffers, through a series of queues. Free buffers are
/// placed in the chunkFreeQueue_ and filled, then pushed to the chunkFilledQueue_.
/// A separate thread pulls filled buffers from chunkFilledQueue_, writes to disk,
/// then recycles the buffers in the chunkFreeQueue_.
/// Before buffered chunk file writing can be used, PrepareChunkBuffers() must be called to allocate
/// the memory and queues.
/// To end the separate thread, a null ChunkBuffer object (o size) is pushed on the
/// chunkFilledQueue_ by a call to EndThread()
class SequelMovieBuffer
{
private:
    uint32_t rows_; // only defined for frame oriented
    uint32_t cols_; // only defined for frame oriented
    size_t frameSize_;
    int numChunkBuffers_ ;
    size_t bufferSize_ ;
    std::unique_ptr<ChunkBuffer> currentBuffer_;
    PacBio::ThreadSafeQueue<std::unique_ptr<ChunkBuffer>> chunkFilledQueue_;
    PacBio::ThreadSafeQueue<std::unique_ptr<ChunkBuffer>> chunkFreeQueue_;
    uint8_t* chunkBufferBase_;
    size_t chunkBufferTotalSize_ = 0;
    uint32_t pixelCount_ = 0;
    uint64_t maxPixels_ = 0;

public:
    // frame oriented
    SequelMovieBuffer(uint32_t rows, uint32_t cols);
    // either frame or trace oriented
    SequelMovieBuffer(const SequelROI& roi, bool traceOriented);

    ~SequelMovieBuffer();

    uint32_t Rows() const { return rows_; }

    uint32_t Cols() const { return cols_; }

    bool BufferReady() const;
    bool QueueReady() const;

    void PrepareChunkBuffers(int numChunkBuffers=2, bool traceOriented=false);

    size_t ChunkBufferNumSamples() const;

    size_t ChunkBufferTotalSize() const
    {
        return chunkBufferTotalSize_;
    }

    size_t NumTiles() const;

    const ChunkBuffer* CurrentChunkBuffer() const { return currentBuffer_.get(); }
    ChunkBuffer* CurrentChunkBuffer()             { return currentBuffer_.get(); }

    const int16_t* PeekFrameFromCurrentChunkBuffer(uint32_t frameIndex) const;

    std::unique_ptr<ChunkBuffer> PopFilledChunk() { return chunkFilledQueue_.Pop(); }

    int GetChunkBufferSize() const;

    void PushFreeChunk(std::unique_ptr<ChunkBuffer>&& chunk) { chunkFreeQueue_.Push(std::move(chunk)); }

    /// gets the next chunk buffer internally. Must be called at the start of a chunk.
    void PrepareNextChunk();

    /// initialize chunk with the first tile (so called "header" tile). resets the pixel counter to 0.
    void SetChunkHeader(const PacBio::Primary::Tile* chunkHeader);

    /// stores frames.
    void AddTileAsFrames(const PacBio::Primary::Tile* tile, uint32_t tileOffset, uint32_t frameStart, uint32_t frameEnd); // 0,framesPerTile

    /// stores traces.
    void AddTileAs1CTraces(const PacBio::Primary::Tile* tile, uint32_t tileOffset, uint32_t internalOffset);
    void AddTileAs2CTraces(const PacBio::Primary::Tile* tile, uint32_t tileOffset, uint32_t internalOffset);

    void FlushChunk(uint32_t framesToWrite = PacBio::Primary::Tile::NumFrames);

    /// \returns true if the buffer has been fully filled with pixels since the last chunk header was received
    bool PixelFull() const;

    /// \returns number of pixels filled since the last chunk header.
    uint64_t NumPixels() const { return pixelCount_; }

    /// \returns the maximum number of pixels this buffer could hold.
    uint64_t MaxPixels() const { return maxPixels_; }

    void EndThread();
};

class SequelMovieFileBase
{
public:
#ifdef OBSOLETE_EVENTS
      SMART_ENUM(EventType_e,
                 Unknown = 0,
                 HotStart = 1,
                 LaserOn = 2,
                 LaserOff = 3,
                 LaserPowerChange = 4
      );
#endif

      enum class SequelMovieType
      {
          Unknown,
          Frame,
          Trace,
          CRC
      };

      SequelMovieFileBase(const std::string& fullFilePath0, const SequelROI& roi0, bool traceOriented, ChipClass chipClass)
              :
              bitsPerPixel(12),
              NFRAMES(0),
              NDARKFRAMES(0),
# ifdef OBSOLETE_EVENTS
              NUM_EVENTS(0),
#endif
              valid(false),
              roi_(roi0.Clone()),
              fullFilePath_(fullFilePath0),
              movieBuffer(roi0,traceOriented),
              limitedNumFrames_(0),
              chipClass_(chipClass),
              frameClass_(DefaultFrameClass(chipClass))
      {
          // std::cout <<"SequelMovieFileBase for " << fullFilePath0 << std::endl;

          roi_->CheckROI();
          saverThread_ = std::thread([this](){ SaverThread(); });
      }

      virtual ~SequelMovieFileBase()
      {
          // std::cout << "virtual ~SequelMovieFileBase()" << std::endl;
          if (saverThread_.joinable())
          {
              std::cout << "saverThread_ was not joined. Derived class destructor must call FinalizeSaverThread()!" << std::endl;
              abort();
              saverThread_.join();
          }
      }

      int GetChunkBufferSize() const;

      uint32_t AddChunk(const PacBio::Primary::Tile* chunkHeader, uint32_t numFrames);

      virtual uint64_t AddFrame(const SequelMovieFrame<int16_t>& frame) = 0;

      virtual uint64_t AddDarkFrame(float exposure, const SequelMovieFrame<int16_t>& frame) = 0;

#ifdef OBSOLETE_EVENTS
      virtual void AddEvent(EventType_e type, uint64_t timestamp) = 0;
#endif
      virtual void SetMeasuredFrameRate(double rate) = 0;

  protected:
      virtual void WriteChunkBuffer(const ChunkBuffer* bigBuffer, const uint32_t numFrames) = 0;

      bool ProcessOneChunkBuffer();

  public:
      virtual uint64_t ReadFrame(uint64_t n, SequelMovieFrame<int16_t>& frame) const = 0;

      virtual void DumpSummary(std::ostream& s, uint32_t frames = 3, uint32_t rows = 3, uint32_t cols = 5) = 0;

      virtual Json::Value DumpJson(uint32_t /*frames*/ = 3, uint32_t /*rows*/ = 3, uint32_t /*cols*/ = 5) = 0;

      virtual uint32_t WriteChunkHeader(const PacBio::Primary::Tile* chunkHeader, uint32_t numFrames) = 0;

      virtual void AddTile(const PacBio::Primary::Tile* tile, uint32_t tileOffset);

      virtual void FlushChunk(uint32_t framesToWrite = PacBio::Primary::Tile::NumFrames);

      virtual SequelMovieType Type() const = 0;

      virtual std::string TypeName() const = 0;

      virtual const SequelROI* GetROI() const
      {
          return roi_.get();
      }

      virtual ChipClass GetChipClass() const { return chipClass_; }

      static int Compare(uint64_t a, uint64_t b, const char* field)
      {
          return SequelMovieFrame<int16_t>::Compare(a, b, field);
      }

#ifdef OBSOLETE_EVENTS
      inline void AddHotStartEvent(double timestamp)
      {
          uint64_t uitimestamp = static_cast<uint64_t>(round( timestamp * 1e9)); // convert to nanoseconds
          AddEvent(EventType_e::HotStart, uitimestamp);
      }
#endif

      virtual void SetFirstFrame(uint64_t /*frameIndex*/) {}
      virtual void AddLaserPowerEvent(const EventObject& eo)
      {
          (void) eo;
          throw PBException("unimplemented");
      }

      virtual void AddHotStartEvent(const EventObject& eo)
      {
          (void) eo;
          PBLOG_WARN << "AddHotStartEvent not implemented, event ignored";
      }

      virtual std::vector<EventObject> ReadAllEvents() { return std::vector<EventObject>(0); }


      virtual void PrepareChunkBuffers(int numChunkBuffers=2) = 0;

      size_t ChunkBufferTotalSize() const
      {
          return movieBuffer.ChunkBufferTotalSize();
      }

      int16_t* ChunkBufferSamplePointer()
      {
          return const_cast<int16_t*>(movieBuffer.PeekFrameFromCurrentChunkBuffer(0));
      }

      size_t ChunkBufferNumSamples() const
      {
          return movieBuffer.ChunkBufferNumSamples();
      }

      size_t ChunkBufferNumTiles() const
      {
          return movieBuffer.NumTiles();
      }

      std::string FullFilePath() const
      {
          return fullFilePath_;
      }

      void LimitFileFrames(uint64_t x)
      {
          limitedNumFrames_ = x;
      }

      /// Returns the lessor of the number of frames in file or the
      /// requested file limit.
      uint64_t LimitedNumFrames() const
      {
          if (limitedNumFrames_ > 0 && limitedNumFrames_ < NFRAMES)
          {
              return limitedNumFrames_;
          }
          else
          {
              return NFRAMES;
          }
      }
      /// thread safe entry point to add a new event object
      void AddEventObject(const EventObject& eo);

protected:
      void FinalizeSaverThread();

  private:
      void SaverThread();


  public:
      uint32_t bitsPerPixel;
      uint64_t NFRAMES;         ///< actual number of frames in the file
      uint32_t NDARKFRAMES;
#ifdef OBSOLETE_EVENTS
      uint32_t NUM_EVENTS;
#endif
      bool valid;
      std::unique_ptr<const SequelROI> roi_;
      std::string fullFilePath_;

  protected:
      SequelMovieBuffer movieBuffer;
      std::thread saverThread_;
      PacBio::IPC::ThreadSafeQueue<EventObject> eventQueue_;

  protected:
      void SetupWorkingFilename(const std::string& filename)
      {
          workingFile_.SetupWorkingFilename(filename);
      }

      void CleanupWorkingFilename()
      {
          workingFile_.CleanupWorkingFilename();
      }

      void AddEventObjectFromThread(const EventObject& eo);

      WorkingFile workingFile_;

  private:
      uint64_t limitedNumFrames_; ///< limited number of frames in the file, requested by calling LimitedFileFrames()

  protected:
      ChipClass chipClass_;
      FrameClass frameClass_;
  };

  void MovieTest();
 }
}





#endif /* FILE_IO_H_ */
