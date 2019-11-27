// Copyright (c) 2014, Pacific Biosciences of California, Inc.
//
// All rights reserved.
//
// THIS SOFTWARE CONSTITUTES AND EMBODIES PACIFIC BIOSCIENCES<92> CONFIDENTIAL
// AND PROPRIETARY INFORMATION.
//
// Disclosure, redistribution and use of this software is subject to the
// terms and conditions of the applicable written agreement(s) between you
// and Pacific Biosciences, where <93>you<94> refers to you or your company or
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
//  SequelMovie functions that could not be put into the class definitions
//  and thus could not be part of the *.h file.
//
// Programmer: Mark Lakata

/************************************************************

 This file is intended for use with HDF5 Library version 1.8

 ************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <sstream>

#include <pacbio/text/String.h>
#include <pacbio/process/ProcessBase.h>

#include <pacbio/primary/EventObject.h>
#include <pacbio/primary/SequelMovie.h>
#include <pacbio/primary/SequelTraceFile.h>
#include <pacbio/primary/SequelMovieFileHDF5.h>

#ifdef WIN32
#else
#include <unistd.h>

#endif

#include <fstream>

using namespace std;

namespace PacBio {
namespace Primary {

  static_assert(sizeof(Tile) == 32768, "bad tile size");


  SequelMovieBuffer::SequelMovieBuffer(uint32_t rows, uint32_t cols)
          : rows_{rows}
          , cols_{cols}
          , frameSize_(0)
          , numChunkBuffers_(0)
          , bufferSize_(0)
          , chunkBufferBase_(nullptr)
          , chunkBufferTotalSize_(0)
          , pixelCount_(0)
  {
      PBLOG_DEBUG << "SequelMovieBuffer::SequelMovieBuffer(rows:" << rows_ <<
                      ", cols_:"<< cols_ << ")";
      frameSize_ = (rows_ * cols_) * sizeof(int16_t);
      maxPixels_ = (rows_ * cols_) * Tile::NumFrames;
  }

  SequelMovieBuffer::SequelMovieBuffer(const SequelROI& roi, bool traceOriented)
          : rows_{0}
          , cols_{0}
          , frameSize_(0)
          , numChunkBuffers_(0)
          , bufferSize_(0)
          , chunkBufferBase_(nullptr)
          , chunkBufferTotalSize_(0)
          , pixelCount_(0)
  {
      PBLOG_DEBUG << "SequelMovieBuffer::SequelMovieBuffer(roi:" << roi <<
        ", traceOriented:"<< traceOriented << ")";

      if (traceOriented)
      {
          frameSize_ = roi.TotalPixels() * sizeof(int16_t);
          maxPixels_ = roi.TotalPixels() * Tile::NumFrames;
      }
      else
      {
          rows_ = roi.SensorROI().PhysicalRows();
          cols_ = roi.SensorROI().PhysicalCols();
          frameSize_ = roi.SensorROI().TotalPixels() * sizeof(int16_t);
          maxPixels_ = roi.SensorROI().TotalPixels() * Tile::NumFrames;
      }
  }

  SequelMovieBuffer::~SequelMovieBuffer()
  {
        if(chunkBufferBase_) PacBio::HugePage::Free(chunkBufferBase_);

  }

  /// \returns true if PrepareNextChunk() has been called
  bool SequelMovieBuffer::BufferReady() const
  {
      return currentBuffer_ && currentBuffer_->Ready();
  }

  /// \returns true if the queues have been created and filled with chunk buffers
  bool SequelMovieBuffer::QueueReady() const
  {
      return numChunkBuffers_ >= 1;
  }

  /// allocates chunk buffers in Huge page memory, and sets up queues for free buffers
  /// and filled buffers.
  void SequelMovieBuffer::PrepareChunkBuffers(int numChunkBuffers, bool traceOriented)
  {
      if (frameSize_ == 0)
      {
          PBLOG_ERROR << "FrameSize ==0!";
          PBLOG_ERROR << "rows:" << rows_ << " cols:" << cols_;
          PBLOG_ERROR << "maxPixels_:" << maxPixels_;
          throw PBException("frameSize_ ==0, SequelMovieBUffer was not set up correctly. ");
      }
      chunkFreeQueue_.Clear();
      chunkFilledQueue_.Clear();
      if (chunkBufferBase_)
      {
          PacBio::HugePage::Free(chunkBufferBase_);
          chunkBufferBase_ = nullptr;
      }

      bufferSize_ = framesPerTile * frameSize_;
      chunkBufferTotalSize_ = bufferSize_ * numChunkBuffers;
      chunkBufferBase_ = PacBio::HugePage::Malloc<uint8_t>(chunkBufferTotalSize_);

      for(int i=0;i<numChunkBuffers;i++)
      {
          uint8_t* ptr = chunkBufferBase_ + i * bufferSize_;
          ChunkBuffer* buf;
          if (traceOriented)
          {
              buf = new ChunkBufferTraceOriented(bufferSize_,ptr);
          }
          else
          {
              buf = new ChunkBufferFrameOriented(bufferSize_,ptr);
          }
          chunkFreeQueue_.Push(std::unique_ptr<ChunkBuffer>(buf));
      }
      numChunkBuffers_ = numChunkBuffers;
  }
  size_t SequelMovieBuffer::ChunkBufferNumSamples() const
  {
      return bufferSize_ / sizeof(uint16_t);
  }
  size_t SequelMovieBuffer::NumTiles() const
  {
      return bufferSize_ / sizeof(Tile);
  }
  const int16_t* SequelMovieBuffer::PeekFrameFromCurrentChunkBuffer(uint32_t frameIndex) const
  {
      if (frameIndex >= PacBio::Primary::Tile::NumFrames) frameIndex = 0;
      return reinterpret_cast<const int16_t*>(&currentBuffer_->Data(0)[frameIndex * frameSize_]);
  }

  /// pops an empty ChunkBuffer from the free queue. This may block if there are no free blocks, which is
  /// as designed.
  void SequelMovieBuffer::PrepareNextChunk()
  {
      if (!QueueReady()) throw PBException("Need to call PrepareChunkBuffers!");
      currentBuffer_ = chunkFreeQueue_.Pop(); // do we need a timeout?
  }

  /// Copies the chunkHeader tile to the ChunkBuffer.
void SequelMovieBuffer::SetChunkHeader(const PacBio::Primary::Tile* chunkHeader)
{
    if (!BufferReady()) throw PBException("Need to call PrepareNextChunk!");
    currentBuffer_->SetHeader(chunkHeader);
    pixelCount_ = 0;
}

  /// Stores the given tile in the current ChunkBuffer.
  void SequelMovieBuffer::AddTileAsFrames(const PacBio::Primary::Tile* tile, uint32_t tileOffset, uint32_t frameStart, uint32_t frameEnd)
  {
      if (!BufferReady()) throw PBException("Need to call PrepareNextChunk!");

      if (currentBuffer_->Data(0) == nullptr || currentBuffer_->BufferByteSize() == 0) throw PBException("currentBuffer is null");

      for (uint32_t frame = frameStart; frame < frameEnd ; frame++)
      {
          size_t pos = frame * (frameSize_) + tileOffset * 64;
          if (pos + 64 > framesPerTile * frameSize_)
          {
              throw PBException("big buffer overflow: pos:" + std::to_string(pos) // + " NROW:" + std::to_string(rows_) + " NCOL:" + std::to_string(cols_)
                  + " frameSize:" + std::to_string(frameSize_));
          }
          memcpy(currentBuffer_->WritableData(0)+pos, tile->data + frame * 64, 64);
          pixelCount_ +=  Tile::NumPixels;
      }

  }

// stores traces in Spider 1C format
void SequelMovieBuffer::AddTileAs1CTraces(const PacBio::Primary::Tile* tile, uint32_t tileOffset, uint32_t internalOffset)
{
    (void) tileOffset; // this should probably be removed from the call signature

    if (!BufferReady()) throw PBException("Need to call PrepareNextChunk!");
    // std::cout << "AddTile " << (int) tile->data[0] << std::endl;

    const size_t srcFrameStride = Tile::NumPixels;
    const size_t dstZmwStride = Tile::NumFrames;

    const uint16_t* const src = reinterpret_cast<const uint16_t*>(tile->data);
    uint16_t* const dst       = reinterpret_cast<uint16_t*>(currentBuffer_->WritableData(internalOffset));
    for (uint32_t frame = 0; frame < Tile::NumFrames; frame++)
    {
        for (uint32_t pixel = 0; pixel < Tile::NumPixels; pixel++)
        {
            uint32_t dstOffset = pixel * dstZmwStride + frame;
            if (dstOffset > sizeof(Tile)/sizeof(uint16_t))
            {
                throw PBException("internal offset problem " + std::to_string(dstOffset));
            }
            dst[dstOffset]                  = src[pixel + srcFrameStride * frame];
        }
    }
}

  // stores traces in Sequel 2C format
  void SequelMovieBuffer::AddTileAs2CTraces(const PacBio::Primary::Tile* tile, uint32_t tileOffset, uint32_t internalOffset)
  {
      (void) tileOffset; // this should probably be removed from the call signature

      if (!BufferReady()) throw PBException("Need to call PrepareNextChunk!");
      // std::cout << "AddTile " << (int) tile->data[0] << std::endl;

      const size_t srcFrameStride = Tile::NumPixels;
      const size_t dstZmwStride = Tile::NumFrames;
      const size_t dstColorOffset = Tile::NumFrames;

      const uint16_t* const src = reinterpret_cast<const uint16_t*>(tile->data);
      uint16_t* const dst       = reinterpret_cast<uint16_t*>(currentBuffer_->WritableData(internalOffset));
      // source tile is in [frame][zmw][colorA] order
      // destination tile is in [zmw][colorB][frame] order
      // colorA order is 0=red,1=green
      // colorB order is 0=green,1=red

      for (uint32_t frame = 0; frame < Tile::NumFrames; frame++)
      {
          for (uint32_t pixel = 0; pixel < Tile::NumPixels; pixel+=2)
          {
              // The sensor output is red first, then green.
              // The trace file wants green first, then red.
              uint32_t dstOffset = pixel * dstZmwStride + frame;
              if (dstOffset > sizeof(Tile)/sizeof(uint16_t))
              {
                  throw PBException("internal offset problem " + std::to_string(dstOffset));
              }

              const int redOffsetInSensorData = 0;
              const int greenOffsetInSensorData = 1;
              //save green pixel first for trace
              dst[dstOffset]                  = src[pixel + greenOffsetInSensorData + srcFrameStride * frame];
              dstOffset += dstColorOffset;
              if (dstOffset > sizeof(Tile)/sizeof(uint16_t))
              {
                  throw PBException("internal offset problem " + std::to_string(dstOffset));
              }
              // save red pixel second for trace
              dst[dstOffset ]                 = src[pixel + redOffsetInSensorData + srcFrameStride * frame];
          }
      }

  }

  void SequelMovieBuffer::FlushChunk(uint32_t framesToWrite)
  {
      // std::cout << "SequelMovieBuffer::FlushChunk(uint32_t framesToWrite:"<< framesToWrite<<")" << std::endl;
      if (!BufferReady()) throw PBException("Need to call PrepareNextChunk!");
//      if (numChunkBuffers_ <= 1) throw PBException("FlushChunk is not designed to be called if numChunkBuffers<=1");

      currentBuffer_->SetFramesToWrite(framesToWrite);
      // std::cout << "pushing chunkFilledQueue_ with framesToWrite:" << currentBuffer_->FramesToWrite() << std::endl;
      chunkFilledQueue_.Push(std::move(currentBuffer_));
  }

void SequelMovieBuffer::EndThread()
{
    PBLOG_DEBUG << "SequelMovieBuffer::EndThread()";
    auto  dummy = std::unique_ptr<ChunkBufferFrameOriented>(new ChunkBufferFrameOriented(0, nullptr));
    dummy->SetFramesToWrite(0);
    chunkFilledQueue_.Push(std::move(dummy));
    PBLOG_DEBUG << "SequelMovieBuffer::EndThread() done";
}

int SequelMovieBuffer::GetChunkBufferSize() const
{
    return chunkFilledQueue_.Size();
}

/// \returns true if the buffer has been fully filled with pixels since the last chunk header was received
bool SequelMovieBuffer::PixelFull() const
{
    if (pixelCount_ > maxPixels_)
    {
        PBLOG_WARN << "pixelCount:" << pixelCount_ << " maxPixels:" << maxPixels_;
    }
    return (pixelCount_ >= maxPixels_);
}

///////

  void SequelMovieFileBase::FlushChunk(uint32_t framesToWrite )
  {
      movieBuffer.FlushChunk(framesToWrite);
  }

  // todo: remove return value, it is not used
  uint32_t SequelMovieFileBase::AddChunk(const PacBio::Primary::Tile* chunkHeader, uint32_t numFrames)
  {
      movieBuffer.PrepareNextChunk();
      movieBuffer.CurrentChunkBuffer()->SetFramesToWrite(numFrames);
      movieBuffer.SetChunkHeader(chunkHeader);
      return numFrames;

  }

  void SequelMovieFileBase::AddTile(const PacBio::Primary::Tile* tile, uint32_t tileOffset)
  {
      if (roi_->ContainsTileOffset(tileOffset))
      {
          movieBuffer.AddTileAsFrames(tile,tileOffset, 0, framesPerTile);
      }
  }

void SequelMovieFileBase::AddEventObject(const EventObject& eo)
{
    PBLOG_DEBUG << "SequelMovieFileBase:: saving event type " << eo.eventType();
    eventQueue_.Push(eo);
}

void SequelMovieFileBase::AddEventObjectFromThread(const EventObject& eo)
{
    PBLOG_DEBUG << "SequelMovieFileBase:: saving event type " << eo.eventType();
    switch (eo.eventType())
    {
    case EventObject::EventType::hotstart:
        AddHotStartEvent(eo);
        break;
    case EventObject::EventType::laserpower:
        AddLaserPowerEvent(eo);
        break;
    default:
        PBLOG_WARN << "event type " << eo.eventType() << " ignored";
        break;
    }
}


/// \returns true if buffer was processed, false if empty buffer was received
bool SequelMovieFileBase::ProcessOneChunkBuffer()
{
    // std::cout << "waiting for movieBuffer.PopFilledChunk" << std::endl;
    std::unique_ptr<ChunkBuffer> chunkBuffer = movieBuffer.PopFilledChunk();
    //std::cout << "get  chunkBuffer from movieBuffer.PopFilledChunk, chunkBuffer.FramesToWrite()="
    //        << chunkBuffer.FramesToWrite() << std::endl;

    if (chunkBuffer->FramesToWrite() == 0) return false;

    //std::cout << "WriteChunkHeader" << chunkBuffer.HeaderTile() << "," << chunkBuffer.FramesToWrite() << std::endl;
    WriteChunkHeader(chunkBuffer->HeaderTile(), chunkBuffer->FramesToWrite());

    //std::cout << "WriteChunkBuffer" << std::endl;
    WriteChunkBuffer(chunkBuffer.get(), chunkBuffer->FramesToWrite());

    //std::cout << "movieBuffer.PushFreeChunk(std::move(chunkBuffer));" << std::endl;
    movieBuffer.PushFreeChunk(std::move(chunkBuffer));

    return true;
}

void SequelMovieFileBase::SaverThread()
{
    PBLOG_DEBUG << "SequelMovieFileBase::SaverThread started" << std::endl;

    bool exit = false;
    while(!exit)
    {
        // this loop will exit when it receives a ChunkBuffer that is empty.
        PBLOG_DEBUG << "SequelMovieFileBase::SaverThread waiting for chunk";
        if ( !ProcessOneChunkBuffer()) exit = true;

        //drain event FIFO
        while(true)
        {
            EventObject eo;
            if (!eventQueue_.TryPop(eo)) break;
            PBLOG_DEBUG << "SequelMovieFileBase::SaverThread got event " << eo.eventType();
            AddEventObjectFromThread(eo);
        }
    }
    PBLOG_DEBUG<< "SequelMovieFileBase::SaverThread ended" << std::endl;
}

void SequelMovieFileBase::FinalizeSaverThread()
{
    PBLOG_DEBUG << "SequelMovieFileBase::FinalizeSaverThread()" << std::endl;

    if (saverThread_.joinable())
    {
        //std::cout << "Flushing Chunk of length 0" << std::endl;
        movieBuffer.EndThread();
        //std::cout << "Joining" << std::endl;
        saverThread_.join();
        //std::cout << "joined" << std::endl;
    }
}

int SequelMovieFileBase::GetChunkBufferSize() const
{
    return movieBuffer.GetChunkBufferSize();
}


}} // namespace
