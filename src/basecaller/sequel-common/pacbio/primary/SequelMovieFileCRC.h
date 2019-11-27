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
//  C++ class wrapper to handle SequelMovieFileCRC
//
// Programmer: Mark Lakata

#ifndef SEQUELACQUISITION_SEQUELMOVIEFILECRC_H_H
#define SEQUELACQUISITION_SEQUELMOVIEFILECRC_H_H

#include <pacbio/primary/SequelMovie.h>

/// Derived movie file that will consist of one CRC value per line of text per movie frame.
/// The file name should be named "*.crc" by convention.

namespace PacBio
{
 namespace Primary
 {
  class SequelMovieFileCRC :
          public SequelMovieFileBase
  {
  private:
      std::ofstream fileout;
      size_t dataSize;
      std::thread t;
      PacBio::ThreadSafeQueue<const uint8_t*> queue;
  public:
      /// opens an output file
      ///
      SequelMovieFileCRC(const std::string& filename, const SequelROI& roi);

      virtual ~SequelMovieFileCRC();

      void SaverThread();

      void PrepareChunkBuffers(int numChunkBuffers=2) override
      {
          movieBuffer.PrepareChunkBuffers(numChunkBuffers, false);
      }

      uint64_t AddFrame(const SequelMovieFrame <int16_t>& frame) override;

      uint64_t AddDarkFrame(float exposure, const SequelMovieFrame <int16_t>& frame) override;

#ifdef OBSOLETE_EVENTS
      void AddEvent(EventType_e type, uint64_t timestamp) override;
#endif

      void SetMeasuredFrameRate(double rate) override;

      /// sort of a stub that just counts precounts frames
      ///
      uint32_t WriteChunkHeader(const PacBio::Primary::Tile* chunkHeader, uint32_t numFrames) override;

  protected:
      /// uses tbb to run multiple threads to crank out the CRCs for all the frames in the big buffer.
      /// The first stage strides the bigBuffer and sends a pointer to the second stage.
      /// The second stage is multithreaded, with each thread running a CRC on a frame.
      /// The last state is single threaded and writes the CRC data to a file.
      void WriteChunkBuffer(const ChunkBuffer* bigBuffer, const uint32_t numFrames)  override;

  public:

      uint64_t ReadFrame(uint64_t n, SequelMovieFrame <int16_t>& frame) const  override;

      void DumpSummary(std::ostream& s, uint32_t frames = 3, uint32_t rows = 3, uint32_t cols = 5)  override;
      Json::Value DumpJson(uint32_t /*frames*/, uint32_t /*rows*/, uint32_t /*cols*/) override;

      SequelMovieType Type() const  override;

      std::string TypeName() const override;
  };

 }
}
#endif //SEQUELACQUISITION_SEQUELMOVIEFILECRC_H_H
