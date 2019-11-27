// Copyright (c) 2015, Pacific Biosciences of California, Inc.
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
//  Description:
/// \brief  Implementation of CRC output writer

#include <pacbio/logging/Logger.h>
#include <pacbio/primary/SequelMovie.h>


#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <math.h>
#include <string.h>
//#if defined(__SSSE3__)
//#include <tmmintrin.h>
//#endif
#ifndef WIN32
#include <unistd.h>
#endif

#include <atomic>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <memory>
#include <random>
#include <vector>
#include <list>

#include <boost/crc.hpp>
#include <tbb/pipeline.h>
#include <tbb/scalable_allocator.h>

#include <pacbio/ipc/ThreadSafeQueue.h>
#include <pacbio/text/String.h>
#include <pacbio/Utilities.h>
#include <pacbio/utilities/ISO8601.h>
#include <pacbio/utilities/PerforceUtil.h>
#include <pacbio/HugePage.h>

#include <pacbio/primary/Tile.h>
#include <pacbio/primary/SequelMovieFileCRC.h>

namespace PacBio
{
 namespace Primary
 {

#if 0
  SequelMovieFileCRC::SequelMovieFileCRC(const std::string& filename, int32_t rows, int32_t cols)
          : SequelMovieFileCRC(filename, SequelROI(0,0,rows,cols))
  {

  }
#endif

  SequelMovieFileCRC::SequelMovieFileCRC(const std::string& filename, const SequelROI& roi0)
          :
          SequelMovieFileBase(filename, roi0, false, ChipClass::UNKNOWN /* who cares */),
          fileout{filename},
          dataSize{roi0.TotalPixels() * sizeof(uint16_t)},
          t{[this]() {
            this->SaverThread();
         }}
  {
  }

  SequelMovieFileCRC::~SequelMovieFileCRC()
  {
      try
      {
          queue.Push(nullptr);
          if (t.joinable())
          {
              t.join();
          }
          FinalizeSaverThread();
      }
      catch (const std::exception& ex)
      {
          PBLOG_ERROR << "SequelMovieFileCRC::~SequelMovieFileCRC() caught exception: " << ex.what();
      }
  }

  void SequelMovieFileCRC::SaverThread()
  {
      std::atomic <uint32_t> n{0};
      int ntoken = 10;
      size_t dataSize0 = dataSize;
      std::ostream& fileout0(fileout);
      auto& queue0 = queue;
      tbb::parallel_pipeline
              (
                      ntoken,
                      tbb::make_filter<void, const uint8_t*>(
                              tbb::filter::serial_in_order,
                              [&n, dataSize0, &queue0](tbb::flow_control& fc) -> const uint8_t* {
                                  const uint8_t* p = queue0.Pop();
                                  if (!p) fc.stop();
                                  return p;
                              })

                      & tbb::make_filter<const uint8_t*, uint32_t>(
                              tbb::filter::parallel,
                              [dataSize0](const uint8_t* data) -> uint32_t {
                                  boost::crc_32_type crcEngine;
                                  crcEngine.reset();
                                  crcEngine.process_bytes(data, dataSize0);
                                  uint32_t crc = crcEngine.checksum();
                                  delete[] data;
                                  return crc;
                              })

                      & tbb::make_filter<uint32_t, void>(
                              tbb::filter::serial_in_order,
                              [&fileout0](uint32_t result) {
                                  fileout0 << result << "\n";
                              })
              );

  }

  uint64_t SequelMovieFileCRC::AddFrame(const SequelMovieFrame<int16_t>& frame)
  {
      uint8_t* p = new uint8_t[frame.DataSize()];
      memcpy(p, frame.data, frame.DataSize());
      queue.Push(p);
      return 0;
  }

  uint64_t SequelMovieFileCRC::AddDarkFrame(float /*exposure*/, const SequelMovieFrame<int16_t>& /*frame*/)
  {
      // we throw away the dark frames
      return 0;
  }

#ifdef OBSOLETE_EVENTS
  void SequelMovieFileCRC::AddEvent(EventType_e /*type*/, uint64_t /*timestamp*/)
  {
      // we throw away events
  }
#endif

  void SequelMovieFileCRC::SetMeasuredFrameRate(double /*rate*/)
  {
      // FrameRate << rate;
  }

  /// sort of a stub that just counts precounts frames
  ///
  uint32_t SequelMovieFileCRC::WriteChunkHeader(const PacBio::Primary::Tile* chunkHeader, uint32_t numFrames)
  {
//      SequelMovieFileBase::AddChunk(chunkHeader, numFrames);
      if (numFrames > chunkHeader->NumFramesInHeader()) numFrames = chunkHeader->NumFramesInHeader();
      NFRAMES += numFrames;
      return numFrames;
  }


  /// uses tbb to run multiple threads to crank out the CRCs for all the frames in the big buffer.
  /// The first stage strides the bigBuffer and sends a pointer to the second stage.
  /// The second stage is multithreaded, with each thread running a CRC on a frame.
  /// The last state is single threaded and writes the CRC data to a file.
  void SequelMovieFileCRC::WriteChunkBuffer(const ChunkBuffer* bigBuffer, const uint32_t numFrames)
  {
      assert(bigBuffer != nullptr);

      std::atomic <uint32_t> n{0};
      int ntoken = 16;
      std::ostream& fileout0(fileout);
      auto errors = bigBuffer->HeaderTile()->ErroredFrames();
      uint32_t errorIndex = 0;
      uint64_t frameIndex = bigBuffer->HeaderTile()->FirstFrameIndex();
#if 0
      for(auto&& s : errors)
      {
          std::cout << "Seen error at " << s << std::endl;
      }
      std::cout << "first frame: " << frameIndex << std::endl;
#endif
      tbb::parallel_pipeline
              (
                      ntoken,
                      tbb::make_filter<void, const uint8_t*>(
                              tbb::filter::serial_in_order,
                              [&n, bigBuffer, numFrames, this](tbb::flow_control& fc) -> const uint8_t* {
                                  if (n < numFrames)
                                  {
                                      const uint8_t* buffer = bigBuffer->Data(0) + dataSize * n;
                                      n++;
                                      return buffer;
                                  }
                                  else
                                  {
                                      fc.stop();
                                      return NULL;
                                  }
                              })

                      & tbb::make_filter<const uint8_t*, uint32_t>(
                              tbb::filter::parallel,
                              [this](const uint8_t* buffer) -> uint32_t {
                                  boost::crc_32_type crcEngine;
                                  crcEngine.reset();
                                  if (roi_->Everything())
                                  {
                                      crcEngine.process_bytes(buffer, dataSize);
                                  }
                                  else
                                  {
#if 1
                                      throw PBException("not supported");
#else
                                      for (int32_t row = roi_.RelativeRowPixelMin(); row < roi_.RelativeRowPixelMax(); row++)
                                      {
                                          const uint8_t* ptr = buffer + sizeof(uint16_t) * row;
                                          ptr += roi_.RelativeColPixelMin() * sizeof(uint16_t);
                                          uint32_t len = roi_.NumPixelCols() * sizeof(uint16_t);
                                          crcEngine.process_bytes(ptr, len);
                                      }
#endif
                                  }
                                  uint32_t crc = crcEngine.checksum();
                                  return crc;
                              })

                      & tbb::make_filter<uint32_t, void>(
                              tbb::filter::serial_in_order,
                              [&fileout0,&errors,&errorIndex,&frameIndex](uint32_t result) {
                                  fileout0 << result;

//                                  std::cout << "checking:"  << errorIndex << " " << errors.size() ;
//                                  if (errorIndex < errors.size()) std::cout << " " << errors[errorIndex] << " "  << frameIndex << std::endl; else std::cout << std::endl;

                                  if (errorIndex < errors.size() && errors[errorIndex] == frameIndex)
                                  {
                                      // marked errored frames with a !
                                      fileout0 << " !";
                                      errorIndex++;
                                  }
                                  frameIndex++;
                                  fileout0 << "\n";
                              })
              );

  }


  uint64_t SequelMovieFileCRC::ReadFrame(uint64_t /*n*/, SequelMovieFrame<int16_t>& /*frame*/) const
  {
      throw PBException("Not supported");
  }

  void SequelMovieFileCRC::DumpSummary(std::ostream& s, uint32_t /*frames*/, uint32_t /*rows*/, uint32_t /*cols*/)
  {
      s << "CRC file " << std::endl;
      s << " NFrames:" << NFRAMES << std::endl;
  }
  Json::Value SequelMovieFileCRC::DumpJson(uint32_t /*frames*/, uint32_t /*rows*/, uint32_t /*cols*/)
  {
      throw PBException("Not supported");
  }

  SequelMovieFileBase::SequelMovieType SequelMovieFileCRC::Type() const
  {
      return SequelMovieType::CRC;
  }

  std::string SequelMovieFileCRC::TypeName() const
  {
      return "CRC";
  }
 };
}
