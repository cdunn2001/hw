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
/// \brief  factory functions to generate various movie-type files for reading
///         and writing. Supports *.mov.h5, *.trc.h5, *.crc


#include <pacbio/primary/SequelMovieFactory.h>

#include <pacbio/primary/SequelMovie.h>
#include <pacbio/primary/SequelTraceFile.h>
#include <pacbio/primary/SequelMovieFileHDF5.h>
#include <pacbio/primary/SequelMovieFileCRC.h>
#include "SequelMovieFactory.h"

namespace PacBio
{
 namespace Primary
 {

  SequelMovieFileBase::SequelMovieType SequelMovieFactory::MovieType(const std::string& filename)
  {
      if (PacBio::Text::String::EndsWith(filename, ".crc"))
          return SequelMovieFileBase::SequelMovieType::CRC;
      else if (PacBio::Text::String::EndsWith(filename, ".trc.h5"))
          return SequelMovieFileBase::SequelMovieType::Trace;
      else if (PacBio::Text::String::EndsWith(filename, ".h5") || filename == "/dev/null")
          return SequelMovieFileBase::SequelMovieType::Frame;
      else
          return SequelMovieFileBase::SequelMovieType::Unknown;
  }

/// Create a output file based on the file extension. Currently supported extensions
/// are .h5, .trc.h5 and .crc.
std::unique_ptr<SequelMovieFileBase> SequelMovieFactory::CreateOutput(const SequelMovieConfig& config)
{
      if (!config.roi) throw PBException("ROI not supplied in SequelMovieConfig");

     auto type = MovieType(config.path);

     if (type == SequelMovieFileBase::SequelMovieType::CRC)
     {
         auto writer0 = new SequelMovieFileCRC(config.path, *config.roi);
         return std::unique_ptr<SequelMovieFileBase>(writer0);
     }
     else if (type == SequelMovieFileBase::SequelMovieType::Trace)
     {
         if (config.numFrames == 0)
         {
             throw PBException("numFrames for trace movie must not be zero in call to CreateOutput");
         }
         auto writer0 = new SequelTraceFileHDF5(config);
         if (!writer0->valid)
         {
             throw PBException("Could not open file " + config.path + " for writing Trace HDF5");
         }

         return std::unique_ptr<SequelMovieFileBase>(writer0);
     }
     else if (type == SequelMovieFileBase::SequelMovieType::Frame)
     {
         auto writer0 = new SequelMovieFileHDF5(config);
         if (!writer0->valid)
         {
             throw PBException("Could not open file " + config.path + " for writing HDF5");
         }
         return std::unique_ptr<SequelMovieFileBase>(writer0);
     }
     throw PBException("Unsupported file extension for output movie file: " + config.path);
}

  /// Create an input file based on the file extension. Currently supported extensions
  /// are .h5 and .trc.h5 .
  std::unique_ptr<SequelMovieFileBase> SequelMovieFactory::CreateInput(const std::string& filename)
  {
      if (PacBio::Text::String::EndsWith(filename, ".crc"))
      {
          throw PBException("Can't open CRC file for read. Use a text reader.");
      }
      else if (PacBio::Text::String::EndsWith(filename, ".trc.h5"))
      {
          auto x = new SequelTraceFileHDF5(filename);
          if (!x->valid)
          {
              throw PBException("Could not open file " + filename + " for reading");
          }
          return std::unique_ptr<SequelMovieFileBase>(x);
      }
      else if (PacBio::Text::String::EndsWith(filename, ".h5"))
      {
          auto x = new SequelMovieFileHDF5(filename);
          if (!x->valid)
          {
              throw PBException("Could not open file " + filename + " for reading");
          }
          return std::unique_ptr<SequelMovieFileBase>(x);
      }
      throw PBException("Unsupported file extension for input movie file: " + filename);
  }

  size_t SequelMovieFactory::EstimatedSize(const std::string& filename, const SequelROI& roi, uint64_t numFrames)
  {
      auto type = SequelMovieFactory::MovieType(filename);

      if (type == SequelMovieFileBase::SequelMovieType::CRC)
      {
          return numFrames * 10; // 9 digits and linefeed per frames
      }

      uint64_t bytesPerPixelToUse = sizeof(uint16_t);
      if (type == SequelMovieFileBase::SequelMovieType::Trace)
      {
          bytesPerPixelToUse = SequelTraceFileHDF5::SizeOfPixel;
      }
      else if (type == SequelMovieFileBase::SequelMovieType::Frame)
      {
          bytesPerPixelToUse = SequelMovieFileHDF5::SizeOfPixel;
      }
      else
      {
          throw PBException("Not supported");
      }
      return numFrames * roi.TotalPixels() * bytesPerPixelToUse * 102 / 100; // 2% overhead
  }

 }
}

