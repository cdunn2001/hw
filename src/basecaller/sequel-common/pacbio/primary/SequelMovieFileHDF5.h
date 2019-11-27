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
//  C++ class wrapper to handle SequelMovieFileHDF5
//
// Programmer: Mark Lakata

#ifndef SEQUELACQUISITION_SEQUELMOVIEFILEHDF5_H_H
#define SEQUELACQUISITION_SEQUELMOVIEFILEHDF5_H_H

#include <pacbio/primary/SequelMovie.h>
#include <pacbio/primary/SequelMovieEventsHDF5.h>
#include <pacbio/primary/SequelMovieConfig.h>

namespace PacBio
{
 namespace Primary
 {

  class EventObject;
  class SequelTraceFileHDF5;

  class SequelMovieFileHDF5 :
          public SequelMovieFileBase
  {
  public:
      static const size_t SizeOfPixel = sizeof(int16_t);
  public:
      uint32_t NROW;
      uint32_t NCOL;

  private:
      SequelHDF5MembersStart _memberStart_;
  public:
      H5::H5File hd5file;

      H5::EnumType eventEnumType;

      H5::DSetCreatPropList framesPropList;
      H5::DSetCreatPropList timesPropList;
      H5::DSetCreatPropList eventPropList;

      // data spaces
      H5::DataSpace scalar;
      H5::DataSpace dim2;
      H5::DataSpace framesSpace;
      H5::DataSpace darkFrameSpace;
      H5::DataSpace frameSpace;
      H5::DataSpace timesSpace;
      H5::DataSpace memFrame; // 1 frame memory space
      H5::DataSpace eventSpace;

      // objects
      H5::Group Movie;
      H5::Attribute FormatVersion;
      H5::Attribute Creator;
      H5::Attribute CreatorVersion;
      H5::Attribute TimeCreated;
      H5::Attribute InstrumentName;
      H5::Attribute CellSN;
      H5::Attribute PaneId;
      H5::Attribute PaneSetId;
      H5::Attribute FrameOffset;
      H5::Attribute FrameSize;
      H5::Attribute CameraConfigKey;

      H5::DataSet Frames;
      H5::Attribute FrameRate;
      H5::Attribute Exposure;
      H5::Attribute Frames_StartIndex;

      H5::DataSet FrameTime;
      H5::Attribute AcqStartTime;

      H5::DataSet DarkFrame;
      H5::Attribute DarkFrame_Exposure;
      H5::DataSet DarkFrameExp;

      H5::DataSet DarkFrameStack;
      H5::Attribute DarkFrameStack_StartIndex;

      H5::DataSet Gain;
      H5::DataSet ReadNoise;

      H5::DataSet AcquisitionXML;

      H5::DataSet ErrorFrames;

      SequelMovieEventsHDF5 events;

  private:
      SequelHDF5MembersEnd _memberEnd_;

  private:
      hsize_t hs_size[3];
      uint64_t NFRAMES_allocated;
      uint64_t errorFrameCount;

  public:
      static const uint32_t chunkSize = 1;
      uint32_t flushCycle = 0;

  private:
     /// this constructor is a private constructor, only called by the other two public constructors. The
     /// fake parameter is to make the overloaded argument signature unique
      SequelMovieFileHDF5( const std::string& fullFileName0, const SequelROI& roi0, bool fake, ChipClass chipClass);
  public:
      SequelMovieFileHDF5(const SequelMovieConfig& config);

      SequelMovieFileHDF5(const std::string& filename);

      virtual ~SequelMovieFileHDF5();

      void SetFirstFrame(uint64_t frameIndex) override { events.laserPowerChanges.SetFirstFrame(frameIndex); }

      void AddLaserPowerEvent(const EventObject& eo) override;

      void PrepareChunkBuffers(int numChunkBuffers=2) override
      {
          movieBuffer.PrepareChunkBuffers(numChunkBuffers, false);
      }

      void OpenFileForRead(const std::string& filename);

      void CreateFileForWrite(const SequelMovieConfig& config);

      void ClearSelectedFrames();

      void AddSelectedFrame(int i);

      uint32_t CountSelectedFrames() const { return selectedFramesPerChunk.size(); }

      int CompareFrames(const SequelMovieFileHDF5& b, bool dataOnly=false);

      uint32_t WriteChunkHeader(const PacBio::Primary::Tile* chunkHeader, uint32_t numFrames) override;

#ifdef OBSOLETE_EVENTS
      void AddEvent(SequelMovieFileBase::EventType_e type, uint64_t timestamp) override;
#endif

      void SetMeasuredFrameRate(double rate) override;

      SequelMovieType Type() const override;

      std::string TypeName() const override;


  protected:
      std::vector <uint16_t> selectedFramesPerChunk;

      void WriteChunkBuffer(const ChunkBuffer* bigBuffer, const uint32_t numFrames) override;

  public:
     // virtual overrides
      uint64_t AddFrame(const SequelMovieFrame <int16_t>& frame) override;

      uint64_t AddDarkFrame(float exposure, const SequelMovieFrame <int16_t>& frame) override;

      /// An override of the ReadFrame virtual function that calls the ReadFrame method below
      /// with options set to ReadFrame_NoOptions.
      uint64_t ReadFrame(uint64_t n, SequelMovieFrame <int16_t>& frame) const override;

      /// Options passed to ReadFrame. These options are bitwise integers and can be logically OR'ed together.
      enum {
          ReadFrame_NoOptions  = 0,
          ReadFrame_Unused     = 1, //!/
          ReadFrame_DontResize = 2, ///! means don't resize the frame, only load the frame that overlaps with the ZMW coordinates
          ReadFrame_Reserved   = 4  ///! this is a bit mask, so each value of the enum must be a power of 2
      };
      /// Reads the n'th frame from a movie into the frame parameter.  If `DontResize` option is used,
      /// the size of the frame is not changed and just the data lying within the row/col size of the frame
      /// is filled in. Otherwise, the frame is resized to the native size of the HDF5 file.
      /// \param n - Load the frame n (zero based from start of movie)
      /// \param frame - reference to the frame instance into which to load the data
      /// \param options - a integer of bit masks to select certain options. See the definition of the enum above for
      ///                  valid bit masks and their meaning.
      /// \returns The number of bytes read from the file into the frame data.
      uint64_t ReadFrame(uint64_t n, SequelMovieFrame <int16_t>& frame, int options) const;

      std::vector<EventObject> ReadAllEvents() override { return events.laserPowerChanges.ReadAllEvents(); }

#ifdef OBSOLETE_EVENTS
      void ReadEvent(int64_t n, EventType_e& type, uint64_t& timestamp) const;

      std::vector <Event> GetEvents();
#endif

      void ReadDarkFrameStack(int64_t n, SequelMovieFrame <int16_t>& frame, float& exposure);

      /// Checks that the movie is correct compared to the CreateTest method.
      /// \param limitErrorsTo - limit error reporting to this many mismatches
      /// \param frameRate - the expected frameRate. If set to 0.0, then don't check time stamps
      uint64_t VerifyTest(uint64_t limitErrorsTo, double frameRate);

      /// \param correctPeriod if false, the period is calculated as 5 seconds for reasons I have forgotten.
      /// If correctPeriod is true, then the period is calculated correctedly for 200 fps.
      void CreateTest(uint32_t nframes, uint32_t nDarkFrames, int bits, bool correctPeriod = false);

      void CreateTestBeta(uint32_t nframes, uint32_t nDarkFrames, int bits);

      void CreateDeltaPattern(uint32_t nframes, uint32_t nDarkFrames, int bits, int spacing);

      void CreateFrameTestPattern(uint32_t nframes, std::function<int16_t(uint32_t,uint32_t,uint32_t)> pattern);

      void DumpSummary(std::ostream& s, uint32_t frames = 3, uint32_t rows = 3, uint32_t cols = 5) override;
      Json::Value DumpJson(uint32_t /*frames*/, uint32_t /*rows*/, uint32_t /*cols*/) override;

      inline uint16_t GetPackedHeaderPduSize(void) const
      {
          return SequelMovieFrame<int16_t>::GetPackedHeaderPduSize(NROW, NCOL);
      }

      inline uint32_t GetPackedRowPduSize(void) const
      {
          return SequelMovieFrame<int16_t>::GetPackedRowPduSize(NROW, NCOL, bitsPerPixel, frameClass_);
      }

      inline uint32_t GetPackedFrameSize(void) const
      {
          return SequelMovieFrame<int16_t>::GetPackedFrameSize(NROW, NCOL, bitsPerPixel, frameClass_);
      }

      static uint32_t PixelPatternBeta(uint32_t frame, uint32_t row, uint32_t col, uint32_t mask)
      {
          uint32_t seed = (row * 997 + col * 991) & mask;
          uint32_t pixel = (frame % 32) + seed;
          if (pixel > mask) pixel = mask;
          return pixel;
      }

      void CopyHeader(const SequelMovieFileHDF5& src);
      void CopyHeader(const SequelTraceFileHDF5& src);
      void CopyHeader(const SequelMovieFileBase* src);
      static void SetMovieDebugStream(std::ostream& os);

      const SequelMovieConfig::MovieChunking& GetChunking() const;

 private:
      uint64_t frames_StartIndex;
      uint32_t cameraConfigKey;
      bool cameraConfigKeySet;
      SequelMovieConfig config_;
  };

 }
}

#endif //SEQUELACQUISITION_SEQUELMOVIEFILEHDF5_H_H
