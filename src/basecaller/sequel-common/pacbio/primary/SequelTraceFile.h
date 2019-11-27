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
/// \brief  declaration of trace file IO


#ifndef _SEQUEL_TRACE_FILE_H_
#define _SEQUEL_TRACE_FILE_H_

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <cstdint>
#include <atomic>
#include <fstream>
#include <iostream>

#include <boost/iostreams/filtering_streambuf.hpp>
#include <boost/iostreams/copy.hpp>
#include <boost/iostreams/filter/gzip.hpp>

#include <pacbio/logging/Logger.h>
#include <pacbio/text/imemstream.h>
#include <pacbio/POSIX.h>
#include <pacbio/dev/AutoTimer.h>

#include <pacbio/primary/AnalogMode.h>
#include <pacbio/primary/SequelMovie.h>
#include <pacbio/primary/ChipLayout.h>
#include <pacbio/primary/Platform.h>
#include <pacbio/primary/SequelMovieEventsHDF5.h>
#include <pacbio/primary/UnitCell.h>
#include <pacbio/primary/WorkingFile.h>
#include <pacbio/primary/SequelMovieConfig.h>

extern unsigned int _home_UNIXHOME_mlakata_Matrices_Full_csv_gz_len;
extern unsigned char _home_UNIXHOME_mlakata_Matrices_Full_csv_gz[];

using namespace PacBio::Dev;

namespace PacBio
{
 namespace Primary
 {

  class StatAccumulator;
  class Tranche;
  class EventObject;

  class SequelTraceFileHDF5 :
          public SequelMovieFileBase
  {
  public:
      enum PlatformId_e
      {
          PlatformId_Astro1       = 1,
          PlatformId_Springfield2 = 2,
          PlatformId_SequelPoC3   = 3,
          PlatformId_SequelAlpha4 = 4,
          PlatformId_Spider5      = 5,
          PlatformId_Benchy6      = 6
      };
      static const size_t SizeOfPixel = sizeof(int16_t);

  private:
      class DataSpace :
              public H5::DataSpace
      {
      public:
          DataSpace(hsize_t dim1)
                  : H5::DataSpace(1, &dim1)
          { }

          DataSpace(hsize_t dim1, hsize_t dim2)
                  : H5::DataSpace()
          {
              hsize_t dim[] = {dim1, dim2};
              this->setExtentSimple(2, dim);
          };

          DataSpace(hsize_t dim1, hsize_t dim2, hsize_t dim3)
                  : H5::DataSpace()
          {
              hsize_t dim[] = {dim1, dim2, dim3};
              this->setExtentSimple(3, dim);
          };
      };

  private:
      std::atomic<uint64_t> NFRAMES_written;
      std::vector<double> frameRateHistory;
      bool spectraUnset_ = false;
#ifdef SUPPORT_VARIANCE
      std::vector<UnitCell> antiZMWs;
      std::map<uint32_t, uint8_t> findAntiZmw;
      std::unique_ptr<StatAccumulator> redStat;
      std::unique_ptr<StatAccumulator> greenStat;
#endif
      mutable std::unique_ptr<SequelSparseROI> originalRoi_;
      mutable double frameRateCache_ = 0;

  public:
      uint32_t NUM_HOLES = 0;
      uint32_t NUM_CHANNELS;

  private:
      SequelHDF5MembersStart _memberStart_;
  public:

      H5::H5File hd5file;
      H5::DSetCreatPropList tracesPropList;
      H5::DataSpace scalar;

      H5::Group TraceData;
      H5::Attribute Representation;
      H5::Attribute TraceOrder;
      H5::Attribute Look;
      H5::Attribute Version;
      H5::Attribute FormatVersion;
      H5::Attribute SoftwareVersion;
      H5::Attribute ChangeListID;
      H5::Attribute DateCreated;
      H5::Attribute NumColors;


      H5::DataSet HoleNumber;
      H5::DataSet HoleXY;
      H5::DataSet HoleChipLook;
      H5::DataSet ActiveChipLooks;
      H5::DataSet HoleXYPlot;
      //  AntiHoles      = Movie.createDataSet      ("AntiHoles"               , int16     , DataSpace(?));
      H5::DataSet Spectra;
      H5::DataSet Traces;
      //Traces         = Movie.createDataSet      ("Traces"                 , float32   , DataSpace(numHoles,numAnalogs,numFrames));
      H5::DataSet HoleStatus; // 0 = normal ZMW, 1 = not normal ZMW
      H5::DataSet HoleType;
      H5::DataSet Variance;
      H5::DataSet VFrameSet;
      H5::DataSet HolePhase;
      H5::DataSet HPFrameSet;
      H5::DataSet ReadVariance;
      H5::DataSet RVFrameSet;
      H5::DataSet IFVarianceScale;
      H5::DataSet OFBackgroundMean;
      H5::DataSet OFBackgroundVariance;

      H5::Group Codec;
      H5::Attribute Codec_Name;
      H5::Attribute Codec_Config;
      H5::Attribute Codec_Units;
      H5::Attribute Codec_BitDepth;
      H5::Attribute Codec_DynamicRange;
      H5::Attribute Codec_Bias;

      H5::Group ScanData;
      H5::Attribute SD_FormatVersion;
      H5::DataSet AcquisitionXML;
      H5::Attribute RI_InstrumentName;
      H5::Attribute RI_PlatformId;
      H5::Attribute RI_PlatformName;
      H5::Attribute MovieName;
      H5::Attribute MoviePath;
      H5::Attribute SequencingChemistry;
      H5::Attribute SequencingKit;
      H5::Attribute BindingKit;

      H5::Attribute Control;
      H5::Attribute IsControlUsed;
      H5::Attribute LaserOnFrame;
      H5::Attribute LaserOnFrameValid;
      H5::Attribute HotStartFrame;
      H5::Attribute HotStartFrameValid;
      H5::Attribute CameraType;
      H5::Attribute CameraGain;
      H5::Attribute AduGain; // aka photoelectronSensitivity
      H5::Attribute CameraBias;
      H5::Attribute CameraBiasStd;
      H5::Attribute FrameRate;
      H5::Attribute NumFrames;
  public:
      H5::Group ChipInfo;
      H5::Attribute LayoutName;
      H5::DataSet FilterMap;
      H5::DataSet ImagePsf;
      H5::DataSet XtalkCorrection;
      H5::DataSet AnalogRefSpectrum;
      H5::DataSet AnalogRefSnr;

      H5::Group DyeSet;
      H5::Attribute NumAnalog;
      H5::Attribute BaseMap;
      H5::DataSet AnalogSpectra;
      H5::DataSet RelativeAmp;
      H5::DataSet ExcessNoiseCV;
      H5::DataSet PulseWidthMean;
      H5::DataSet IpdMean;
      // added in 2.5
      H5::DataSet Pw2SlowStepRatio;
      H5::DataSet Ipd2SlowStepRatio;

      SequelMovieEventsHDF5 events;

      // Only exists for simulated trace files.
      H5::Group GroundTruth;
      H5::DataSet StateCovariance;
      H5::DataSet StateMean;

  private:
      SequelHDF5MembersEnd _memberEnd_;
      mutable std::unique_ptr<const ChipLayout> chipLayout_;
      mutable std::vector<UnitCell> holeXY_;
      mutable struct
      {
          bool enabled_ = true;
          uint64_t frameBase_ = 0XFFFFFFFFFFFFFFFFULL;
          hsize_t size_[3];
          std::vector<int16_t> pixels_;
      } cache_;


 public:
      bool Supports2_3 = false;
      bool Supports2_4 = false;
      bool Supports2_5 = false;
      bool simulated = false;
 public:
      static const uint32_t chunkSize = 1;
 public:
     void SetChunking(const SequelMovieConfig::TraceChunking& chunking);
     SequelMovieConfig::TraceChunking GetChunking() const;
     void SetGzipCompression(int level);
 private:
     SequelMovieConfig config_;

  private:
      SequelTraceFileHDF5(const std::string& filename, uint64_t nFrames, const SequelROI& roi0, bool fake,
                       ChipClass chipClass);
  public:
      SequelTraceFileHDF5(const std::string& filename);
      SequelTraceFileHDF5(const SequelMovieConfig& config);
      ~SequelTraceFileHDF5() override;

#if 1
      void SetAnalogs(const AnalogSet& analogs, const ChipLayout& layout);
#else
      void SetBaseMap(const std::string& baseMapExt, uint32_t numChannels, uint32_t numAnalogs, const double spectralAngle[], const double amplitudes[],const ChipLayout& layout);
#endif

     void PrepareChunkBuffers(int numChunkBuffers=2) override
     {
         movieBuffer.PrepareChunkBuffers(numChunkBuffers, true);
     }

     //overrides
      uint64_t AddFrame(const SequelMovieFrame<int16_t>& frame) override;
      uint64_t AddDarkFrame(float exposure, const SequelMovieFrame<int16_t>& frame) override;
#ifdef OBSOLETE_EVENTS
      void AddEvent(SequelMovieFileBase::EventType_e type, uint64_t timestamp) override;
#endif
      void SetMeasuredFrameRate(double rate) override;
      void WriteChunkBuffer(const ChunkBuffer* bigBuffer, const uint32_t numFrames) override;
      uint64_t ReadFrame(uint64_t n, SequelMovieFrame<int16_t>& frame) const override;
      enum {
          ReadFrame_NoOptions  = 0,
          ReadFrame_Condensed  = 1, ///! means to ignore ZMW coordinates, and just load every pixel of the frame with consecutive ZMWs
                                    ///! Until the frame is filled.
          ReadFrame_DontResize = 2, ///! means don't resize the frame, only load the frame that overlaps with the ZMW coordinates
          ReadFrame_Reserved   = 4  ///! this is a bit mask, so each value of the enum must be a power of 2
      };
     /// Reads the n'th frame from a trace file into the frame parameter.  If `DontResize` option is used,
     /// the size of the frame is not changed and just the data lying within the row/col size of the frame
     /// is filled in. Otherwise, the frame is resized to the native size of the HDF5 file.
     /// \param n - Load the frame n (zero based from start of movie)
     /// \param frame - reference to the frame instance into which to load the data
     /// \param options - a integer of bit masks to select certain options. See the definition of the enum above for
     ///                  valid bit masks and their meaning.
     /// \returns The value 1, although this value may change in the future.
      uint64_t ReadFrame(uint64_t n, SequelMovieFrame <int16_t>& frame, int options) const;
      void DumpSummary(std::ostream& s, uint32_t frames = 3, uint32_t rows = 3, uint32_t cols = 5) override;
      Json::Value DumpJson(uint32_t /*frames*/, uint32_t /*rows*/, uint32_t /*cols*/) override;
      uint32_t WriteChunkHeader(const PacBio::Primary::Tile* chunkHeader, uint32_t numFrames) override;
      void AddTile(const PacBio::Primary::Tile* tile, uint32_t tileOffset) override;
      SequelMovieType Type() const override
      {
          return SequelMovieType::Trace;
      }

      std::string TypeName() const override
      {
          return "Trace";
      }
      const SequelROI* GetROI() const override;
      void SetFirstFrame(uint64_t frameIndex) override { events.laserPowerChanges.SetFirstFrame(frameIndex); }
      void AddLaserPowerEvent(const EventObject& eo) override;
      std::vector<EventObject> ReadAllEvents() override { return events.laserPowerChanges.ReadAllEvents(); }

      // non virtual
      void OpenFileForRead(const std::string& filename);
      void CreateFileForWrite(const SequelMovieConfig& config);

      void ReadDarkFrameStack(uint64_t n, SequelMovieFrame<int16_t>& frame, float& exposure);
      void CreateTest(uint32_t nframes, uint32_t nDarkFrames, int bits);
      void DetermineAntiZMWs(const SequelROI& roi, const ChipLayout& layout);
      void CalculateVariance();
      uint32_t ReadTile(uint32_t zmwOffset, uint64_t frameOffset, Tile* tile);
      uint32_t ReadTranche(uint32_t zmwOffset, uint64_t frameOffset, Tranche* tranche,uint32_t numFrames);
      // Similar to ReadTranche, but the number of frames read is configurable.  
      uint32_t ReadLaneSegment(uint32_t zmwOffset, uint64_t frameOffset, uint32_t maxReadLen, int16_t* dest);
      uint32_t Read1CZmwLaneSegment(uint32_t zmwOffset, uint32_t numZmws, uint64_t frameOffset, uint32_t numFrames, int16_t* dest) const;
      uint32_t Read1CTrace(uint32_t zmwOffset, uint64_t frameOffset, std::vector<int16_t>& traceData) const;
      uint32_t Read2CTrace(uint32_t zmwOffset, uint64_t frameOffset, std::vector<std::pair<int16_t,int16_t> >& traceData) const;
      void Write1CTrace(uint32_t zmwOffset, uint64_t frameOffset, const std::vector<int16_t>& traceData);
      void Write2CTrace(uint32_t zmwOffset, uint64_t frameOffset, const std::vector<std::pair<int16_t,int16_t> >& traceData);
      const ChipLayout& GetChipLayout() const;
      const std::vector<UnitCell>& GetUnitCells() const;

      /// compare this trc file with b.
      /// \param b - the comparison file object
      /// \param subsetFlag - only compare the ZMWs in this object to those in B. Ignore ZMWS in b that are not in this.
      int CompareTraces(const SequelTraceFileHDF5& b, bool subsetFlag = false);

      /// \returns a integer that represents the platform that was used to capture the data.
      PlatformId_e GetPlatformId() const
      {
          int32_t p;
          RI_PlatformId >> p;
          return static_cast<PlatformId_e>(p);
      }

      /// \returns a Sequel Platform enum, based on the internal platform integer
      Platform GetPlatform() const
      {
          switch (GetPlatformId())
          {
          case SequelTraceFileHDF5::PlatformId_Astro1:
          case SequelTraceFileHDF5::PlatformId_Springfield2:
              throw PBException("Ha ha very funny. This code base does not support Astro or Springfield. PlatformId="+ std::to_string(static_cast<int>(GetPlatformId())));
          case SequelTraceFileHDF5::PlatformId_SequelPoC3:
          case SequelTraceFileHDF5::PlatformId_SequelAlpha4:
              return Platform::Sequel1PAC1;
              break;
          case SequelTraceFileHDF5::PlatformId_Spider5:
              return Platform::Spider;
              break;
          case SequelTraceFileHDF5::PlatformId_Benchy6:
              return Platform::Benchy;
              break;
          default:
//              PBLOG_ERROR << "Unsupported PlatformId_e: " << static_cast<int>(GetPlatformId());
              throw PBException("Unsupported PlatformId_e: " + std::to_string(static_cast<int>(GetPlatformId())));
          }
      }

 private:
#if 1
     void SetSpectra(const AnalogSet& analogs);
#else
            void SetSpectra(const std::string& baseMapExt, uint32_t numChannels, uint32_t numAnalogs, const double spectralAngle[], const double amplitudes[]);
#endif
      void UpdateLayout(const ChipLayout& layout);
  };
 }
}


#endif // _SEQUEL_TRACE_FILE_H_
