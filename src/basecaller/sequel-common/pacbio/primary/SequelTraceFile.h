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
#include <pacbio/primary/SequelHDF5.h>
#include <pacbio/primary/UnitCell.h>

extern unsigned int _home_UNIXHOME_mlakata_Matrices_Full_csv_gz_len;
extern unsigned char _home_UNIXHOME_mlakata_Matrices_Full_csv_gz[];

using namespace PacBio::Dev;

namespace PacBio
{
 namespace Primary
 {

  class StatAccumulator;
  class EventObject;

  class SequelTraceFileHDF5
  {
  public:

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

      // Only exists for simulated trace files.
      H5::Group GroundTruth;
      H5::DataSet StateCovariance;
      H5::DataSet StateMean;

  private:
      SequelHDF5MembersEnd _memberEnd_;
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
     void SetGzipCompression(int level);
 private:

  private:
      SequelTraceFileHDF5(const std::string& filename, uint64_t nFrames, bool fake);
  public:
      SequelTraceFileHDF5(const std::string& filename);
      ~SequelTraceFileHDF5();

      enum {
          ReadFrame_NoOptions  = 0,
          ReadFrame_Condensed  = 1, ///! means to ignore ZMW coordinates, and just load every pixel of the frame with consecutive ZMWs
                                    ///! Until the frame is filled.
          ReadFrame_DontResize = 2, ///! means don't resize the frame, only load the frame that overlaps with the ZMW coordinates
          ReadFrame_Reserved   = 4  ///! this is a bit mask, so each value of the enum must be a power of 2
      };

      // non virtual
      void OpenFileForRead(const std::string& filename);

      void CalculateVariance();
      uint32_t Read1CZmwLaneSegment(uint32_t zmwOffset, uint32_t numZmws, uint64_t frameOffset, uint32_t numFrames, int16_t* dest) const;
      const std::vector<UnitCell>& GetUnitCells() const;

      /// compare this trc file with b.
      /// \param b - the comparison file object
      /// \param subsetFlag - only compare the ZMWs in this object to those in B. Ignore ZMWS in b that are not in this.
      int CompareTraces(const SequelTraceFileHDF5& b, bool subsetFlag = false);

      /// \returns a integer that represents the platform that was used to capture the data.

 private:
#if 1
     void SetSpectra(const AnalogSet& analogs);
#else
            void SetSpectra(const std::string& baseMapExt, uint32_t numChannels, uint32_t numAnalogs, const double spectralAngle[], const double amplitudes[]);
#endif
    public: 
        size_t NFRAMES = 0;
  };
 }
}

#endif // _SEQUEL_TRACE_FILE_H_
