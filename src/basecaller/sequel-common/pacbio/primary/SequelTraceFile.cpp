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
/// \brief  Implementation of trace file IO



#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <exception>
#include <vector>
#include <iomanip>
#include <fstream>
#include <iostream>
#include <atomic>
#include <numeric>

#include <boost/filesystem/path.hpp>
#include <boost/numeric/conversion/cast.hpp>

#include <pacbio/logging/Logger.h>
#include <pacbio/text/imemstream.h>
#include <pacbio/text/String.h>
#include <pacbio/POSIX.h>
#include <pacbio/dev/AutoTimer.h>

#include <pacbio/primary/SequelTraceFile.h>

using namespace PacBio::Dev;
using namespace PacBio::Text;

namespace PacBio
{
 namespace Primary
 {

  static const int GREEN_TRACE_OFFSET = 0;
  static const int RED_TRACE_OFFSET = 1;
#ifdef SUPPORT_VARIANCE
  static const int numBlocks = 1;
#endif

  // data types

  // data types

#define int16()      H5::PredType::STD_I16LE
#define uint16()     H5::PredType::STD_U16LE
#define uint32()     H5::PredType::STD_U32LE
#define uint64()     H5::PredType::STD_U64LE
#define float32()    H5::PredType::IEEE_F32LE
#define float64()    H5::PredType::IEEE_F64LE

#define uint8()      H5::PredType::STD_U8LE
#define int32()      H5::PredType::STD_I32LE

  class StatAccumulator
  {
  public:

      StatAccumulator() :
              sum(0),
              sumsq(0),
              n(0)
      {

      }
      void clear()
      {
          sum = 0;
          sumsq = 0;
          n = 0 ;
      }
      void Add(int32_t p)
      {
          // std::cout << "Stat.Add " << std::hex << p << std::dec << " " << sum << " " << sumsq << " " << n << std::endl;
          sum += p;
          sumsq += (double)p * p;
          n++;
      }
      double Mean() const
      {
          if (n == 0) throw PBException("Divide by zero, no data to calculate mean.");
          return sum/n;
      }
      double Variance() const
      {
          if (n == 0) throw PBException("Divide by zero, no data to calculate variance.");
          return (sumsq - sum * sum / n) / n;
      }
      double sum = 0;
      double sumsq = 0;
      double n = 0;
  };

  SequelTraceFileHDF5::SequelTraceFileHDF5(const std::string& /*fullFileName0*/, uint64_t nFrames, bool /*fake*/)
              : NFRAMES_written(0)
              , frameRateHistory{}
              , spectraUnset_{false}
#ifdef SUPPORT_VARIANCE
              , antiZMWs{}
              , findAntiZmw{}
              , redStat(new StatAccumulator)
              , greenStat(new StatAccumulator)
#endif
              , NUM_HOLES(0)
              , NUM_CHANNELS(0)
  {
      NFRAMES = nFrames;
  }

  SequelTraceFileHDF5::SequelTraceFileHDF5(const std::string& filename)
              : SequelTraceFileHDF5(filename, 0, false)
  {
      OpenFileForRead(filename);
  }


      SequelTraceFileHDF5::~SequelTraceFileHDF5()
      {
      }



      void SequelTraceFileHDF5::OpenFileForRead(const std::string& filename)
      {
          DisableDefaultErrorHandler noErrors;
          try
          {
              {
                  LOCK_HDF5();

                  hd5file.openFile(filename.c_str(), H5F_ACC_RDONLY);

                  TraceData = hd5file.openGroup("/TraceData");
                  Representation = TraceData.openAttribute("Representation");
                  TraceOrder = TraceData.openAttribute("TraceOrder");
                  Look = TraceData.openAttribute("Look");
                  bool Supports2_2 = false;
                  bool Supports1_x = true;
                  Supports2_3 = false;
                  if (TraceData.attrExists("Version"))
                  {
                      Version = TraceData.openAttribute("Version");
                      std::string versionString;
                      Version >> versionString;
                      if (versionString == "Otto 0.5")
                      {


                      }
                      Supports2_2 = String::Contains(versionString, "POC");

                  }
                  else
                  {
                      Supports1_x = false;
                  }


                  if (Supports2_2)
                  {
                      FormatVersion = TraceData.openAttribute("FormatVersion");
                      SoftwareVersion = TraceData.openAttribute("SoftwareVersion");
                      ChangeListID = TraceData.openAttribute("ChangeListID");
                      NumColors = TraceData.openAttribute("NumColors");
                  }
                  DateCreated = TraceData.openAttribute("DateCreated");


                  HoleNumber = TraceData.openDataSet("HoleNumber");
                  HoleXY = TraceData.openDataSet("HoleXY");
                  //  AntiHoles      = TraceData.openDataSet      ("AntiHoles"       );

                  try
                  {
                      Spectra = TraceData.openDataSet("Spectra");
                      Traces = TraceData.openDataSet("Traces");
                      HoleStatus = TraceData.openDataSet("HoleStatus");
                      Variance = TraceData.openDataSet("Variance");
                      VFrameSet = TraceData.openDataSet("VFrameSet");
                      ReadVariance = TraceData.openDataSet("ReadVariance");
                      RVFrameSet = TraceData.openDataSet("RVFrameSet");
                      IFVarianceScale = TraceData.openDataSet("IFVarianceScale");

                      hsize_t chunk_dims[3];
                      tracesPropList = Traces.getCreatePlist();
                      if (H5D_CHUNKED == tracesPropList.getLayout())
                      {
                          tracesPropList.getChunk(3, chunk_dims);
                      }
                      else
                      {
                          // no chunking, just set values to 1 in case someone asks.
                          chunk_dims[0] = 1;
                          chunk_dims[1] = 1;
                          chunk_dims[2] = 1;
                      }
                      //tracesPropList.getDeflate(compressionLevel_);


                      if (Supports1_x)
                      {
                          HoleChipLook = TraceData.openDataSet("HoleChipLook");
                          ActiveChipLooks = TraceData.openDataSet("ActiveChipLooks");
                          HoleXYPlot = TraceData.openDataSet("HoleXYPlot");
                          HolePhase = TraceData.openDataSet("HolePhase");
                          HPFrameSet = TraceData.openDataSet("HPFrameSet");
                          OFBackgroundMean = TraceData.openDataSet("OFBackgroundMean");
                          OFBackgroundVariance = TraceData.openDataSet("OFBackgroundVariance");
                      }
                      if (Supports2_2)
                      {
                          HoleType = TraceData.openDataSet("HoleType");
                      }
                  }
                  catch (H5::GroupIException)
                  {
                      PBLOG_WARN << "Trace file does not have a dataset that depends on the Spectra dataset";
                  }

                  if (Supports1_x)
                  {
                      Codec = TraceData.openGroup("Codec");
                      Codec_Name = Codec.openAttribute("Name");
                      Codec_Config = Codec.openAttribute("Config");
                      Codec_Units = Codec.openAttribute("Units");
                      Codec_BitDepth = Codec.openAttribute("BitDepth");
                      Codec_DynamicRange = Codec.openAttribute("DynamicRange");
                      Codec_Bias = Codec.openAttribute("Bias");
                  }

                  ScanData = hd5file.openGroup("/ScanData");
                  if (Supports1_x)
                  {
                      SD_FormatVersion = ScanData.openAttribute("FormatVersion");
                  }
                  if (Supports2_2)
                  {
                      AcquisitionXML = ScanData.openDataSet("AcquisitionXML");
                  }
                  auto RunInfo = ScanData.openGroup("RunInfo");
                  RI_InstrumentName = RunInfo.openAttribute("InstrumentName");
                  MovieName = RunInfo.openAttribute("MovieName");
                  if (Supports1_x)
                  {
                      SequencingKit = RunInfo.openAttribute("SequencingKit");
                      BindingKit = RunInfo.openAttribute("BindingKit");
                      Control = RunInfo.openAttribute("Control");
                      IsControlUsed = RunInfo.openAttribute("IsControlUsed");
                  }
                  if (Supports2_2)
                  {
                      MoviePath = RunInfo.openAttribute("MoviePath");
                      SequencingChemistry = RunInfo.openAttribute("SequencingChemistry");
                  }

                  auto AcqParams = ScanData.openGroup("AcqParams");
                  LaserOnFrame = AcqParams.openAttribute("LaserOnFrame");
                  HotStartFrame = AcqParams.openAttribute("HotStartFrame");
                  CameraType = AcqParams.openAttribute("CameraType");
                  CameraGain = AcqParams.openAttribute("CameraGain");
                  AduGain = AcqParams.openAttribute("AduGain");
                  FrameRate = AcqParams.openAttribute("FrameRate");
                  NumFrames = AcqParams.openAttribute("NumFrames");
                  if (Supports1_x)
                  {
                      LaserOnFrameValid = AcqParams.openAttribute("LaserOnFrameValid");
                      HotStartFrameValid = AcqParams.openAttribute("HotStartFrameValid");
                  }
                  if (Supports2_2)
                  {
                      CameraBias = AcqParams.openAttribute("CameraBias");
                      CameraBiasStd = AcqParams.openAttribute("CameraBiasStd");
                  }

                  ChipInfo = ScanData.openGroup("ChipInfo");

                  try
                  {   // to determine if the dataset exists in the group
                      ChipInfo.openDataSet("FilterMap");
                      Supports2_3 = true;
                  }
                  catch (H5::GroupIException)
                  {
                      Supports2_3 = Supports2_4 = Supports2_5 = false;
                  }

                  LayoutName = ChipInfo.openAttribute("LayoutName");
                  if (Supports2_3)
                  {
                      FilterMap = ChipInfo.openDataSet("FilterMap");
                      ImagePsf = ChipInfo.openDataSet("ImagePsf");
                      XtalkCorrection = ChipInfo.openDataSet("XtalkCorrection");
                      AnalogRefSpectrum = ChipInfo.openDataSet("AnalogRefSpectrum");
                      AnalogRefSnr = ChipInfo.openDataSet("AnalogRefSnr");
                  }

                  DyeSet = ScanData.openGroup("DyeSet");
                  NumAnalog = DyeSet.openAttribute("NumAnalog");
                  BaseMap = DyeSet.openAttribute("BaseMap");
                  if (Supports2_3)
                  {
                      AnalogSpectra = DyeSet.openDataSet("AnalogSpectra");
                      RelativeAmp = DyeSet.openDataSet("RelativeAmp");
                      ExcessNoiseCV = DyeSet.openDataSet("ExcessNoiseCV");

                      // PulseWidthMean and IpdMean data sets were added to the
                      // ScanData group for Jaguar (5.1).
                      try
                      {
                          PulseWidthMean = DyeSet.openDataSet("PulseWidthMean");
                          Supports2_4 = true;
                      }
                      catch (H5::GroupIException)
                      {
                          Supports2_4 = false;
                      }
                      if (Supports2_4)
                      {
                          IpdMean = DyeSet.openDataSet("IpdMean");
                      }

                      try
                      {
                          Pw2SlowStepRatio = DyeSet.openDataSet("Pw2SlowStepRatio");
                          Supports2_5 = true;
                      }
                      catch (H5::GroupIException)
                      {
                          Supports2_5 = false;
                      }
                      if (Supports2_5)
                      {
                          Ipd2SlowStepRatio = DyeSet.openDataSet("Ipd2SlowStepRatio");
                      }
                  }

                  if (Supports2_2)
                  {
                      AcquisitionXML = ScanData.openDataSet("AcquisitionXML");
                  }


                  std::vector<uint8_t> traceOrder(3);
                  TraceOrder >> traceOrder;
                  if (traceOrder[0] != 0 || traceOrder[1] != 1 || traceOrder[2] != 2)
                  {
                      throw PBException("Nonstandard TraceOrder attribute not supported. Only supports 0,1,2");
                  }

                  try
                  {
                      auto d = GetDims(Traces);
                      if (d.size() != 3) throw PBException("wrong Traces rank: " + std::to_string(d.size()));

                      NUM_HOLES = static_cast<uint32_t>(d[0]);
                      NUM_CHANNELS = static_cast<uint32_t>(d[1]);
                      NFRAMES = static_cast<uint32_t>(d[2]);
                  }
                  catch(H5::Exception)
                  {
                      PBLOG_WARN << "Tracefile does not have Traces data set";
                      NUM_HOLES = 0;
                      NUM_CHANNELS = 0;
                      NFRAMES =0 ;
                  }

                  NFRAMES_written = NFRAMES;
              }

              try
              {
                  GroundTruth = hd5file.openGroup("/GroundTruth");
                  StateCovariance = GroundTruth.openDataSet("StateCovariance");
                  StateMean = GroundTruth.openDataSet("StateMean");
                  simulated = true;
              }
              catch (H5::Exception)
              {

              }


          }
          catch(H5::Exception& ex)
          {
              H5::Exception::printErrorStack();
              throw ex;
          }
      }

      void SequelTraceFileHDF5::SetGzipCompression(int level)
      {
          if (level < 0 || level >9)
          {
              throw PBException("Invalid HDF5 compression level: " + std::to_string(level));
          }
      }

  void SequelTraceFileHDF5::CalculateVariance()
  {
#ifdef SUPPORT_VARIANCE
      // this "variance" calculation is not useful at all and not the correct calculation to be stored
      // in the "Variance" data field. This code is left here as a template for later improvement.
      LOCK_HDF5();
      {
          try
          {
              double redVariance = redStat->Variance();
              double greenVariance = greenStat->Variance();

              PBLOG_DEBUG << "Red Variance " << redVariance << " n points:" << redStat->n;
              PBLOG_DEBUG << "Green Variance " << greenVariance << " n points:" << redStat->n;

              uint16_t numChannels;
              NumColors >> numChannels;

              if (numChannels != 2)
              {
                  throw  PBException("Only Sequel 2-color supported");
              }
              std::vector<float> variance(NUM_HOLES * numBlocks * numChannels);

              for (uint32_t i = 0; i < NUM_HOLES; i++)
              {
                  variance[i * (numBlocks * numChannels) + GREEN_TRACE_OFFSET] = greenVariance;
                  variance[i * (numBlocks * numChannels) + RED_TRACE_OFFSET] = redVariance;
              }

              Variance << variance;
          }
          catch (const std::exception&)
          {
              PBLOG_ERROR << "Variances not calculated due to no data";
          }
      }
#endif
  }

#if 1
  void SequelTraceFileHDF5::SetSpectra(const AnalogSet& analogs)
  {
      using std::vector;

      LOCK_HDF5();
      uint32_t numAnalogs = analogs.size();
      uint32_t numChannels = analogs[0].numFilters;

      vector<float> spectra(numAnalogs * NUM_HOLES * numChannels);

      PBLOG_INFO << "Copying " << numAnalogs << " analogs over " << numChannels <<
                 " channels to trc.h5 file";


      for (const auto& analog: analogs)
      {
          PBLOG_DEBUG << "SetAnalogs sorted " << analog;
      }


      std::string baseMap = "";
      boost::multi_array<float, 2> analogSpectra(boost::extents[numAnalogs][numChannels]);
      vector<float> relativeAmp(numAnalogs);
      vector<float> excessNoiseCV(numAnalogs);
      vector<float> pulseWidth(numAnalogs);
      vector<float> ipd(numAnalogs);
      vector<float> pw2SlowStepRatio(numAnalogs);
      vector<float> ipd2SlowStepRatio(numAnalogs);
      //std::vector<float> analogRefSpectrum(numChannels);

      for (uint32_t i = 0; i < numAnalogs; i++)
      {
          uint32_t z = i * NUM_HOLES * numChannels;
          const auto& analog = analogs[i];

          baseMap += analog.baseLabel;

          for (uint32_t j = 0; j < NUM_HOLES; j++)
          {
              spectra[z + j * numChannels + GREEN_TRACE_OFFSET] = analog.dyeSpectrum[0];
              if (numChannels == 2)
                    spectra[z + j * numChannels + RED_TRACE_OFFSET] = analog.dyeSpectrum[1];
          }

          for (unsigned int j = 0; j < numChannels; j++)
          {
              analogSpectra[i][j] = analog.dyeSpectrum[j];
          }

          relativeAmp[i] = analog.RelativeAmplitude();
          excessNoiseCV[i] = analog.excessNoiseCV;
          pulseWidth[i] = analog.pulseWidth;
          ipd[i] = analog.interPulseDistance;
          pw2SlowStepRatio[i] = analog.pw2SlowStepRatio;
          ipd2SlowStepRatio[i] = analog.ipd2SlowStepRatio;
      }

      PBLOG_TRACE << "basemap: " << baseMap ;

      Spectra << spectra;
      BaseMap << baseMap;

      AnalogSpectra << analogSpectra;
      RelativeAmp   << relativeAmp;
      ExcessNoiseCV << excessNoiseCV;
      PulseWidthMean << pulseWidth;
      IpdMean << ipd;

      if (Supports2_5)
      {
          Pw2SlowStepRatio << pw2SlowStepRatio;
          Ipd2SlowStepRatio << ipd2SlowStepRatio;
      }
  }
#else
  void SequelTraceFileHDF5::SetSpectra(const std::string& baseMapExt, uint32_t numChannels, uint32_t numAnalogs, const double spectralAngle[], const double amplitude[])
  {

      if (baseMapExt.size() != numAnalogs)
      {
          throw PBException("BaseMap passed to AddSpectra is not consistent, " + baseMapExt + ".size != " + std::to_string(numAnalogs));
      }

      std::vector<float> spectra(numAnalogs * NUM_HOLES * numChannels);


      std::vector<int> sorted(numAnalogs);
      for (uint32_t i = 0; i < numAnalogs; i++)
      {
          sorted[i] = i; // unsorted order!
          PBLOG_DEBUG << "SetSpectra unsorted " << sorted[i] << " " << spectralAngle[sorted[i]] << " " << amplitude[sorted[i]] << " " <<baseMapExt[sorted[i]];
      }

      std::sort<>(sorted.begin(), sorted.end(), [&spectralAngle, &amplitude,&baseMapExt](int a, int b){
          return (spectralAngle[a] < spectralAngle[b]) ||
                  (spectralAngle[a] == spectralAngle[b] && amplitude[a] > amplitude[b]) ||
                  (spectralAngle[a] == spectralAngle[b] && amplitude[a] == amplitude[b] && baseMapExt[a] < baseMapExt[b]);
      });

      for (uint32_t i = 0; i < numAnalogs; i++)
      {
          PBLOG_DEBUG << "SetSpectra sorted " << sorted[i] << " " << spectralAngle[sorted[i]] << " " << amplitude[sorted[i]] << " " <<baseMapExt[sorted[i]];
      }
      std::string baseMap = "";

      if (numChannels != 2)
      {
          throw PBException("Sorry, this version does not support anything but C=2. numChannels was " + std::to_string(numChannels));
      }

      for (uint32_t i = 0; i < numAnalogs; i++)
      {
          uint32_t z = i * NUM_HOLES * numChannels;

          int offset = sorted[i];

          baseMap += baseMapExt[offset];
          auto angle = spectralAngle[offset]; // should be radians
          double ratio = std::tan(angle);
          float green = static_cast<float>(1.0 / (1.0 + ratio));
          float red = 1.0F - green;
          for (uint32_t j = 0; j < NUM_HOLES; j++)
          {
              spectra[z + j * numChannels + GREEN_TRACE_OFFSET] = green;
              spectra[z + j * numChannels + RED_TRACE_OFFSET] = red;
          }
      }

      PBLOG_TRACE << "sorted basemap generated " << baseMap << ", from " << baseMapExt;

      Spectra << spectra;
      BaseMap << baseMap;
  }
#endif

uint32_t SequelTraceFileHDF5::Read1CZmwLaneSegment(uint32_t zmwOffset, uint32_t numZmws, uint64_t frameOffset, uint32_t numFramesIn, int16_t* dest) const
{
    LOCK_HDF5();
    assert(numFramesIn > 0);
    hsize_t numFrames = numFramesIn;
    if (frameOffset + numFrames > NFRAMES)
    {
        numFrames = NFRAMES - frameOffset;
    }
    std::vector<int16_t> pixels(numFramesIn * numZmws * NUM_CHANNELS);

    hsize_t offset[3];
    offset[0] = zmwOffset;
    offset[1] = 0;
    offset[2] = frameOffset;

    hsize_t size[3];
    size[0] = numZmws;
    size[1] = NUM_CHANNELS;
    size[2] = numFrames;

    H5::DataSpace fspace1 = Traces.getSpace();
    fspace1.selectHyperslab(H5S_SELECT_SET, size, offset);

    H5::DataSpace memTile(3, size);
    Traces.read(&pixels[0], H5::PredType::NATIVE_INT16, memTile, fspace1);

    for(uint32_t iframe = 0; iframe < numFrames; iframe++)
    {
        for (uint32_t izmw = 0; izmw < numZmws; izmw++)
        {
            dest[(iframe*numZmws) + izmw] = pixels[(izmw*numFrames) + iframe];
        }
    }

    return numFrames;
}

const std::vector<UnitCell>& SequelTraceFileHDF5::GetUnitCells() const
{
    LOCK_HDF5();
    if (holeXY_.size() == 0)
    {
        static std::vector<int16_t> holexy(NUM_HOLES * 2 /* x + y coordinates */);
        HoleXY >> holexy; // read the whole hole thing
        holeXY_.resize(NUM_HOLES);
        for(uint32_t i=0;i<NUM_HOLES;i++)
        {
            holeXY_[i].x = holexy[i*2+0];
            holeXY_[i].y = holexy[i*2+1];
        }
    }
    return holeXY_;
}

#if 0
  void SequelTraceFileHDF5::SetSequelCell()
  {
      FilterMap          = ChipInfo.createAttribute("FilterMap", float32(),DataSpace(numChannels));
      ImagePsf           = ChipInfo.createDataSet("ImagePsf", float32(),DataSpace(numChannels,5,5));
      XtalkCorrection    = ChipInfo.createDataSet("XtalkCorrection", float32(),DataSpace(7,7));
      AnalogRefSpectrum  = ChipInfo.createDataSet("AnalogRefSpectrum", float32(),DataSpace(numChannels));
      AnalogRefSnr       = ChipInfo.createDataSet("AnalogRefSnr", float32(),scalar);

      std::vector<uint16_t> filterMap({1,0});
      FilterMap << filterMap;

      AnalogSpectra  = DyeSet.createDataSet("AnalogSpectra", float32(),DataSpace(numAnalogs,numChannels));
      RelativeAmp    = DyeSet.createDataSet("RelativeAmp", float32(),DataSpace(numAnalogs));
      ExcessNoiseCV  = DyeSet.createDataSet("ExcessNoiseCV", float32(),DataSpace(numAnalogs));
  }
#endif

}}

