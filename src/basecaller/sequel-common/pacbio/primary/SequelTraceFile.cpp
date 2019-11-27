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

#include <pacbio/primary/EventObject.h>
#include <pacbio/primary/Tranche.h>
#include <pacbio/primary/SequelMovie.h>
#include <pacbio/primary/ChipLayout.h>
#include <pacbio/primary/SequelTraceFile.h>
#include <pacbio/primary/AnalogMode.h>
#include <pacbio/primary/PrimaryConfig.h>


using namespace PacBio::Dev;
using namespace PacBio::Text;

namespace PacBio
{
 namespace Primary
 {

  static const int GREEN_TRACE_OFFSET = 0;
  static const int RED_TRACE_OFFSET = 1;
  static const int numBlocks = 1;


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

#ifdef OBSOLETE_EVENTS
  static H5::EnumType eventEnumType(sizeof(SequelMovieFileBase::EventType_e));
#endif
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

  SequelTraceFileHDF5::SequelTraceFileHDF5(const std::string& fullFileName0, uint64_t nFrames, const SequelROI& roi0,
                                           bool /*fake*/, ChipClass chipClass)
              : SequelMovieFileBase(fullFileName0, roi0, true, chipClass)
              , NFRAMES_written(0)
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
              : SequelTraceFileHDF5(filename, 0, SequelROI::Null() , false, ChipClass::UNKNOWN)
  {
      OpenFileForRead(filename);
  }

  SequelTraceFileHDF5::SequelTraceFileHDF5(const SequelMovieConfig& config)
              : SequelTraceFileHDF5(config.path, config.numFrames, *config.roi, false, config.chipClass)
  {
      spectraUnset_ = true; // reminder to call SetBaseMap soon!
      config_ = config;
      CreateFileForWrite(config);
      NumFrames << config.numFrames;
  }

      SequelTraceFileHDF5::~SequelTraceFileHDF5()
      {
      // std::cout << "SequelTraceFileHDF5::~SequelTraceFileHDF5()" << std::endl;
          try
          {
              FinalizeSaverThread();

              if (spectraUnset_)
              {
                  PBLOG_WARN << "SequelTraceFileHDF5: Basemap was not set for trace file, file will not be useable";
              }

              LOCK_HDF5();
              if (workingFile_.IsOpenForWrite())
              {
                  CalculateVariance();
              }

              if (NFRAMES_written != NFRAMES)
              {
                  PBLOG_WARN << "SequelTraceFileHDF5: written frames " << NFRAMES_written << " not equal to expected frames " <<
                                                                                 NFRAMES;
              }

              // For 5.1.0, we are defining the /ScanData/AcqParams/FrameRate
              // attribute as the frame rate _setting_ (i.e., the rate requested by
              // the system configuration). This is consistent with RS software.
              // It also enables us to ensure that the same value is provided to
              // the basecaller whether it's run on instrument of re-analyzing a
              // trace file.
    #if 0
              if (frameRateHistory.size() > 0)
              {
                  std::sort(frameRateHistory.begin(), frameRateHistory.end());
                  int offset = static_cast<int>(frameRateHistory.size() / 4);
                  auto quartile1 = frameRateHistory.begin() + offset;
                  auto quartile3 = frameRateHistory.end() - offset;
                  auto count = std::distance(quartile1, quartile3);
                  if (count > 0)
                  {
                      double sum = std::accumulate(quartile1, quartile3, 0.0);
                      double mean = sum / count;
                      PBLOG_DEBUG << "SequelTraceFileHDF5: Frame Rate at closing was " << mean;
                      FrameRate << mean;
                  }
              }
    #endif

              hd5file.close();

              CleanupWorkingFilename();
          }
          catch(const std::exception& ex)
          {
              PBLOG_ERROR << "SequelTraceFileHDF5::~SequelTraceFileHDF5() caught exception: " << ex.what();
          }
          catch(...)
          {
              std::cerr << "Uncaught exception caught in ~SequelTraceFileHDF5 " << PacBio::GetBackTrace(5);
              PBLOG_FATAL << "Uncaught exception caught in ~SequelTraceFileHDF5 " << PacBio::GetBackTrace(5);
              PacBio::Logging::PBLogger::Flush();
              std::terminate();
          }
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
                      config_.trace.chunking.zmw = chunk_dims[0];
                      config_.trace.chunking.channel = chunk_dims[1];
                      config_.trace.chunking.frame = chunk_dims[2];
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
                  RI_PlatformId = RunInfo.openAttribute("PlatformId");
                  RI_PlatformName = RunInfo.openAttribute("PlatformName");
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
              events.OpenForRead(hd5file);

              try
              {
                  auto& chipLayout = this->GetChipLayout();
                  PBLOG_DEBUG << "ChipLayout:" << chipLayout.Name();
                  chipClass_ = chipLayout.GetChipClass();
              }
              catch(const std::exception& ex)
              {
                  PBLOG_WARN << ex.what();
                  PBLOG_WARN << "ChipLayout is not supported, will mark it as UNKNOWN";
                  chipClass_ = ChipClass::UNKNOWN;
              }
              valid = true;

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

      void SequelTraceFileHDF5::CreateFileForWrite(const SequelMovieConfig& config)
      {
          roi_->CheckROI();
          SetupWorkingFilename(config.path);

          SetChunking(config.trace.chunking);
          SetGzipCompression(config.trace.compression);

          {
              LOCK_HDF5();
              hd5file = H5::H5File(workingFile_, H5F_ACC_TRUNC);

              NDARKFRAMES = 0;
              NFRAMES_written = 0;
              NUM_HOLES = roi_->CountHoles();


              int numChipLooks = 1;

              // onenote:http://sharepoint/progmgmt/Sequel/OneNote/Sequel/Cell/PoC%20Team/Straw%20Man.one#Converting%20movie%20to%20trace&section-id={02CCE69A-A1BF-4091-8329-6FA5AA15B631}&page-id={30F59BC4-65BE-4631-BA81-342944BCA155}&end

              // objects
              TraceData = hd5file.createGroup("/TraceData");
              Representation = TraceData.createAttribute("Representation", SeqH5string(), scalar);
              TraceOrder = TraceData.createAttribute("TraceOrder", uint8(), DataSpace(3));
              Look = TraceData.createAttribute("Look", uint8(), scalar);
              Version = TraceData.createAttribute("Version", SeqH5string(), scalar);
              FormatVersion = TraceData.createAttribute("FormatVersion", SeqH5string(), scalar);
              SoftwareVersion = TraceData.createAttribute("SoftwareVersion", SeqH5string(), scalar);
              ChangeListID = TraceData.createAttribute("ChangeListID", SeqH5string(), scalar);
              DateCreated = TraceData.createAttribute("DateCreated", SeqH5string(), scalar);
              NumColors = TraceData.createAttribute("NumColors", uint16(), scalar);


              HoleNumber = TraceData.createDataSet("HoleNumber", uint32(), DataSpace(NUM_HOLES));
              HoleXY = TraceData.createDataSet("HoleXY", int16(), DataSpace(NUM_HOLES, 2));
              HoleChipLook = TraceData.createDataSet("HoleChipLook", int16(), DataSpace(NUM_HOLES));
              ActiveChipLooks = TraceData.createDataSet("ActiveChipLooks", int16(), DataSpace(numChipLooks));
              HoleXYPlot = TraceData.createDataSet("HoleXYPlot", float32(), DataSpace(NUM_HOLES, 2));
              //  AntiHoles      = TraceData.createDataSet      ("AntiHoles"               , int16()     , DataSpace(?));
              //Traces         = TraceData.createDataSet      ("Traces"                  , uint8()     , DataSpace(NUM_HOLES,numAnalogs,NFRAMES));
              //Traces         = TraceData.createDataSet      ("Traces"                  , uint8()     , DataSpace(NUM_HOLES,numAnalogs,NFRAMES));
              HoleStatus = TraceData.createDataSet("HoleStatus", uint8(), DataSpace(NUM_HOLES));
              HoleType = TraceData.createDataSet("HoleType", uint8(), DataSpace(NUM_HOLES));

              Codec = TraceData.createGroup("Codec");
              Codec_Name = Codec.createAttribute("Name", SeqH5string(), scalar);
              Codec_Config = Codec.createAttribute("Config", SeqH5string(), DataSpace(3));
              Codec_Units = Codec.createAttribute("Units", SeqH5string(), scalar);
              Codec_BitDepth = Codec.createAttribute("BitDepth", uint8(), scalar);
              Codec_DynamicRange = Codec.createAttribute("DynamicRange", float32(), scalar);
              Codec_Bias = Codec.createAttribute("Bias", float32(), scalar);

              ScanData = hd5file.createGroup("/ScanData");
              SD_FormatVersion = ScanData.createAttribute("FormatVersion", SeqH5string(), scalar);
              AcquisitionXML = ScanData.createDataSet("AcquisitionXML", SeqH5string(), scalar);
              auto RunInfo = ScanData.createGroup("RunInfo");
              RI_PlatformId = RunInfo.createAttribute("PlatformId", uint32(), scalar);
              RI_PlatformName = RunInfo.createAttribute("PlatformName", SeqH5string(), scalar);
              RI_InstrumentName = RunInfo.createAttribute("InstrumentName", SeqH5string(), scalar);
              MovieName = RunInfo.createAttribute("MovieName", SeqH5string(), scalar);
              MoviePath = RunInfo.createAttribute("MoviePath", SeqH5string(), scalar);
              SequencingChemistry = RunInfo.createAttribute("SequencingChemistry", SeqH5string(), scalar);
              SequencingKit = RunInfo.createAttribute("SequencingKit", SeqH5string(), scalar);
              BindingKit = RunInfo.createAttribute("BindingKit", SeqH5string(), scalar);
              Control = RunInfo.createAttribute("Control", SeqH5string(), scalar);
              IsControlUsed = RunInfo.createAttribute("IsControlUsed", uint8(), scalar);
              auto AcqParams = ScanData.createGroup("AcqParams");
              LaserOnFrame = AcqParams.createAttribute("LaserOnFrame", int32(), scalar);
              LaserOnFrameValid = AcqParams.createAttribute("LaserOnFrameValid", uint8(), scalar);
              HotStartFrame = AcqParams.createAttribute("HotStartFrame", int32(), scalar);
              HotStartFrameValid = AcqParams.createAttribute("HotStartFrameValid", uint8(), scalar);
              CameraType = AcqParams.createAttribute("CameraType", int32(), scalar);
              CameraGain = AcqParams.createAttribute("CameraGain", float32(), scalar);
              AduGain = AcqParams.createAttribute("AduGain", float32(), scalar);
              CameraBias = AcqParams.createAttribute("CameraBias", float32(), scalar);
              CameraBiasStd = AcqParams.createAttribute("CameraBiasStd", float32(), scalar);
              FrameRate = AcqParams.createAttribute("FrameRate", float32(), scalar);
              NumFrames = AcqParams.createAttribute("NumFrames", uint32(), scalar);
              ChipInfo = ScanData.createGroup("ChipInfo");
              LayoutName = ChipInfo.createAttribute("LayoutName", SeqH5string(), scalar);
              DyeSet = ScanData.createGroup("DyeSet");
              NumAnalog = DyeSet.createAttribute("NumAnalog", uint16(), scalar);
              BaseMap = DyeSet.createAttribute("BaseMap", SeqH5string(), scalar);
              std::vector<int16_t> activeChipLooks(numChipLooks);
              activeChipLooks[0] = 0;

              ActiveChipLooks << activeChipLooks;


              // write 'em
              Representation << "Camera";
              std::vector<uint8_t> traceOrder{0, 1, 2};
              TraceOrder << traceOrder;
              Look << (uint8_t) 0;
              Version << "SequelPOC MovieToTrace 2.3.0";
              FormatVersion << "2.3";
              SoftwareVersion << "1.0.0.0";
              ChangeListID << PacBio::Utilities::PerforceUtil::ChangeNumberAsString();
              DateCreated << PacBio::Utilities::ISO8601::TimeString();
              SD_FormatVersion << "2.3.0";


              RI_PlatformId << 0;
              RI_PlatformName << "unknown";

              SequencingChemistry << "unknown";
              SequencingKit << "NA";
              BindingKit << "NA";
              Control << "NA";
              RI_InstrumentName << PacBio::POSIX::gethostname();
              IsControlUsed << 0;
              MoviePath << workingFile_.FinalFilename();
              boost::filesystem::path p(workingFile_.FinalFilename());
              MovieName << p.filename().stem().stem().string();

              LaserOnFrame << 0;
              LaserOnFrameValid << (uint8_t) 0;
              HotStartFrame << 0;
              HotStartFrameValid << (uint8_t) 0;
              CameraType << 200;
              CameraGain << 1.0;


              Codec_Name << "FixedPoint";
              std::vector<std::string> configNames{"BitDepth", "DynamicRange", "Bias"};
              Codec_Config << configNames;
              Codec_Units << "Counts";
              //FixedPoint (e.g., 16-bit) uses the following very simple algorithm to compress:
              // CompressedTrace =  (uint16) round( max (min ( ( trace - bias) * cR, 65535 ), 0 ) ),
              // where cR is compressionRatio, defined as cR = (2^BitDepth - 1) / (dR - Bias), and
              // dR is the DynamicRange parameter.
              Codec_BitDepth << 16;
              Codec_DynamicRange << 65535.0;
              Codec_Bias << 0.0;
              // dR = 65535
              // cR = 65535/65535 = 1.0 (unity)


              AduGain << 1.0; // photoelectronSensitivity
              CameraGain << 1.0;

#if 0
              for (int i = 0; i < numAnalogs; i++)
              {
                  auto Analogx = DyeSet.createGroup(String::Format("Analog[%d]", i));
                  FrameRate = AcqParams.createAttribute("FrameRate", float32(), scalar);
                  auto Base = Analogx.createAttribute("Base", SeqH5string(), scalar);
                  Base << baseMap.substr(i, 1); // take  one character
                  auto Label = Analogx.createAttribute("Label", SeqH5string(), scalar);
                  Label << String::Format("Analog-%d-label", i);
                  auto Nucleotide = Analogx.createAttribute("Nucleotide", SeqH5string(), scalar);
                  Nucleotide << String::Format("dTxP%d", i);
                  auto Wavelength = Analogx.createAttribute("Wavelength", float32(), scalar);
                  Wavelength << 600.0; // fix me
              }
#endif
          }

          events.CreateForWrite(hd5file,GetPrimaryConfig().GetLaserNames());

          valid = true;
      }

      void SequelTraceFileHDF5::SetChunking(const SequelMovieConfig::TraceChunking& chunking)
      {
          config_.trace.chunking = chunking;
          config_.trace.chunking.channel = NUM_CHANNELS; // ignore the given channel chunking
      }

      SequelMovieConfig::TraceChunking SequelTraceFileHDF5::GetChunking() const
      {
          return config_.trace.chunking;
      }

      void SequelTraceFileHDF5::SetGzipCompression(int level)
      {
          if (level < 0 || level >9)
          {
              throw PBException("Invalid HDF5 compression level: " + std::to_string(level));
          }
          config_.trace.compression = level;
      }

// useful for Sequel, may not be useful for anything but 2C, where green is the complement of red...
   void SequelTraceFileHDF5::SetAnalogs(const AnalogSet& analogs, const ChipLayout& layout)
   {

      LOCK_HDF5();

      int numAnalogs = analogs.size();
      if (numAnalogs <=0)
      {
          throw PBException("No analogs passed to SequelTraceFileHDF5::SetAnalogs. "
                            "At least 1, preferably 4, must be passed");
      }
      uint32_t numChannels = boost::numeric_cast<uint32_t>(analogs.front().numFilters);

      NUM_CHANNELS = numChannels;

      NumAnalog << numAnalogs;
      NumColors << numChannels;

      Spectra = TraceData.createDataSet("Spectra", float32(), DataSpace(numAnalogs, NUM_HOLES, numChannels));
      std::vector<float> spectra(numAnalogs * NUM_HOLES * numChannels); // don't care if it is uninitialized. actually will fill it with noise which is not bad!

      if (NUM_HOLES >= config_.trace.chunking.zmw && NFRAMES >= config_.trace.chunking.frame)
      {
          config_.trace.chunking.channel = NUM_CHANNELS;
           hsize_t chunk_dims[3] = { config_.trace.chunking.zmw , config_.trace.chunking.channel, config_.trace.chunking.frame};
           tracesPropList.setChunk(3, chunk_dims);
           if ( config_.trace.compression >9)
           {
               throw PBException("Invalid HDF5 compression level: " + std::to_string(config_.trace.compression));
           }
           tracesPropList.setDeflate(config_.trace.compression);
      }
      else
      {
          // don't both with chunking
      }
      int fill_val = 0;
      tracesPropList.setFillValue(H5::PredType::NATIVE_INT, &fill_val);

      Traces = TraceData.createDataSet("Traces", int16(), DataSpace(NUM_HOLES, numChannels, NFRAMES), tracesPropList);
      static_assert(SizeOfPixel == sizeof(int16_t),"size consistency check");


      Variance = TraceData.createDataSet("Variance", float32(), DataSpace(NUM_HOLES, numBlocks, numChannels));
      VFrameSet = TraceData.createDataSet("VFrameSet", uint32(), DataSpace(numBlocks));
      HolePhase = TraceData.createDataSet("HolePhase", float32(), DataSpace(NUM_HOLES, numBlocks, numChannels));
      HPFrameSet = TraceData.createDataSet("HPFrameSet", uint32(), DataSpace(numBlocks));
      ReadVariance = TraceData.createDataSet("ReadVariance", float32(), DataSpace(NUM_HOLES, numBlocks, numChannels));
      RVFrameSet = TraceData.createDataSet("RVFrameSet", uint32(), DataSpace(numBlocks));
      IFVarianceScale = TraceData.createDataSet("IFVarianceScale", float32(),
                                                DataSpace(NUM_HOLES, numBlocks, numChannels));
      OFBackgroundMean = TraceData.createDataSet("OFBackgroundMean", float32(),
                                                 DataSpace(NUM_HOLES, numBlocks, numChannels));
      OFBackgroundVariance = TraceData.createDataSet("OFBackgroundVariance", float32(),
                                                     DataSpace(NUM_HOLES, numBlocks, numChannels));

      Supports2_3 = true; // always support it when creating it
      if (Supports2_3)
      {
          FilterMap          = ChipInfo.createDataSet("FilterMap",         uint16(), DataSpace(numChannels));
          ImagePsf           = ChipInfo.createDataSet("ImagePsf",          float32(),DataSpace(numChannels,5,5));
          XtalkCorrection    = ChipInfo.createDataSet("XtalkCorrection",   float32(),DataSpace(7,7));
          AnalogRefSpectrum  = ChipInfo.createDataSet("AnalogRefSpectrum", float32(),DataSpace(numChannels));
          AnalogRefSnr       = ChipInfo.createDataSet("AnalogRefSnr",      float32(),scalar);

          if (numChannels == 2)
          {
              std::vector<uint16_t> filterMap({1, 0});
              FilterMap << filterMap;
          }
          else if (numChannels == 1)
          {
              std::vector<uint16_t> filterMap({1});
              FilterMap << filterMap; // is this useful?
          }
          else
          {
              throw PBException("not supported");
          }
          AnalogSpectra  = DyeSet.createDataSet("AnalogSpectra",           float32(),DataSpace(numAnalogs,numChannels));
          RelativeAmp    = DyeSet.createDataSet("RelativeAmp",             float32(),DataSpace(numAnalogs));
          ExcessNoiseCV  = DyeSet.createDataSet("ExcessNoiseCV",           float32(),DataSpace(numAnalogs));
      }

      Supports2_4 = true;
      if (Supports2_4)
      {
          PulseWidthMean = DyeSet.createDataSet("PulseWidthMean", float32(), DataSpace(numAnalogs));
          IpdMean = DyeSet.createDataSet("IpdMean", float32(), DataSpace(numAnalogs));
      }
      Supports2_5 = true;
      if (Supports2_5)
      {
          Pw2SlowStepRatio  = DyeSet.createDataSet("Pw2SlowStepRatio" , float32(), DataSpace(numAnalogs));
          Ipd2SlowStepRatio = DyeSet.createDataSet("Ipd2SlowStepRatio", float32(), DataSpace(numAnalogs));
      }

      std::vector<int16_t> holeChipLook(NUM_HOLES);
      std::vector<uint32_t> aZero(numBlocks);
      aZero[0] = 0;
      std::vector<float> zeros(NUM_HOLES * numBlocks * numChannels);
      std::vector<float> ones(NUM_HOLES * numBlocks * numChannels);

      // fill in default values.
      for(SequelROI::Enumerator e(*roi_,0,NUM_HOLES);e;e++)
      {
          uint32_t i = e.Index();
          holeChipLook[i] = 0;
          for(uint16_t color =0; color<numChannels;color++)
          {
              zeros[i * (numBlocks * numChannels) + color] = 0;
              ones[i * (numBlocks * numChannels) + color] = 1.0;
          }
      }


      HoleChipLook    << holeChipLook;
      HolePhase       << zeros;
      IFVarianceScale << ones;
      Variance        << ones;
      OFBackgroundMean<< zeros;
      OFBackgroundVariance<<zeros;

      Spectra    << spectra;
      VFrameSet  << aZero;
      HPFrameSet << aZero;
      RVFrameSet << aZero;

      UpdateLayout(layout);

      SetSpectra(analogs);
      spectraUnset_ = false;
  }

  void SequelTraceFileHDF5::UpdateLayout(const ChipLayout& layout)
  {

      LOCK_HDF5();

      const int numSpatialDimensions = 2; // x,y
      std::vector<uint32_t> holes(NUM_HOLES);
      std::vector<int16_t>  holexy(NUM_HOLES * numSpatialDimensions);
      std::vector<float>    holeXYPlot(NUM_HOLES * numSpatialDimensions);
      std::vector<uint8_t>  holeStatus(NUM_HOLES);
      std::vector<uint8_t>  holeType(NUM_HOLES);


      LayoutName << layout.Name();


      uint32_t j = 0;
      for(SequelROI::Enumerator e(*roi_,0,NUM_HOLES);e;e++,j+=2)
      {
          uint32_t i = e.Index();
          std::pair<RowPixels, ColPixels> pix = e.UnitCellPixelOrigin();

          UnitCell hole(layout.ConvertAbsoluteRowPixelToUnitCellX(pix.first),
                        layout.ConvertAbsoluteColPixelToUnitCellY(pix.second));

          holes[i] = (((uint32_t) hole.x) << 16) | hole.y;
          holeXYPlot[j + 0] = holexy[j + 0] = hole.x;
          holeXYPlot[j + 1] = holexy[j + 1] = hole.y;

          auto features = layout.GetUnitCellFeatures(hole);
          const uint8_t normalZmw = 0;
          const uint8_t notZmw = 1;
          holeStatus[i] = ChipLayout::IsSequencing(features) ? normalZmw : notZmw;
          holeType[i]   = static_cast<uint8_t>(layout.GetUnitCellIntType(hole));
      }


      HoleNumber << holes;
      HoleXY     << holexy;
      HoleXYPlot << holeXYPlot;
      HoleStatus << holeStatus;
      HoleType   << holeType;
  }

  uint32_t SequelTraceFileHDF5::WriteChunkHeader(const PacBio::Primary::Tile* chunkHeader, uint32_t numFrames)
  {
      // the trace file does nothing with the chunk header, so it can be ignored.
      (void) chunkHeader;
      return numFrames;
  }

  void SequelTraceFileHDF5::AddTile(const PacBio::Primary::Tile* tile, uint32_t tileOffset)
  {
      uint32_t internalOffset;
      if (roi_->ContainsTileOffset(tileOffset,&internalOffset))
      {
          if (NUM_CHANNELS == 1)
          {
              movieBuffer.AddTileAs1CTraces(tile, tileOffset, internalOffset);
          }
          else if (NUM_CHANNELS == 2)
          {
              movieBuffer.AddTileAs2CTraces(tile, tileOffset, internalOffset);
          }
          else
          {
              throw PBException("Not supported");
          }
      }
  }


  // this is called through FlushChunk from SequelMovieBase.
  void SequelTraceFileHDF5::WriteChunkBuffer(const ChunkBuffer* bigBuffer, const uint32_t /*numFrames*/)
  {
      if (NUM_CHANNELS == 0) throw PBException("NUM_CHANNELS was not set before calling WriteChunkBuffer");

      uint64_t framesToWriteThisChunk = NFRAMES - NFRAMES_written;
      if (framesToWriteThisChunk > Tile::NumFrames)
      {
          framesToWriteThisChunk = Tile::NumFrames;
      }
      // AutoTimer _("bigbuffer",512*sizeof(uint16_t)*NUM_HOLES*2,"bytes");

      //std::cout << "WriteChunkBuffer() START bigBuffer:" << (void*)bigBuffer << " numFrame:" << numFrames << std::endl;
      LOCK_HDF5();
      {
          const size_t numTiles = movieBuffer.NumTiles();

          // advance over the entire ROI.
          const int numUnitCellsPerTile = Tile::NumPixels / NUM_CHANNELS;
          assert(numUnitCellsPerTile == 16 || numUnitCellsPerTile == 32);
          for(SequelROI::Enumerator e(*roi_,0,NUM_HOLES);e;e+=numUnitCellsPerTile)
          {
              uint32_t zmw = e.Index();
              std::pair<RowPixels, ColPixels> pix = e.UnitCellPixelOrigin();
              uint32_t row = pix.first.Value();
              uint32_t col = pix.second.Value();

              uint32_t rowOffset = row - roi_->SensorROI().PhysicalRowOffset();
              uint32_t colOffset = col - roi_->SensorROI().PhysicalColOffset();

              auto tileOffset = (rowOffset * roi_->SensorROI().PhysicalCols() + colOffset) / Tile::NumPixels;
              uint32_t internalOffset;
              if ( !roi_->ContainsTileOffset(tileOffset,&internalOffset))
              {
                  throw PBException("internal inconsistency. tileOffset should be in ROI.");
              }
              if (internalOffset >= numTiles)
              {
                  std::stringstream ss;
                  ss << "tileOffset more than numtiles. tileOffset:" << internalOffset << ", numTiles:" << numTiles << "\n";
                  ss << "rowOffset:" << rowOffset << ", colOffset:" << colOffset << "\n";
                  ss << (*roi_);
                  throw PBException(ss.str());
              }

              const uint8_t* src = bigBuffer->Data(internalOffset);

              hsize_t dstOffset[3];
              dstOffset[0] = zmw;
              dstOffset[1] = 0;
              dstOffset[2] = NFRAMES_written;

              hsize_t dstSize[3];
              dstSize[0] = numUnitCellsPerTile;
              dstSize[1] = NUM_CHANNELS;
              dstSize[2] = framesToWriteThisChunk;

              H5::DataSpace fspace1 = Traces.getSpace();
              fspace1.selectHyperslab(H5S_SELECT_SET, dstSize, dstOffset);

              if (framesToWriteThisChunk == Tile::NumFrames)
              {
                  // std::cout << "zmw:" << zmw << " dst offset:" << offset[0] << "," << offset[1] << "," << offset[2] << " from src:" << (void*)src << std::endl;

                  H5::DataSpace memTile(3, dstSize);
                  Traces.write(src, H5::PredType::NATIVE_INT16, memTile, fspace1);
              } else {
                  hsize_t srcSize[3];
                  srcSize[0] = numUnitCellsPerTile;
                  srcSize[1] = NUM_CHANNELS;
                  srcSize[2] = Tile::NumFrames;

                  // std::cout << "zmw:" << zmw << " dst offset:" << offset[0] << "," << offset[1] << "," << offset[2] << " from src:" << (void*)src << std::endl;

                  H5::DataSpace memTile(3, srcSize);
                  hsize_t srcOffset[3];
                  srcOffset[0] = 0;
                  srcOffset[1] = 0;
                  srcOffset[2] = 0;
                  memTile.selectHyperslab(H5S_SELECT_SET, dstSize, srcOffset);

                  Traces.write(src, H5::PredType::NATIVE_INT16, memTile, fspace1);

              }
#if 0
              {
                  std::cout << "R" <<row<<"C"<<col<< " " << std::hex << (int) src[0]<< " " << (int) src[1]<< std::dec << std::endl;

                  if (row == 63 && col == 96)
                  {
                      std::cout << std::hex;
                      for(int i=0;i<numUnitCellsPerTile * 2 * Tile::NumFrames*sizeof(uint16_t);i++)
                      {
                          std::cout << " " << (int) src[i];
                      }
                      std::cout << std::dec;
                      std::cout << std::endl;
                  }
              }
#endif

#ifdef SUPPORT_VARIANCE
              uint32_t index = row * 2048 + col;
              auto x = findAntiZmw.find(index);
              if (x != findAntiZmw.end())
              {

                  PBLOG_TRACE << "flushing ANTIZMW:" << index << " " << std::hex << x->first << " " <<
                              (int) x->second << std::dec;

                  const uint16_t* pixels = reinterpret_cast<const uint16_t*>(src) + x->second * Tile::NumFrames;

                  for (uint32_t i = 0; i < framesToWriteThisChunk; i++)
                  {
                      // the pixels have already been sorted so that green comes before red
                      int32_t g = pixels[i + GREEN_TRACE_OFFSET * Tile::NumFrames];
                      int32_t r = pixels[i + RED_TRACE_OFFSET * Tile::NumFrames];
                      // PBLOG_TRACE << " adding r,g to variance calc " << g << " " << r;
                      redStat->Add(r);
                      greenStat->Add(g);
                  }
              }
#endif
          }

          NFRAMES_written += framesToWriteThisChunk;
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

  uint64_t SequelTraceFileHDF5::AddFrame(const SequelMovieFrame<int16_t>& /*frame*/)
  {
      throw PBException("Not implemented");

  }

  uint64_t SequelTraceFileHDF5::AddDarkFrame(float /*exposure*/, const SequelMovieFrame<int16_t>& /*frame*/)
  {
      throw PBException("Not implemented");
      // ReadVariance
      // OFBackgroundMean
      //OFBackgroundVariance
      // Variance = ReadVariance + OFBackgroundVariance

      return 0;
  }

#ifdef OBSOLETE_EVENTS
  void SequelTraceFileHDF5::AddEvent(SequelMovieFileBase::EventType_e type, uint64_t timestamp)
  {
      LOCK_HDF5();

      switch (type)
      {
      case EventType_e::HotStart:
          HotStartFrame << timestamp;
          HotStartFrameValid << 1;
          break;
      case EventType_e::LaserOff:
          PBLOG_WARN << "LaserOff event not stored";
          break;
      case EventType_e::LaserOn:
          LaserOnFrame << timestamp;
          LaserOnFrameValid << 1;
          break;
      default:
          PBLOG_WARN << "Event ignored, event type number:" << type;
      }

      // this could be handled by keeping the last 2 frames in memory
      // and interpolating the "frame" of the event using those 2 frames
  }
#endif

  void SequelTraceFileHDF5::SetMeasuredFrameRate(double rate)
  {
      frameRateHistory.push_back(rate);
  }

  void SequelTraceFileHDF5::ReadDarkFrameStack(uint64_t /*n*/, SequelMovieFrame<int16_t>& /*frame*/, float& /*exposure*/)
  {
      throw PBException("Not implemented");
  }

  void SequelTraceFileHDF5::CreateTest(uint32_t /*nframes*/, uint32_t /*nDarkFrames*/, int /*bits*/)
  {
      // initialize data
      throw PBException("Not implemented");

#if 0
        uint16_t mask = (1<< bits) -1;

        SequelMovieFrame<int16_t> frame(NROW,NCOL);
        for(uint32_t n=0;n<nframes;n++)
        {
            for(uint32_t i=0;i<NROW;i++)
                for(uint32_t j=0;j<NCOL;j++)
                    frame.data[i*NCOL + j] = ((100*n + i*10 + j) & mask);

            frame.timestamp = static_cast<uint64_t>(n) * 5000000ULL;

            this->AddFrame(frame);
        }


        for(uint32_t n=0;n<nDarkFrames;n++){
            for(uint32_t i=0;i<NROW;i++)
                for(uint32_t j=0;j<NCOL;j++)
                    frame.data[i*NCOL + j] = (n & mask);
            this->AddDarkFrame(static_cast<float>(pow(0.1,n)),frame);
        }
#endif
  }

  void SequelTraceFileHDF5::DumpSummary(std::ostream& s, uint32_t /*frames*/, uint32_t /*rows*/, uint32_t /*cols*/)
      {
          LOCK_HDF5();
          auto p = s.precision();
          s.precision(100);
          s << FormatVersion;
          s << Representation;
          s << TraceOrder;
          s << Look;
          s << Version;
          s << SoftwareVersion;
          s << ChangeListID;
          s << DateCreated;


          s << SD_FormatVersion;
          s << RI_PlatformId   ;
          s << RI_PlatformName ;

          s << SequencingChemistry;
          s << SequencingKit ;
          s << BindingKit        ;
          s << Control          ;
          s << RI_InstrumentName;
          s << IsControlUsed;
          s << MovieName;

          s << LaserOnFrame;
          s << LaserOnFrameValid;
          s << HotStartFrame;
          s << HotStartFrameValid;
          s << CameraType;
          s << CameraGain;
          s << AduGain;
          s << LayoutName;
          s << NumAnalog;
          s << BaseMap;

          uint32_t numAnalogs ;
          NumAnalog >> numAnalogs;

#if 0
        for(int i=0;i<numAnalogs;i++)
        {
            auto Analogx = DyeSet.createDataset(String::Format("Analog[%d]",i );
            Analogx.createAttribute("Base",string);
        }
#endif
        s.precision(p);
  }

Json::Value SequelTraceFileHDF5::DumpJson(uint32_t /*frames*/, uint32_t /*rows*/, uint32_t /*cols*/)
{
    LOCK_HDF5();
    Json::Value jv;

    jv << FormatVersion
     << Representation;

    jv
     << TraceOrder
     << Look
     << Version
     << SoftwareVersion
     << ChangeListID
     << DateCreated

     << SD_FormatVersion
     << RI_PlatformId
     << RI_PlatformName

     << SequencingChemistry
     << SequencingKit
     << BindingKit
     << Control
     << RI_InstrumentName
     << IsControlUsed
     << MovieName

     << LaserOnFrame
     << LaserOnFrameValid
     << HotStartFrame
     << HotStartFrameValid
     << CameraType
     << CameraGain
     << AduGain
     << LayoutName
     << NumAnalog
     << BaseMap;

    // Reference SNR and spectrum
    float analogRefSnr;
    AnalogRefSnr >> analogRefSnr;
    jv["refDwsSnr"] = analogRefSnr;

    std::vector<float> analogRefSpectrum;
    AnalogRefSpectrum >> analogRefSpectrum;
    Json::Value refSpectrum(Json::arrayValue);
    for (const auto& r : analogRefSpectrum) refSpectrum.append(r);
    jv["refSpectrum"] = refSpectrum;

    // Analog metadata
    boost::multi_array<float, 2> analogSpectra;
    std::vector<float> relativeAmpl;
    std::vector<float> excessNoiseCV;
    std::vector<float> ipdMean;
    std::vector<float> pwMean;

    AnalogSpectra >> analogSpectra;
    RelativeAmp >> relativeAmpl;
    ExcessNoiseCV >> excessNoiseCV;

    if (Supports2_4)
    {
        IpdMean >> ipdMean;
        PulseWidthMean >> pwMean;
    }
    else
    {
        ipdMean = { { 13.0f, 18.0f, 14.0f, 14.0f } };   // frames
        pwMean  = { { 10.0f,  7.0f,  9.0f,  9.0f } };   // frames

        float frameRate;
        FrameRate >> frameRate;

        // Convert to seconds.
        for (unsigned int i = 0; i < ipdMean.size(); ++i)
        {
            ipdMean[i] /= frameRate;
            pwMean[i]  /= frameRate;
        }
    }

    uint16_t numAnalogs;
    std::string baseMap;
    NumAnalog >> numAnalogs;
    BaseMap >> baseMap;

    Json::Value analogs(Json::arrayValue);
    for (int i = 0; i < numAnalogs; i++)
    {
        Json::Value analog;
        analog["base"] = Json::Value(std::string(1, baseMap[i]));
        Json::Value analogSpectrum(Json::arrayValue);
        for (const auto& a : analogSpectra[i]) analogSpectrum.append(a);
        analog["spectrumValues"] = analogSpectrum;
        analog["intraPulseXsnCV"] = excessNoiseCV[i]; // fixme
        analog["interPulseXsnCV"] = excessNoiseCV[i]; // fixme
        analog["relativeAmplitude"] = relativeAmpl[i];
        analog["ipdMeanSeconds"] = ipdMean[i];
        analog["pulseWidthMeanSeconds"] = pwMean[i];
        analogs.append(analog);
    }
    jv["analogs"] = analogs;

    double photoelectronSensitivity;
    AduGain >> photoelectronSensitivity;
    jv["photoelectronSensitivity"] = photoelectronSensitivity;

#if 0
                    // fixme todo
                    //traceFile->
                    //control.chipId =
                    //instrumentName
#endif
     return jv;
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


  void SequelTraceFileHDF5::DetermineAntiZMWs(const SequelROI& /*roi*/, const ChipLayout& /*layout*/)
  {
#ifdef SUPPORT_VARIANCE
      antiZMWs = layout.GetUnitCellListByPredicate([](const UnitCell&uc){ return ChipLayout::IsAntiZMW(uc); }, roi);
      PBLOG_INFO << "AntiZMWs determined, count:" << antiZMWs.size();
      if (antiZMWs.size() > 0)
      {
          PBLOG_INFO << " First AntiZMW: " << antiZMWs[0].ExcelCell();
      }
      findAntiZmw.clear();
      redStat->clear();
      greenStat->clear();
      for(auto& cell : antiZMWs)
      {
          uint32_t pixelCol = layout.ConvertUnitCellToColPixels(cell).Value();
          uint32_t pixelRow = layout.ConvertUnitCellToRowPixels(cell).Value();
          uint32_t pixelColHigh = pixelCol & ~0x1F;
          uint8_t  pixelColLow = static_cast<uint8_t>(pixelCol &  0x1F);
          uint32_t index = pixelRow * 2048 + pixelColHigh;

          PBLOG_DEBUG << " AntiZMW at " << cell.ExcelCell() << "/unit(" << cell.x << "," << cell.y << ") pixel(" << pixelRow << "," << pixelCol << ")" ;

          findAntiZmw[index] = pixelColLow;
      }
#endif
  }


  static const uint32_t halfTilePixels = Tile::NumPixels/2;

  /// Reads a single tile's worth of data, starting at zmwOffset and frameOffset, into the memory
  /// pointed to by `tile`.
  /// Returns a zwmOffset to the next tile. The caller can use this returned value to
  /// iterate over all tiles of the file.
uint32_t SequelTraceFileHDF5::ReadTile(uint32_t zmwOffset, uint64_t frameOffset, Tile* tile)
  {
      uint32_t nextZmwOffset = zmwOffset;

      try {
          if (frameOffset + Tile::NumFrames > NFRAMES)
          {
              throw PBExceptionErrno("File does not contain enough frames for a full tile at frameOffset:" +
                                             std::to_string(frameOffset) + ", NFRAMES:" + std::to_string(NFRAMES));
          }
          LOCK_HDF5();
          static std::vector<int16_t> pixels(Tile::NumPixels * Tile::NumFrames);
          assert(pixels.size()*sizeof(int16_t) == sizeof(Tile));
          const int numUnitCellsPerTile = Tile::NumPixels / NUM_CHANNELS;
          nextZmwOffset += numUnitCellsPerTile;

          hsize_t offset[3];
          offset[0] = zmwOffset;
          offset[1] = 0;
          offset[2] = frameOffset;

          hsize_t size[3];
          size[0] = numUnitCellsPerTile;
          size[1] = NUM_CHANNELS;
          size[2] = Tile::NumFrames;

#if 0
          PBLOG_INFO << "Loading  SequelTraceFileHDF5::ReadTile:" << " zmwOffset:" << zmwOffset << " tile:" << (void*) tile <<
                      " frameOffset: " << frameOffset <<
                      " size[0]:" << size[0] <<
                      " size[1]:" << size[1] <<
                      " size[2]:" << size[2];
#endif

          H5::DataSpace fspace1 = Traces.getSpace();
          fspace1.selectHyperslab(H5S_SELECT_SET, size, offset);

          H5::DataSpace memTile(3, size);
          Traces.read(&pixels[0], H5::PredType::NATIVE_INT16, memTile, fspace1);

          int16_t* dst = reinterpret_cast<int16_t*>(tile);

          for(uint32_t iframe=0;iframe<Tile::NumFrames;iframe++)
          {
              for (int32_t izmw = 0; izmw < numUnitCellsPerTile; izmw++)
              {
                  if (NUM_CHANNELS == 2)
                  {
                      // Sequel (interleaved Red/Green)

                      // Red
                      dst[iframe * Tile::NumPixels + izmw * PIXEL_SIZE + 0] =
                              pixels[(izmw * Tile::NumFrames * PIXEL_SIZE + Tile::NumFrames) + iframe];

                      // Green
                      dst[iframe * Tile::NumPixels + izmw * PIXEL_SIZE + 1] =
                              pixels[(izmw * Tile::NumFrames * PIXEL_SIZE +               0) + iframe];

                  }
                  else if (NUM_CHANNELS == 1 )
                  {
#ifndef SPIDER_INTERLEAVED
                      // Spider (contiguous Red/Red)

                      // Red
                      dst[iframe * Tile::NumPixels + izmw]      = pixels[(izmw * Tile::NumFrames) + iframe];

                      // Red (duplicated)
                      dst[iframe * Tile::NumPixels + izmw + halfTilePixels] = dst[iframe * Tile::NumPixels + izmw];
#else
                      // Spider (interleaved)

                      // Red
                      dst[iframe * Tile::NumPixels + izmw * PIXEL_SIZE + 0] = pixels[(izmw * Tile::NumFrames) + iframe];
                      dst[iframe * Tile::NumPixels + izmw * PIXEL_SIZE + 1] = pixels[(izmw * Tile::NumFrames) + iframe];
#endif
                  }
              }
          }
      }
      catch(...)
      {
          std::string msg = "SequelTraceFileHDF5::ReadTile(" + std::to_string(zmwOffset) + "," + std::to_string(frameOffset) +") failed";
          std::throw_with_nested( std::runtime_error(msg) );
      }
      return nextZmwOffset;
  }

  uint32_t SequelTraceFileHDF5::ReadTranche(uint32_t zmwOffset, uint64_t frameOffset, Tranche* tranche,uint32_t numFrames0)
  {
      LOCK_HDF5();
      hsize_t numFrames = numFrames0;
      if (frameOffset + numFrames > NFRAMES)
      {
          numFrames = NFRAMES - frameOffset;
      }

      static std::vector<int16_t> pixels(numFrames * zmwsPerTranche * NUM_CHANNELS);
//      const int numUnitCellsPerTile = Tile::NumPixels / NUM_CHANNELS;

      hsize_t offset[3];
      offset[0] = zmwOffset;
      offset[1] = 0;
      offset[2] = frameOffset;

      hsize_t size[3];
      size[0] = zmwsPerTranche;
      size[1] = NUM_CHANNELS;
      size[2] = numFrames;

#if 0
      PBLOG_INFO << "Loading Tranche:" << " zmwOffset:" << zmwOffset <<
                  " frameOffset: " << frameOffset <<
                  " size[0]:" << size[0] <<
                  " size[1]:" << size[1] <<
                  " size[2]:" << size[2];
#endif
      H5::DataSpace fspace1 = Traces.getSpace();
      fspace1.selectHyperslab(H5S_SELECT_SET, size, offset);

      H5::DataSpace memTile(3, size);
      Traces.read(&pixels[0], H5::PredType::NATIVE_INT16, memTile, fspace1);

      PacBio::Primary::Tranche::Pixelz* dst = tranche->GetTraceDataPointer();
      if (!dst) { throw PBException("Trace.data pointer was null."); }

      for (uint32_t iframe = 0; iframe < numFrames; iframe++)
      {
          for (uint32_t izmw = 0; izmw < zmwsPerTranche; izmw++)
          {
              if (NUM_CHANNELS == 2)
              {
                  // Sequel (interleaved Red/Green)

                  // Red
                  dst->pixels.int16[iframe * Tile::NumPixels + izmw * PIXEL_SIZE + 0] =
                          pixels[(izmw * numFrames * PIXEL_SIZE + numFrames) + iframe];

                  // Green
                  dst->pixels.int16[iframe * Tile::NumPixels + izmw * PIXEL_SIZE + 1] =
                          pixels[(izmw * numFrames * PIXEL_SIZE +         0) + iframe];

              }
              else
              {
                  // Spider (contiguous Red/Red)

#ifndef SPIDER_INTERLEAVED
                  // Red
                  dst->pixels.int16[iframe * Tile::NumPixels + izmw] = pixels[(izmw * numFrames) + iframe];

                  // Red (duplicated)
                  dst->pixels.int16[iframe * Tile::NumPixels + izmw + halfTilePixels] = dst->pixels.int16[iframe * Tile::NumPixels + izmw];
#else
                  // Spider (interleaved)

                  // Red
                  dst->pixels.int16[iframe * Tile::NumPixels + izmw * PIXEL_SIZE + 0] = pixels[(izmw * numFrames) + iframe];
                  dst->pixels.int16[iframe * Tile::NumPixels + izmw * PIXEL_SIZE + 1] = pixels[(izmw * numFrames) + iframe];
#endif

              }
          }
      }


      static std::vector<uint32_t> holes(NUM_HOLES);
      if (holes.size() == 0)
      {
          HoleNumber >> holes;
      }

      uint32_t zmwNumber = holes[zmwOffset];

      TrancheTitle& title(tranche->Title());
      title.FrameCount(numFrames);
      title.FrameIndexStart(frameOffset);
      title.TimeStampStart(0); // fixme
      title.TimeStampDelta(defaultExposureUsec); // fixme
      title.ConfigWord(0);
      title.SuperChunkIndex(frameOffset/numFrames);
      title.StopStatus(Tranche::StopStatusType::NOT_STOPPED);

      tranche->ZmwLaneIndex(zmwOffset/zmwsPerTranche);
      tranche->ZmwIndex(zmwOffset);
      tranche->ZmwNumber(zmwNumber);


      return numFrames;
  }

  uint32_t SequelTraceFileHDF5::ReadLaneSegment(uint32_t zmwOffset, uint64_t frameOffset, uint32_t maxReadLen, int16_t* dest) {

      LOCK_HDF5();
      assert(maxReadLen > 0);
      hsize_t numFrames = maxReadLen;
      if (frameOffset + numFrames > NFRAMES)
      {
          numFrames = NFRAMES - frameOffset;
      }
      std::vector<int16_t> pixels(maxReadLen * zmwsPerTranche * NUM_CHANNELS);

      hsize_t offset[3];
      offset[0] = zmwOffset;
      offset[1] = 0;
      offset[2] = frameOffset;

      hsize_t size[3];
      size[0] = zmwsPerTranche;
      size[1] = NUM_CHANNELS;
      size[2] = numFrames;

      H5::DataSpace fspace1 = Traces.getSpace();
      fspace1.selectHyperslab(H5S_SELECT_SET, size, offset);

      H5::DataSpace memTile(3, size);
      Traces.read(&pixels[0], H5::PredType::NATIVE_INT16, memTile, fspace1);

      for(uint32_t iframe=0;iframe<numFrames;iframe++)
      {
          for (uint32_t izmw = 0; izmw < 16; izmw++)
          {
              if (NUM_CHANNELS == 2)
              {
                  // Sequel (interleaved Red/Green)

                  // Red
                  dest[iframe*Tile::NumPixels + izmw*PIXEL_SIZE + 0] = pixels[(izmw*numFrames*PIXEL_SIZE + numFrames) + iframe ];

                  // Green
                  dest[iframe*Tile::NumPixels + izmw*PIXEL_SIZE + 1] = pixels[(izmw*numFrames*PIXEL_SIZE +         0) + iframe ];

              }
              else
              {
#ifndef SPIDER_INTERLEAVED
                  // Spider (contiguous Red/Red)

                  // Red
                  dest[iframe*Tile::NumPixels + izmw] = pixels[(izmw*numFrames) + iframe ];

                  // Red (duplicated)
                  dest[iframe*Tile::NumPixels + izmw + halfTilePixels] = dest[iframe*Tile::NumPixels + izmw];
#else
                  // Spider (interleaved)

                  // Red
                  dest[iframe*Tile::NumPixels + izmw * PIXEL_SIZE + 0] = pixels[(izmw*numFrames) + iframe ];
                  dest[iframe*Tile::NumPixels + izmw * PIXEL_SIZE + 1] = pixels[(izmw*numFrames) + iframe ];
#endif
              }

          }
      }

      return numFrames;
  }

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


uint32_t SequelTraceFileHDF5::Read1CTrace(uint32_t zmwOffset, uint64_t frameOffset, std::vector<int16_t>& traceData) const
{
    if (NUM_CHANNELS != 1) throw PBException("Can't read 1C trace from trace file, actual number of channels:" +
                                                     std::to_string(NUM_CHANNELS));
    LOCK_HDF5();
    hsize_t numFrames;
    if (traceData.size() == 0)
    {
        numFrames = NFRAMES;
        traceData.resize(numFrames);
    }
    else
    {
        numFrames = traceData.size();
    }

    hsize_t offset[3];
    offset[0] = zmwOffset;
    offset[1] = 0;
    offset[2] = frameOffset;

    hsize_t size[3];
    size[0] = 1; // 1 ZMW
    size[1] = NUM_CHANNELS;
    size[2] = numFrames;

    H5::DataSpace fspace1 = Traces.getSpace();
    fspace1.selectHyperslab(H5S_SELECT_SET, size, offset);

    H5::DataSpace memTile(3, size);
    Traces.read(&traceData[0], H5::PredType::NATIVE_INT16, memTile, fspace1);

    return numFrames;
}
  uint32_t SequelTraceFileHDF5::Read2CTrace(uint32_t zmwOffset, uint64_t frameOffset, std::vector<std::pair<int16_t,int16_t> >& traceData) const
  {
      if (NUM_CHANNELS != 2) throw PBException("Can't read 2C trace from trace file, actual number of channels:" +
                                               std::to_string(NUM_CHANNELS));

      LOCK_HDF5();
      hsize_t numFrames;
      if (traceData.size() == 0)
      {
          numFrames = NFRAMES;
          traceData.resize(numFrames);
      }
      else
      {
          numFrames = traceData.size();
      }

      static std::vector<int16_t> pixels(numFrames * NUM_CHANNELS);

      hsize_t offset[3];
      offset[0] = zmwOffset;
      offset[1] = 0;
      offset[2] = frameOffset;

      hsize_t size[3];
      size[0] = 1; // 1 ZMW
      size[1] = NUM_CHANNELS;
      size[2] = numFrames;

      H5::DataSpace fspace1 = Traces.getSpace();
      fspace1.selectHyperslab(H5S_SELECT_SET, size, offset);

      H5::DataSpace memTile(3, size);
      Traces.read(&pixels[0], H5::PredType::NATIVE_INT16, memTile, fspace1);

      for(uint32_t iframe=0;iframe<numFrames;iframe++)
      {
          // Green
          traceData[iframe].first = pixels[0 + iframe];

          // Red
          traceData[iframe].second =
              (NUM_CHANNELS == 2)
              ? pixels[numFrames + iframe]
              : traceData[iframe].first;
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


uint64_t SequelTraceFileHDF5::ReadFrame(uint64_t n, SequelMovieFrame <int16_t>& frame) const
{
    return ReadFrame(n,frame,ReadFrame_NoOptions);
}

uint64_t SequelTraceFileHDF5::ReadFrame(uint64_t n, SequelMovieFrame <int16_t>& frame, int options) const
{
    const SequelSensorROI sensorROI = GetChipLayout().GetSensorROI();
    const bool condensed = (options & ReadFrame_Condensed) != 0;
    const bool dontResize = (options & ReadFrame_DontResize) != 0;
    if ( !condensed && ! dontResize &&
        (sensorROI.PhysicalCols() != frame.NCOL || sensorROI.PhysicalRows() != frame.NROW))
    {
        frame.Resize(sensorROI.PhysicalRows(), sensorROI.PhysicalCols());
        frame.SetDefaultValue(0);
    }
    const int frameRows = static_cast<int>(frame.NROW);
    const int frameCols = static_cast<int>(frame.NCOL);

    if (n >= NFRAMES)
    {
        throw PBException("n out of range " + std::to_string(n) + " " + std::to_string(NFRAMES));
    }
    if (cache_.enabled_)
    {
        const uint64_t cacheFrames = 512;

        if (cache_.frameBase_ != (n & ~(cacheFrames-1)))
        {
            PBLOG_DEBUG << "SequelTraceFileHDF5::ReadFrame reading  cache";
            cache_.frameBase_ = (n & ~(cacheFrames-1));
            hsize_t offset[3];
            offset[0] = 0; // zww
            offset[1] = 0; // color
            offset[2] = cache_.frameBase_; // frame

            hsize_t toLoad = NFRAMES -  cache_.frameBase_;
            if (toLoad > cacheFrames) toLoad= cacheFrames;
            cache_.size_[0] = NUM_HOLES;
            cache_.size_[1] = NUM_CHANNELS;
            cache_.size_[2] = toLoad;

            cache_.pixels_.resize(cache_.size_[0] * cache_.size_[1] * cache_.size_[2]);

            H5::DataSpace fspace1 = Traces.getSpace();
            fspace1.selectHyperslab(H5S_SELECT_SET, cache_.size_, offset);

            H5::DataSpace memTile(3, cache_.size_);
            Traces.read(&cache_.pixels_[0], H5::PredType::NATIVE_INT16, memTile, fspace1);
        } else {
            PBLOG_DEBUG << "SequelTraceFileHDF5::ReadFrame reusing cache";
        }

        int16_t* dst = frame.data;

        const ChipLayout& layout(GetChipLayout());
        const std::vector<UnitCell>& uc(GetUnitCells());

        uint32_t nOffset = n % cacheFrames;

        if (condensed)
        {
            uint32_t izmw = 0;
            for (int irow = 0; irow < frameRows; irow++)
            {
                if (NUM_CHANNELS == 2)
                {
                    for (int icol = 0; icol < frameCols; icol += 2)
                    {
                        dst[irow * frameCols + icol + 0] = cache_.pixels_[(izmw * 2 + 1) * cache_.size_[2] +
                                                                          nOffset]; // red
                        dst[irow * frameCols + icol + 1] = cache_.pixels_[(izmw * 2 + 0) * cache_.size_[2] +
                                                                          nOffset]; // green
                        izmw++;
                        if (izmw >= NUM_HOLES)
                        {
                            izmw = 0;
                        }
                    }
                }
                else if (NUM_CHANNELS == 1)
                {
                    throw PBException("not implmented yet");
                }
            }
        }
        else
        {
            if (NUM_CHANNELS == 2)
            {
                for (uint32_t izmw = 0; izmw < NUM_HOLES; izmw++)
                {
                    int irow = layout.ConvertUnitCellToRowPixels(uc[izmw]).Value();
                    int icol = layout.ConvertUnitCellToColPixels(uc[izmw]).Value();
                    if (irow >=0 && irow < frameRows &&
                        icol >=0 && icol < frameCols)
                    {
                        dst[irow * frameCols + icol + 0] = cache_.pixels_[(izmw * 2 + 1) * cache_.size_[2] +
                                                                          nOffset]; // red
                        dst[irow * frameCols + icol + 1] = cache_.pixels_[(izmw * 2 + 0) * cache_.size_[2] +
                                                                          nOffset]; // green
                    }
                }
            }
            else if (NUM_CHANNELS == 1)
            {
                for (uint32_t izmw = 0; izmw < NUM_HOLES; izmw++)
                {
                    int irow = layout.ConvertUnitCellToRowPixels(uc[izmw]).Value();
                    int icol = layout.ConvertUnitCellToColPixels(uc[izmw]).Value();
                    if (irow >=0 && irow < frameRows &&
                        icol >=0 && icol < frameCols)
                    {
                        dst[irow * frameCols + icol + 0] = cache_.pixels_[izmw * cache_.size_[2] + nOffset];
                    }
                }
            }
        }
    }
    else
    {
        PBLOG_DEBUG << "SequelTraceFileHDF5::ReadFrame Not using cache";

        static std::vector<int16_t> pixels(NUM_HOLES * 2 /*colors*/);

        hsize_t offset[3];
        offset[0] = 0;
        offset[1] = 0; // color
        offset[2] = n;

        hsize_t size[3];
        size[0] = NUM_HOLES;
        size[1] = NUM_CHANNELS;
        size[2] = 1;

        H5::DataSpace fspace1 = Traces.getSpace();
        fspace1.selectHyperslab(H5S_SELECT_SET, size, offset);

        H5::DataSpace memTile(3, size);
        Traces.read(&pixels[0], H5::PredType::NATIVE_INT16, memTile, fspace1);


        int16_t* dst = frame.data;

        const ChipLayout& layout(GetChipLayout());
        const std::vector<UnitCell>& uc(GetUnitCells());

        if (NUM_CHANNELS == 2)
        {
            for (uint32_t izmw = 0; izmw < NUM_HOLES; izmw++)
            {
                int irow = layout.ConvertUnitCellToRowPixels(uc[izmw]).Value();
                int icol = layout.ConvertUnitCellToColPixels(uc[izmw]).Value();
                if (irow >=0 && irow < frameRows &&
                    icol >=0 && icol < frameCols)
                {
                    dst[irow * frameCols + icol + 0] = pixels[izmw * 2 + 1]; // red
                    dst[irow * frameCols + icol + 1] = pixels[izmw * 2 + 0]; // green
                }
            }
        }
        else if (NUM_CHANNELS == 1)
        {
            throw PBException("not implemnted yet");
        }
    }
    frame.index = n;
    if (frameRateCache_ == 0)
    {
        FrameRate >> frameRateCache_;
        if (frameRateCache_ <= 0) frameRateCache_ = 1;
    }
    frame.timestamp = static_cast<uint64_t>((n / frameRateCache_) * 1e6);

    return 1;
}

const ChipLayout& SequelTraceFileHDF5::GetChipLayout() const
{
    if (!chipLayout_)
    {
        std::string layoutName;
        {
            LOCK_HDF5();

            LayoutName >> layoutName;
        }
        chipLayout_ = ChipLayout::Factory(layoutName);
    }
    return *chipLayout_;
}


  const SequelROI* SequelTraceFileHDF5::GetROI() const
  {
      if (!originalRoi_)
      {
          SequelSparseROI* roi;
          originalRoi_.reset(roi = new SequelSparseROI(GetChipLayout().GetSensorROI()));

          std::vector<uint32_t> holes(NUM_HOLES);

          const std::vector<UnitCell>& unitcells(SequelTraceFileHDF5::GetUnitCells());

          {
              LOCK_HDF5();

              HoleNumber >> holes;
          }

          const ChipLayout& layout = GetChipLayout();
          std::map<uint32_t,uint32_t> lanes;
          for (uint32_t i = 0; i < NUM_HOLES; i++)
          {
              const UnitCell& hole(unitcells[i]);
              uint32_t expectedNumber = (((uint32_t) hole.x) << 16) | hole.y;
              if (expectedNumber != holes[i])
              {
                  PBLOG_WARN << "Something fishy. Numbering doesn't match coordinate" << expectedNumber << " " <<
                             holes[i];
              }

              // ROIs are required to be specified with exact multiples of 32
              // pixels, but that is either 16 or 32 zmw depending on sequel
              // vs spider
              uint32_t zmwsPerPixelLane = zmwsPerTranche;
              if (layout.ColPixelsPerZmw() == 1) 
                  zmwsPerPixelLane *= 2;
              
              // Bitwise logic below only works for powers of 2
              assert(zmwsPerPixelLane == 16 || zmwsPerPixelLane == 32);
              uint32_t lane = (((uint32_t) hole.x) << 16) | ( hole.y & ~(zmwsPerPixelLane - 1)) ;
              uint32_t laneOffset = hole.y & (zmwsPerPixelLane - 1);
              auto l = lanes.find(lane);
              if (l == lanes.end())
              {
                  lanes[lane] = 1<<laneOffset;
              }
              else
              {
                  l->second |= 1 << laneOffset;
              }
          }
          const uint32_t fullPixelLaneBitmask = layout.ColPixelsPerZmw() == 2 ? 0xFFFF : 0xFFFFFFFF;
          for(const auto& l : lanes)
          {
              if (l.second != fullPixelLaneBitmask)
              {
                  // verify that all 16 ZMWS are present
                  throw PBException("trace file has invalid ROI. " + std::to_string(l.first) + " " + std::to_string(l.second));
              }
              UnitCell hole(l.first);

              RowPixels row = layout.ConvertUnitCellToRowPixels(hole);
              ColPixels col = layout.ConvertUnitCellToColPixels(hole);

              roi->AddRectangle(row, col.Value() & ~31, 1, 32);
          }
          roi->PostAddRectangle();
      }
      return originalRoi_.get();
  }
// writes 1-color trace data (Spider only)
void SequelTraceFileHDF5::Write1CTrace(uint32_t zmwOffset, uint64_t frameOffset, const std::vector<int16_t >& traceData)
{
    LOCK_HDF5();
    hsize_t numFrames = traceData.size();
    if (frameOffset + numFrames > NFRAMES)
    {
        numFrames = NFRAMES - frameOffset;
    }

    hsize_t offset[3];
    offset[0] = zmwOffset;
    offset[1] = 0; // color offset
    offset[2] = frameOffset;

    hsize_t size[3];
    size[0] = 1; // num ZMWs
    size[1] = 1; // num colors
    size[2] = numFrames;

    H5::DataSpace fspace1 = Traces.getSpace();
    fspace1.selectHyperslab(H5S_SELECT_SET, size, offset);

    H5::DataSpace memTile(3, size);
    Traces.write(&traceData[0], H5::PredType::NATIVE_INT16, memTile, fspace1);

#if 0
    static std::vector<uint32_t> holes(NUM_HOLES);
      if (holes.size() == 0)
      {
          HoleNumber >> holes;
      }

      uint32_t zmwNumber = holes[zmwOffset];
#endif
}

  // writes 2-color trace data (Sequel only)
  void SequelTraceFileHDF5::Write2CTrace(uint32_t zmwOffset, uint64_t frameOffset, const std::vector<std::pair<int16_t,int16_t> >& traceData)
  {
      LOCK_HDF5();
      hsize_t numFrames = traceData.size();
      if (frameOffset + numFrames > NFRAMES)
      {
          numFrames = NFRAMES - frameOffset;
      }
      // frames are contiguous, then colors. So this is laid out
      // { green[nFrames],red[nFrames] }
      // which is the transpose of the input data.
      static std::vector<int16_t> pixels(numFrames * 2 /*colors*/);

      hsize_t offset[3];
      offset[0] = zmwOffset;
      offset[1] = 0; // color
      offset[2] = frameOffset;

      hsize_t size[3];
      size[0] = 1;
      size[1] = 2; // colors
      size[2] = numFrames;

      for(uint32_t iframe=0;iframe<numFrames;iframe++)
      {
          pixels[            iframe ] = traceData[iframe].first;  // green
          pixels[numFrames + iframe ] = traceData[iframe].second; // red
      }

      H5::DataSpace fspace1 = Traces.getSpace();
      fspace1.selectHyperslab(H5S_SELECT_SET, size, offset);

      H5::DataSpace memTile(3, size);
      Traces.write(&pixels[0], H5::PredType::NATIVE_INT16, memTile, fspace1);

#if 0
      static std::vector<uint32_t> holes(NUM_HOLES);
      if (holes.size() == 0)
      {
          HoleNumber >> holes;
      }

      uint32_t zmwNumber = holes[zmwOffset];
#endif
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

// works for both Sequel and Spider
int SequelTraceFileHDF5::CompareTraces(const SequelTraceFileHDF5& b, bool subsetFlag)
{
    int errors = 0;

    std::vector<uint32_t> holesThis;
    this->HoleNumber >> holesThis;

    std::vector<uint32_t> holesB;
    b.HoleNumber >> holesB;

    if (subsetFlag)
    {
        size_t missingHoles = 0;
        for(auto holeThis : holesThis)
        {
            if (std::find(holesB.begin(),holesB.end(), holeThis) == holesB.end())
            {
                if (missingHoles < 10)
                {
                    PBLOG_WARN << "Hole 0x" << holeThis << " not found in B";
                }
                missingHoles++;
            }
        }
        errors += missingHoles;
    }
    else
    {
        errors += Compare(this->NUM_HOLES, b.NUM_HOLES, "file NUM_HOLES count mismatch");
    }
    errors += Compare(this->NUM_CHANNELS, b.NUM_CHANNELS, "file NUM_CHANNELS count mismatch");
    if (errors)
    {
        std::cout << "Can not continue comparison, different sets of ZMWs" << std::endl;
        return errors;
    }

    if (subsetFlag)
    {
        if (this->NFRAMES > b.NFRAMES)
        {
            PBLOG_WARN << " More frames in this " << this->NFRAMES << " than in B " << b.NFRAMES;
          //  errors++;
        }
    }
    else
    {
        errors += Compare(this->NFRAMES, b.NFRAMES, "Frame count mismatch, will use smaller count");
    }

    uint32_t numFrames = std::min(this->NFRAMES, b.NFRAMES);

    PBLOG_INFO << "this NFRAMES:" << this->NFRAMES << " b.NFRAMES:" << b.NFRAMES << " numFrames: " << numFrames;

    for (uint32_t nThis = 0; nThis < NUM_HOLES; nThis++)
    {
        uint32_t holeThis = holesThis[nThis];

        auto iterB = std::find(holesB.begin(), holesB.end(), holeThis);
        if (iterB == holesB.end())
        {
            throw PBException("Internal inconsistency in code");
        }
        uint32_t nB = iterB - holesB.begin();

        if (NUM_CHANNELS == 2)
        {
            std::vector<std::pair<int16_t, int16_t>> aTrace;
            std::vector<std::pair<int16_t, int16_t>> bTrace;
            aTrace.resize(numFrames);
            bTrace.resize(numFrames);


            this->Read2CTrace(nThis, 0, aTrace);
            b.Read2CTrace(nB, 0, bTrace);
            bool fail = (aTrace != bTrace);
            errors += (fail ? 1 : 0);
            if (fail)
            {
                PBLOG_WARN << "Comparison of ZMW 0x" << std::hex << holeThis <<
                            std::dec << " (fileA.offset:" << nThis << " fileB.offset:" << nB << ") failed";

                PBLOG_INFO << " Dumping numFrames:" << numFrames;
                for (uint32_t f = 0; f < numFrames; f++)
                {
                    bool xfail = (aTrace[f].first != bTrace[f].first) || (aTrace[f].second != bTrace[f].second);
                    PBLOG_INFO << "[" << f << "] " << aTrace[f].first << "," << aTrace[f].second
                                << (xfail ? " != " : " == ")
                                << bTrace[f].first << "," << bTrace[f].second;
                }
                break;
            }
        }
        else if (NUM_CHANNELS == 1)
        {
            std::vector<int16_t> aTrace;
            std::vector<int16_t> bTrace;
            aTrace.resize(numFrames);
            bTrace.resize(numFrames);

            this->Read1CTrace(nThis, 0, aTrace);
            b.Read1CTrace(nB, 0, bTrace);
            bool fail = (aTrace != bTrace);
            errors += (fail ? 1 : 0);
            if (fail)
            {
                PBLOG_WARN << "Comparison of ZMW offset " << nThis << " to offset " << nB << " failed";

                for (uint32_t f = 0; f < numFrames; f++)
                {
                    bool xfail = (aTrace[f] != bTrace[f]);
                    PBLOG_INFO << "[" << f << "] " << aTrace[f] << "," << aTrace[f] << " "
                                << (xfail ? " FAILED" : "");
                }
                break;
            }
        }
        else
        {
            throw PBException("NUM_CHANNELS not supported");
        }
    }
    PBLOG_INFO << "Total errors: " << errors << std::endl;

    return errors;
}


void SequelTraceFileHDF5::AddLaserPowerEvent(const EventObject& eo)
{
    if (eo.eventType() != EventObject::EventType::laserpower)
    {
        throw PBException("AddLaerPowerEvent called with event that is not a laserpower change");
    }
    events.laserPowerChanges.AddEvent(eo);
}

}}

