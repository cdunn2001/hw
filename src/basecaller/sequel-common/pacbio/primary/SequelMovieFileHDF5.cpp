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
/// \brief  Implementation of frame-based *.mov.h5 handling class.

#include <pacbio/primary/SequelMovieFileHDF5.h>
#include <pacbio/primary/SequelTraceFile.h>

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
#include <stdexcept>
#include <memory>
#include <vector>
#include <list>

#include <tbb/scalable_allocator.h>

#include <pacbio/primary/Tile.h>
#include <pacbio/ipc/ThreadSafeQueue.h>
#include <pacbio/text/String.h>
#include <pacbio/Utilities.h>
#include <pacbio/utilities/ISO8601.h>
#include <pacbio/utilities/PerforceUtil.h>
#include <pacbio/logging/Logger.h>
#include <pacbio/primary/HDF5cpp.h>
#include <pacbio/primary/EventObject.h>
#include <pacbio/primary/PrimaryConfig.h>
#include <pacbio/primary/SequelMovieConfig.h>

namespace PacBio
{
 namespace Primary
 {


  // data types

#define int16()      H5::PredType::STD_I16LE
#define uint16()     H5::PredType::STD_U16LE
#define uint32()     H5::PredType::STD_U32LE
#define uint64()     H5::PredType::STD_U64LE
#define float32()    H5::PredType::IEEE_F32LE
#define float64()    H5::PredType::IEEE_F64LE

//adding a debug stream
  static std::ofstream devnull; 
  static std::ostream* movieDebugStream = &devnull;

  void SequelMovieFileHDF5::SetMovieDebugStream(std::ostream& os)
  {
    movieDebugStream = &os;
  }


  /// Construct a movie file with the given ROI.
  SequelMovieFileHDF5::SequelMovieFileHDF5(const std::string& fullFileName0, const SequelROI& roi0 , bool /*fake*/,
                                           ChipClass chipClass)
          : SequelMovieFileBase(fullFileName0, roi0, false, chipClass)
          , NROW(0)
          , NCOL(0)
#ifdef OBSOLETE_EVENTS
          , eventEnumType(sizeof(SequelMovieFileBase::EventType_e))
#endif
          , frames_StartIndex(0)
          , cameraConfigKey(0)
          , cameraConfigKeySet(false)
  {
      LOCK_HDF5();

      hs_size[0] = 1;
      hs_size[1] = NROW;
      hs_size[2] = NCOL;
      memFrame = H5::DataSpace(3, hs_size);

      if (roi_->Type() != SequelROI::ROI_Type_e::Rectangular)
      {
          throw PBException("SequelMovieFileHDF5 does not support any ROI other than SequelRectangularROI" +
                            roi_->Type().toString());
      }

      const SequelRectangularROI* rectROI = dynamic_cast< const SequelRectangularROI*>(&roi0);
      NROW = rectROI->NumPixelRows();
      NCOL = rectROI->NumPixelCols();
  }

  /// Read an existing HDF movie file
  SequelMovieFileHDF5::SequelMovieFileHDF5(const std::string& filename)
          : SequelMovieFileHDF5(filename, SequelROI::Null(), false, ChipClass::UNKNOWN)
  {
          OpenFileForRead(filename);
  }

  /// Create a new HDF movie file
  SequelMovieFileHDF5::SequelMovieFileHDF5(const SequelMovieConfig& config)
          : SequelMovieFileHDF5(config.path, *config.roi, false, config.chipClass)
  {
      PBLOG_DEBUG << "SequelMovieFileHDF5::SequelMovieFileHDF5(" << config.path << ", roi:" << *config.roi << ")";
      config_ = config;
      CreateFileForWrite(config);
  }


  // destructor
  SequelMovieFileHDF5::~SequelMovieFileHDF5()
  {
      // std::cout << "SequelMovieFileHDF5::~SequelMovieFileHDF5()" << std::endl;

      try
      {
          FinalizeSaverThread();

          {
              LOCK_HDF5();
              hd5file.close();
          }

          CleanupWorkingFilename();
      }
      catch(const std::exception& ex)
      {
          // std::cout << "SequelMovieFileHDF5::~SequelMovieFileHDF5() caught exception: " << ex.what() << std::endl;
          PBLOG_ERROR << "SequelMovieFileHDF5::~SequelMovieFileHDF5() caught exception: " << ex.what();
      }
      catch(...)
      {
          std::cerr << "Uncaught exception caught in ~SequelMovieFileHDF5 " << PacBio::GetBackTrace(5);
          PBLOG_FATAL << "Uncaught exception caught in ~SequelMovieFileHDF5 " << PacBio::GetBackTrace(5);
          PacBio::Logging::PBLogger::Flush();
          std::terminate();
      }
  }


  /// open file for read. load all attributes and dataset definitions.
  void SequelMovieFileHDF5::OpenFileForRead(const std::string& filename)
  {
      {
          LOCK_HDF5();

          DisableDefaultErrorHandler noErrors;

          {
              // the HDF5 error message is horrible. It doesn't display the name of the file if it can not open the file
              std::ifstream canary(filename);
              if (!canary)
              {
                  throw PBException(
                          "Could not open file " + filename + " for reading. Check existence and permissions.");
              }
          }
          try
          {

              hd5file.openFile(filename.c_str(), H5F_ACC_RDONLY);

              Movie = hd5file.openGroup("/Movie");
              FormatVersion = Movie.openAttribute("FormatVersion");
              Creator = Movie.openAttribute("Creator");
              CreatorVersion = Movie.openAttribute("CreatorVersion");
              TimeCreated = Movie.openAttribute("TimeCreated");
              InstrumentName = Movie.openAttribute("InstrumentName");
              CellSN = Movie.openAttribute("CellSN");
              if (Movie.attrExists("PaneId")) PaneId = Movie.openAttribute("PaneId");
              if (Movie.attrExists("PaneSetId")) PaneSetId = Movie.openAttribute("PaneSetId");
              CameraConfigKey = Movie.openAttribute("CameraConfigKey");
              if (Movie.attrExists("FrameOffset")) FrameOffset = Movie.openAttribute("FrameOffset");
              FrameSize = Movie.openAttribute("FrameSize");

              Frames = Movie.openDataSet("Frames");
              FrameRate = Frames.openAttribute("FrameRate");
              Exposure = Frames.openAttribute("Exposure");
              Frames_StartIndex = Frames.openAttribute("StartIndex");

              FrameTime = Movie.openDataSet("FrameTime");
              AcqStartTime = FrameTime.openAttribute("AcquisitionStartTime");

              DarkFrame = Movie.openDataSet("DarkFrame");
              DarkFrame_Exposure = DarkFrame.openAttribute("Exposure");

              try
              {
                  DarkFrameExp = Movie.openDataSet("DarkFrameExp");
              }
              catch (H5::GroupIException&)
              {
              }
              catch (H5::Exception&)
              {
              }

              {
                  hsize_t chunk_dims[3];
                  framesPropList = Frames.getCreatePlist();
                  if (H5D_CHUNKED == framesPropList.getLayout())
                  {
                      framesPropList.getChunk(3, chunk_dims);
                  }
                  else
                  {
                      // no chunking, just set values to 1 in case someone asks.
                      chunk_dims[0] = 1;
                      chunk_dims[1] = 1;
                      chunk_dims[2] = 1;
                  }
                  config_.movie.chunking.frame = chunk_dims[0];
                  config_.movie.chunking.row   = chunk_dims[1];
                  config_.movie.chunking.col   = chunk_dims[2];
              }
              std::string creator;
              Creator >> creator;
              PBLOG_INFO << "Movie " << filename << " Creator:" << creator;
              for (const auto s : ChipClass::allValues())
              {

                  if (PacBio::Text::String::Contains(creator, s.toString()))
                  {
                      chipClass_ = s;
                      PBLOG_INFO << "ChipClass set to " << chipClass_.toString();
                      break;
                  }
              }

              try
              {
                  DarkFrameStack = Movie.openDataSet("DarkFrameStack");
                  darkFrameSpace = DarkFrameStack.getSpace();
                  int rank = darkFrameSpace.getSimpleExtentNdims();
                  if (rank != 3)
                  {
                      throw PBException("wrong darkFrameSpace rank: " + std::to_string(rank));
                  }
                  hsize_t dims[3];
                  rank = darkFrameSpace.getSimpleExtentDims(dims);
                  DarkFrameStack_StartIndex
                          = DarkFrameStack.openAttribute("StartIndex");

                  NDARKFRAMES = static_cast<uint32_t>(dims[0]);
              }
              catch (H5::Exception&)
              {
                  NDARKFRAMES = 0;
              }

              Gain = Movie.openDataSet("Gain");
              ReadNoise = Movie.openDataSet("ReadNoise");

              AcquisitionXML = Movie.openDataSet("AcquisitionXML");
#ifdef OBSOLETE_EVENTS
              try
              {
                  EventType = Movie.openDataSet("EventType");
                  eventEnumType = H5::EnumType(EventType);

                  hsize_t dims[1];
                  EventType.getSpace().getSimpleExtentDims(dims);

                  NUM_EVENTS = static_cast<uint32_t>(dims[0]);
              }
              catch (H5::GroupIException&)
              { NUM_EVENTS = 0; }
              try
              {
                  EventTime = Movie.openDataSet("EventTime");
                  hsize_t dims[1];
                  EventTime.getSpace().getSimpleExtentDims(dims);
                  if (NUM_EVENTS != static_cast<uint32_t>(dims[0]))
                  {
                      throw PBException("EventType and EventTime datasets do not agree in dimension");
                  }
              }
              catch (H5::GroupIException&)
              { NUM_EVENTS = 0; }
#endif

              auto dims = GetDims(Frames);
              if (dims.size() != 3)
              {
                  throw PBException("wrong Frames rank: " + std::to_string(dims.size()));
              }

              NFRAMES = static_cast<uint32_t>(dims[0]);
              NROW = static_cast<uint32_t>(dims[1]);
              NCOL = static_cast<uint32_t>(dims[2]);

              //printf("Found Num Frames:%ld Num Cols:%d Num Rows:%d\n",NFRAMES,NCOL,NROW);
              *movieDebugStream << "Found NFRAMES:" << NFRAMES << " NCOL:" << NCOL << " NROW:" << NROW << std::endl;
              uint32_t frameOffset[2] = {0, 0};
              uint32_t frameSize[2] = {0, 0};
              
              // TODO: this is disabled because the implementation is not finished. 
              if (Movie.attrExists("FrameOffset") && false)
              {
                  *movieDebugStream << "Reading FrameOffset\n";
                  //printf("Reading FrameOffset\n");
                  FrameOffset.read(uint32(), frameOffset);
                  //printf("After FrameOffset\n");
                  *movieDebugStream << "After FrameOffset\n";
                  // ??
              }
              if (true)
              {
                  //printf("Checking FrameSize\n");
                  *movieDebugStream << "Checking FrameSize\n";
                  FrameSize.read(uint32(), frameSize);
                  if (frameSize[0] != NROW)
                  {
                      throw PBException("row mismatch");
                  }
                  if (frameSize[1] != NCOL)
                  {
                      throw PBException("col mismatch");
                  }
              }
              uint32_t pixelsPerRow = 0;
              uint32_t pixelsPerCol = 0;
              if (chipClass_ == ChipClass::UNKNOWN)
              {
                  switch (frameSize[0])
                  {
                  case Sequel::maxPixelRows:
                      chipClass_ = ChipClass::Sequel;
                      break;
                  case Spider::maxPixelRows:
                      chipClass_ = ChipClass::Spider;
                      break;
                  default:
                      PBLOG_WARN << "Unrecognized row size in movie:" << frameSize[0] << ", defaulting to Sequel";
                      chipClass_ = ChipClass::Sequel;
                  }
              }

              auto pixelsPerZmw = PixelsPerZmw(chipClass_);
              pixelsPerRow = pixelsPerZmw.first;
              pixelsPerCol = pixelsPerZmw.second;

              SequelSensorROI sensorROI(frameOffset[0],
                                        frameOffset[1],
                                        frameSize[0],
                                        frameSize[1],
                                        pixelsPerRow, pixelsPerCol);
              roi_.reset(new SequelRectangularROI(frameOffset[0],
                                                  frameOffset[1],
                                                  frameSize[0],
                                                  frameSize[1],
                                                  sensorROI));

              if (NDARKFRAMES > 0)
              {
                  hsize_t dimsDf[3];
                  int ndims = darkFrameSpace.getSimpleExtentDims(dimsDf);
                  if (ndims == 3)
                  {
                      if (dimsDf[1] != NROW)
                          throw PBException(
                                  PacBio::Text::String::Format(
                                          "rows mismatch between data frame (%d) and dark frames (%d)",
                                          NROW, dimsDf[1]));
                      if (dimsDf[2] != NCOL)
                          throw PBException(
                                  PacBio::Text::String::Format(
                                          "cols mismatch between data frame (%d) and dark frames (%d)",
                                          NCOL, dimsDf[2]));
                  }
              }

              hs_size[0] = 1;
              hs_size[1] = NROW;
              hs_size[2] = NCOL;
              memFrame = H5::DataSpace(3, hs_size);

              Frames_StartIndex >> frames_StartIndex;
              CameraConfigKey >> cameraConfigKey;
              cameraConfigKeySet = true;

              errorFrameCount = 0;
              PBLOG_DEBUG << "looking at ErrorFrames";
              try
              {
                  ErrorFrames = Movie.openDataSet("ErrorFrames");
              }
              catch (H5::GroupIException&)
              {
              }
              catch (H5::Exception&)
              {
              }

              valid = true;
          }
          catch (H5::Exception& ex)
          {
              H5::Exception::printErrorStack();
              throw ex;
          }
      }
      events.OpenForRead(hd5file);
  }


  /// create a new file. Create all attributes and datasets
  void SequelMovieFileHDF5::CreateFileForWrite(const SequelMovieConfig& config)
  {
      SetupWorkingFilename(config.path);

      LOCK_HDF5();

      if (NROW == 0 || NCOL == 0) throw PBException("rows=" + std::to_string(NROW) + ", cols=" + std::to_string(NCOL));
      if (NROW > 4096 || NCOL > 4096)
          throw PBException("rows=" + std::to_string(NROW) + ", cols=" + std::to_string(NCOL));
#if 0
      H5::FileAccPropList fapl;
//    	fapl.setCache(11,521,256*1024*1024,1.0);
      hid_t fapl_id = fapl.getId();
      size_t alignment = 512;
      size_t block_size = 512;
      size_t cbuf_size = 512 * 1024;
      H5Pset_fapl_direct( fapl_id,  alignment,  block_size,  cbuf_size );
      hd5file = H5::H5File(filename.c_str(), H5F_ACC_TRUNC, H5::FileCreatPropList::DEFAULT, fapl);
#else
      hd5file = H5::H5File(workingFile_, H5F_ACC_TRUNC);
#endif
      NFRAMES = 0;
      NDARKFRAMES = 0;
      NFRAMES_allocated = 0;
      errorFrameCount = 0;


      hsize_t chunk_dims[3];
      if (config.movie.chunking.frame == 0)
      {
          chunk_dims[0] = chunkSize;
          chunk_dims[1] = NROW;
          chunk_dims[2] = NCOL;
      }
      else
      {
          chunk_dims[0] = config.movie.chunking.frame;
          chunk_dims[1] = config.movie.chunking.row;
          chunk_dims[2] = config.movie.chunking.col;
      }
      PBLOG_DEBUG << "SequelMovieFileHDF5::CreateFileForWrite() filename:" << workingFile_.FinalFilename()
                      << " Chunk Sizes:" << chunk_dims[0] << "," << chunk_dims[1] << "," << chunk_dims[2];
      framesPropList.setChunk(3, chunk_dims);

      if (config.movie.compression >9)
      {
          throw PBException("Invalid HDF5 compression level: " + std::to_string(config.movie.compression));
      }
      framesPropList.setDeflate(config.movie.compression);

      int fill_val = 0;
      framesPropList.setFillValue(H5::PredType::NATIVE_INT, &fill_val);

      timesPropList.setChunk(1, chunk_dims);
      timesPropList.setFillValue(H5::PredType::NATIVE_INT, &fill_val);
#if 0
      hid_t dapl_id = H5Pcreate(H5P_DATASET_ACCESS);
      H5Pset_chunk_cache(dapl_id,521,16*1024*1024,1.0);
#endif
      hsize_t startSpace[3];
      startSpace[0] = 0;
      startSpace[1] = NROW;
      startSpace[2] = NCOL;

      hsize_t maxSpace[3];
      maxSpace[0] = H5S_UNLIMITED;
      maxSpace[1] = NROW;
      maxSpace[2] = NCOL;
      hsize_t dim2dim[1] = {2};

      hs_size[0] = 1;
      hs_size[1] = NROW;
      hs_size[2] = NCOL;
      memFrame = H5::DataSpace(3, hs_size);


      dim2 = H5::DataSpace(1, dim2dim);
      framesSpace = H5::DataSpace(3, startSpace, maxSpace);
      frameSpace = H5::DataSpace(2, startSpace + 1);
      timesSpace = H5::DataSpace(1, startSpace, maxSpace);
      darkFrameSpace = H5::DataSpace(3, startSpace, maxSpace);
      eventSpace = H5::DataSpace(1, startSpace, maxSpace);

      // objects
      Movie = hd5file.createGroup("/Movie");
      FormatVersion = Movie.createAttribute("FormatVersion", SeqH5string(), scalar);
      Creator = Movie.createAttribute("Creator", SeqH5string(), scalar);
      CreatorVersion = Movie.createAttribute("CreatorVersion", SeqH5string(), scalar);
      TimeCreated = Movie.createAttribute("TimeCreated", SeqH5string(), scalar);
      InstrumentName = Movie.createAttribute("InstrumentName", SeqH5string(), scalar);
      CellSN = Movie.createAttribute("CellSN", SeqH5string(), scalar);
      PaneId = Movie.createAttribute("PaneId", uint16(), scalar);
      PaneSetId = Movie.createAttribute("PaneSetId", uint16(), scalar);
      FrameOffset = Movie.createAttribute("FrameOffset", uint32(), dim2);
      FrameSize = Movie.createAttribute("FrameSize", uint32(), dim2);
      CameraConfigKey = Movie.createAttribute("CameraConfigKey", uint32(), scalar);

      Frames = Movie.createDataSet("Frames", int16(), framesSpace, framesPropList);
      FrameRate = Frames.createAttribute("FrameRate", float64(), scalar);
      Exposure = Frames.createAttribute("Exposure", float64(), scalar);
      Frames_StartIndex = Frames.createAttribute("StartIndex", uint64(), scalar);

      FrameTime = Movie.createDataSet("FrameTime", uint64(), timesSpace, timesPropList);
      AcqStartTime = FrameTime.createAttribute("AcquisitionStartTime", SeqH5string(), scalar);

      ErrorFrames = Movie.createDataSet("ErrorFrames", uint64(), timesSpace, timesPropList);

      DarkFrame = Movie.createDataSet("DarkFrame", float32(), frameSpace);
      DarkFrame_Exposure
              = DarkFrame.createAttribute("Exposure", float64(), scalar);

      DarkFrameStack = Movie.createDataSet("DarkFrameStack", int16(), darkFrameSpace, framesPropList);
      DarkFrameStack_StartIndex
              = DarkFrameStack.createAttribute("StartIndex", uint64(), scalar);

      DarkFrameExp = Movie.createDataSet("DarkFrameExp", float32(), timesSpace, timesPropList);
      Gain = Movie.createDataSet("Gain", float32(), frameSpace);
      ReadNoise = Movie.createDataSet("ReadNoise", float32(), frameSpace);

      AcquisitionXML = Movie.createDataSet("AcquisitionXML", SeqH5string(), scalar);

#ifdef OBSOLETE_EVENTS
      int v;
      eventEnumType.insert("Unknown", (v = EventType_e::Unknown, &v));
      eventEnumType.insert("HotStart", (v = EventType_e::HotStart, &v));
      eventEnumType.insert("LaserOn", (v = EventType_e::LaserOn, &v));
      eventEnumType.insert("LaserOff", (v = EventType_e::LaserOff, &v));

      eventPropList.setChunk(1, chunk_dims);
      eventPropList.setFillValue(eventEnumType, &fill_val);

      EventType = Movie.createDataSet("EventType", eventEnumType, eventSpace, eventPropList);
      EventTime = Movie.createDataSet("EventTime", uint64(), eventSpace, eventPropList);
#endif

      // write 'em
      FormatVersion << "1.0";
      Creator << chipClass_.toString();
      CreatorVersion << "1.0.0.0." + PacBio::Utilities::PerforceUtil::ChangeNumberAsString();
      TimeCreated << PacBio::Utilities::ISO8601::TimeString();
      PaneId << 1;
      PaneSetId << 1;
      FrameOffset << std::pair<uint32_t, uint32_t>(0, 0); // can be written over by user code
      FrameSize <<
      std::pair<uint32_t, uint32_t>(NROW, NCOL); // don't over write this, since it must match Frames dimensions

      SequelMovieFrame<float> darkFrame = SequelMovieFrame<float>(NROW, NCOL);
      darkFrame.SetDefaultValue(0.0F);
      SequelMovieFrame<float> gainFrame = SequelMovieFrame<float>(NROW, NCOL);
      gainFrame.SetDefaultValue(1.0F);
      SequelMovieFrame<float> noiseFrame = SequelMovieFrame<float>(NROW, NCOL);
      noiseFrame.SetDefaultValue(0.707F);

      DarkFrame << darkFrame;
      Gain << gainFrame;
      ReadNoise << noiseFrame;

      events.CreateForWrite(hd5file,GetPrimaryConfig().GetLaserNames());

      valid = true;

  }


#define COPY(x,type) { if ( src.x.getId()!=0 && x.getId() != 0) { type s; src.x >> s; x << s; }}

  void SequelMovieFileHDF5::CopyHeader(const SequelMovieFileHDF5& src)
  {
      LOCK_HDF5();

      COPY(FormatVersion, std::string);
      COPY(Creator, std::string);
      COPY(CreatorVersion, std::string);
      COPY(TimeCreated, std::string);
      COPY(InstrumentName, std::string);
      COPY(CellSN, std::string);
      COPY(PaneId, uint16_t);
      COPY(PaneSetId, uint16_t);
      COPY(FrameRate, double);
      COPY(Exposure, double);
      COPY(AcquisitionXML, std::string);

//      COPY(DarkFrameStack, uint64_t);
//      COPY(DarkFrame, std::string);
//      COPY(Gain, std::string);
//      COPY(ReadNoise, std::string);
  }
#undef COPY

#define COPY(x,y,type) { if ( src.y.getId()!=0 && x.getId() != 0) { type s; src.y >> s; x << s; }}
  void SequelMovieFileHDF5::CopyHeader(const SequelTraceFileHDF5& /*src*/)
  {
      //COPY(FormatVersion, std::string);
      //COPY(Creator, std::string);
      //COPY(CreatorVersion, std::string);
      //COPY(TimeCreated, DateCreated, std::string);
      //COPY(InstrumentName, RI_InstrumentName, std::string);
      //COPY(CellSN, MovieName, std::string);
      //COPY(PaneId, uint16_t);
      //COPY(PaneSetId, uint16_t);
      //COPY(FrameRate, FrameRate, double);
      //COPY(Exposure, double);
      //COPY(AcquisitionXML, std::string);

      //COPY(DarkFrameStack, uint64_t);
      //COPY(DarkFrame, std::string);
      //COPY(Gain, std::string);
      //COPY(ReadNoise, std::string);
  }

  void SequelMovieFileHDF5::CopyHeader(const SequelMovieFileBase* src)
  {
      if (src->Type() == SequelMovieFileBase::SequelMovieType::Frame)
      {
          CopyHeader(*dynamic_cast<const SequelMovieFileHDF5*>(src));
      }
      else if (src->Type() == SequelMovieFileBase::SequelMovieType::Trace)
      {
          CopyHeader(*dynamic_cast<const SequelTraceFileHDF5*>(src));
      }
      else
      {
          throw PBException("not supported");
      }
  }

  /// add Chunk (up to 512 frames) to the movie
  uint32_t SequelMovieFileHDF5::WriteChunkHeader(const PacBio::Primary::Tile* chunkHeader, uint32_t numFrames)
  {
//      SequelMovieFileBase::AddChunk(chunkHeader, numFrames);

      PBLOG_DEBUG << "*** SequelMovieFileHDF5::WriteChunkHead(" << (void*) chunkHeader << ",numFrames=" << numFrames;

      if (selectedFramesPerChunk.size() == 0)
      {
          if (numFrames > chunkHeader->NumFramesInHeader())
          {
              numFrames = chunkHeader->NumFramesInHeader();
          }
      }
      else
      {
          numFrames = selectedFramesPerChunk.size();
          if (chunkHeader->NumFramesInHeader() != Tile::NumFrames)
          {
              throw PBException("chunk size must be 512 frames long to use the SelecteFrames feature, was " + std::to_string(chunkHeader->NumFramesInHeader()));
          }
      }

      PBLOG_DEBUG << "***    numFrames=" << numFrames;

      uint64_t NFRAME_start = NFRAMES;

      NFRAMES += numFrames;

      LOCK_HDF5();

      // extend the file
      while (NFRAMES > NFRAMES_allocated)
      {
          NFRAMES_allocated += chunkSize;
          hsize_t newSize[3];
          newSize[0] = NFRAMES_allocated;
          newSize[1] = NROW;
          newSize[2] = NCOL;

          Frames.extend(newSize);
          FrameTime.extend(newSize);

      }

      // write timestamps
      if (selectedFramesPerChunk.size() == 0)
      {
          uint64_t deltaTime;
          if (chunkHeader->NumFramesInHeader() < 2)
              deltaTime = 0;
          else
              deltaTime = (chunkHeader->LastFrameTimeStamp() - chunkHeader->FirstFrameTimeStamp()) /
                          (chunkHeader->NumFramesInHeader() - 1);
          uint64_t timestamp = chunkHeader->FirstFrameTimeStamp();
          for (uint64_t frameOffset = NFRAME_start; frameOffset < NFRAMES; frameOffset++, timestamp += deltaTime)
          {
              hsize_t offset[1];
              offset[0] = frameOffset;

              H5::DataSpace memFrameTime(1, hs_size);
              H5::DataSpace fspace2 = FrameTime.getSpace();
              fspace2.selectHyperslab(H5S_SELECT_SET, hs_size, offset);
              FrameTime.write(&timestamp, H5::PredType::NATIVE_UINT64, memFrameTime, fspace2);
          }

          //
          if (!cameraConfigKeySet)
          {
              cameraConfigKey = chunkHeader->FirstFrameConfig();
              CameraConfigKey << cameraConfigKey;

              frames_StartIndex = chunkHeader->FirstFrameIndex();
              Frames_StartIndex << frames_StartIndex;
              cameraConfigKeySet = true;
          }
      }
      return numFrames;
  }

  /// clear the the list of selected frames. the frame selection can be used to select out only certain frames
  /// from a chunk.
  void SequelMovieFileHDF5::ClearSelectedFrames()
  {
      selectedFramesPerChunk.resize(0);
  }

  /// add the offset to the frame selection. the frame selection can be used to select out only certain frames
  /// from a chunk.
  void SequelMovieFileHDF5::AddSelectedFrame(int i)
  {
      if (i <= -(int) PacBio::Primary::Tile::NumFrames || i >= (int) PacBio::Primary::Tile::NumFrames)
      {
          throw PBException(
                  "AddSelectedFrame frame offset must be between -PacBio::Primary::Tile::NumFrames and PacBio::Primary::Tile::NumFrames, was " +
                  std::to_string(i));
      }
      if (i < 0) i += PacBio::Primary::Tile::NumFrames;
      selectedFramesPerChunk.push_back(i);
  }


  /// write the chunk buffer to the file
  void SequelMovieFileHDF5::WriteChunkBuffer(const ChunkBuffer* bigBuffer, const uint32_t numFrames)
  {
      LOCK_HDF5();
      {
          PBLOG_DEBUG << "WriteChunkBuffer(" << (void*)bigBuffer << ", numFrames:" << numFrames << ")" <<
                  " selectedFramesPerChunk.size():" << selectedFramesPerChunk.size() <<
                  " NFRAMES:" << NFRAMES << std::endl;

          const Tile* headerTile(bigBuffer->HeaderTile());
          if (selectedFramesPerChunk.size() == 0)
          {
              uint32_t framesToWrite = numFrames;
              if (framesToWrite > headerTile->NumFramesInHeader())
              {
                  framesToWrite = headerTile->NumFramesInHeader();
              }
              if (framesToWrite > NFRAMES)
              {
                  throw PBException("Can't write " + std::to_string(framesToWrite) + " to movie file allocated to " + std::to_string(NFRAMES) + " frames");
              }

              if (roi_->Everything())
              {
                  hsize_t offset[3];
                  offset[0] = NFRAMES - framesToWrite;
                  offset[1] = 0;
                  offset[2] = 0;

                  hsize_t size[3];
                  size[0] = framesToWrite;
                  size[1] = NROW;
                  size[2] = NCOL;

                  H5::DataSpace fspace1 = Frames.getSpace();
                  fspace1.selectHyperslab(H5S_SELECT_SET, size, offset);

                  H5::DataSpace memTile(3, size);
                  Frames.write(bigBuffer->Data(0), H5::PredType::INTEL_I16, memTile, fspace1);
              }
              else if (roi_->Type() == SequelROI::ROI_Type_e::Rectangular)
              {
                  const SequelRectangularROI* rectROI = static_cast<const SequelRectangularROI*>(roi_.get());
                  // this is where the frames will be written to the file.
                  // rewind a few frames from the allocate end.
                  hsize_t destOffset[3];
                  destOffset[0] = NFRAMES - framesToWrite;
                  destOffset[1] = 0;
                  destOffset[2] = 0;

                  // this is where the frames are stored in the big buffer memory.
                  hsize_t srcOffset[3];
                  srcOffset[0] = 0;
                  srcOffset[1] = rectROI->RelativeRowPixelMin();
                  srcOffset[2] = rectROI->RelativeColPixelMin();

                  // this is the size of the frames in big buffer memory. it basically
                  // sets the stride correctly for the hyperslab mapping.
                  hsize_t srcSize[3];
                  srcSize[0] = PacBio::Primary::Tile::NumFrames;
                  srcSize[1] = rectROI->SensorROI().PhysicalRows();
                  srcSize[2] = rectROI->SensorROI().PhysicalCols();

                  // this is the reduced size that is copied from the src to dest
                  hsize_t hyperSlabSize[3];
                  hyperSlabSize[0] = framesToWrite;
                  hyperSlabSize[1] = NROW;
                  hyperSlabSize[2] = NCOL;

                  if (rectROI->RelativeRowPixelMin() + static_cast<int>(NROW) > rectROI->RelativeRowPixelMax()) throw PBException("row too big");
                  if (rectROI->RelativeColPixelMin() + static_cast<int>(NCOL) > rectROI->RelativeColPixelMax()) throw PBException("col too big");
                  //std::cout << "writing: src" << memOffset[0] << "," << memOffset[1] << "," << memOffset[2];
                  //std::cout << " " << size[0] << "," << size[1] << "," << size[2];
                  //std::cout <<"  dst: "  << offset[0] << "," << offset[1] << "," << offset[2] << " LLL " << numFrames << std::endl;


                  H5::DataSpace fileSpace = Frames.getSpace();
#if 0
                  if(hyperSlabSize[0] != 512) throw PBException("bad hyperSlabSize[0] " + std::to_string(hyperSlabSize[0]));
                  if(hyperSlabSize[1] != 64) throw PBException("bad hyperSlabSize[1] " + std::to_string(hyperSlabSize[1]));
                  if(hyperSlabSize[2] != 128) throw PBException("bad hyperSlabSize[2] " + std::to_string(hyperSlabSize[2]));
                  if(destOffset[0] != 0 && destOffset[0] != 512 ) throw PBException("bad destOffset[0] " + std::to_string(destOffset[0]));
                  if(destOffset[1] !=0 ) throw PBException("bad destOffset[1] " + std::to_string(destOffset[1]));
                  if(destOffset[2] !=0 ) throw PBException("bad destOffset[2] " + std::to_string(destOffset[2]));
#endif
                  fileSpace.selectHyperslab(H5S_SELECT_SET, hyperSlabSize, destOffset);

                  H5::DataSpace memTileSpace(3, srcSize);
                  memTileSpace.selectHyperslab(H5S_SELECT_SET, hyperSlabSize, srcOffset);

                  Frames.write(bigBuffer->Data(0), H5::PredType::INTEL_I16, memTileSpace, fileSpace);
              }
              else
              {
                  throw PBException("Cant write to selecting ROI type" + roi_->Type().toString());
              }
          }
          else
          {
              if (!roi_->Everything())throw PBException("Not implemented");
              if (numFrames > selectedFramesPerChunk.size())
              {
                  throw PBException("Can't write more frames + " + std::to_string(numFrames) + " than selected " + std::to_string(selectedFramesPerChunk.size()));
              }
              hsize_t offset[3];
              offset[0] = NFRAMES - numFrames;
              offset[1] = 0;
              offset[2] = 0;

              hsize_t size[3];
              size[0] = 1;
              size[1] = NROW;
              size[2] = NCOL;

              H5::DataSpace fspace1 = Frames.getSpace();
              H5::DataSpace memTile(3, size);
              for (uint32_t index : selectedFramesPerChunk)
              {
                  if (index < Tile::NumFrames)
                  {
                      fspace1.selectHyperslab(H5S_SELECT_SET, size, offset);
                      Frames.write(bigBuffer->Data(0) + sizeof(int16_t) * NROW * NCOL * index, H5::PredType::INTEL_I16,
                                   memTile, fspace1);
                      offset[0]++;
                  }
                  else
                  {
                      PBLOG_WARN << " selected index " << index << " is out of range 0:" << Tile::NumFrames << std::endl;
                  }
              }
          }
          std::vector<uint64_t> errors = headerTile->ErroredFrames();
          if (errors.size())
          {
              if (errorFrameCount < 1000)
              {
                  hsize_t fileSize[1];
                  fileSize[0] = errors.size() + errorFrameCount;
                  ErrorFrames.extend(fileSize);

                  hsize_t fileOffset[1];
                  fileOffset[0] = errorFrameCount;

                  hsize_t memSize[1];
                  memSize[0] = errors.size();

                  H5::DataSpace memSpace(1, memSize);
                  H5::DataSpace fileSpace = ErrorFrames.getSpace();
                  fileSpace.selectHyperslab(H5S_SELECT_SET, memSize, fileOffset);
                  ErrorFrames.write(&errors[0], H5::PredType::NATIVE_UINT64, memSpace, fileSpace);

                  errorFrameCount = fileSize[0];
              }
              else
              {
                  PBLOG_WARN << "Too many bad frames, truncating at 1000";
              }
          }
      }
      //std::cout << "WriteChunkBuffer() DONE bigBuffer:" << (void*)bigBuffer << " numFrame:" << numFrames << std::endl;
  }

  // add a single frame to the movie.
  uint64_t SequelMovieFileHDF5::AddFrame(const SequelMovieFrame<int16_t>& frame)
  {
      LOCK_HDF5();
      {
          hsize_t offset[3];
          offset[0] = NFRAMES;
          offset[1] = 0;
          offset[2] = 0;

          NFRAMES++;

          if (NFRAMES >= NFRAMES_allocated)
          {
              NFRAMES_allocated += chunkSize;
              hsize_t newSize[3];
              newSize[0] = NFRAMES_allocated;
              newSize[1] = NROW;
              newSize[2] = NCOL;

              Frames.extend(newSize);
              FrameTime.extend(newSize);
          }
          H5::DataSpace fspace1 = Frames.getSpace();
          fspace1.selectHyperslab(H5S_SELECT_SET, hs_size, offset);
          Frames.write(frame.data, H5::PredType::INTEL_I16, memFrame, fspace1);

          uint64_t timestamp = frame.timestamp;
          H5::DataSpace memFrameTime(1, hs_size);
          H5::DataSpace fspace2 = FrameTime.getSpace();
          fspace2.selectHyperslab(H5S_SELECT_SET, hs_size, offset);
          FrameTime.write(&timestamp, H5::PredType::NATIVE_UINT64, memFrameTime, fspace2);

          if (!cameraConfigKeySet)
          {
              cameraConfigKey = frame.cameraConfiguration;
              CameraConfigKey << cameraConfigKey;

              frames_StartIndex = frame.index;
              Frames_StartIndex << frames_StartIndex;
              cameraConfigKeySet = true;

          }

#if 0
            if (flushCycle == 0)
            {
				//Frames.flush(H5F_SCOPE_LOCAL);
				hd5file.flush(H5F_SCOPE_GLOBAL);
				int* fd;
				hd5file.getVFDHandle((void**)&fd);
				if (*fd > 0 && *fd<20) {
					fsync(*fd);
				} else {
					std::cout << "fd is funny: " << *fd << std::endl;
				}
            }
            flushCycle++;
            if (flushCycle >= 10) flushCycle = 0;
#endif

          return hs_size[0] * hs_size[1] * hs_size[2] * sizeof(int16_t);
      }
  }

  /// add a dark frame to the movie
  uint64_t SequelMovieFileHDF5::AddDarkFrame(float exposure, const SequelMovieFrame<int16_t>& frame)
  {
      LOCK_HDF5();
      {
          hsize_t offset[3];
          offset[0] = NDARKFRAMES;
          offset[1] = 0;
          offset[2] = 0;

          NDARKFRAMES++;

          hsize_t newSize[3];
          newSize[0] = NDARKFRAMES;
          newSize[1] = NROW;
          newSize[2] = NCOL;


          DarkFrameStack.extend(newSize);
          H5::DataSpace fspace1 = DarkFrameStack.getSpace();
          fspace1.selectHyperslab(H5S_SELECT_SET, hs_size, offset);
          DarkFrameStack.write(frame.data, H5::PredType::INTEL_I16, memFrame, fspace1);

          H5::DataSpace memFrameTime(1, hs_size);
          DarkFrameExp.extend(newSize);
          H5::DataSpace fspace2 = DarkFrameExp.getSpace();
          fspace2.selectHyperslab(H5S_SELECT_SET, hs_size, offset);
          DarkFrameExp.write(&exposure, H5::PredType::NATIVE_FLOAT, memFrameTime, fspace2);

          if (NDARKFRAMES <= 1)
          {
              // write index and timestamp ONLY on first frame of movie (i.e. NDARKFRAMES==1)
              DarkFrameStack_StartIndex << frame.index;
              auto x = DarkFrameStack.createAttribute("TimeStamp",uint64(),scalar);
              x << frame.timestamp;
          }
          return hs_size[0] * hs_size[1] * hs_size[2] * sizeof(int16_t);

      }
  }

#ifdef OBSOLETE_EVENTS
/// add an event to the movie
  void SequelMovieFileHDF5::AddEvent(SequelMovieFileBase::EventType_e type, uint64_t timestamp)
  {
      LOCK_HDF5();
      {
          hsize_t offset[1];
          offset[0] = NUM_EVENTS;

          NUM_EVENTS++;

          hsize_t newSize[1];
          newSize[0] = NUM_EVENTS;

          EventType.extend(newSize);
          EventTime.extend(newSize);

          H5::DataSpace memSpace(1, hs_size);
          H5::DataSpace fileSpace = EventType.getSpace();

          fileSpace.selectHyperslab(H5S_SELECT_SET, hs_size, offset);
          EventType.write(&type, eventEnumType, memSpace, fileSpace);
          EventTime.write(&timestamp, H5::PredType::NATIVE_UINT64, memSpace, fileSpace);
      }
  }
#endif

  void SequelMovieFileHDF5::SetMeasuredFrameRate(double rate)
  {
      LOCK_HDF5();
      FrameRate << rate;
  }

/// read  an existing frame from the movie
uint64_t SequelMovieFileHDF5::ReadFrame(uint64_t n, SequelMovieFrame<int16_t>& frame) const
{
    return ReadFrame(n,frame, ReadFrame_NoOptions);
}

uint64_t SequelMovieFileHDF5::ReadFrame(uint64_t n, SequelMovieFrame <int16_t>& frame, int options) const
  {
      if (static_cast<uint64_t>(n) >= NFRAMES)
      {
            throw PBException(
                    "Requested frame " + std::to_string(n) + " is greater than NFRAMES = " + std::to_string(NFRAMES));
      }

      // check if any bits that are not supported are set.
      if (options & ~(ReadFrame_DontResize))
      {
          std::stringstream ss;
          ss << "Illegal option bits set: 0x"  << std::hex << options;
          throw PBException(ss.str());
      }

      if (frame.NROW == 0 && frame.NCOL == 0)
      {
          frame.Resize(NROW, NCOL);
      }
      if ((frame.NROW != NROW || frame.NCOL != NCOL) && !(ReadFrame_DontResize & options))
      {
          PBLOG_ERROR << "frame.NROW:" << frame.NROW << " NROW:" << NROW;
          PBLOG_ERROR << "frame.NCOL:" << frame.NCOL << " NCOL:" << NCOL;
          throw PBException("row/col mismatch.");
      }

      LOCK_HDF5();
      {

          const hsize_t offset[3] = {n,0,0};

          // read the frame pixel data
          const hsize_t destinationSize[3] = {1,frame.NROW, frame.NCOL};
          H5::DataSpace destinationSpace = H5::DataSpace(3, destinationSize);
          H5::DataSpace sourceSpace = Frames.getSpace();
          sourceSpace.selectHyperslab(H5S_SELECT_SET, destinationSize, offset);
          Frames.read(frame.data, H5::PredType::INTEL_I16, destinationSpace, sourceSpace);

          // read the frame timestamp (1 value)
          H5::DataSpace destinationFrameTime(1, hs_size);
          H5::DataSpace sourceSpace2 = FrameTime.getSpace();
          sourceSpace2.selectHyperslab(H5S_SELECT_SET, hs_size, offset);
          FrameTime.read(&frame.timestamp, H5::PredType::NATIVE_UINT64, destinationFrameTime, sourceSpace2);

          frame.index = frames_StartIndex + n;

          if (cameraConfigKeySet)
          {
              frame.cameraConfiguration = cameraConfigKey;
          }
          return destinationSize[0] * destinationSize[1] * destinationSize[2] * sizeof(int16_t);
      }
  }



#ifdef OBSOLETE_EVENTS
  void SequelMovieFileHDF5::ReadEvent(int64_t n, EventType_e& type, uint64_t& timestamp) const
  {
      LOCK_HDF5();
      {
          hsize_t offset[1];
          offset[0] = n;

          H5::DataSpace memSingle(1, hs_size);
          H5::DataSpace fspace = EventType.getSpace();
          fspace.selectHyperslab(H5S_SELECT_SET, hs_size, offset);

          uint16_t value;
          EventType.read(&value, H5::PredType::INTEL_U16, memSingle, fspace);
          EventTime.read(&timestamp, H5::PredType::NATIVE_UINT64, memSingle, fspace);
          type = static_cast<EventType_e::RawEnum>(value);
      }
  }

  std::vector <SequelMovieFileHDF5::Event> SequelMovieFileHDF5::GetEvents()
  {
      std::vector <Event> l(NUM_EVENTS);
      for (uint32_t i = 0; i < NUM_EVENTS; i++)
      {
          Event& e = l[i];
          ReadEvent(i, e.type, e.timestamp);
      }
      return l;
  }
#endif

  void SequelMovieFileHDF5::ReadDarkFrameStack(int64_t n, SequelMovieFrame<int16_t>& frame, float& exposure)
  {
      if (frame.NROW == 0 && frame.NCOL == 0)
      {
          frame.Resize(NROW, NCOL);
      }
      if (frame.NROW != NROW || frame.NCOL != NCOL)
      {
          PBLOG_ERROR << "frame.NROW:" << frame.NROW << " NROW:" << NROW;
          PBLOG_ERROR << "frame.NCOL:" << frame.NCOL << " NCOL:" << NCOL;
          throw PBException("row/col mismatch");
      }

      LOCK_HDF5();
      {

          hsize_t offset[3];
          offset[0] = n;
          offset[1] = 0;
          offset[2] = 0;

          H5::DataSpace fspace1 = DarkFrameStack.getSpace();
          fspace1.selectHyperslab(H5S_SELECT_SET, hs_size, offset);
          DarkFrameStack.read(frame.data, H5::PredType::INTEL_I16, memFrame, fspace1);
          //frame.timestamp = ? ; fix me

          H5::DataSpace memSingle(1, hs_size);
          H5::DataSpace fspace2 = DarkFrameExp.getSpace();
          fspace2.selectHyperslab(H5S_SELECT_SET, hs_size, offset);

          DarkFrameExp.read(&exposure, H5::PredType::NATIVE_FLOAT, memSingle, fspace2);

          if (n == 0)
          {
              DarkFrameStack_StartIndex >> frame.index;
              if (DarkFrameStack.attrExists("TimeStamp"))
              {
                  auto x = DarkFrameStack.openAttribute("TimeStamp");
                  x >> frame.timestamp;
              }
          }
      }
  }


  uint64_t SequelMovieFileHDF5::VerifyTest(uint64_t maxErrorsReported, double frameRate)
  {
      uint32_t timestamp = 0;
      uint64_t errors = 0;
      const bool checkTimeStamp = (frameRate != 0.0);

      const uint64_t periodMicroseconds = (frameRate!=0) ? static_cast<uint64_t>(std::round(1e6/frameRate)) : 0;

      SequelMovieFrame<int16_t> frame(NROW, NCOL);
      int nn = 0;
      for (uint32_t n = 0; n < NFRAMES; n++)
      {
          if (!(n & 0x3ff)) std::cout << n << std::endl;

          ReadFrame(n, frame);

          bool checkit = false;

          if (checkTimeStamp)
          {
              // there appears to be a bug somewhere in the data path where the timestamp gets truncated to 32 bits
              uint32_t ftimestamp = frame.timestamp & 0xFFFFFFFFULL;

              if (ftimestamp == timestamp)
              {
                  //std::cout << "OK!" << n << " " << ftimestamp << std::endl;
                  checkit = true;
              }
              else
              {
                  if (errors < maxErrorsReported)
                  {
                      std::cout << "Skipped Frame n:" << n << " timestamp:" << std::dec << ftimestamp << " expected:" <<
                              timestamp << std::endl;
                  }

                  int skipped;
                  for (skipped = 0;
                       timestamp != ftimestamp && skipped < 100; skipped++, timestamp += periodMicroseconds);
                  if (skipped >= 100)
                  {
                      std::cout << "corrupted timestamp" << std::endl;
                  }
                  else
                  {
                      errors += skipped;
                      nn += skipped;
                      timestamp = ftimestamp;
                      checkit = true;
                  }
              }
          }
          else
          {
              checkit = true;
          }

          if (checkit)
          {
              for (uint32_t i = 0; i < NROW; i++)
              {
                  for (uint32_t j = 0; j < NCOL; j++)
                  {
                      int16_t actual = static_cast<int16_t>(frame.data[i * NCOL + j] & 0x3FF);
                      int16_t expected;
                      if (checkTimeStamp)
                      {
                          expected = static_cast<int16_t>((100 * nn + i * 10 + j) & 0x3FF);
                      }
                      else
                      {
                          expected = static_cast<int16_t>((100 * n + i * 10 + j) & 0x3FF);
                      }
                      if (actual != expected)
                      {
                          if (errors < maxErrorsReported)
                          {
                              std::cout << "Frame:" << n  << " Pixel:[" << i << "," << j << "] = " << actual << "!="
                                      << expected << std::endl;
                          }
                          errors++;
                      }
                  }
              }
          }
          timestamp += periodMicroseconds;
          nn++;
      }
      std::cout << "errors: " << errors << std::endl;
      if (errors >= maxErrorsReported)
      {
          std::cout << " an additional " << (errors - maxErrorsReported ) << " we omitted from this report!" << std::endl;
      }
      return errors;
  }

  void SequelMovieFileHDF5::CreateTest(uint32_t nframes, uint32_t nDarkFrames, int bits, bool correctPeriod)
  {
      double frameRate = 200.0;

      // initialize data
      InstrumentName << "simulator";
      CellSN << "n/a";
      FrameRate << frameRate;
      Exposure << 1 / frameRate;
      DarkFrame_Exposure << 1 / 201.1;
      DarkFrameStack_StartIndex << 0x000000000000ULL;
      Frames_StartIndex << 0x010203040506ULL;
      AcqStartTime << PacBio::Utilities::ISO8601::TimeString();
      FrameOffset << std::pair<uint32_t, uint32_t>(0, 0);
      AcquisitionXML << "<foo>Hello Elias</foo>";

      uint16_t mask = static_cast<uint16_t>((1 << bits) - 1);
      // the correct units for the timestamp is microseconds, but in previous code it was assumed to be picoseconds.
      const uint64_t periodMicroseconds = correctPeriod?
                                          static_cast<uint64_t>(std::round(1e6/frameRate)) :
                                          5000000ULL;


      SequelMovieFrame<int16_t> frame(NROW, NCOL);
      for (uint32_t n = 0; n < nframes; n++)
      {
          for (uint32_t i = 0; i < NROW; i++)
              for (uint32_t j = 0; j < NCOL; j++)
                  frame.data[i * NCOL + j] = static_cast<int16_t>((100 * n + i * 10 + j) & mask);

          frame.timestamp = static_cast<uint64_t>(n) * periodMicroseconds;
          frame.index = n;
          frame.cameraConfiguration = 0xDEADBEEF;
          this->AddFrame(frame);
      }


      for (uint32_t n = 0; n < nDarkFrames; n++)
      {
          for (uint32_t i = 0; i < NROW; i++)
              for (uint32_t j = 0; j < NCOL; j++)
                  frame.data[i * NCOL + j] = static_cast<int16_t>(n & mask);
          frame.index = n;
          frame.cameraConfiguration = 0xDEADBEEF;
          this->AddDarkFrame(static_cast<float>(pow(0.1, n)), frame);
      }
  }


  void SequelMovieFileHDF5::CreateTestBeta(uint32_t nframes, uint32_t /*nDarkFrames*/, int bits)
  {
      // initialize data
      InstrumentName << "CreateTestBeta";
      CellSN << "n/a";
      FrameRate << Sequel::defaultFrameRate;
      Exposure << 1 / Sequel::defaultFrameRate;
      DarkFrame_Exposure << 1 / Sequel::defaultFrameRate;
      DarkFrameStack_StartIndex << 0x000000000000ULL;
      Frames_StartIndex << 0x010203040506ULL;
      AcqStartTime << PacBio::Utilities::ISO8601::TimeString();
      FrameOffset << std::pair<uint32_t, uint32_t>(0, 0);
      AcquisitionXML << "<foo></foo>";

      const uint16_t mask = static_cast<uint16_t>((1 << bits) - 1);


      SequelMovieFrame<int16_t> frame(NROW, NCOL);
      for (uint32_t n = 0; n < nframes; n++)
      {
          for (uint32_t i = 0; i < NROW; i++)
              for (uint32_t j = 0; j < NCOL; j++)
              {
                  frame.data[i * NCOL + j] = static_cast<int16_t>(PixelPatternBeta(n,i,j,mask));
              }

          frame.timestamp = static_cast<uint64_t>(n) * 5000000ULL;
          frame.index = n;
          frame.cameraConfiguration = 0xDEADBEEF;

          this->AddFrame(frame);
      }
  }

  void SequelMovieFileHDF5::CreateDeltaPattern(uint32_t nframes, uint32_t /*nDarkFrames*/, int bits, int spacing)
  {
      // initialize data
      InstrumentName << "simulator";
      CellSN << "n/a";
      FrameRate << 200.0;
      Exposure << 1 / 200.0;
      DarkFrame_Exposure << 1 / 201.1;
      DarkFrameStack_StartIndex << 0x000000000000ULL;
      Frames_StartIndex << 0x010203040506ULL;
      AcqStartTime << PacBio::Utilities::ISO8601::TimeString();
      FrameOffset << std::pair<uint32_t, uint32_t>(0, 0);
      AcquisitionXML << "<foo>Hello Elias</foo>";

      uint16_t value = static_cast<uint16_t>(1 << (bits - 1));

      SequelMovieFrame<int16_t> frame(NROW, NCOL);

      for (uint32_t n = 0; n < nframes; n++)
      {
          memset(frame.data, 0, sizeof(int16_t) * NROW * NCOL);
          for (uint32_t i = n % spacing; i < NROW; i += spacing)
              for (uint32_t j = n % spacing; j < NCOL; j += spacing)
                  frame.data[i * NCOL + j] = value;

          frame.timestamp = static_cast<uint64_t>(n) * 5000000ULL;
          frame.index = n;
          frame.cameraConfiguration = 0xDEADBEEF;

          this->AddFrame(frame);
      }
  }

/// Creates a new movie of `nframes` frames, using the functor to generate the test pattern based on row, column and
/// frame index.
  void SequelMovieFileHDF5::CreateFrameTestPattern(uint32_t nframes, std::function<int16_t(uint32_t,uint32_t,uint32_t)> pattern)
  {
      // initialize data
      InstrumentName << "simulator";
      CellSN << "n/a";
      FrameRate << 80.0;
      Exposure << 1 / 80.0;
      uint64_t deltaTime = 1000000ULL/80.0;
      DarkFrame_Exposure << 1 / 80.1;
      DarkFrameStack_StartIndex << 0x000000000000ULL;
      Frames_StartIndex << 0;
      AcqStartTime << PacBio::Utilities::ISO8601::TimeString();
      FrameOffset << std::pair<uint32_t, uint32_t>(0, 0);
      AcquisitionXML << "<foo>Hello Elias</foo>";

      SequelMovieFrame<int16_t> frame(NROW, NCOL);
      for (uint32_t n = 0; n < nframes; n++)
      {
          for (uint32_t i = 0; i < NROW; i++)
              for (uint32_t j = 0; j < NCOL; j++)
                  frame.data[i * NCOL + j] = pattern(i,j,n);

          frame.timestamp = static_cast<uint64_t>(n) * deltaTime;
          frame.index = n;
          frame.cameraConfiguration = 0xDEADBEEF;

          this->AddFrame(frame);
      }
  }

  void SequelMovieFileHDF5::DumpSummary(std::ostream& s, uint32_t frames , uint32_t rows , uint32_t cols )
  {
      auto p = s.precision();
      s.precision(100);
      s << FormatVersion;
      s << Creator;
      s << CreatorVersion;
      s << TimeCreated;
      s << InstrumentName;
      s << CellSN;
      s << PaneId;
      s << PaneSetId;
      s << CameraConfigKey;
      s << FrameOffset;
      s << FrameSize;

      s << "Frames/" << FrameRate;
      s << "Frames/" << Exposure;
      s << "Frames/" << Frames_StartIndex;

      s << "FrameTime/" << AcqStartTime;

      s << "DarkFrame/" << DarkFrame_Exposure;
      s << "DarkFrame/" << DarkFrameStack_StartIndex;
      s << AcquisitionXML;

      s << "Dimensions "
      << NFRAMES << " x "
      << NROW << " x "
      << NCOL << std::endl;


      SequelMovieFrame<int16_t> frame(NROW, NCOL); // buffer used to load data

      s << "\nFrames DATASET:  ----------\n" << " (limited to " << frames << ")\n";
      for (uint32_t n = 0; n < NFRAMES; n++)
      {
          if (n < frames || n > NFRAMES - frames)
          {
              this->ReadFrame(n, frame);
              s << "Frame #:" << n << "\n";

              frame.DumpSummary(s, "  ", rows, cols);

              s << std::endl;
          }
      }

      SequelMovieFrame<float> fframe(NROW, NCOL);
      Gain >> fframe;
      s << "Gain: ---------------\n";
      fframe.DumpSummary(s, "  ");

#ifdef OBSOLETE_EVENTS
      s << "Events: -------------\n";
      auto events = GetEvents();
      for (auto&& event :events)
      {
          s << " @ " << event.timestamp << " -> " << event.GetEventTypeName() << "(" << static_cast<uint64_t>(event.type) <<
          ")\n";
      }
#endif
      s.precision(p);
  }

Json::Value SequelMovieFileHDF5::DumpJson(uint32_t /*frames*/, uint32_t /*rows*/, uint32_t /*cols*/)
{
    throw PBException("Not supported");
}

SequelMovieFileBase::SequelMovieType SequelMovieFileHDF5::Type() const
  {
      return SequelMovieType::Frame;
  }

  std::string SequelMovieFileHDF5::TypeName() const
  {
      return "Frame";
  }

  int SequelMovieFileHDF5::CompareFrames(const SequelMovieFileHDF5& b, bool dataOnly)
  {
      int errors = 0;
      errors += Compare(this->NROW, b.NROW, "file Row count mismatch");
      errors += Compare(this->NCOL, b.NCOL, "file Col count mismatch");
      if (errors)
      {
          std::cout << "Can not continue comparison" << std::endl;
          return errors;
      }
      errors += Compare(this->NFRAMES, b.NFRAMES, "Frame count mismatch");

      SequelMovieFrame<int16_t> aFrame(NROW, NCOL); // buffer used to load data
      SequelMovieFrame<int16_t> bFrame(NROW, NCOL); // buffer used to load data
      for (uint32_t n = 0; n < NFRAMES; n++)
      {
          this->ReadFrame(n, aFrame);
          b.ReadFrame(n, bFrame);
          errors += aFrame.Compare(bFrame, 0, dataOnly);
      }
      std::cout << "Total errors: " << errors << std::endl;
      return errors;
  }

void SequelMovieFileHDF5::AddLaserPowerEvent(const EventObject& eo)
{
    if (eo.eventType() != EventObject::EventType::laserpower)
    {
          throw PBException("AddLaerPowerEvent called with event that is not a laserpower change");
    }
#ifdef OBSOLETE_EVENTS
    SequelMovieFileBase::EventType_e type = SequelMovieFileBase::EventType_e::LaserPowerChange;
    uint64_t timestamp = eo.timestamp_epoch() * 1e9;
    AddEvent(type, timestamp);
#else
    events.laserPowerChanges.AddEvent(eo);
#endif
}

const SequelMovieConfig::MovieChunking& SequelMovieFileHDF5::GetChunking() const
{
      return config_.movie.chunking;
}

}}
