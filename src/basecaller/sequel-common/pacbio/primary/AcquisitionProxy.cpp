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
/// \brief  A proxy for the pa-acq server which generates tranches. This generates
///  tranches from "chunk" .int16 files or .trc.h5 files, or just zeros.
//
// Programmer: Mark Lakata

#include <pacbio/logging/Logger.h>
#include <pacbio/text/String.h>

#include <pacbio/primary/Acquisition_Setup.h>
#include <pacbio/primary/AcquisitionProxy.h>
#include <pacbio/primary/Tranche.h>
#include <pacbio/primary/SequelDefinitions.h>
#include <pacbio/primary/ChipLayoutRTO3.h>
#include <pacbio/primary/SmrtSensor.h>
#include <cmath>

namespace PacBio {
namespace Primary {

//
// AcquisitionProxy methods
//

AcquisitionProxy::AcquisitionProxy( size_t framesPerTranche)
    : numReplicatedLanes_(0)
    , numReplicatedSuperchunks_(0)
    , numSourceLanes_(1)
    , numSourceSuperchunks_(1)
    , framesPerTranche_(framesPerTranche)
    , laneIndex(0)
    , chunkIndex_(0)
    , frames_(163840)
    , timeStampDelta_(defaultExposureUsec)
    , configWord_(0x12345678)
    , cache_(false)
    , chipClass_(ChipClass::Sequel)
    , frameRate_(0)
{
    PBLOG_DEBUG << "AcquisitionProxy constructor: numReplicatedLanes_ " << numReplicatedLanes_
                << ", numReplicatedSuperchunks_ " << numReplicatedSuperchunks_ << ", IsTerminus: " << IsTerminusChunk()
                << " chipClass:" << chipClass_.toString();
}

AcquisitionProxy::~AcquisitionProxy()
{
    for (auto&& x : pixelCache_)
    {
        for (auto&& y : x)
        {
            delete y;
        }
    }
}

std::vector<LaserPowerChange> AcquisitionProxy::LaserPowerChanges() const
{
    // FIXME: Is there a source of laser power change events for this "base implementation"?
    return std::vector<LaserPowerChange>();
}


size_t AcquisitionProxy::FillNext(Tranche* tranche)
{
    if (numReplicatedSuperchunks_ == 0 || numReplicatedLanes_ == 0)
    {
        throw PBException("need to set numReplicatedSuperchunks_ and/or numReplicatedLanes_");
    }
    if (selectedLanes_.size() == 0)
    {
        throw PBException("SelectedLanes is empty. Either ROI was empty, or SetROI was not called.");
    }
    if (chunkIndex_ >= numReplicatedSuperchunks_) return 0;


    size_t nFrames = IsTerminusChunk()
                     ? (((frames_-1) % framesPerTranche_)+1)
                     : framesPerTranche_;

    uint32_t stridedLaneOffset = laneIndex * numMics_ + micOffset_;
    uint32_t stridedLane       = selectedLanes_[stridedLaneOffset];
    uint32_t wrappedSuperchunk = chunkIndex_ % numSourceSuperchunks_;
    uint32_t wrappedLane       = stridedLane % numSourceLanes_;

    switch(chipClass_) // this is messy
    {
    case ChipClass::Sequel:
        tranche->Create2C2ATitle();
        break;
    case ChipClass::Spider:
        tranche->Create1C4A_CA_Title();
        break;
    default:
        throw PBException("not handled");
    }

    PBLOG_TRACE << "AcquisitionProxy::FillNext(tranche:" << (void*)tranche << ") wrappedLane:" << wrappedLane <<
                    " wrappedSuperchunk:" << wrappedSuperchunk << " stridedLane:" << stridedLane <<
                    " laneIndex:" << laneIndex;

    uint32_t nFramesRead;
    if (cache_)
    {
        PBLOG_TRACE << "read from cache, super:" << wrappedSuperchunk << " lane:"<<wrappedLane << " stridedLaneOffset:"
                        <<stridedLaneOffset << " stridedLane:"<<stridedLane;
        assert(wrappedLane < pixelCache_[wrappedSuperchunk].size());
        tranche->AssignTraceDataPointer( pixelCache_[wrappedSuperchunk][wrappedLane]);
        nFramesRead = nFrames;
    }
    else
    {
        nFramesRead = LoadTranche(tranche->GetTraceDataPointer(), wrappedLane, wrappedSuperchunk);

        assert(nFramesRead > 0);
        if (nFramesRead < nFrames)
        {
            // repeat if not enough.
            int i = 0;
            auto* pixels = tranche->GetTraceDataPointer()->pixels.simd;
            while (nFramesRead < nFrames)
            {
                pixels[nFramesRead] = pixels[i];
                nFramesRead++;
                i++;
            }
        }
    }

    PBLOG_TRACE << " Read Trace lane" << laneIndex << " superchunk:" << chunkIndex_ << " frames:" << nFrames;

    AssignTrancheMetadata(tranche, nFrames);

    laneIndex++;
    if (laneIndex >= numReplicatedLanes_)
    {
        laneIndex = 0;
        chunkIndex_++;
    }

    return nFramesRead;
}

void AcquisitionProxy::AssignTrancheMetadata(Tranche* tranche, const size_t nFrames)
{
    uint32_t zmwLaneIndex = laneIndex;
    uint32_t stridedLaneOffset = laneIndex * numMics_ + micOffset_;
    uint32_t zmwIndex = zmwLaneIndex * zmwsPerTranche;
    uint32_t zmwNumber = zmwNumbers_[stridedLaneOffset];

    tranche->type = Tranche::MessageType::Data;
    TrancheTitle& title = tranche->Title();
    title.FrameCount(nFrames);
    title.SuperChunkIndex(chunkIndex_);
    uint64_t frameIndex = chunkIndex_ * framesPerTranche_;
    title.FrameIndexStart(frameIndex); // NOT nFrames!

    title.TimeStampDelta(timeStampDelta_);
    title.TimeStampStart(frameIndex * timeStampDelta_);
    title.ConfigWord(configWord_);

    tranche->StopStatus(IsTerminusChunk()
                     ? ITranche::StopStatusType::NORMAL
                     : ITranche::StopStatusType::NOT_STOPPED);
    tranche->ZmwIndex(zmwIndex);
    tranche->ZmwNumber(zmwNumber);
    tranche->ZmwLaneIndex(zmwLaneIndex);

    for (uint32_t i = 0; i < zmwsPerTranche; i++)
    {
        tranche->calls[i]->Reset(tranche->ZmwIndex() + i);
    }
}


void AcquisitionProxy::EnableCache()
{
    cache_ = false;

    pixelCache_.resize(numSourceSuperchunks_);
    for(uint32_t super=0; super < numSourceSuperchunks_; super++)
    {
        size_t cacheLanes = std::min(numSourceLanes_, numReplicatedLanes_);
        PBLOG_INFO << "Caching superchunk super:" << super << " lanes: 0 to "  << cacheLanes;
        auto& x = pixelCache_[super];
        x.resize(cacheLanes);
        for(uint32_t lane = 0; lane < cacheLanes; lane++)
        {
            Tranche::Pixelz* data = Tranche::Pixelz::Factory(framesPerTranche_);
            x[lane] = data;
            size_t framesRead = LoadTranche(data,lane,super);

            // Need to fill empty frames with something so we don't have uninitialized data.
            // This is necessary if we're tiling out in time at all.
            assert(framesRead > 0);
            int i = 0;
            auto* pixels = data->pixels.simd;
            while (framesRead < framesPerTranche_)
            {
                pixels[framesRead] = pixels[i];
                framesRead++;
                i++;
            }
//            TrancheHelper th(3);
//            std::cout << th << pixels;
        }
    }

    cache_ = true;
}

//
// AcquisitionProxy methods
//

  uint32_t AcquisitionProxy::LoadTranche(Tranche::Pixelz* data, uint32_t /*lane*/, uint32_t /*super*/)
  {
      memset( data, 0, Tranche::Pixelz::Sizeof(framesPerTranche_));
      chipClass_ = ChipClass::Sequel;
      return framesPerTranche_;
  }

  /// apply the given ROI to the selectedLanes vector.
  /// The ROI is enumerated over pixels. Every 32 pixels, a new selected lane is added
  /// to selectedLanes_. The lane counting convention starts with lane #0 at pixel (0,0),
  /// and increases along a row (+1 lane for every 32 pixels), then wraps around to the next row at the end of the
  /// physical (sensor) ROI.
  void AcquisitionProxy::SetROIUsingSourceROI(const SequelROI& maskROI, const PacBio::Primary::ChipLayout& layout, SequelROI* sourceROI)
  {

      PBLOG_DEBUG << "AcquisitionProxy::SetROI sourceROI: " << *sourceROI;
      outputROI_.reset(new SequelSparseROI(layout.GetSensorROI()));

      selectedLanes_.clear();
      size_t numZmws = sourceROI->CountZMWs();
      zmwNumbers_.clear();

      uint32_t lane = 0;
      // When reading from the trace file we always do 16 wide zmw lanes, 
      // regardless of spider vs sequel
      for (SequelROI::Enumerator e(*sourceROI, 0, numZmws); e; e += zmwsPerTranche)
      {
          if (maskROI.ContainsPixel(e.GetPixelCoord()))
          {
              selectedLanes_.push_back(lane);
              try
              {
                  outputROI_->AddZMW(e.GetPixelCoord());
              }
              catch(const std::exception& /*ex*/)
              {
                  PBLOG_ERROR << "While adding ZMW (rowPixel,colPixel)=" << e.GetPixelCoord() << " an "
                              " exception was thrown ... ";
                  throw;
              }
              e.GetPixelCoord();
              zmwNumbers_.push_back(layout.ConvertPixelCoordToUnitCell(e.GetPixelCoord()).Number());

              PBLOG_DEBUG << "Added row:" << e.GetPixelRow().Value() << ", col:" << e.GetPixelCol().Value() <<
              "->" << lane;
          }
          lane++;
      }
  }

  std::unique_ptr<SequelROI> AcquisitionProxy::GetSourceROI() const
  {
      SequelSparseROI* sroi;
      std::unique_ptr<SequelROI> roi(sroi = new SequelSparseROI(SequelSensorROI::SequelAlpha()));
      sroi->AddRectangle(0,0,1,2048);
      sroi->AddRectangle(1,0,1,1152);
      return roi;
  }

  uint32_t AcquisitionProxy::GetSourceFrames() const
  {
      return 163840;
  }
  uint32_t AcquisitionProxy::GetFirstZmwNumber() const {
      return 0;
  }

  PacBio::Primary::ChipLayout& AcquisitionProxy::ChipLayout() const
  {
      if (!chipLayout_)
      {
          chipLayout_.reset(new PacBio::Primary::ChipLayoutRTO3);
      }
      return *chipLayout_;
  }


//
// AcquisitionProxyChunkFile methods
//

AcquisitionProxyChunkFile::AcquisitionProxyChunkFile(size_t framesPerTranche, const std::string& chunkFile)
        : AcquisitionProxy(framesPerTranche)        // Defer the base configuration pending avail of file metadata.
        , src_(chunkFile)           // Opens the chunk file and reads the main header.
{
    numSourceLanes_ = src_.NumLanes();
    numSourceSuperchunks_ = src_.NumChunks();


    // Lanes and chunks are determined by the file
    std::ifstream fstrm(chunkFile.c_str());

    if (!fstrm.good())
        throw PBException("Cannot open input chunk file");

    Chunking dims;
    // Read the file header
    fstrm >> dims;

    assert(dims.sizeofSample == 64); // Only 512-bit SIMD for the basecaller
    assert(dims.chunkNum==numSourceSuperchunks_);
    assert(dims.laneNum==numSourceLanes_);

    PBLOG_INFO << "Chunk File:" << chunkFile << " numSourceLanes" << numSourceLanes_ << " numSuperchunks:" << numSourceSuperchunks_;
}

std::unique_ptr<SequelROI> AcquisitionProxyChunkFile::GetSourceROI() const
{
    const auto sensorROI = SequelSensorROI::SequelAlpha();
    uint32_t pixels = numSourceLanes_ * Tile::NumPixels;
    uint32_t rows = pixels / sensorROI.PhysicalCols(); // round down
    uint32_t cols = pixels % sensorROI.PhysicalCols();
    std::unique_ptr<SequelROI> roi;
    SequelSparseROI* sparseROI;
    roi.reset(sparseROI = new SequelSparseROI(sensorROI));
    sparseROI->AddRectangle(0,0,rows,sensorROI.PhysicalCols());
    if (cols != 0)
    {
        sparseROI->AddRectangle(rows,0,1,cols);
    }
    return move(roi);
}

  uint32_t AcquisitionProxyChunkFile::GetSourceFrames() const
  {
      return 16384 * numSourceSuperchunks_;
  }

  uint32_t AcquisitionProxyChunkFile::GetFirstZmwNumber() const {
      return 0;
  }

  uint32_t AcquisitionProxyChunkFile::LoadTranche(Tranche::Pixelz* data, uint32_t lane, uint32_t super)
{
    using VIn = Tranche::Pixelz::SimdPixel;
    bool chunkCompleted;
    bool terminus;

    PBLOG_DEBUG << "AcquisitionProxyChunkFile::LoadTranche(" << lane <<","<< super<<");";

    chipClass_ = ChipClass::Sequel; // I don't think we need to support Spider, as this file format is not really used.

    // Note: for chunk files, blocks==chunks, there's no distinction.
    size_t pixelLaneIndex;
    size_t chunkIndex;
    const uint32_t maxAttempts = src_.NumLanes() * src_.NumChunks();
    uint32_t attempts = 0;
    while(true)
    {
        uint64_t numFrames = src_.NextBlock(pixelLaneIndex, chunkIndex, chunkCompleted, terminus);
        PBLOG_DEBUG << "NextBlock(" << pixelLaneIndex <<","<< chunkIndex<<","<< chunkCompleted<<","<< terminus<<") sizeofsample:" << src_.SizeOfSample();
        assert(numFrames == framesPerTranche_ || numFrames== 0);

        if (pixelLaneIndex == lane && chunkIndex == super)
        {
            // Read the data directly into the tranche buffer
            VIn* dst = reinterpret_cast<VIn*>(data->pixels.tiles[0].data /* REALLY? */);
            size_t nFramesRead = src_.ReadBlock(dst, numFrames);
            assert(nFramesRead == numFrames);
            PBLOG_DEBUG << "ReadBlock(" << numFrames << "/" << nFramesRead << "," << src_.SizeOfSample() <<")";

            return nFramesRead;
        }
        if (chunkIndex > super)
        {
            PBLOG_DEBUG << "Rewind()";
            src_.Rewind();
        }
        else
        {
            PBLOG_DEBUG << "SkipBlock()";
            src_.SkipBlock();
        }
        if (++attempts > maxAttempts)
        {
            throw PBException("Could not find lane/superchunk in chunk file");
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////


AcquisitionProxyTraceFile::AcquisitionProxyTraceFile(size_t framesPerTranche, const std::string &traceFile) :
    AcquisitionProxy(framesPerTranche),
    src_(traceFile),
    timeStampDelta_(defaultExposureUsec),
    configWord_(0x12345678)
{
    assert(src_.NUM_HOLES % zmwsPerTranche == 0);
    numSourceLanes_ = src_.NUM_HOLES / zmwsPerTranche;
    numSourceSuperchunks_ = (src_.NFRAMES + framesPerTranche_ - 1) / framesPerTranche_;
    chipClass_ = ChipLayout().GetChipClass();
    src_.FrameRate >> frameRate_;
    PBLOG_DEBUG << "Opened traceFile: " << traceFile << " chipClass:" << chipClass_.toString();
}

std::unique_ptr<SequelROI> AcquisitionProxyTraceFile::GetSourceROI() const
{
    if (relax_)
    {
        // the ZMW numbers are all bogus. Remap to (0,0) and raster order after that
        SequelSparseROI* sroi = new SequelSparseROI(ChipLayout().GetSensorROI());
        std::unique_ptr<SequelROI> roi(sroi);
        PixelCoord coord(sroi->SensorROI().PhysicalRowOffset(),sroi->SensorROI().PhysicalColOffset());
        for(uint32_t i = 0; i < src_.NUM_HOLES; i += zmwsPerTranche)
        {
            sroi->AddZMW(coord);
            coord.col = coord.col + zmwsPerTranche * ChipLayout().PixelsPerZmw();
            if (coord.col.Value() >= sroi->SensorROI().PhysicalColMax())
            {
                coord.col = sroi->SensorROI().PhysicalColOffset();
                coord.row = coord.row + 1;
            }
        }
        return roi;
    }
    try{
        return std::unique_ptr<SequelROI>(src_.GetROI()->Clone());
    }
    catch(...)
    {
        std::cerr << "\n\nExcept processing trc.h5 file. I suggest trying running with --relax option if this is not a valid trc.h5 file\n\n" << std::endl;
        throw;
    }
}


uint32_t AcquisitionProxyTraceFile::GetSourceFrames() const
{
    return src_.NFRAMES;
}

uint32_t AcquisitionProxyTraceFile::GetFirstZmwNumber() const
{
    if (relax_)
    {
        return 0x00200020;
    }
    std::vector<uint32_t> holeNumbers;
    src_.HoleNumber >> holeNumbers;
    return holeNumbers[0];
}

  PacBio::Primary::ChipLayout& AcquisitionProxyTraceFile::ChipLayout() const
  {
      if (!chipLayout_)
      {
          std::string layoutName;

          src_.LayoutName >> layoutName;
          chipLayout_ = PacBio::Primary::ChipLayout::Factory(layoutName);
      }
      return *chipLayout_;
  }

std::vector<LaserPowerChange>
AcquisitionProxyTraceFile::LaserPowerChanges() const
{
    return src_.events.laserPowerChanges.ReadAllLaserChanges();
}


uint32_t AcquisitionProxyTraceFile::LoadTranche(Tranche::Pixelz* data, uint32_t lane, uint32_t superchunk)
{
    size_t zmwIndex = (lane * zmwsPerTranche) % src_.NUM_HOLES;
    size_t frameIndex = (superchunk * framesPerTranche_) % src_.NFRAMES;
    Tranche tempTranche;
    tempTranche.AssignTraceDataPointer( data);
    switch(chipClass_) // this is messy
    {
    case ChipClass::Sequel:
        tempTranche.Create2C2ATitle();
        break;
    case ChipClass::Spider:
        tempTranche.Create1C4A_CA_Title();
        break;
    default:
        throw PBException("not handled");
    }
    size_t nFramesRead = src_.ReadTranche(zmwIndex, frameIndex, &tempTranche, framesPerTranche_);

    return nFramesRead;
}

  /// modify the setup object with information gleaned from the trace file
  Acquisition::Setup AcquisitionProxyTraceFile::UpdateSetup()
  {
      using std::vector;

      Acquisition::Setup setup(src_.GetChipLayout());
      setup.PostImportSmrtSensor(); // this loads the SmrtSensor based on chipLayoutName

      PBLOG_INFO << "Updating acquisition setup with information from the trace file";
      SmrtSensor& smrtSensor(setup.smrtSensor());
      double photoelectronSensitivity;
      src_.AduGain >> photoelectronSensitivity;
      setup.photoelectronSensitivity = photoelectronSensitivity;
      smrtSensor.PhotoelectronSensitivity(photoelectronSensitivity);
      float frameRate;
      src_.FrameRate >> frameRate;      // Hz
      smrtSensor.FrameRate(frameRate);
      setup.expectedFrameRate = frameRate;

      // TODO: For completeness, we should factor in the CameraGain as well.
      // The units of CameraGain, however, are undocumented.
      // In the Sequel context, its value should always be 1.0.
      // Curtis Gehman, 2016-04-12

      uint16_t numAnalogs = 4;
      if (src_.Supports2_3)
      {
          vector<uint16_t> filterMap;
          boost::multi_array<float, 3> imagePsf;
          boost::multi_array<float, 2> xtalkCorrection;
          vector<float> analogRefSpectrum;
          float analogRefSnr;
          boost::multi_array<float, 2> analogSpectra;
          vector<float> relativeAmpl;
          vector<float> excessNoiseCV;

          src_.FilterMap >> filterMap;
          src_.ImagePsf >> imagePsf;

          if (imagePsf.size() == 1 && setup.chipClass() == ChipClass::Spider)
          {
              const auto shape = imagePsf.shape();
              PBLOG_DEBUG << "Resizing the psfs to have two channels to match Sequel.";
              imagePsf.resize(boost::extents[2][shape[1]][shape[2]]);
              imagePsf[1] = imagePsf[0];
          }

          src_.XtalkCorrection >> xtalkCorrection;
          src_.AnalogRefSpectrum >> analogRefSpectrum;
          src_.AnalogRefSnr >> analogRefSnr;

          src_.AnalogSpectra >> analogSpectra;
          src_.RelativeAmp >> relativeAmpl;
          src_.ExcessNoiseCV >> excessNoiseCV;

          // SE-41 Work around bug in pa-acq.
          // Until 3.2.0.?????, it always wrote all zeros for excessNoiseCV in the trc.h5 file.
          const auto encvZeroCount = std::count(excessNoiseCV.cbegin(), excessNoiseCV.cend(), 0.0f);
          if (boost::numeric_cast<size_t>(encvZeroCount) == excessNoiseCV.size())
          {
              static const float encvDefault = 0.1f;
              PBLOG_INFO << "Suspect that hard zero is invalid value for ExcessNoiseCV."
                         << " Using default value of "
                         << encvDefault << '.';
              std::fill(excessNoiseCV.begin(), excessNoiseCV.end(), encvDefault);
          }

          // These setters will throw if an invariant is violated.
          smrtSensor.FilterMap(filterMap.cbegin(), filterMap.cend());
          smrtSensor.RefDwsSnr(analogRefSnr);
          smrtSensor.RefSpectrum(analogRefSpectrum.cbegin(), analogRefSpectrum.cend());

          // Also update Setup object itself.

          setup.refDwsSnr = analogRefSnr;
          setup.refSpectrum.resize(analogRefSpectrum.size());
          for (size_t r = 0; r < analogRefSpectrum.size(); r++) setup.refSpectrum[r] = analogRefSpectrum[r];

          std::string baseMap;
          src_.NumAnalog >> numAnalogs;
          src_.BaseMap >> baseMap;

          setup.analogs.clear();
          for (unsigned int i = 0; i < numAnalogs; ++i)
          {
              AnalogMode analog(2);
              analog.baseLabel = baseMap[i];
              analog.dyeSpectrum[0] = analogSpectra[i][0];
              analog.dyeSpectrum[1] = (src_.NUM_CHANNELS == 2) ? analogSpectra[i][1] : 0;
              analog.excessNoiseCV = excessNoiseCV[i];
              analog.relAmplitude = relativeAmpl[i];
              setup.analogs.push_back(analog);
          }

          setup.crosstalkFilter = Kernel(xtalkCorrection);

          setup.psfs.clear();
          for (const auto& psf : imagePsf)
          {
              Kernel k(psf);
              setup.psfs.push_back(k);
          }
      }
      else
      {
          PBLOG_WARN << "Pre version 2.3 format trc.h5 file, using default analogs";

          PBLOG_WARN << "Old format trc.h5 file, creating dummy PSFs";
          setup.PostImport();
      }

      vector<float> pulseWidth;
      vector<float> ipd;
      if (src_.Supports2_4)
      {
          // These quantities stored in the trace file with units of seconds.
          src_.PulseWidthMean >> pulseWidth;
          src_.IpdMean >> ipd;
          if (pulseWidth.size() != numAnalogs)
          {
              std::ostringstream msg;
              msg << "Bad size of PulseWidthMean read from trace file, "
                  << pulseWidth.size()
                  << "; expected " << numAnalogs << '.';
              throw PBException(msg.str());
          }
          if (ipd.size() != numAnalogs)
          {
              std::ostringstream msg;
              msg << "Bad size of IpdMean read from trace file, "
                  << ipd.size()
                  << "; expected " << numAnalogs << '.';
              throw PBException(msg.str());
          }
      }
      else
      {
          PBLOG_NOTICE << "Trace file does not include PulseWidthMean or"
                          " IpdMean. Using 5.0.0 defaults.";
          pulseWidth = { { 10.0f, 7.0f, 9.0f, 9.0f } }; // frames
          ipd = { { 13.0f, 18.0f, 14.0f, 14.0f } };     // frames
          // Convert units from frames to seconds.
          for (unsigned int i = 0; i < numAnalogs; ++i)
          {
              pulseWidth[i] /= frameRate;
              ipd[i] /= frameRate;
          }
      }
      assert(pulseWidth.size() == numAnalogs);
      assert(ipd.size() == numAnalogs);

      vector<float> pw2SlowStepRatio;
      vector<float> ipd2SlowStepRatio;
      if (src_.Supports2_5)
      {
          src_.Pw2SlowStepRatio >> pw2SlowStepRatio;
          src_.Ipd2SlowStepRatio >> ipd2SlowStepRatio;
      }
      else
      {
          PBLOG_NOTICE << "Trace file does not include Pw2SlowStepRatio or Ipd2SlowStepRatio.";
          pw2SlowStepRatio = {0.0,0.0,0.0,0.0};
          ipd2SlowStepRatio = {0.0,0.0,0.0,0.0};
      }
      assert(pw2SlowStepRatio.size() == numAnalogs);
      assert(ipd2SlowStepRatio.size() == numAnalogs);

      if (setup.analogs.size() < numAnalogs)
      {
        PBLOG_ERROR << "setup.analogs.size():" << setup.analogs.size() <<
          " numAnalogs:" << numAnalogs;
        throw PBException("internal inconsistency, can't continue");
      }
      for (unsigned int i = 0; i < numAnalogs; ++i)
      {
          auto& analog = setup.analogs[i];
          analog.pulseWidth = pulseWidth[i];
          analog.interPulseDistance = ipd[i];
          analog.pw2SlowStepRatio = pw2SlowStepRatio[i];
          analog.ipd2SlowStepRatio = ipd2SlowStepRatio[i];
      }
      return setup;
  }

Acquisition::Setup AcquisitionProxy::UpdateSetup()
  {
      // warning: this base implementation only supports Sequel.
      // This base implementation is largely useless because it is only used to sanity check the basecaller
      // with default "test-pattern" trace data that doesn't even contain pulses.
      // todo: change AcquisitionProxy class to be pure abstract. Get rid of this dumb baseclass implementations.

      const auto sequelLayout = ChipLayout::Factory("SequEL_4.0_RTO3");
      Acquisition::Setup setup(sequelLayout.get());
      setup.PostImportSmrtSensor();

      if (setup.NumFilters() != 2)
      {
          throw PBException("NumFilters must be 2, were " + std::to_string(setup.NumFilters()));
      }

      SmrtSensor& smrtSensor(setup.smrtSensor());
      // These setters will throw if an invariant is violated.
      smrtSensor.FilterMap({{1, 0}});
      smrtSensor.RefDwsSnr( defaultRefDwsSnr );
      smrtSensor.RefSpectrum({{0.0f, 1.0f}});
      smrtSensor.PhotoelectronSensitivity( defaultPhotoelectronSensitivity );
      smrtSensor.FrameRate(Sequel::defaultFrameRate);

      // Fill in some dummy values for pulse width and IPD to avoid an
      // exception in the basecaller.
      setup.PostImport();

      return setup;
  }


//
// Factory method for creating an AcquisitionProxy
//

AcquisitionProxy* CreateAcquisitionProxy(size_t framesPerTranche, const std::string& file)
{
    AcquisitionProxy* ret;
    if (PacBio::Text::String::EndsWith(file,".int16"))
    {
        ret = new AcquisitionProxyChunkFile(framesPerTranche, file);
    }
    else if (PacBio::Text::String::EndsWith(file,".trc.h5"))
    {
        ret = new AcquisitionProxyTraceFile(framesPerTranche, file);
    }
    else if (file == "")
    {
        ret = new AcquisitionProxy(framesPerTranche);
    }
    else
    {
        throw PBException("File format not supported:" + file);
    }
    return ret;
}

}} // ::PacBio::Primary
