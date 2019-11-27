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
//  C++ source for Sequel Calibration File object
//
// Programmer: Mark Lakata

#include <assert.h>
#include <stdint.h>

#include <string>
#include <map>


#include <pacbio/primary/SequelMovieFileHDF5.h>
#include <pacbio/logging/Logger.h>
#include <pacbio/utilities/SmartEnum.h>
#include <pacbio/primary/SequelCalibrationFile.h>
#include <pacbio/primary/HDF5cpp.h>
#include "SequelMovieConfig.h"

namespace PacBio {
namespace Primary  {

template<class H5Obj, typename T>
void CreateAttribute(H5Obj& dataset, const std::string& name, const T value)
{
    H5::DataType type = GetType<T>();
    H5::DataSpace scalar;
    auto attr  = dataset.createAttribute(name.c_str(),type,scalar);
    attr << value;
}

template<typename T, class H5Obj>
T ReadAttribute(const H5Obj& dataset, const char* name)
{
    auto attr  = dataset.openAttribute(name);
    T s;
    attr >> s;
    return s;
}


SequelCalibrationFileBase::SequelCalibrationFileBase()
        : numRows_(0),
          numCols_(0)
{
}

SequelCalibrationFileBase::SequelCalibrationFileBase(const SequelMovieConfig& config, const uint32_t rows , const uint32_t cols )
        : numRows_(rows),
          numCols_(cols)
{
    config_ = config;
}



void SequelCalibrationFileBase::OpenFileForRead(const std::string& filename)
{
  LOCK_HDF5();
  {
      DisableDefaultErrorHandler noErrors;
      try
      {
          hd5file.openFile(filename.c_str(), H5F_ACC_RDONLY);
          PBLOG_DEBUG << "Opened " << filename << " as calibration file for read";

          Cal = hd5file.openGroup("/" + GroupName());
          for (const auto& x : SetsAllValues())
          {
              datasets[x] = Cal.openDataSet(x.c_str());
              PBLOG_DEBUG << " Cal." << x << " opened for read";

              auto dims = GetDims(datasets[x]);

              if (numRows_ == 0)
              {
                  numRows_ = dims[0];
              }
              else if (numRows_ != dims[0])
              {
                  throw PBException("Inconsistent dimensions of Cal datasets");
              }
              if (numCols_ == 0)
              {
                  numCols_ = dims[1];
              }
              else if (numCols_ != dims[1])
              {
                  throw PBException("Inconsistent dimensions of Cal datasets");
              }
          }

      }
      catch (H5::Exception& ex)
      {
#ifndef WIN32
          H5::Exception::printErrorStack();
#endif
          throw PBException(ex.getDetailMsg() + ", filename:" + filename);
      }
  }
}

void SequelCalibrationFileBase::CreateFileForWrite(const std::string& filename)
{
  LOCK_HDF5();
  {
      hd5file = H5::H5File(filename.c_str(), H5F_ACC_TRUNC);

      float fill_val = 0;
      framesPropList_.setFillValue(H5::PredType::NATIVE_FLOAT, &fill_val);
      hsize_t chunk_dims[3];
      if (config_.movie.chunking.row == 0)
      {
          chunk_dims[0] = numRows_;
          chunk_dims[1] = numCols_;
      }
      else
      {
          chunk_dims[0] = config_.movie.chunking.row;
          chunk_dims[1] = config_.movie.chunking.col;
      }
      PBLOG_DEBUG << "SequelCalibrationFileBase::CreateFileForWrite() filename:" << filename
                      << " Chunk Sizes:" << chunk_dims[0] << "," << chunk_dims[1];
      framesPropList_.setChunk(2, chunk_dims);
      framesPropList_.setDeflate(config_.movie.compression);

      hsize_t space[2];
      space[0] = numRows_;
      space[1] = numCols_;
      H5::DataSpace framesSpace(2, space);

      Cal = hd5file.createGroup("/" + GroupName());
      for (const auto& x : SetsAllValues())
      {
          datasets[x] = Cal.createDataSet(x.c_str(), float32(), framesSpace, framesPropList_);
      }
  }
}

void SequelCalibrationFileBase::Write1(const H5::DataSet dataset, const SequelMovieFrame<float>& frame)
{
    hsize_t oneFrame[2];
    oneFrame[0] = numRows_;
    oneFrame[1] = numCols_;

    hsize_t offset[2];
    offset[0] = 0;
    offset[1] = 0;

    H5::DataSpace fileSpace = dataset.getSpace();
    fileSpace.selectHyperslab(H5S_SELECT_SET, oneFrame, offset);

    H5::DataSpace memFrame(2, oneFrame);

    dataset.write(frame.data, H5::PredType::NATIVE_FLOAT, memFrame, fileSpace);

    H5::DataSpace scalar;
    auto frameStartIndexAttr  = dataset.createAttribute("Frame_StartIndex",H5::PredType::STD_U64LE,scalar);
    frameStartIndexAttr << frame.index;

    auto frameTimeStampAttr  = dataset.createAttribute("Frame_TimeStamp",H5::PredType::STD_U64LE,scalar);
    frameTimeStampAttr << frame.timestamp;

    auto frameConfigWordAttr  = dataset.createAttribute("Frame_ConfigWord",H5::PredType::STD_U32LE,scalar);
    frameConfigWordAttr << frame.cameraConfiguration;
}

void SequelCalibrationFileBase::Read1(const H5::DataSet dataset,  SequelMovieFrame<float>& frame)
{
    hsize_t oneFrame[2];
    oneFrame[0] = numRows_;
    oneFrame[1] = numCols_;

    if (frame.NROW != numRows_ || frame.NCOL != numCols_)
    {
        if (frame.NROW !=0 || frame.NCOL !=0)
        {
            throw PBException("Can't read into a frame that has been sized already.");
        }
        frame.Resize(numRows_,numCols_);
    }

    PBLOG_DEBUG << "Reading dataset from calibration, size: (" << numRows_ <<","<<numCols_<<")";

    hsize_t offset[2];
    offset[0] = 0;
    offset[1] = 0;

    H5::DataSpace fileSpace = dataset.getSpace();
    fileSpace.selectHyperslab(H5S_SELECT_SET, oneFrame, offset);

    H5::DataSpace memFrame(2, oneFrame);

    dataset.read(frame.data, H5::PredType::NATIVE_FLOAT, memFrame, fileSpace);

    auto frameStartIndexAttr  = dataset.openAttribute("Frame_StartIndex");
    frameStartIndexAttr >> frame.index;

    auto frameTimeStampAttr  = dataset.openAttribute("Frame_TimeStamp");
    frameTimeStampAttr >> frame.timestamp;

    auto frameConfigWordAttr  = dataset.openAttribute("Frame_ConfigWord");
    frameConfigWordAttr >> frame.cameraConfiguration;
}

H5::DataSet SequelCalibrationFileBase::OpenOrCreateDataSet(const std::string& name)
{
    LOCK_HDF5();
    {
        hsize_t space[2];
        space[0] = numRows_;
        space[1] = numCols_;
        H5::DataSpace framesSpace(2, space);
        H5::DataSet dataset;

        try
        {
            DisableDefaultErrorHandler noErrors;
            dataset = Cal.openDataSet(name.c_str());
        }
        catch (H5::GroupIException&)
        {
            dataset = Cal.createDataSet(name.c_str(), float32(), framesSpace, framesPropList_);
        }
        return dataset;
    }
}

/////////////////////

SequelCalibrationFileHDF5::SequelCalibrationFileHDF5(const std::string& filepath)
        : SequelCalibrationFileBase()
{
    OpenFileForRead(filepath);
}

SequelCalibrationFileHDF5::SequelCalibrationFileHDF5(const SequelMovieConfig& config,
        const uint32_t rows,
        const uint32_t cols)
        : SequelCalibrationFileBase(config,rows,cols)
{
    CreateFileForWrite(config.path);
}

void SequelCalibrationFileHDF5::Write(const Sets set, const SequelMovieFrame<float>& frame)
{
    auto& dataset = datasets[set.toString()];
    LOCK_HDF5();
    {
        Write1( dataset, frame);
    }
}

void SequelCalibrationFileHDF5::Read(const Sets set, SequelMovieFrame<float>& frame)
{
    LOCK_HDF5();
    {
        auto& dataset = datasets[set.toString()];
        Read1(dataset, frame);
    }
}

std::string SequelCalibrationFileHDF5::GetDarkFrameMeanDataSetName(double exposure) const
{
    return PacBio::Text::String::Format("DarkFrameMean_%6lf", exposure);
}

std::string SequelCalibrationFileHDF5::GetDarkFrameSigmaDataSetName(double exposure) const
{
    return PacBio::Text::String::Format("DarkFrameSigma_%6lf", exposure);
}

std::string SequelCalibrationFileHDF5::GetFrameRateAttributeName() const
{
    return "FrameRate";
}
std::string SequelCalibrationFileHDF5::GetPhotoelectronSensitivityAttributeName() const
{
    return "PhotoelectronSensitivity";
}


void SequelCalibrationFileHDF5::WriteDarkFrameMean(const double exposure,  const SequelMovieFrame<float>& frame)
{
    std::string x = GetDarkFrameMeanDataSetName(exposure);

    auto dataset = OpenOrCreateDataSet(x);
    LOCK_HDF5();
    {
        Write1(dataset, frame);
    }
}

void SequelCalibrationFileHDF5::WriteDarkFrameSigma(const double exposure, const SequelMovieFrame<float>& frame)
{
    std::string x = GetDarkFrameSigmaDataSetName(exposure);
    auto dataset = OpenOrCreateDataSet(x);
    LOCK_HDF5();
    {
        Write1(dataset, frame);
    }
}

void SequelCalibrationFileHDF5::WriteFrameRate(double frameRate)
{
    LOCK_HDF5();
    const auto& name = GetFrameRateAttributeName();
    CreateAttribute(Cal, name, frameRate);
}

void SequelCalibrationFileHDF5::WritePhotoelectronSensitivity(double pe)
{
    LOCK_HDF5();
    const auto& name = GetPhotoelectronSensitivityAttributeName();
    CreateAttribute(Cal, name, pe);
}


/// returns false if dark frame mean is not there
bool SequelCalibrationFileHDF5::ReadDarkFrameMean(const double exposure, SequelMovieFrame<float>& frame)
{
    LOCK_HDF5();
    {
        std::string x = GetDarkFrameMeanDataSetName(exposure);
        try
        {
            DisableDefaultErrorHandler noErrors;
            auto dataset = Cal.openDataSet(x.c_str());
            Read1(dataset, frame);
            return true;
        }
        catch (H5::Exception& ex )
        {
            PBLOG_DEBUG << "Can't open data set " << x << " from cal file: " << ex.getDetailMsg();
            return false;
        }
    }
}

bool SequelCalibrationFileHDF5::ReadDarkFrameSigma(const double exposure, SequelMovieFrame<float>& frame)
{
    LOCK_HDF5();
    {
        std::string x = GetDarkFrameSigmaDataSetName(exposure);
        try
        {
            DisableDefaultErrorHandler noErrors;
            auto dataset = Cal.openDataSet(x.c_str());
            Read1(dataset, frame);
            return true;
        }
        catch (H5::Exception& ex)
        {
            PBLOG_DEBUG << "Can't open data set " << x << " from cal file: " << ex.getDetailMsg();
            return false;
        }
    }
}

double SequelCalibrationFileHDF5::ReadFrameRate() const
{
    LOCK_HDF5();
    return ReadAttribute<double>(Cal, GetFrameRateAttributeName().c_str());
}

double SequelCalibrationFileHDF5::ReadPhotoelectronSensitivity() const
{
    LOCK_HDF5();
    return ReadAttribute<double>(Cal, GetPhotoelectronSensitivityAttributeName().c_str());
}

/////////////


SequelLoadingFileHDF5::SequelLoadingFileHDF5(const std::string& filepath)
        : SequelCalibrationFileBase()
{
    OpenFileForRead(filepath);
}

SequelLoadingFileHDF5::SequelLoadingFileHDF5(const SequelMovieConfig& config,
        const uint32_t rows,
        const uint32_t cols
)
        : SequelCalibrationFileBase(config, rows,cols)
{
    CreateFileForWrite(config.path);
}

std::string SequelLoadingFileHDF5::GetLoadingFrameMeanDataSetName() const
{
    return PacBio::Text::String::Format("LoadingMean");
}

std::string SequelLoadingFileHDF5::GetLoadingFrameVarianceDataSetName() const
{
    return PacBio::Text::String::Format("LoadingVariance");
}

bool SequelLoadingFileHDF5::ReadMean(PacBio::Primary::SequelMovieFrame<float>& frame)
{
    LOCK_HDF5();
    {
        std::string x = GetLoadingFrameMeanDataSetName();
        try
        {
            DisableDefaultErrorHandler noErrors;
            auto dataset = Cal.openDataSet(x.c_str());
            Read1(dataset, frame);
            return true;
        }
        catch (H5::Exception& ex )
        {
            PBLOG_DEBUG << "Can't open data set " << x << " from cal file: " << ex.getDetailMsg();
            return false;
        }
    }
}

bool SequelLoadingFileHDF5::ReadVariance(PacBio::Primary::SequelMovieFrame<float>& frame)
{
    LOCK_HDF5();
    {
        std::string x = GetLoadingFrameVarianceDataSetName();
        try
        {
            DisableDefaultErrorHandler noErrors;
            auto dataset = Cal.openDataSet(x.c_str());
            Read1(dataset, frame);
            return true;
        }
        catch (H5::Exception& ex )
        {
            PBLOG_DEBUG << "Can't open data set " << x << " from cal file: " << ex.getDetailMsg();
            return false;
        }
    }
}

void SequelLoadingFileHDF5::WriteMean(const PacBio::Primary::SequelMovieFrame<float>& frame,
                                      uint64_t numFrames,
                                      float frameRate)
{
    std::string x = GetLoadingFrameMeanDataSetName();

    auto dataset = OpenOrCreateDataSet(x);
    LOCK_HDF5();
    {
        Write1(dataset, frame);
        CreateAttribute(dataset,"NumFrames",numFrames);
        CreateAttribute(dataset,"FrameRate",frameRate);
    }
}
void SequelLoadingFileHDF5::WriteVariance(const PacBio::Primary::SequelMovieFrame<float>& frame,
                                          uint64_t numFrames,
                                          float frameRate)
{
    std::string x = GetLoadingFrameVarianceDataSetName();

    auto dataset = OpenOrCreateDataSet(x);
    LOCK_HDF5();
    {
        Write1(dataset, frame);
        CreateAttribute(dataset,"NumFrames",numFrames);
        CreateAttribute(dataset,"FrameRate",frameRate);
    }
}

float SequelLoadingFileHDF5::ReadFrameRate() const
{
    auto dataset = Cal.openDataSet(GetLoadingFrameMeanDataSetName());
    return ReadAttribute<float>(dataset,"FrameRate");
}

uint64_t SequelLoadingFileHDF5::ReadNumFrames() const
{
    auto dataset = Cal.openDataSet(GetLoadingFrameMeanDataSetName());
    return ReadAttribute<uint64_t>(dataset,"NumFrames");
}


}}
