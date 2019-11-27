#pragma once

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
//  C++ header for Sequel Calibration File object
//
// Programmer: Mark Lakata

#include <assert.h>
#include <stdint.h>

#include <string>
#include <map>

#include <pacbio/primary/SequelMovieConfig.h>
#include <pacbio/primary/SequelMovie.h>
#include <pacbio/utilities/SmartEnum.h>
#include <pacbio/primary/HDF5cpp.h>

namespace PacBio {
namespace Primary {


class SequelCalibrationFileBase
{
public:

    SequelCalibrationFileBase(); ////< open for read
    SequelCalibrationFileBase(const SequelMovieConfig& config, const uint32_t rows, const uint32_t cols); ///< open for write
    virtual ~SequelCalibrationFileBase() {}
    uint32_t NROW() const { return numRows_;}
    uint32_t NCOL() const { return numCols_;}

    uint64_t FrameIndex() const; ///< the uint64 frame index value of the first frame captured.
    double FrameTimeStamp() const; ///< the epoch time stamp of the first frame.
    virtual std::string GroupName() const = 0;
public:
    H5::H5File hd5file;
    H5::Group Cal;
    std::map<std::string, H5::DataSet> datasets;

protected:
    virtual const std::vector<std::string>& SetsAllValues() = 0;
    void OpenFileForRead(const std::string& filename);
    void CreateFileForWrite(const std::string& filename);
    void Write1(const H5::DataSet dataset, const SequelMovieFrame<float>& frame);
    void Read1( const H5::DataSet dataset,       SequelMovieFrame<float>& frame);
    H5::DataSet OpenOrCreateDataSet(const std::string& name);

protected:
    uint32_t numRows_;
    uint32_t numCols_;

private:
    SequelHDF5MembersStart _memberStart_;
    SequelHDF5MembersEnd _memberEnd_;
    H5::DSetCreatPropList framesPropList_;
    SequelMovieConfig config_;
};

class SequelCalibrationFileHDF5 : public SequelCalibrationFileBase
{
public:
    SMART_ENUM(Sets, DarkOffSetZero, DarkOffSetSlope, DarkNoiseZero, DarkNoiseSlope);

public:

    SequelCalibrationFileHDF5(const std::string& filepath);
    SequelCalibrationFileHDF5(const SequelMovieConfig& config,
            const uint32_t rows,
            const uint32_t cols
            );
    ~SequelCalibrationFileHDF5() override {}

public: // virtual overrides
    const std::vector<std::string>& SetsAllValues() override { return Sets::allValuesAsStrings();  }
    std::string GroupName() const override { return "Cal";}

public:
    void Write(const Sets set, const SequelMovieFrame<float>& frame);
    void Read(const Sets set,  SequelMovieFrame<float>& frame);

    std::string GetDarkFrameMeanDataSetName(double exposure) const;
    std::string GetDarkFrameSigmaDataSetName(double exposure) const;
    std::string GetFrameRateAttributeName() const;
    std::string GetPhotoelectronSensitivityAttributeName() const;
    void WriteDarkFrameMean(const double exposure,  const SequelMovieFrame<float>& frame);
    void WriteDarkFrameSigma(const double exposure, const SequelMovieFrame<float>& frame);
      /// \returns false on failure to load
    bool ReadDarkFrameMean(const double exposure,  SequelMovieFrame<float>& frame);
     /// \returns false on failure to load
    bool ReadDarkFrameSigma(const double exposure,  SequelMovieFrame<float>& frame);
    void WriteFrameRate(double frameRate);
    void WritePhotoelectronSensitivity(double pe);
    double ReadFrameRate() const;
    double ReadPhotoelectronSensitivity() const;
};

class SequelLoadingFileHDF5 : public SequelCalibrationFileBase
{
public:
    SMART_ENUM(Sets, LoadingMean, LoadingVariance);

public:
    SequelLoadingFileHDF5(const std::string& filepath);
    SequelLoadingFileHDF5(const SequelMovieConfig& config,
            const uint32_t rows,
            const uint32_t cols
            );
    ~SequelLoadingFileHDF5() override {}

public: // virtual overrides
    const std::vector<std::string>& SetsAllValues() override { return Sets::allValuesAsStrings();  }
    std::string GroupName() const override { return "Loading";}

public:
    std::string GetLoadingFrameMeanDataSetName() const;
    std::string GetLoadingFrameVarianceDataSetName() const;
    void WriteMean(const SequelMovieFrame<float>& frame,
                   uint64_t numFrames,
                   float frameRate);
    void WriteVariance(const SequelMovieFrame<float>& frame,
                       uint64_t numFrames,
                       float frameRate);

    /// \returns false on failure to load
    bool ReadMean(SequelMovieFrame<float>& frame);

    /// \returns false on failure to load
    bool ReadVariance(SequelMovieFrame<float>& frame);
    float ReadFrameRate() const;
    uint64_t ReadNumFrames() const;
};

}} // namespace

