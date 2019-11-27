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
/// \brief  A class that represents the meta data stored in the "acquisition/setup" message
///  sent from PAWS.
//
// Programmer: Mark Lakata

#pragma once
#include <stdint.h>
#include <string>
#include <vector>
#include <memory>

#include <json/json.h>

#include <pacbio/logging/Logger.h>
#include <pacbio/text/PBXml.h>
#include <pacbio/smrtdata/Readout.h>
#include <pacbio/smrtdata/MetricsVerbosity.h>
#include <pacbio/utilities/Finally.h>
#include <pacbio/primary/AnalogMode.h>
#include <pacbio/primary/ChipLayout.h>
#include <pacbio/primary/Kernel.h>
#include <pacbio/primary/PrimaryConfig.h>
#include <pacbio/primary/SequelROI.h>
#include <pacbio/primary/SmrtSensor.h>
#include <pacbio/primary/UnitCell.h>
#include <pacbio/process/ConfigurationBase.h>

namespace PacBio {
namespace Primary  {

class AcquisitionMetaData
{
public:
    static std::unique_ptr<SequelRectangularROI> ParseXmlForROI(const std::string& xmlText);
    static std::unique_ptr<SequelROI> ParseJSONForSequencingROI(const Json::Value& roiMetaData,
                                                                const SequelSensorROI& sensorROI);
    static std::unique_ptr<SequelROI> ParseJSONForTraceFileROI(const Json::Value& roiMetaData,
                                                               const SequelSensorROI& sensorROI);
    static SequelSensorROI ParseJSONForSensorROI(const Json::Value& roiMetaData,
                                                 const SequelSensorROI& sensorROI);
    static void ParseXmlForChemistry(const std::string& xmlText, std::string& baseMap,
                                     std::vector<double>& spectralAngle, std::vector<double>& amplitude,
                                     std::vector<double>& excessNoiseCV, std::vector<double>& ipd,
                                     std::vector<double>& pw);
};


namespace Acquisition {

   /// A struct that represents a type-safe view of the data in the JSON object that is passed from
   /// PAWS to pa-acq, and then passed on to pa-t2b and pa-bw.
   /// The struct is modifiable. pa-acq inserts the crosstalk coefficient kernel, and pa-t2b inserts
   /// its version number.

class Setup : public PacBio::Process::ConfigurationObject
{
private:
    // don't allow Load to be called in user code. I fell into that trap.
    // Use "Parse" instead.
    using PacBio::Process::ConfigurationObject::Load;

public:
    /// The ChipLayout must be known at construction time. Either pass the chiplayout object to the constructor,
    /// or pass the name of the chip layout as the "chipLayoutName" of the JSON object.
    /// Do not use the obsoleted version that uses ChipClass as an argument.
    explicit Setup(const ChipLayout& chipLayout);
    explicit Setup(const ChipLayout* chipLayout) : Setup(*chipLayout) {}
    explicit Setup(const Json::Value& json);
    explicit Setup(const ChipClass) = delete;

    /// copy constructor. This copies the underlying COnfigurationObject (JSON based) and
    /// then all the specialized C++ members.
    Setup(const Setup& a ) : ConfigurationObject(),
                             analogs(a.analogs),
                             crosstalkFilter(a.crosstalkFilter),
                             psfs(a.psfs),
                             sensorROI(a.sensorROI),
                             sequencingROI(a.sequencingROI ? a.sequencingROI->Clone() : nullptr),
                             fileROI(a.fileROI ? a.fileROI->Clone() : nullptr),
                             pSmrtSensor_(a.pSmrtSensor_ ? a.pSmrtSensor_->Clone(): nullptr)

    {
        CopyCompletion(a);
        //PBLOG_NOTICE << "Setup.Copy Constructor"; // I hate C++11 copy elision unpredictability...
    }

    /// assignment
    Setup& operator=(const Setup& a)
    {
        analogs = a.analogs;
        crosstalkFilter = a.crosstalkFilter;
        psfs = a.psfs;
        sensorROI = a.sensorROI;
        sequencingROI.reset(a.sequencingROI->Clone());
        fileROI.reset(a.fileROI->Clone());
        pSmrtSensor_.reset(a.pSmrtSensor_ ? a.pSmrtSensor_->Clone(): nullptr);
        CopyCompletion(a);
        MarkChanged();
        //PBLOG_NOTICE << "Setup.Copy Assignment";  // I hate C++11 copy elision unpredictability...
        return *this;
    }

    void CopyCompletion(const Setup& a)
    {
        Copy(a);
        {
            PacBio::Utilities::Finally final([this]() { copyInProgress_ = false; });
            copyInProgress_ = true;
            PostImportAll();
        }
    }

   ADD_ENUM(ChipClass, chipClass, ChipClass::DONT_CARE);

    // members
    /// A timestamped name. Typically starts with "m" then instrument number
    /// then datecode. example: "m54001_20180101_134512"
   ADD_PARAMETER(std::string, movieContext, "");

    /// chipLayoutName is the official layout name of the chip, that is mapped to a unit cell list
   ADD_PARAMETER(std::string,chipLayoutName,""); // "SequEL_4.0_RTO3");

   /// token is an arbitrary string used to ensure proper closure of setup/start/stop pairs.
   ADD_PARAMETER(std::string,token,"");

   /// The path to the HDF5 output file (trc.h5 or mov.h5). Set to "" if no output is desired.
   ADD_PARAMETER(std::string,hdf5output,"");

   /// The path to the BAZ output file. Set to "" if no output is desired.
   ADD_PARAMETER(std::string,bazfile,"");

   /// The arbitrary 32 bit integer coming from the sensor. TODO. what does this do?
   ADD_PARAMETER(uint32_t,cameraConfigKey,0);

   /// The instrument name
   ADD_PARAMETER(std::string, instrumentName,"n/a");

   /// The chip ID
   ADD_PARAMETER(std::string, chipId,"n/a");

   /// The number of frames expected in this acquisition. because an acquisition can be stopped at any time
   /// this is not necessarily the length of the actual acquisition.
   ADD_PARAMETER(uint64_t,numFrames,0);

   /// An enum value that describes what should be saved to the BAZ file. Could pulses, or bases. See enum definition.
   ADD_ENUM(PacBio::SmrtData::Readout, readout, PacBio::SmrtData::Readout::BASES );

   /// the verbosity of metrics that should be generated.
   ADD_ENUM(PacBio::SmrtData::MetricsVerbosity, metricsVerbosity, PacBio::SmrtData::MetricsVerbosity::MINIMAL);

   /// version of the basecaller, inserted by the basecaller and extracted by the basewriter, to be written to BAZ file
   ADD_PARAMETER(std::string,basecallerVersion,"?");

   /// opaque string that represents the run meta data. This is not explicitly parsed, but simply written intact
   /// into the HDF5 trace or movie file.
   ADD_PARAMETER(std::string, acquisitionXML,"");

   /// The desired frame rate, in frames per second
   ADD_PARAMETER(double, expectedFrameRate, 0.0);

   /// The desired exposure time, in seconds
   ADD_PARAMETER(double, exposure, defaultExposureSec);

   /// a string consisting of the nucleotide labels
   ADD_PARAMETER(std::string, baseMap,"");

   /// If dryrun is true, all checks that are normally done during a real acquisition are performed, but
   /// the acquisition is not started. These include checking for necessary disk space, and that there are no
   /// errors in the JSON object.
   ADD_PARAMETER(bool, dryrun, false);

   /// A bool that if true, remaps the first 16 ROWS of the incoming frame (VROI1) into output pixel rows 1143-1128
   /// and the remaining rows (VROI2 and VROI3) into pixel rows 0 to 1127. Defaults to false.
   /// This should be true until the Sensor FPGA firmware is changed to perform this remapping.
   ADD_PARAMETER(bool, remapVroiOption,false);

   ADD_PARAMETER(double, minSnr, defaultMinSnr);

   ADD_PARAMETER(double, refDwsSnr, defaultRefDwsSnr);

   ADD_ARRAY(float, refSpectrum); // see default constructor for defaults

   ADD_PARAMETER(double, photoelectronSensitivity, defaultPhotoelectronSensitivity);

    /// A vector that contains the number of pixel lanes allocated per T2B processor. For Sequel, this vector
    /// vector is size 3 for the 3 MIC coprocessors. For PAC2 and Spider, this vector is size 1 for the
    /// host processor.
    ADD_ARRAY(uint32_t, numPixelLanes);
    ADD_ARRAY(uint32_t, numZmwLanes);

    ADD_PARAMETER(std::string, darkFrameCalFile, ""); // "" means no calibration

public:
    /// information about the SMRT sensor
    const SmrtSensor& smrtSensor() const
    {
        if (!pSmrtSensor_ ) throw PBException("SmrtSensor not constructed");
        return *pSmrtSensor_;
    }

    SmrtSensor& smrtSensor()
    {
        return const_cast<SmrtSensor&>( const_cast<const Setup*>(this)->smrtSensor());
    }
        /// a list of all analogs.
    AnalogSet analogs;

    ///  the derived or explicit crosstalk filter coefficients.
    Kernel crosstalkFilter;

    /// a vector of PSFs (pixel spread function) for each filter wavelength, sorted in wavelength order (green first
    /// then red, etc).
    std::vector<Kernel> psfs;

    void RenderCrosstalkFilter(Json::Value& originalJson);

    static Kernel RenderCrosstalkFilter(const std::vector<Kernel>& psfs);

private:
    /// the readout range of the sensor, usually the same as the physical range, but could be reduced for high frame rates.
    SequelSensorROI sensorROI;

    /// the ROI of the data that is sent through the base calling pipeline.
    std::unique_ptr<SequelROI> sequencingROI;

    /// The ROI of the data that is sent to the HDF5 file (trc.h5 or mov.h5)
    std::unique_ptr<SequelROI> fileROI;

public:
    /// This sets the physical size of the sensor. It also deletes the sequencing ROI and file ROIs.
    void SetSensorROIAndResetOtherROIs(const SequelSensorROI& roi) { sensorROI = roi; sequencingROI.reset(); fileROI.reset(); }

    /// This sets the physical size of the sensor. It doesn't change other fields.
    void OverwriteSensorROI(const SequelSensorROI& roi) { sensorROI = roi; UpdateRoiMetaData(); }

    /// This sets the sequencing ROI and refreshes some internal JSON settings
    void SetSequencingROI(const SequelROI& roi) { sequencingROI.reset(roi.Clone()); UpdateRoiMetaData(); }

    /// This sets the file (HDF5) ROI and refreshes some internal JSON settings
    void SetFileROI(const SequelROI& roi) { fileROI.reset(roi.Clone()); UpdateRoiMetaData(); }

    const SequelSensorROI& SensorROI() const { return sensorROI; }
    const SequelROI& SequencingROI() const { if (!sequencingROI) throw PBException("no sequencing ROI"); return *sequencingROI; }
    const SequelROI& FileROI() const { if (!fileROI) throw PBException("no file ROI"); return *fileROI; }

    /// loads data from the JSON object. Note that existing data in the Setup object will be erased.
    void Parse(const Json::Value& json);

    /// Returns a vector of all unit cells, as pairs of bit flags (see ChipLayout::SequelUnitFeature) and UnitCell coordinates.
    std::vector<std::pair<ChipLayout::UnitFeature, PacBio::Primary::UnitCell>> GetUnitCellFeatureList() const;

    /// Returns all ZMWs (by ZMW Number) in the ROI specified by the trace file ROI
    std::vector<uint32_t> GetTraceFileZmwMap() const;

    /// Returns all ZMWs (by ZMW Number) in the ROI specified by the sequencing ROI
    std::vector<uint32_t> GetSequencingZmwMap() const;

    /// Returns a ZMW map simply based on the ROI passed.
    std::vector<uint32_t> GetZmwMap(const SequelROI& ) const;

    void SetJsonTraceMetaData(const Json::Value& json);

    /// Returns number of unique wavelength analogs. Usually 2 for Sequel, and 4 for RS2.
    uint32_t NumAnalogWavelengths() const;

    std::shared_ptr<const PacBio::Primary::ChipLayout> GetChipLayout() const;

    std::unique_ptr<SequelROI> GetHdf5Roi() const
    {
        return AcquisitionMetaData::ParseJSONForTraceFileROI(Json()["roiMetaData"],GetSensorROI());
    }
    /// \returns the ROI for the sequencing or BAZ pipeline
    std::unique_ptr<SequelROI> GetBazRoi() const
    {
        return AcquisitionMetaData::ParseJSONForSequencingROI(Json()["roiMetaData"],GetSensorROI());
    }

#if 1
    SequelSensorROI GetSensorROI() const;
#endif

    const Json::Value& GreenPsf() const { return Json()["greenPsf"]; }

    const Json::Value& RedPsf() const { return Json()["redPsf"]; }

    void SetPsfs(const Json::Value&v ) { Json()["psfs"] = v; Json().removeMember("crosstalkFilter"); }
    void SetCrosstalkFilter(const Json::Value&v ) { Json()["crosstalkFilter"] = v; Json().removeMember("psfs"); }

    /// Basecaller configuration. Returns a JSON formatted string.
    std::string BasecallerConfig() const ;

    void PostImport() override;

    uint32_t NumFilters() const;

    bool DoHdf5Output() const
    {
        return hdf5output != "" && fileROI->TotalPixels() > 0;
    }

    /// Fill in the Run Metadata XML that can be consumed by pa-ws. The XML run definition will be formatted
    /// for a single acquisition, based on the chipClass0. This is never used by production code. It is only
    /// for use with test fixtures.
    static std::string CreateDummyXML(const ChipClass chipClass0);
    void CreateDummyAnalogs();

    /// Reads the coefficients from JSON. The format is
      /// [ [ coeff_r1c1, coeff_r1c2, ..],[coeff_r2c1...,],...[coeff_rNc1,...] ]
      ///
    void SetFilterCoefficientsFromJson(const std::string& json)
    {
        Json::Value v;
        Json::Reader reader;
        reader.parse(json,v,false);
        if (v.size() <1 || v.size() > 7)
        {
            throw PBException("Bad filter coefficients in JSON:" + json);
        }
        SetCrosstalkFilter(v);
    }

    void SetDefaultPsfs();
    void SetDefaultSequelAnalogs();
    void SetDefaultSpiderAnalogs();

    /// There are a few settings that need to be set in pairs, like "bazFile" and "sequencingROI". This function
    /// will remove settings if they are only partially set.
    void Sanitize();

    /// \returns number of ZMWS in the sequencing ROI, or 0 if no ROI exists.
    uint32_t NumSequencingZmws() const
    {
        return (sequencingROI)? sequencingROI->CountZMWs() : 0;
    }

    /// formats the setup for an output stream
    friend std::ostream& operator<<(std::ostream& s, const Setup& setup);

    /// looks at the JSON for SmrtSensor information
    void PostImportSmrtSensor();

    // protected methods are used to hide these functions from normal usage, but enable unit tests to
    // access the methods directly.
protected:

    void PostImportCrosstalk();

private:
    /// Parses the JSON for Analogs.
    void ParseForAnalogs();
    void UpdateRoiMetaData();
    void SetNumFilters(uint32_t numFilters);
    void SetMissingDefaults();
    void PostImportAnalogs();
private:
    void SortAnalogs();

    uint32_t numAnalogWavelengths_;
    mutable std::shared_ptr<const PacBio::Primary::ChipLayout> chipLayout_;
    std::unique_ptr<SmrtSensor> pSmrtSensor_;

    bool copyInProgress_ = false;
};

   std::ostream& operator<<(std::ostream& s, const Setup& setup);

   // Helper function
   std::string PrettyJSON(const Setup& acqSetup);

} // namespace Acquisition

}}
