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


#include <memory>
#include <array>
#include <set>
#include <algorithm>

#include <pacbio/ipc/JSON.h>
#include <pacbio/text/String.h>
#include <pacbio/logging/Logger.h>
#include <pacbio/primary/Acquisition_Setup.h>
#include <pacbio/primary/ChipLayout.h>
#include <pacbio/primary/ChipLayoutRTO2.h>
#include <pacbio/primary/ChipLayoutRTO3.h>
#include <pacbio/primary/ChipLayoutSpider1p0NTO.h>
#include <pacbio/smrtdata/Readout.h>
#include <pacbio/smrtdata/MetricsVerbosity.h>

using namespace std;
using namespace PacBio::IPC;
using namespace PacBio::Primary;
using namespace PacBio::SmrtData;
using namespace PacBio::Text;



namespace PacBio {
namespace Primary {
namespace Acquisition {

/// constructor
Setup::Setup(const ChipLayout& chipLayout0)
        :
        sensorROI(chipLayout0.GetSensorROI()),
        // the default is to sequence full chips:
        sequencingROI(new SequelRectangularROI(sensorROI)),
        // the default is to not save any traces, so null ROI is used:
        fileROI(SequelROI::Null().Clone())
{
    chipLayoutName = chipLayout0.Name();
    chipClass = chipLayout0.GetChipClass();
    SetDefaultPsfs();
}

/// constructor based on exist Json object.
/// The JSON object MUST define the chipLayoutName field with a valid chip layout name.
Setup::Setup(const Json::Value& json)
        : Setup(
                ChipLayout::Factory(
                // look for chipLayoutName member first, then LayoutName
                json.get("chipLayoutName", json.get("LayoutName","")).asString()
        ).get())
{
    Parse(json);
}

   /// fills setup struct with the Json values.
   void Setup::Parse(const Json::Value& jsonConst)
   {
       PBLOG_DEBUG << "Acquisition::SetupParse:: json=" << jsonConst;

       {
           Json::Value json = jsonConst;

           // convert these members to upper case
           if (json.isMember("readout")) json["readout"] = String::ToUpper(json["readout"].asString());
           if (json.isMember("metricsVerbosity"))
               json["metricsVerbosity"] = String::ToUpper(json["metricsVerbosity"].asString());
           /// this is an allowed misspelling to support backwards compat
           if (json.isMember("hd5output") && json.isMember("hdf5output") &&
               json["hd5output"].asString() != json["hdf5output"].asString())
           {
             throw PBException("Both hd5output and hdf5output fields defined in the acquisition/setup message and they are inconsistent");
           }
           if (json.isMember("hd5output"))
           {
             json["hdf5output"] = json["hd5output"];
             json.removeMember("hd5output");
           }
           if (json.isMember("LayoutName"))
           {
               json["chipLayoutName"] = json["LayoutName"].asString();
               json.removeMember("LayoutName");
           }
           Load(json);
       }

       PBLOG_DEBUG << "Acquisition::SetupParse:: modified final json=" << Json();

       // CorrectChipLayoutName();

       try
       {
           if (sensorROI.PhysicalRows() == 0)
           {
               sensorROI = GetChipLayout()->GetSensorROI();
           }

           // numZmws is a hidden JSON field. I don't think we are supporting it any more
           // but here is the legacy code. It should always be undefined or 0.
           uint32_t numZmws = Json().get("numzmws", 0).asInt();
           if (numZmws == 0)
           {
               /// if the number of ZMWs is not explicitly defined, we get the
               /// ZMW information from the ROI
               if (Json()["roiMetaData"].isNull() && acquisitionXML() != "")
               {
                   PBLOG_INFO << "Getting ROIS from the acquisitionXML";
                   sequencingROI = AcquisitionMetaData::ParseXmlForROI(acquisitionXML()) ;
                   fileROI.reset(sequencingROI->Clone());
               }
               else
               {
                   const auto& roiMetaData = Json()["roiMetaData"];
                   sensorROI     = AcquisitionMetaData::ParseJSONForSensorROI(roiMetaData, sensorROI);
                   sequencingROI = AcquisitionMetaData::ParseJSONForSequencingROI(roiMetaData,sensorROI);
                   fileROI       = AcquisitionMetaData::ParseJSONForTraceFileROI(roiMetaData,sensorROI);

                   // the default for 3.0.* was true, the default for 3.1 is false
                   remapVroiOption = roiMetaData.get("remapVroiOption", false).asBool();
               }
           }
           else
           {
               // This is legacy code. We might want to throw an exception here in the future.
               // I'll leave it here for now.
               SequelRectangularROI roi(sensorROI);
               sequencingROI.reset(new SequelRectangularROI(roi));
               fileROI.reset(new SequelRectangularROI(roi));
               if (numZmws != roi.CountZMWs())
               {
                   throw PBException("inconsistent ZMW counts:" + std::to_string(numZmws) + " vs " + std::to_string(roi.CountZMWs()));
               }
           }

           if (exposure() <= 0)
           {
               throw PBException("Non positive exposure " + std::to_string(exposure) + " is not allowed");
           }
           if (expectedFrameRate() == 0.0)
           {
               expectedFrameRate = 1.0 / exposure();
           }
           if (expectedFrameRate() <= 0)
           {
               throw PBException("Non positive expected frameRate " + std::to_string(expectedFrameRate()) + " is not allowed");
           }
           if (expectedFrameRate() > 1.01/exposure()) // allow 1% disagreement to avoid roundoff issues
           {
               throw PBException("Frame rate " + std::to_string(expectedFrameRate) + " is not compatible with exposure " +  std::to_string(exposure) );
           }

           PostImport();
       }
       catch(const std::exception& )
       {

           throw;
       }
   }

void Setup::SetMissingDefaults()
{
    if (!Json()["crosstalkFilter"].isArray())
    {
        Json()["crosstalkFilter"] = PacBio::IPC::ParseJSON(R"([ [1] ])");
    }
    if (Json()["psfs"].size() == 0)
    {
        PBLOG_WARN << "acquisition/setup JSON does not contain PSFs. Unity PSFs will be used.";
        Json()["psfs"] = PacBio::IPC::ParseJSON(R"([ [ [1] ], [ [ 1 ] ] ] )");
    }

    if (Json()["analogs"].isNull() || Json()["analogs"].size() == 0)
    {
        PBLOG_WARN << "acquisition/setup JSON does not contain analogs. Will supply default value, which are probably WRONG.";

        if (chipClass() == ChipClass::Spider )
        {
            SetDefaultSpiderAnalogs();
            //throw PBException("No analogs found in JSON or XML, and no default analogs are available for Spider.");
        }
        else if (chipClass() == ChipClass::Sequel || chipClass() == ChipClass::DONT_CARE)
        {
            // these values were copied from TraceBlockAnalyzer.cpp. They are basically only used for testing.
            // Note that 'pulseWidthMeanSeconds' and 'ipdMeanSeconds' are not set here. They can only be set
            // if expectedFrameRate is know.
            Json()["analogs"] = PacBio::IPC::ParseJSON( R"(
            [
                {
                    "base" : "T",
                    "intraPulseXsnCV" : 0.10,
                    "relativeAmplitude" : 0.85,
                    "spectrumValues" : [ 0.42, 0.58 ]
                },
                {
                    "base" : "G",
                    "intraPulseXsnCV" : 0.10,
                    "relativeAmplitude" : 0.544,
                    "spectrumValues" : [ 0.42, 0.58 ]
                },
                {
                    "base" : "C",
                    "intraPulseXsnCV" : 0.10,
                    "relativeAmplitude" : 1.0,
                    "spectrumValues" : [ 0.06, 0.94 ]
                },
                {
                    "base" : "A",
                    "intraPulseXsnCV" : 0.10,
                    "relativeAmplitude" : 0.50,
                    "spectrumValues" : [ 0.06, 0.94 ]
                }
                ]
            )" );
        }
        else
        {
            throw PBException("can't supply defaults for chipclass:" + chipClass().toString());
        }
    }
}

// copy JSON settings to C++ object
void Setup::PostImportCrosstalk()
{
    crosstalkFilter = Kernel(Json()["crosstalkFilter"]);

    psfs.clear();
    for (const auto& jsonPsf : Json()["psfs"])
    {
        Kernel k(jsonPsf);
        psfs.push_back(k);
    }
}

void Setup::PostImportAnalogs()
{
    if (Json()["analogs"].isNull())
    {
        throw PBException("Internal inconsistency. The JSON [analogs] field must be filled it.");
    }

    if (chipClass() == ChipClass::Spider )
    {
        // do nothing here.
    }
    else if (chipClass() == ChipClass::Sequel || chipClass() == ChipClass::DONT_CARE)
    {
        if (expectedFrameRate() > 0)
        {
            // these values were copied from TraceBlockAnalyzer.cpp. They are basically only used for testing.
            const std::map<char, int> pwInFrames = {{'T', 10},
                                                    {'G', 7},
                                                    {'C', 9},
                                                    {'A', 9}};
            const std::map<char, int> ipdInFrames = {{'T', 13},
                                                     {'G', 18},
                                                     {'C', 14},
                                                     {'A', 14}};

            for (auto& analog : Json()["analogs"])
            {
                if (!analog.isMember("base"))
                {
                    throw PBException("Analog configuration does not have 'base' member:" + PacBio::IPC::RenderJSON(analog));
                }
                const char base = analog["base"].asString()[0];
                if (!analog.isMember("pulseWidthMeanSeconds"))
                    analog["pulseWidthMeanSeconds"] = pwInFrames.at(base) / static_cast<float>(expectedFrameRate);
                if (!analog.isMember("ipdMeanSeconds"))
                    analog["ipdMeanSeconds"] = ipdInFrames.at(base) / static_cast<float>(expectedFrameRate);
            }
        }
    }

    ParseForAnalogs();

    if (analogs.size() != 4)
    {
        throw PBException("There were only " + std::to_string(analogs.size()) + " analogs specified, should be 4");
    }

}

   uint32_t Setup::NumFilters() const
   {
       if (pSmrtSensor_) return pSmrtSensor_->NumFilters();
       return 0;
   }

 void Setup::SetNumFilters(uint32_t numFilters)
 {
     if (pSmrtSensor_ && pSmrtSensor_->NumFilters() == numFilters) return; // don't change anything if smrtSensor already exists

     pSmrtSensor_.reset(new SmrtSensor(numFilters));
 }

 void Setup::PostImportSmrtSensor()
 {
     const auto& cl = GetChipLayout();

     const auto fm = cl->FilterMap();

     PBLOG_DEBUG << "chiplayout:" << cl->Name() << " fm:" << PacBio::Text::String::AsArray(fm.cbegin(),fm.cend());

     SetNumFilters(fm.size());

     // These setters will throw if an invariant is violated.
     smrtSensor().FilterMap(fm.begin(), fm.end());

     if (refSpectrum.size() == 0)
     {
         if (fm.size() == 1)
         {
             refSpectrum.resize(1);
             refSpectrum[0] = 1.0f;
         }
         if (fm.size() == 2)
         {
             refSpectrum.resize(2);
             refSpectrum[0] = 0.10197792f;
             refSpectrum[1] = 0.8980221f;
         }
     }

     smrtSensor().RefSpectrum(refSpectrum.begin(), refSpectrum.end());
     PBLOG_DEBUG << " refSpectrum:" << PacBio::Text::String::AsArray(smrtSensor().RefSpectrum().cbegin(), smrtSensor().RefSpectrum().cend());

     if (photoelectronSensitivity() <= 0)
     {
         throw PBException("Non positive photoelectronSensitivity" + std::to_string(photoelectronSensitivity()) + " is not allowed");
     }
     smrtSensor().PhotoelectronSensitivity(photoelectronSensitivity);
     smrtSensor().FrameRate(expectedFrameRate);

     //SmrtSensor& ImagePsf(const MultiArray<float, 3>& value);
#if 0
     smrtSensor.ImagePsf(psfs);
     smrtSensor.XtalkCorrection(crosstalkFilter.AsMultiArray()); // fixme this is stupid
#endif

     smrtSensor().RefDwsSnr(refDwsSnr);
     PBLOG_DEBUG << "refdwssnr:" << refDwsSnr << " smrtSensor().RefDwsSnr():" << smrtSensor().RefDwsSnr();
 }

 void Setup::PostImport()
   {
       ConfigurationObject::PostImport();

       if (!copyInProgress_)
       {
           /// this fills in any missing JSON fields with reasonable defaults
           SetMissingDefaults();

           if (expectedFrameRate() == 0.0 && exposure() > 0.0)
           {
               expectedFrameRate = 1.0 / exposure();
           }
           else if (expectedFrameRate() > 0.0 && exposure() == 0.0)
           {
               exposure = 1.0/ expectedFrameRate();
           }

           /// these update the C++ structures from the JSON
           PostImportCrosstalk();
           PostImportAnalogs();
           PostImportSmrtSensor();
       }

       if (numPixelLanes.size() != numZmwLanes.size())
       {
           throw PBException("internal inconsistency: " + std::to_string(numPixelLanes.size()) + " " + std::to_string(numZmwLanes.size()));
       }
   }

   std::string Setup::BasecallerConfig() const {
       return PacBio::IPC::RenderJSON(Json()["basecaller"]);
   }

   std::vector<std::pair<ChipLayout::UnitFeature, PacBio::Primary::UnitCell>> Setup::GetUnitCellFeatureList() const
   {
       return GetChipLayout()->GetUnitCellFeatureList(*sequencingROI);
   }


   std::vector<uint32_t> Setup::GetZmwMap(const SequelROI& roi) const
   {
       std::vector<uint32_t> zmwMap;

       const auto cellList = GetChipLayout()->GetUnitCellList(roi);
       for (auto& cell : cellList)
       {
           zmwMap.push_back(cell.ID());
       }
       return zmwMap;
   }

   std::shared_ptr<const PacBio::Primary::ChipLayout>
   Setup::GetChipLayout() const
   {
       // TODO: Switching the chip layout like this seems to violate const semantics.
       if (chipLayout_ && chipLayout_->Name() != chipLayoutName())
       {
           // if the chip layout name changed, then throw away the old chip layout
           PBLOG_WARN << "The Acquisition::Setup was called with different chipLayoutNames. Previously it was " <<
                       chipLayout_->Name() << " and now it is " << chipLayoutName();
           PBLOG_INFO << "Destroying " << chipLayout_->Name();
           chipLayout_.reset();
       }
       if (!chipLayout_)
       {
           PBLOG_INFO << "Constructing chiplayout " << chipLayoutName() << " for use in Setup object";
           chipLayout_ = ChipLayout::Factory(chipLayoutName);
       }
       return chipLayout_;
   }

   std::vector<uint32_t> Setup::GetSequencingZmwMap() const
   {
       return GetZmwMap(SequencingROI());
   }

   std::vector<uint32_t> Setup::GetTraceFileZmwMap() const
   {
       return GetZmwMap(FileROI());
   }

#if 0
    // this is deprecated and will be deleted soon.
   void Setup::ParseXMLForAnalogs(const std::string& xml)
   {
       PBLOG_INFO << " Setup::ParseXMLForAnalogs. Importing XML analog data";
       // this hack is gross. This whole function should be deleted. TODO

       std::vector<double>  spectralAngle;
       std::vector<double>  amplitude;
       std::vector<double>  excessNoiseCV;
       std::vector<double>  ipd;
       std::vector<double>  pw;

       std::string baseMap1 = "";

       analogs.clear();

       AcquisitionMetaData::ParseXmlForChemistry(xml, baseMap1, spectralAngle, amplitude, excessNoiseCV, ipd, pw);

       std::set<double> channels;


       for (uint32_t i=0;i<baseMap1.size();i++)
       {
           char base = baseMap1[i];
           double angle = spectralAngle.at(i);
           if (angle == 0)
           {
               // assume if there is no spectralAngle, it will default to 0.0 and this means 1 color
               const std::array<float, 1> spectrum = {1.0f};

               AnalogMode analog(base, spectrum,
                                 static_cast<float>(amplitude[i]),
                                 static_cast<float>(excessNoiseCV[i]),
                                 static_cast<float>(ipd[i]),
                                 static_cast<float>(pw[i]));
               analogs.push_back(analog);
           }
           else
           {
               double green = cos(angle);
               double red = sin(angle);
               double norm = green + red; // yes it is linear normalization, not quadratic!
               assert (norm != 0);
               green /= norm;
               red /= norm;
               // now green + red = 1.0000


               const std::array<float, 2> spectrum = {static_cast<float>(green),
                                                      static_cast<float>(red)};

               AnalogMode analog(base, spectrum,
                                 static_cast<float>(amplitude[i]),
                                 static_cast<float>(excessNoiseCV[i]),
                                 static_cast<float>(ipd[i]),
                                 static_cast<float>(pw[i]));
               analogs.push_back(analog);
           }
           channels.insert(angle);
       }

       baseMap = baseMap1;
       SortAnalogs();
       numAnalogWavelengths_ = channels.size();
   }
#endif

void Setup::SetJsonTraceMetaData(const Json::Value& traceMetadata)
{
    Json()["analogs"] = traceMetadata["analogs"];

    if (traceMetadata.isMember("refDwsSnr")) refDwsSnr = traceMetadata["refDwsSnr"].asDouble();

    if (traceMetadata.isMember("refSpectrum"))
    {
        if (!traceMetadata["refSpectrum"].isArray())
            throw PBException("refSpectrum must be a JSON array, was " +
                              PacBio::IPC::RenderJSON(traceMetadata["refSpectrum"]));
        Json()["refSpectrum"] = traceMetadata["refSpectrum"];
    }
}

   void Setup::ParseForAnalogs()
   {
       std::string baseMap1 = "";
       std::set<double> channels;

       analogs.clear();

       for (const auto& z : Json()["analogs"])
       {
           // use the AnalogConfigEx JSON wrapper class to parse the JSON,
           // then place it in a POD struct (AnalogMode) for efficiency.
           AnalogConfigEx ac;
           ac.Load(z);
           ac.Normalize();
           baseMap1 += ac.base;

           AnalogMode analog(ac);

           analogs.push_back(analog);
           channels.insert(ac.wavelength);
       }

       baseMap = baseMap1;
       SortAnalogs();
       numAnalogWavelengths_ = channels.size();
   }

   void Setup::SortAnalogs()
   {
       baseMap = "";
       std::sort(analogs.begin(), analogs.end(), [](const AnalogMode&a , const AnalogMode& b)
       {
          if (a.SpectralAngle() < b.SpectralAngle()) return true;
          if (a.SpectralAngle() == b.SpectralAngle())
          {
              return a.RelativeAmplitude() > b.RelativeAmplitude();
          }
          else
          {
              return false;
          }
       });
       std::string baseMap1 = "";
       for(const auto& c : analogs)
       {
           baseMap1 += c.baseLabel;
       }
       baseMap = baseMap1;
   }

   uint32_t Setup::NumAnalogWavelengths() const
    {
        return numAnalogWavelengths_;
    }

   std::ostream& operator<<(std::ostream& s, const Setup& setup)
   {
       if (setup.sequencingROI)
       {
           s << "sequencingROI:" << setup.SequencingROI() << std::endl;
       }
       else
       {
           s << "sequencingROI: none";
       }
       if (setup.fileROI)
       {
           s << "traceFileROI:" << setup.FileROI() << std::endl;
       }
       else
       {
           s << "traceFileROI: none";
       }
       auto x = setup.GetTraceFileZmwMap();
       s << "traceFileZmwMap count:" << x.size() << std::endl;
       for (const auto& a : setup.analogs)
       {
           s << "Analog: " << a << std::endl;
       }
       return s;
   }

SequelSensorROI Setup::GetSensorROI() const
 {
     SequelSensorROI defaultSensorROI = chipLayout_->GetSensorROI();
     return AcquisitionMetaData::ParseJSONForSensorROI(Json()["roiMetaData"],defaultSensorROI);
 }

 ///
 /// Returns a JSON formatted string with extra properly formatted members
 ///
 std::string PrettyJSON(const Setup& acqSetup)
 {
     Json::Value v = acqSetup.Json();

     uint32_t numChannels = 0;
     if (acqSetup.analogs.size() > 0) numChannels = acqSetup.analogs[0].numFilters;
     else numChannels = acqSetup.psfs.size();

     // AcqParams:
     v["AcqParams"]["AduGain"] = acqSetup.smrtSensor().PhotoelectronSensitivity();

     // Chipinfo:
     const auto& fm = acqSetup.GetChipLayout()->FilterMap();
     for (unsigned int j = 0; j < fm.size(); ++j)
     {
         v["ChipInfo"]["FilterMap"][j] = fm[j];
     }
     v["ChipInfo"]["ImagePsf"] = Json::arrayValue;
     if (numChannels != acqSetup.psfs.size())
     {
         if (acqSetup.chipClass() == ChipClass::Spider && (numChannels == 1 && acqSetup.psfs.size() == 2))
         {
             // legacy ugliness.
         }
         else
         {
             PBLOG_ERROR << "numChannels:" << numChannels << " acqSetup.psfs.size(): " <<
                 acqSetup.psfs.size();
             throw PBException("numChannels not equal to acqSetup.psfs.size()");
         }
     }
     for (unsigned int j = 0; j < numChannels; ++j)
     {
         v["ChipInfo"]["ImagePsf"][j] = Json::arrayValue;
         const auto ma = acqSetup.psfs[j].AsMultiArray();
         for (unsigned int k = 0; k < ma.size(); ++k)
         {
             v["ChipInfo"]["ImagePsf"][j][k] = Json::arrayValue;
             for (unsigned int l = 0; l < ma[k].size(); ++l)
             {
                 v["ChipInfo"]["ImagePsf"][j][k][l] = ma[k][l];
             }
         }
     }
     v["ChipInfo"]["CrosstalkFilter"] = Json::arrayValue;
     for (int row = 0; row < acqSetup.crosstalkFilter.NumRows(); row++)
     {
         v["ChipInfo"]["CrosstalkFilter"][row] = Json::arrayValue;
         for (int col = 0; col < acqSetup.crosstalkFilter.NumCols(); col++)
         {
             v["ChipInfo"]["CrosstalkFilter"][row][col] = acqSetup.crosstalkFilter(row, col);
         }
     }
     v["ChipInfo"]["AnalogRefSnr"] = v["refDwsSnr"];
     v["ChipInfo"]["AnalogRefSpectrum"] = v["refSpectrum"];
     v["ChipInfo"]["LayoutName"] = v["chipLayoutName"];

     // DyeSet:
     v["DyeSet"]["NumAnalogs"] = acqSetup.analogs.size();

     std::string baseMap = "";
     v["DyeSet"]["AnalogSpectra"] = Json::arrayValue;
     v["DyeSet"]["ExcessNoiseCV"] = Json::arrayValue;
     v["DyeSet"]["IpdMean"] = Json::arrayValue;
     v["DyeSet"]["PulseWidthMean"] = Json::arrayValue;
     v["DyeSet"]["RelativeAmp"] = Json::arrayValue;
     for (unsigned int i = 0; i < acqSetup.analogs.size(); ++i)
     {
         const auto& analog = acqSetup.analogs[i];
         baseMap += analog.baseLabel;
         v["DyeSet"]["AnalogSpectra"][i] = Json::arrayValue;
         for (unsigned int j = 0; j < numChannels; ++j)
         {
             v["DyeSet"]["AnalogSpectra"][i][j] = analog.dyeSpectrum[j];
         }
         v["DyeSet"]["RelativeAmp"][i] = analog.RelativeAmplitude();
         v["DyeSet"]["ExcessNoiseCV"][i] = analog.excessNoiseCV;
         v["DyeSet"]["PulseWidthMean"][i] = analog.pulseWidth;
         v["DyeSet"]["IpdMean"][i] = analog.interPulseDistance;
     }
     v["DyeSet"]["BaseMap"] = baseMap;

     // These don't seem to be updated properly in the trc.h5:
     // ! Update after the fact?
     // v["RunInfo"]["SequencingChemistry"] = "unknown";
     // v["RunInfo"]["BindingKit"] = "NA";
     // v["RunInfo"]["SequencingKit"] = "NA";
     // v["AcqParams"]["CameraBias"] = 0.0;
     // v["AcqParams"]["CameraBiasStd"] = 0.0;
     // v["AcqParams"]["CameraGain"] = 1.0;
     // v["AcqParams"]["CameraType"] = 200;
     //
     // These don't seem that important to have in the baz header:
     // v["RunInfo"]["Control"] = "NA";
     // v["InstrumentName"] = PacBio::POSIX::gethostname();
     // v["IsControlUsed"] = 0;
     // v["MovieName"] = "";
     // v["MoviePath"] = "";
     // v["PlatformID"] = 0;
     // v["PlatformName"] = "unkown";
     //
     // These don't seem plausible in the bam header:
     // // Framerate is added elsewhere
     // // ! Needs to be updated after the fact
     // v["AcqParams"]["HotStartFrame"] = 0;
     // v["AcqParams"]["HotStartFrameValid"] = 0;
     // // ! Needs to be updated after the fact
     // v["AcqParams"]["LaserOnFrame"] = 0;
     // v["AcqParams"]["LaserOnFrameValid"] = 0;
     // v["AcqParams"]["NumFrames"] = 0;
     // //AcquisitionXML
     // v["acquisitionXML"] = acquisitionXML();

     v.removeMember("token");
     v.removeMember("acquisitionXML");
     v.removeMember("analogs");
     v.removeMember("baseMap");
     v.removeMember("crosstalkFilter");
     v.removeMember("psfs");
     v.removeMember("chipLayoutName");

     return PacBio::IPC::RenderJSON(v);
 }

/// Create a XML blob that is usually created by ICS. This sets up the dye parameters.
/// I'd *really* like to get rid of this.
std::string Setup::CreateDummyXML(const ChipClass chipClass0)
{
    std::string xmlString = R"(
<?xml version="1.0" encoding="utf-8"?>
  <PacBioDataModel>
    <ProjectContainer>
      <Runs>
        <Run>
           <Collections>
      <CollectionMetadata CreatedAt="0001-01-01T00:00:00" ModifiedAt="0001-01-01T00:00:00" Status="Ready" InstrumentId="Inst1234" InstrumentName="Inst1234">
      <AutomationParameters>
)";
    if (chipClass0 == ChipClass::Sequel)
    {
        // these values are copied from BasecallerConfig.h SetSequelDefaults()
        xmlString += R"(
      <AutomationParameter xsi:type="SequencingChemistry" Name="2C2A" CreatedAt="0001-01-01T00:00:00" ModifiedAt="0001-01-01T00:00:00">
        <DyeSet Name="Super duper dye set" CreatedAt="0001-01-01T00:00:00" ModifiedAt="0001-01-01T00:00:00">
        <Analogs>
        <Analog Name="AAnalog" CreatedAt="0001-01-01T00:00:00" ModifiedAt="0001-01-01T00:00:00" Base="A" SpectralAngle="1.4446" RelativeAmplitude="0.9" Wavelength="665"
             IntraPulseXsnCV="0.1" IpdMeanSeconds="0.175" PulseWidthMeanSeconds="0.1125" />
        <Analog Name="CAnalog" CreatedAt="0001-01-01T00:00:00" ModifiedAt="0001-01-01T00:00:00" Base="C" SpectralAngle="1.4446" RelativeAmplitude="1.8" Wavelength="665"
             IntraPulseXsnCV="0.1" IpdMeanSeconds="0.175" PulseWidthMeanSeconds="0.1125" />
        <Analog Name="GAnalog" CreatedAt="0001-01-01T00:00:00" ModifiedAt="0001-01-01T00:00:00" Base="G" SpectralAngle="0.8205" RelativeAmplitude="1.3" Wavelength="600"
             IntraPulseXsnCV="0.1" IpdMeanSeconds="0.225" PulseWidthMeanSeconds="0.0875" />
        <Analog Name="TAnalog" CreatedAt="0001-01-01T00:00:00" ModifiedAt="0001-01-01T00:00:00" Base="T" SpectralAngle="0.8205" RelativeAmplitude="3.3" Wavelength="600"
             IntraPulseXsnCV="0.1" IpdMeanSeconds="0.1625" PulseWidthMeanSeconds="0.125" />
        </Analogs>
        </DyeSet>
      </AutomationParameter>)";
    }
    else if (chipClass0 == ChipClass::Spider )
    {
        xmlString += R"(
      <AutomationParameter xsi:type="SequencingChemistry" Name="1C4A" CreatedAt="0001-01-01T00:00:00" ModifiedAt="0001-01-01T00:00:00">
        <DyeSet Name="Spider duper dye set" CreatedAt="0001-01-01T00:00:00" ModifiedAt="0001-01-01T00:00:00">
        <Analogs>
        <Analog Name="AAnalog" CreatedAt="0001-01-01T00:00:00" ModifiedAt="0001-01-01T00:00:00" Base="A" RelativeAmplitude="0.9" Wavelength="600" />
        <Analog Name="CAnalog" CreatedAt="0001-01-01T00:00:00" ModifiedAt="0001-01-01T00:00:00" Base="C" RelativeAmplitude="1.8" Wavelength="600" />
        <Analog Name="GAnalog" CreatedAt="0001-01-01T00:00:00" ModifiedAt="0001-01-01T00:00:00" Base="G" RelativeAmplitude="1.3" Wavelength="600" />
        <Analog Name="TAnalog" CreatedAt="0001-01-01T00:00:00" ModifiedAt="0001-01-01T00:00:00" Base="T" RelativeAmplitude="3.3" Wavelength="600" />
        </Analogs>
        </DyeSet>
      </AutomationParameter>)";
    }
    else if (chipClass0 == ChipClass::DONT_CARE)
    {
        // don't do anything
    }
    else
    {
        throw PBException("Unsupported chipclass:" + chipClass0.toString());
    }

    xmlString += R"(</AutomationParameters>
      </CollectionMetadata>
      </Collections>
      </Run>
      </Runs>
      </ProjectContainer>
      </PacBioDataModel>
      )";
    return xmlString;
}

/// This analog set is ancient, and should NOT BE USED FOR BASECALLING. It is useful only for using pa-acq-cli to
/// capture a trc.h5 or mov.h5 file. I would delete it, but it will probably break a bunch of tests.
void Setup::CreateDummyAnalogs()
{
    if ( chipClass() == ChipClass::Sequel)
    {
        Json()["analogs"] = ParseJSON(R"(
          [
          {"analogName":"AAnalog", "base":"A", "spectralAngle":1.4446,"wavelength":665, "relativeAmplitude":0.9,
             "intraPulseXsnCV":0.1, "ipdMeanSeconds":0.175,  "pulseWidthMeanSeconds":0.1125},
          {"analogName":"CAnalog", "base":"C", "spectralAngle":1.4446,"wavelength":665, "relativeAmplitude":1.8,
             "intraPulseXsnCV":0.1, "ipdMeanSeconds":0.175,  "pulseWidthMeanSeconds":0.1125},
          {"analogName":"GAnalog", "base":"G", "spectralAngle":0.8205,"wavelength":600, "relativeAmplitude":1.3,
             "intraPulseXsnCV":0.1, "ipdMeanSeconds":0.225,  "pulseWidthMeanSeconds":0.0875},
          {"analogName":"TAnalog", "base":"T", "spectralAngle":0.8205,"wavelength":600, "relativeAmplitude":3.3,
             "intraPulseXsnCV":0.1, "ipdMeanSeconds":0.1625, "pulseWidthMeanSeconds":0.125}
          ]
          )");
    }
    else if ( chipClass() == ChipClass::Spider )
    {
        Json()["analogs"] = ParseJSON(R"(
          [
          {"analogName":"AAnalog", "base":"A", "spectrumValues":[1.0], "wavelength":600, "relativeAmplitude":0.9},
          {"analogName":"CAnalog", "base":"C", "spectrumValues":[1.0], "wavelength":600, "relativeAmplitude":1.8},
          {"analogName":"GAnalog", "base":"G", "spectrumValues":[1.0], "wavelength":600, "relativeAmplitude":1.3},
          {"analogName":"TAnalog", "base":"T", "spectrumValues":[1.0], "wavelength":600, "relativeAmplitude":3.3}
          ]
          )");
    }
    PostImportAnalogs();
}


void Setup::SetDefaultPsfs()
{
    Json::Value psfsUnity;
    // unity PSFs
    psfsUnity[0][0][0] = 1;
    psfsUnity[1][0][0] = 1;
    SetPsfs(psfsUnity);
}

void Setup::SetDefaultSequelAnalogs()
{
  Json()["analogs"] = ParseJSON(R"(
  [
  {"analogName":"AAnalog", "base":"A", "spectralAngle":1.3756,"wavelength":665, "relativeAmplitude":0.9},
  {"analogName":"CAnalog", "base":"C", "spectralAngle":1.3756,"wavelength":665, "relativeAmplitude":1.8},
  {"analogName":"GAnalog", "base":"G", "spectralAngle":0.5855,"wavelength":600, "relativeAmplitude":1.3},
  {"analogName":"TAnalog", "base":"T", "spectralAngle":0.5855,"wavelength":600, "relativeAmplitude":3.3}
  ]
  )");
}

/// Used for unit testing only!
void Setup::SetDefaultSpiderAnalogs()
{
    Json()["analogs"] = ParseJSON(R"(
  [
  {"analogName" : "AAnalog","base" : "A","diffusionXsnCV" : 0.24,"interPulseXsnCV" : 0.1,"intraPulseXsnCV" : 0.1,"ipd2SlowStepRatio":0,"ipdMeanSeconds":0.16,
       "pulseWidthMeanSeconds" : 0.166, "pw2SlowStepRatio" : 3.2,"relativeAmplitude" : 0.68,"spectrumValues" : [ 1 ], "wavelength" : 665 },
  {"analogName" : "CAnalog","base" : "C","diffusionXsnCV" : 0.36,"interPulseXsnCV" : 0.1,"intraPulseXsnCV" : 0.1,"ipd2SlowStepRatio": 0,"ipdMeanSeconds" : 0.19,
       "pulseWidthMeanSeconds" : 0.209, "pw2SlowStepRatio" : 3.2,"relativeAmplitude" : 1.00,"spectrumValues" : [ 1 ], "wavelength" : 665 },
  {"analogName" : "GAnalog","base" : "G","diffusionXsnCV" : 0.27,"interPulseXsnCV" : 0.1,"intraPulseXsnCV" : 0.1,"ipd2SlowStepRatio": 0,"ipdMeanSeconds" : 0.11,
       "pulseWidthMeanSeconds" : 0.193, "pw2SlowStepRatio" : 3.2,"relativeAmplitude" : 0.27,"spectrumValues" : [ 1 ], "wavelength" : 665 },
  {"analogName" : "TAnalog","base" : "T","diffusionXsnCV" : 0.34,"interPulseXsnCV" : 0.1,"intraPulseXsnCV" : 0.1,"ipd2SlowStepRatio": 0,"ipdMeanSeconds" : 0.16,
       "pulseWidthMeanSeconds" : 0.163, "pw2SlowStepRatio" : 3.2,"relativeAmplitude" : 0.43,"spectrumValues" : [ 1 ], "wavelength" : 665 }
  ]
  )");
}

// this is called whenever the ROIs are modified, so as to keep the JSON in sync with
// the new ROIs.
void Setup::UpdateRoiMetaData()
{
    Json::Value& roiMetaData = Json()["roiMetaData"];

#if 0
    if (fileROI && sensorROI != fileROI->SensorROI())
    {
        PBLOG_WARN << "sensorROI:" << sensorROI;
        PBLOG_WARN << "fileROI->sensorROI:" << fileROI->SensorROI();
        throw PBException("FileROI sensorROI is not compatible with sensorROI. Please set sensorROI of setup object first.");
    }
    if (sequencingROI && sensorROI != sequencingROI->SensorROI())
    {
        throw PBException("sequencingROI sensorROI is not compatible with sensorROI. Please set sensorROI of setup object first.");
    }

    roiMetaData["sensorPixelRowMin"]  = sensorROI.PhysicalRowOffset();
    roiMetaData["sensorPixelRowSize"] = sensorROI.PhysicalRows();
    roiMetaData["sensorPixelColMin"]  = sensorROI.PhysicalColOffset();
    roiMetaData["sensorPixelColSize"] = sensorROI.PhysicalCols();
#endif

    if (fileROI)
    {
        roiMetaData["traceFilePixelROI"] = fileROI->GetJson();
    }
    else
    {
        roiMetaData.removeMember("traceFilePixelROI");
    }
    if (sequencingROI)
    {
        roiMetaData["sequencingPixelROI"] = sequencingROI->GetJson();
    }
    else
    {
        roiMetaData.removeMember("sequencingPixelROI");
    }

    roiMetaData["remapVroiOption"] = false; // this is obsolete
}

void Setup::Sanitize()
{
    if (bazfile() == "")
    {
        if (sequencingROI)
        {
            PBLOG_WARN << "removing sequencing ROI from Acquisition::Setup as no bazFile was set.";
        }
        sequencingROI.reset();
    }
    else if (!sequencingROI)
    {
        if (bazfile() == "")
        {
            PBLOG_WARN << "setting sequencing ROI in Acquisition::Setup to full chip.";
        }
        sequencingROI.reset( new SequelRectangularROI(sensorROI));
    }

    if (hdf5output() == "")
    {
        if (fileROI)
        {
            PBLOG_WARN << "Removing file ROI from Acquisition::Setup as no hdf5output was set.";
        }
        fileROI.reset();
    }
    else if (!fileROI)
    {
        if (hdf5output() == "")
        {
            PBLOG_WARN << "Setting file ROI in Acquisition::Setup to null.";
        }
        fileROI.reset( new SequelRectangularROI(SequelROI::Null()));
    }

    UpdateRoiMetaData();
}


void Setup::RenderCrosstalkFilter(Json::Value& originalJson)
{
    if (crosstalkFilter.NumRows() <= 1 && psfs.size() > 0)
    {
        // calculate the crosstalk filter based on the channel PSFs.
        // For now, we only support 2 channels and we average the PSFs together.
        if (psfs.size() != 2)
        {
            PBLOG_ERROR << "AcquisitionSetup with bad PSFs:\n" << Json();
            throw PBException("Must have exactly 2 PSFs (green,red) to calculate crosstalk decovolution kernel,"
                              " size was " + std::to_string(psfs.size()));
        }
        crosstalkFilter = RenderCrosstalkFilter(psfs);
        PBLOG_INFO << "crosstalk filter = meanPsf^-1:\n" << crosstalkFilter;

        // Only correct spectrum values if running Sequel.
        if (GetPrimaryConfig().chipClass() == ChipClass::Sequel)
        {
            // Now correct the spectrum values
            // in both the "setup" and the originalJson analog set.
            auto& jsonAnalogs = originalJson["analogs"];

            if (jsonAnalogs.isArray())
            {
                if (analogs.size() != jsonAnalogs.size())
                {
                    throw PBException("internal inconsistency between JSON and Acquisition::Setup");
                }
                for (auto& setupAnalog : analogs)
                {
                    // Find the json entry corresponding to this analog
                    uint32_t ja = 0;
                    for (; ja < jsonAnalogs.size(); ++ja)
                    {
                        auto& analogJson = jsonAnalogs[ja];
                        std::string base = analogJson["base"].asString();
                        if (setupAnalog.baseLabel == base.at(0)) break;
                    }

                    // Make sure it was found
                    if (ja == jsonAnalogs.size())
                    {
                        throw PBException("analog label inconsistency between JSON and Acquisition::Setup");
                    }
                    auto& jsonAnalog = jsonAnalogs[ja];

                    // Compute the x-talk corrected value
                    Spectrum raw(setupAnalog.dyeSpectrum);
                    Spectrum corrected = CorrectSpectrum(crosstalkFilter, raw);

                    // save corrected values to C++ object
                    setupAnalog.dyeSpectrum = corrected;

                    // save corrected values to JSON object
                    Json::Value& spectrumValues = jsonAnalog["spectrumValues"];
                    spectrumValues[0] = corrected[0];
                    spectrumValues[1] = corrected[1];
                }
            }
            else
            {
                PBLOG_WARN << "PSFs given, but no analogs found of which to correct the spectral values";
            }
        }
    }
    else
    {
        PBLOG_INFO << "crosstalk filter:\n" << crosstalkFilter;
    }
}

Kernel Setup::RenderCrosstalkFilter(const std::vector<Kernel>& psfs0)
{
    // calculate the crosstalk filter based on the channel PSFs.

    // For now, we only support 2 channels and we average the PSFs together.
    if (psfs0.size() != 2)
    {
        throw PBException("Must have exactly 2 PSFs (green,red) to calculate crosstalk decovolution kernel,"
                          " size was " + std::to_string(psfs0.size()));
    }
    for (const auto& psf: psfs0)
    {
        if (psf.Sum() < 0.9 || psf.Sum() > 1.1)
        {
            throw PBException("PSFs are not normalized. Sum of PSF coefficients must be between 0.9 and 1.1");
        }
    }

    Kernel gPsf(psfs0[0].Resize(5,5).Normalize(1.0));
    Kernel rPsf(psfs0[1].Resize(5,5).Normalize(1.0));

    Kernel sumPsf = gPsf + rPsf;
    Kernel meanPsf = sumPsf.Normalize(1.0);
    PBLOG_INFO << "gPsf:" << gPsf;
    PBLOG_INFO << "rPsf:" << rPsf;
    PBLOG_INFO << "meanPsf:" << meanPsf;

    Kernel crosstalkFilter1 = meanPsf.InvertTo7x7().Flip();
    PBLOG_INFO << "crosstalk filter = meanPsf^-1:\n" << crosstalkFilter1;

    return crosstalkFilter1;
}


} // namespace

  ////////////////////////////////////////////////////////////////////////////////

  std::unique_ptr<SequelRectangularROI> AcquisitionMetaData::ParseXmlForROI(const std::string& /*xmlText*/)
  {
      throw PBException("unsupported");
#if 0
      PBLOG_TRACE << "SaverThread::ParseXmlForROI xmlText:" << xmlText;

      PacBio::Text::PBXml xml(xmlText);
      auto z = xml.Down("PacBioDataModel")
              .Down("ProjectContainer")
              .Down("Runs")
              .Down("Run")
              .Down("Collections")
              .Down("CollectionMetadata");
      auto z1 = z.AsPugiXmlNode("AutomationParameters");
      auto SensorPixelRowMin = z1.find_child_by_attribute("AutomationParameter", "Name", "SensorPixelRowMin");
      auto SensorPixelColMin = z1.find_child_by_attribute("AutomationParameter", "Name", "SensorPixelColMin");
      auto SensorPixelRowSize = z1.find_child_by_attribute("AutomationParameter", "Name", "SensorPixelRowSize");
      auto SensorPixelColSize = z1.find_child_by_attribute("AutomationParameter", "Name", "SensorPixelColSize");

      auto SequencingPixelRowMin = z1.find_child_by_attribute("AutomationParameter", "Name", "SequencingPixelRowMin");
      auto SequencingPixelColMin = z1.find_child_by_attribute("AutomationParameter", "Name", "SequencingPixelColMin");
      auto SequencingPixelRowSize = z1.find_child_by_attribute("AutomationParameter", "Name", "SequencingPixelRowSize");
      auto SequencingPixelColSize = z1.find_child_by_attribute("AutomationParameter", "Name", "SequencingPixelColSize");

      RowPixels rowMin(SequencingPixelRowMin.attribute("SimpleValue").as_uint(0));
      ColPixels colMin(SequencingPixelColMin.attribute("SimpleValue").as_uint(0));
      RowPixels rows(SequencingPixelRowSize.attribute("SimpleValue").as_uint(1144));
      ColPixels cols(SequencingPixelColSize.attribute("SimpleValue").as_uint(2048));

      SequelSensorROI sensorROI(SensorPixelRowMin.attribute("SimpleValue").as_uint(0),
                                SensorPixelColMin.attribute("SimpleValue").as_uint(0),
                                SensorPixelRowSize.attribute("SimpleValue").as_uint(Sequel::maxPixelRows),
                                SensorPixelColSize.attribute("SimpleValue").as_uint(Sequel::maxPixelCols),
                                Sequel::numPixelRowsPerZmw,
                                Sequel::numPixelColsPerZmw
      );

      SequelRectangularROI* roi = new SequelRectangularROI(rowMin, colMin, rows, cols, sensorROI);
      std::unique_ptr<SequelRectangularROI> ret(roi);

      if ((roi->NumPixelCols() & 31) != 0)
      {
          PBLOG_INFO << *roi;
          throw PBException("columns must be multiple of 32");
      }

      return ret ;
#endif
  }

  SequelSensorROI AcquisitionMetaData::ParseJSONForSensorROI(const Json::Value& roiMetaData,
                                                             const SequelSensorROI& sensorROI)
  {
      if (roiMetaData.isMember("sensorPixelROI"))
      {
          return SequelSensorROI(roiMetaData["sensorPixelROI"][0].asInt(),
                                 roiMetaData["sensorPixelROI"][1].asInt(),
                                 roiMetaData["sensorPixelROI"][2].asInt(),
                                 roiMetaData["sensorPixelROI"][3].asInt(),
                                 sensorROI.NumPixelRowsPerZmw(),
                                 sensorROI.NumPixelColsPerZmw()
          );
      }
      else
      {
          return SequelSensorROI(roiMetaData.get("sensorPixelRowMin", 0).asUInt(),
                                 roiMetaData.get("sensorPixelColMin", 0).asUInt(),
                                 roiMetaData.get("sensorPixelRowSize", sensorROI.PhysicalRows()).asUInt(),
                                 roiMetaData.get("sensorPixelColSize", sensorROI.PhysicalCols()).asUInt(),
                                 sensorROI.NumPixelRowsPerZmw(),
                                 sensorROI.NumPixelColsPerZmw()
          );
      }
  }

  std::unique_ptr<SequelROI> AcquisitionMetaData::ParseJSONForSequencingROI(const Json::Value& roiMetaData,
                                                                            const SequelSensorROI& sensorROI)
  {
      std::unique_ptr<SequelROI> ret;

      if (roiMetaData.isMember("sequencingPixelROI"))
      {
          SequelSparseROI* roi = new SequelSparseROI(roiMetaData["sequencingPixelROI"], sensorROI);
          ret.reset(roi);
      }
      else
      {
          RowPixels rowMin(roiMetaData.get("sequencingPixelRowMin", 0).asUInt());
          ColPixels colMin(roiMetaData.get("sequencingPixelColMin", 0).asUInt());
          RowPixels rows(roiMetaData.get("sequencingPixelRowSize", sensorROI.PhysicalRows()).asUInt());
          ColPixels cols(roiMetaData.get("sequencingPixelColSize", sensorROI.PhysicalCols()).asUInt());

          if ((cols.Value() & 31) != 0)
          {
              throw PBException("columns must be multiple of 32, was " + std::to_string(cols.Value()));
          }

          SequelRectangularROI* roi = new SequelRectangularROI(rowMin, colMin, rows, cols, sensorROI);
          ret.reset(roi);
          roi->CheckROI();
      }


      return ret;
  }

  std::unique_ptr<SequelROI> AcquisitionMetaData::ParseJSONForTraceFileROI(const Json::Value& roiMetaData,
                                                                           const SequelSensorROI& sensorROI)
  {
      std::unique_ptr<SequelROI> ret;

      if (roiMetaData.isMember("traceFilePixelROI"))
      {
          // We allow 4 different JSON specifications for ROI.  These are all done for the convenience of the end user
          // as essentially they could all be specified as a SequelSparseROI, but knowing that the ROI is just a
          // rectangle makes computation simpler and also simpler to print to a iostream for debugging.
          //
          // case 1: `null` - If the JSON is `null`, then this is assumed to be a 0 pixel rectangular ROI.
          // case 2: `[x_origin, y_origin, x_size, y_size]` - a rectangular ROI
          // case 3: `[[x_origin, y_origin, x_size, y_size]]` - this is technically a sparse ROI, but equivalent to a rectangular ROI
          // case 4: `[[x_origin1, y_origin1, x_size1, y_size1],[x_origin2,y_origin2,x_size2,y_size2],...] - a sparse ROI
          //  for case 4, we allow the SequelSparseROI constructor to throw if the JSON format is invalid, so
          //              this code doesn't sanity check it.  Basically, anything that can't be identified as a rectangular ROI
          //              is tossed to the SequelSparseROI constructor to deal with it.
          SequelROI* roi;
          const Json::Value& roiJson(roiMetaData["traceFilePixelROI"]);
          if (roiJson.isNull())
          {
              // case 1 - null ROI than can be represented by a Rectangular ROI
              roi = new SequelRectangularROI(0,0,0,0, sensorROI);
          }
          else if (roiJson.isArray() && roiJson.size() == 4 && roiJson[0].isInt())
          {
              // case 2 - rectangular ROI
              roi = new SequelRectangularROI(roiJson, sensorROI);
          }
          else if (roiJson.isArray() &&
                   roiJson.size() == 1 &&
                   roiJson[0].isArray() &&
                   roiJson[0].size() == 4 &&
                   roiJson[0][0].isInt())
          {
              // case 3 - sparse ROI than degenerates to rectangular ROI (i.e. one rectangle)
              roi = new SequelRectangularROI(roiJson[0], sensorROI);
          }
          else
          {
              // case 4 - multiple rectangles in ROI
              roi = new SequelSparseROI(roiJson, sensorROI);
          }
          ret.reset(roi);
      }
      else
      {
          // very old way of specifying ROI that is probably not in use any more.
          RowPixels rowMin(roiMetaData.get("traceFilePixelRowMin", 0).asUInt());
          ColPixels colMin(roiMetaData.get("traceFilePixelColMin", 0).asUInt());
          RowPixels rows(roiMetaData.get("traceFilePixelRowSize", 256).asUInt());
          ColPixels cols(roiMetaData.get("traceFilePixelColSize", 256).asUInt());

          if ((cols.Value() & 31) != 0)
          {
              throw PBException("columns must be multiple of 32, was " + std::to_string(cols.Value()));
          }

          SequelRectangularROI* roi = new SequelRectangularROI(rowMin, colMin, rows, cols, sensorROI);
          ret.reset(roi);
          roi->CheckROI();
      }

      return ret;
  }


  void AcquisitionMetaData::ParseXmlForChemistry(const std::string& xmlText, std::string& baseMap,
                                                 std::vector<double>& spectralAngle, std::vector<double>& amplitude,
                                                 std::vector<double>& excessNoiseCV, std::vector<double>& ipd,
                                                 std::vector<double>& pw)
  {
      baseMap = "";
      spectralAngle.clear();
      amplitude.clear();
      excessNoiseCV.clear();
      ipd.clear();
      pw.clear();

      PacBio::Text::PBXml xml(xmlText);
      auto z = xml.Down("PacBioDataModel")
              .Down("ProjectContainer")
              .Down("Runs")
              .Down("Run")
              .Down("Collections")
              .Down("CollectionMetadata");
      auto z1 = z.AsPugiXmlNode("AutomationParameters");
      auto zz = z1.find_child_by_attribute("AutomationParameter", "xsi:type", "SequencingChemistry");
      auto zzz = zz.child("DyeSet").child("Analogs");
      for (auto zzzz = zzz.child("Analog"); zzzz; zzzz = zzzz.next_sibling("Analog"))
      {
          baseMap += zzzz.attribute("Base").as_string("X");
          spectralAngle.push_back(zzzz.attribute("SpectralAngle").as_double(0.0));
          amplitude.push_back(zzzz.attribute("RelativeAmplitude").as_double(0.0));
          excessNoiseCV.push_back(zzzz.attribute("IntraPulseXsnCV").as_double(0.0));
          ipd.push_back(zzzz.attribute("IpdMeanSeconds").as_double(0.0));
          pw.push_back(zzzz.attribute("PulseWidthMeanSeconds").as_double(0.0));
      }
  }

}} // namespaces
