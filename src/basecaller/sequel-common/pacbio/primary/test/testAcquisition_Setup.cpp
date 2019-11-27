//
// Created by mlakata on 7/10/15.
//


#include <gtest/gtest.h>

#include <pacbio/dev/gtest-extras.h>
#include <pacbio/primary/Acquisition_Setup.h>
#include <pacbio/ipc/JSON.h>
#include <pacbio/PBException.h>
#include <pacbio/logging/Logger.h>
#include <pacbio/primary/ChipLayoutSpider.h>

using namespace PacBio::Primary;
using namespace PacBio::Logging;

class Acquisition_SetupTest : public ::testing::Test
{
    void SetUp() override
    {
        // clean up any previous tests that mucked with the primaryConfig
        ResetPrimaryConfig();
        GetPrimaryConfig().platform = Platform::Sequel1PAC1;
        GetPrimaryConfig().chipClass = ChipClass::Sequel;
        sequelLayout = ChipLayout::Factory("SequEL_4.0_RTO3");
        spiderLayout = ChipLayout::Factory("Spider_1p0_NTO");
    }
    void TearDown() override
    {
        // be nice to following tests that rely on primaryConfig
        ResetPrimaryConfig();
    }
public:
    std::unique_ptr<ChipLayout> sequelLayout;
    std::unique_ptr<ChipLayout> spiderLayout;
};

TEST_F(Acquisition_SetupTest,JsonParsing)
{
    LogSeverityContext _(LogLevel::ERROR);

    const char* payload = R"(

{"token":"1234","hdf5output":"/data/pa/foo.trc.h5","cameraConfigKey":567,
"bazfile":"hello.baz",
 "instrumentName":"Barney","chipId":"exp5","numFrames":51200,"exposure":0.01,
 "roiMetaData":{
    "sequencingPixelRowMin" : 0,
    "sequencingPixelColMin" : 0,
    "sequencingPixelRowSize": 1,
    "sequencingPixelColSize": 32,
    "traceFilePixelRowMin" : 1,
    "traceFilePixelColMin" : 32,
    "traceFilePixelRowSize": 2,
    "traceFilePixelColSize": 64,
    "sensorPixelRowMin" : 0,
    "sensorPixelColMin" :0,
    "sensorPixelRowSize":1144,
    "sensorPixelColSize":2048
},
"analogs":[
    {"analogName":"AAnalog", "base":"A", "spectralAngle":1.3756,"wavelength":665, "relativeAmplitude":0.9},
    {"analogName":"CAnalog", "base":"C", "spectralAngle":1.3756,"wavelength":665, "relativeAmplitude":1.8},
    {"analogName":"GAnalog", "base":"G", "spectralAngle":0.5855,"wavelength":600, "relativeAmplitude":1.3},
    {"analogName":"TAnalog", "base":"T", "spectralAngle":0.5855,"wavelength":600, "relativeAmplitude":3.3}
],
 "acquisitionXML":"<some>xml</some>",
 "basecallerVersion": "1.2.3",
 "basecaller": {},
 "chipClass": "Sequel",
 "chipLayoutName": "SequEL_4.0_RTO3",
 "readout":"bases_without_qvs","metricsVerbosity":"high",
 "crosstalkFilter":[[ 0.0,1.0,2.0],[3.0,4,5],[6,7,8]],
 "photoelectronSensitivity": 0.8,
 "refDwsSnr" : 23.0,
 "refSpectrum" :
  [
    0.10197792202234268,
    0.8980221152305603
  ]
}
)";


    Json::Value json = PacBio::IPC::ParseJSON(payload);

    Acquisition::Setup setup(json);

    EXPECT_EQ("hello.baz",setup.bazfile());
    EXPECT_EQ("1234",setup.token());
    EXPECT_EQ("/data/pa/foo.trc.h5", setup.hdf5output());
    EXPECT_EQ(567,setup.cameraConfigKey());
    EXPECT_EQ("Barney",setup.instrumentName());
    EXPECT_EQ("exp5",setup.chipId());
    EXPECT_EQ(51200,setup.numFrames());
    EXPECT_FLOAT_EQ(0.01,setup.exposure());
    EXPECT_FLOAT_EQ(100.0,setup.expectedFrameRate());

    auto zmws = setup.GetSequencingZmwMap();
    EXPECT_EQ(16,zmws.size());
    zmws = setup.GetTraceFileZmwMap();
    EXPECT_EQ(64,zmws.size());

    EXPECT_EQ(PacBio::SmrtData::Readout::BASES_WITHOUT_QVS, setup.readout());
    EXPECT_EQ(PacBio::SmrtData::MetricsVerbosity::HIGH,setup.metricsVerbosity());
    ASSERT_EQ(PacBio::Primary::SequelROI::ROI_Type_e::Rectangular,setup.SequencingROI().Type());
    const SequelRectangularROI* rectSeqRoi = dynamic_cast<const SequelRectangularROI*>(&setup.SequencingROI());
    EXPECT_NE(nullptr, rectSeqRoi);
    EXPECT_EQ(1, rectSeqRoi->NumPixelRows());
    EXPECT_EQ(32, rectSeqRoi->NumPixelCols());

    ASSERT_EQ(PacBio::Primary::SequelROI::ROI_Type_e::Rectangular,setup.FileROI().Type());
    const SequelRectangularROI* rectTraceRoi = dynamic_cast<const SequelRectangularROI*>(&setup.FileROI());
    EXPECT_NE(nullptr, rectTraceRoi);
    EXPECT_EQ(2, rectTraceRoi->NumPixelRows());
    EXPECT_EQ(64, rectTraceRoi->NumPixelCols());

    EXPECT_EQ("1.2.3", setup.basecallerVersion());
    EXPECT_EQ("<some>xml</some>",setup.acquisitionXML());
    ASSERT_EQ(3, setup.crosstalkFilter.NumRows());
    ASSERT_EQ(3, setup.crosstalkFilter.NumCols());
    EXPECT_DOUBLE_EQ(0, setup.crosstalkFilter(0,0));
    EXPECT_DOUBLE_EQ(1, setup.crosstalkFilter(0,1));
    EXPECT_DOUBLE_EQ(2, setup.crosstalkFilter(0,2));
    EXPECT_DOUBLE_EQ(3, setup.crosstalkFilter(1,0));
    EXPECT_DOUBLE_EQ(4, setup.crosstalkFilter(1,1));
    EXPECT_DOUBLE_EQ(5, setup.crosstalkFilter(1,2));
    EXPECT_DOUBLE_EQ(6, setup.crosstalkFilter(2,0));
    EXPECT_DOUBLE_EQ(7, setup.crosstalkFilter(2,1));
    EXPECT_DOUBLE_EQ(8, setup.crosstalkFilter(2,2));

#ifdef WITHOUT_ANALOGMODE
    EXPECT_DOUBLE_EQ(1.3756, setup.analogSpectralAngle[0]);
    EXPECT_DOUBLE_EQ(1.3756, setup.analogSpectralAngle[1]);
    EXPECT_DOUBLE_EQ(0.5855, setup.analogSpectralAngle[2]);
    EXPECT_DOUBLE_EQ(0.5855, setup.analogSpectralAngle[3]);
    EXPECT_DOUBLE_EQ(665, setup.analogWavelength[0]);
    EXPECT_DOUBLE_EQ(665, setup.analogWavelength[1]);
    EXPECT_DOUBLE_EQ(600, setup.analogWavelength[2]);
    EXPECT_DOUBLE_EQ(600, setup.analogWavelength[3]);
    EXPECT_DOUBLE_EQ(0.9, setup.analogAmplitude[0]);
    EXPECT_DOUBLE_EQ(1.8, setup.analogAmplitude[1]);
    EXPECT_DOUBLE_EQ(1.3, setup.analogAmplitude[2]);
    EXPECT_DOUBLE_EQ(3.3, setup.analogAmplitude[3]);
#else
    EXPECT_FLOAT_EQ(0.5855, setup.analogs[0].SpectralAngle());
    EXPECT_FLOAT_EQ(0.5855, setup.analogs[1].SpectralAngle());
    EXPECT_FLOAT_EQ(1.3756, setup.analogs[2].SpectralAngle());
    EXPECT_FLOAT_EQ(1.3756, setup.analogs[3].SpectralAngle());
    EXPECT_FLOAT_EQ(3.3, setup.analogs[0].RelativeAmplitude());
    EXPECT_FLOAT_EQ(1.3, setup.analogs[1].RelativeAmplitude());
    EXPECT_FLOAT_EQ(1.8, setup.analogs[2].RelativeAmplitude());
    EXPECT_FLOAT_EQ(0.9, setup.analogs[3].RelativeAmplitude());
#endif
    EXPECT_EQ("TGCA",setup.baseMap());
    EXPECT_EQ(2,setup.NumAnalogWavelengths());
    EXPECT_FLOAT_EQ(0.8,setup.smrtSensor().PhotoelectronSensitivity());

    EXPECT_FLOAT_EQ(23.0, setup.smrtSensor().RefDwsSnr());

    ASSERT_EQ(2,setup.smrtSensor().FilterMap().size());
    EXPECT_EQ(1,setup.smrtSensor().FilterMap()[0]);
    EXPECT_EQ(0,setup.smrtSensor().FilterMap()[1]);

    ASSERT_EQ(2,setup.smrtSensor().RefSpectrum().size());
    EXPECT_FLOAT_EQ(0.10197792202234268, setup.smrtSensor().RefSpectrum()[0]);
    EXPECT_FLOAT_EQ(0.8980221152305603, setup.smrtSensor().RefSpectrum()[1]);

}

TEST_F(Acquisition_SetupTest,JsonParsingAndModify)
{
    LogSeverityContext logSeverity(LogLevel::ERROR);
    const char* payload = R"(

{"token":"1234","hdf5output":"/data/pa/foo.trc.h5","cameraConfigKey":567,
"bazfile":"hello.baz",
 "instrumentName":"Barney","chipId":"exp5","numFrames":51200,"exposure":0.01,
 "roiMetaData":{
    "sequencingPixelRowMin" : 0,
    "sequencingPixelColMin" : 0,
    "sequencingPixelRowSize": 1,
    "sequencingPixelColSize": 32,
    "traceFilePixelRowMin" : 1,
    "traceFilePixelColMin" : 32,
    "traceFilePixelRowSize": 2,
    "traceFilePixelColSize": 64,
    "sensorPixelRowMin" : 0,
    "sensorPixelColMin" :0,
    "sensorPixelRowSize":1144,
    "sensorPixelColSize":2048
},
"analogs":[
    {"analogName":"AAnalog", "base":"A", "spectralAngle":1.3756,"wavelength":665, "relativeAmplitude":0.9},
    {"analogName":"CAnalog", "base":"C", "spectralAngle":1.3756,"wavelength":665, "relativeAmplitude":1.8},
    {"analogName":"GAnalog", "base":"G", "spectralAngle":0.5855,"wavelength":600, "relativeAmplitude":1.3},
    {"analogName":"TAnalog", "base":"T", "spectralAngle":0.5855,"wavelength":600, "relativeAmplitude":3.3}
],
 "acquisitionXML":"<some>xml</some>",
"basecallerVersion": "1.2.3",
 "basecaller": {},
 "chipClass": "Sequel",
 "chipLayoutName": "SequEL_4.0_RTO3",
 "readout":"bases_without_qvs","metricsVerbosity":"high",
 "crosstalkFilter":[[ 0.0,1.0,2.0],[3.0,4,5],[6,7,8]]
}
)";


    Json::Value json = PacBio::IPC::ParseJSON(payload);

    //PBLOG_INFO << "json: " << json;
    
    Acquisition::Setup setup(json);
    
    //PBLOG_INFO << "setup: " << setup;

    EXPECT_EQ("hello.baz",setup.bazfile());
    EXPECT_EQ("1234",setup.token());
    EXPECT_EQ("/data/pa/foo.trc.h5", setup.hdf5output());
    EXPECT_EQ(567,setup.cameraConfigKey());
    EXPECT_EQ("Barney",setup.instrumentName());
    EXPECT_EQ("exp5",setup.chipId());
    EXPECT_EQ(51200,setup.numFrames());
    EXPECT_FLOAT_EQ(0.01,setup.exposure());
    EXPECT_FLOAT_EQ(100.0,setup.expectedFrameRate());

    auto zmws = setup.GetSequencingZmwMap();
    EXPECT_EQ(16,zmws.size());
    zmws = setup.GetTraceFileZmwMap();
    EXPECT_EQ(64,zmws.size());

    EXPECT_EQ(PacBio::SmrtData::Readout::BASES_WITHOUT_QVS, setup.readout());
    EXPECT_EQ(PacBio::SmrtData::MetricsVerbosity::HIGH,setup.metricsVerbosity());
    ASSERT_EQ(PacBio::Primary::SequelROI::ROI_Type_e::Rectangular,setup.SequencingROI().Type());
    const SequelRectangularROI& rectSeqRoi = dynamic_cast<const SequelRectangularROI&>(setup.SequencingROI());
    EXPECT_EQ(1, rectSeqRoi.NumPixelRows());
    EXPECT_EQ(32, rectSeqRoi.NumPixelCols());

    ASSERT_EQ(PacBio::Primary::SequelROI::ROI_Type_e::Rectangular,setup.FileROI().Type());
    const SequelRectangularROI& rectTraceRoi = dynamic_cast<const SequelRectangularROI&>(setup.FileROI());
    EXPECT_EQ(2, rectTraceRoi.NumPixelRows());
    EXPECT_EQ(64, rectTraceRoi.NumPixelCols());

    EXPECT_EQ("1.2.3", setup.basecallerVersion());
    EXPECT_EQ("<some>xml</some>",setup.acquisitionXML());
    ASSERT_EQ(3, setup.crosstalkFilter.NumRows());
    ASSERT_EQ(3, setup.crosstalkFilter.NumCols());
    EXPECT_DOUBLE_EQ(0, setup.crosstalkFilter(0,0));
    EXPECT_DOUBLE_EQ(1, setup.crosstalkFilter(0,1));
    EXPECT_DOUBLE_EQ(2, setup.crosstalkFilter(0,2));
    EXPECT_DOUBLE_EQ(3, setup.crosstalkFilter(1,0));
    EXPECT_DOUBLE_EQ(4, setup.crosstalkFilter(1,1));
    EXPECT_DOUBLE_EQ(5, setup.crosstalkFilter(1,2));
    EXPECT_DOUBLE_EQ(6, setup.crosstalkFilter(2,0));
    EXPECT_DOUBLE_EQ(7, setup.crosstalkFilter(2,1));
    EXPECT_DOUBLE_EQ(8, setup.crosstalkFilter(2,2));

    EXPECT_FLOAT_EQ(1.3756, setup.analogs[2].SpectralAngle());
    EXPECT_FLOAT_EQ(1.3756, setup.analogs[3].SpectralAngle());
    EXPECT_FLOAT_EQ(0.5855, setup.analogs[0].SpectralAngle());
    EXPECT_FLOAT_EQ(0.5855, setup.analogs[1].SpectralAngle());
    EXPECT_FLOAT_EQ(0.9, setup.analogs[3].RelativeAmplitude());
    EXPECT_FLOAT_EQ(1.8, setup.analogs[2].RelativeAmplitude());
    EXPECT_FLOAT_EQ(1.3, setup.analogs[1].RelativeAmplitude());
    EXPECT_FLOAT_EQ(3.3, setup.analogs[0].RelativeAmplitude());

    EXPECT_EQ("TGCA",setup.baseMap());
    EXPECT_EQ(2,setup.NumAnalogWavelengths());

    // The JSON analog set
    auto& jsonAnalogs = json["analogs"];

    ASSERT_EQ(setup.analogs.size(), jsonAnalogs.size());

    // Modify the spectra of the original json values in such
    // a way to verify the correct pairing of json and setup.
    //
    for (const auto& setupAnalog : setup.analogs)
    {
        // Find the json entry corresponding to this analog
        unsigned int ja = 0;
        for (; ja < jsonAnalogs.size(); ++ja)
        {
            //const auto& jsonAnalogLabel = jsonAnalogs[ja]["base"];
            //PBLOG_INFO << "setupAnalog: " << setupAnalog.baseLabel << ", jsonAnalog: " << jsonAnalogLabel
            //           << ", " << (setupAnalog.baseLabel == jsonAnalogLabel.asString().at(0));

#if 0
            if (setupAnalog.baseLabel == jsonAnalogs[ja]["base"].asString().at(0)) break;
#else
            auto& analogJson = jsonAnalogs[ja];
            std::string base = (analogJson["baseMap"].isString()) ? analogJson["baseMap"].asString() : analogJson["base"].asString();
            if (setupAnalog.baseLabel == base.at(0)) break;
#endif
        }

        // Make sure it was found
        ASSERT_LT(ja, jsonAnalogs.size());

        // Modify the json against the setup-analog value
        auto& jsonAnalog = jsonAnalogs[ja];

        Json::Value& spectralAngle = jsonAnalog["spectralAngle"];
        Json::Value& relativeAmplitude = jsonAnalog["relativeAmplitude"];

        double zeroAngle = spectralAngle.asDouble() - setupAnalog.SpectralAngle();
        float zeroRelAmp = relativeAmplitude.asFloat() - setupAnalog.RelativeAmplitude();
        
        //PBLOG_INFO << "zeroAngle = " << zeroAngle << ", zeroRelAmp = " << zeroRelAmp;

        spectralAngle = zeroAngle;
        relativeAmplitude = zeroRelAmp;        
    }

    //PBLOG_INFO << "modified json: " << json;

    // Parse the modified json 
    Acquisition::Setup setupMod(json);

    // Verify the correct modification
    for (const auto& modAnalog : setupMod.analogs)
    {
        EXPECT_FLOAT_EQ(0.0, modAnalog.RelativeAmplitude());
        EXPECT_NEAR(0.0, modAnalog.SpectralAngle(), 1e-6);
    }
}


TEST_F(Acquisition_SetupTest,JsonParsing2)
{
    LogSeverityContext logSeverity(LogLevel::ERROR);
    const char* payload = R"(
    {
        "cellId":"00000133635767018458122835","movieContext":"mUnset_150901_175719.trc.h5","movieName":"",
        "roiData":{
            "sensorPixelRowMin"    :0 , "sensorPixelRowSize"    :1144, "sensorPixelColMin"    :0 , "sensorPixelColSize"    :2048,
            "sequencingPixelRowMin":32, "sequencingPixelRowSize":1144, "sequencingPixelColMin":64, "sequencingPixelColSize":2048,
            "traceFilePixelRowMin" :32, "traceFilePixelRowSize" :256 , "traceFilePixelColMin" :64, "traceFilePixelColSize" :256
         },
        "crosstalkFilter":[[8.26E-05,0.000372,0.0008122,0.0011329,0.0009037,0.0004714,0.0001231],
        [0.0003206,-0.0008694,-0.0033729,-0.0070469,-0.0040712,-0.0018659,0.0004553],
        [0.0006739,-0.0007131,-0.0280701,-0.0951478,-0.0280312,-0.0049404,0.0008782],
        [0.0010642,-0.0063655,1.5123618,-0.0782312,-0.009702,0.0010642],
        [0.0008782,-0.0044638,-0.0298692,-0.0936265,-0.0295229,-0.0021659,0.0006739],
        [0.0004553,-0.0018659,-0.0040484,-0.0055941,-0.0033729,-0.0017998,0.0003206],
        [0.0001231,0.0004714,0.0009037,0.0011329,0.0008122,0.000372,8.26E-05]],
        "analogs":[{"spectralAngle":1.3756,"relativeAmplitude":0.9,"base":"A","analogName":"AAnalog","wavelength":665.0},
        {"spectralAngle":1.3756,"relativeAmplitude":1.8,"base":"C","analogName":"CAnalog","wavelength":665.0},
        {"spectralAngle":0.5855,"relativeAmplitude":1.3,"base":"G","analogName":"GAnalog","wavelength":600.0},
        {"spectralAngle":0.5855,"relativeAmplitude":3.3,"base":"T","analogName":"TAnalog","wavelength":600.0}],
        "basecaller": {},
        "chipClass": "Sequel",
        "LayoutName": "SequEL_4.0_RTO3"  /* old style */
    }
)";

    Json::Value json = PacBio::IPC::ParseJSON(payload);

    Acquisition::Setup setup(json);

    EXPECT_EQ("n/a", setup.instrumentName());
    EXPECT_EQ("n/a", setup.chipId());


// the default if exposure is not set in the JSON
    EXPECT_FLOAT_EQ(1/Sequel::defaultFrameRate, setup.exposure());
    EXPECT_FLOAT_EQ(Sequel::defaultFrameRate, setup.expectedFrameRate());

    auto zmws = setup.GetSequencingZmwMap();
    EXPECT_EQ(1171456, zmws.size());
    zmws = setup.GetTraceFileZmwMap();
    EXPECT_EQ(32768, zmws.size());

    EXPECT_EQ(PacBio::SmrtData::Readout::BASES, setup.readout());
    ASSERT_EQ(PacBio::Primary::SequelROI::ROI_Type_e::Rectangular, setup.SequencingROI().Type());
    const SequelRectangularROI& rectSeqRoi = dynamic_cast<const SequelRectangularROI&>(setup.SequencingROI());
    EXPECT_EQ(1144, rectSeqRoi.NumPixelRows());
    EXPECT_EQ(2048, rectSeqRoi.NumPixelCols());

    ASSERT_EQ(PacBio::Primary::SequelROI::ROI_Type_e::Rectangular, setup.FileROI().Type());
    const SequelRectangularROI& rectTraceRoi = dynamic_cast<const SequelRectangularROI&>(setup.FileROI());
    EXPECT_EQ(256, rectTraceRoi.NumPixelRows());
    EXPECT_EQ(256, rectTraceRoi.NumPixelCols());

    ASSERT_EQ(7, setup.crosstalkFilter.NumRows());
    ASSERT_EQ(7, setup.crosstalkFilter.NumCols());
    EXPECT_DOUBLE_EQ(8.26E-05, setup.crosstalkFilter(0,0));
    EXPECT_DOUBLE_EQ(0.000372, setup.crosstalkFilter(0,1));
    EXPECT_DOUBLE_EQ(0.0008122, setup.crosstalkFilter(0,2));
    EXPECT_DOUBLE_EQ(0.0003206, setup.crosstalkFilter(1,0));

    EXPECT_EQ("TGCA", setup.baseMap());
    EXPECT_EQ(2, setup.NumAnalogWavelengths());
    EXPECT_FALSE(setup.remapVroiOption());
}


TEST_F(Acquisition_SetupTest,NewROIParsing)
{
    LogSeverityContext logSeverity(LogLevel::ERROR);
    const char* payload = R"(

{"token":"1234","hdf5output":"/data/pa/foo.trc.h5","cameraConfigKey":567,
"bazfile":"hello.baz",
 "instrumentName":"Barney","chipId":"exp5","numFrames":51200,"exposure":0.01,"expectedFrameRate":80,
 "roiMetaData":{
    "sequencingPixelROI": [[ 0, 32, 1, 128]],
    "traceFilePixelROI": [2, 64, 3, 96],
    "sensorPixelRowMin" : 0,
    "sensorPixelColMin" :0,
    "sensorPixelRowSize":1144,
    "sensorPixelColSize":2048,
    "remapVroiOption": false
}, "acquisitionXML":"<some>xml</some>",
"baseCallerVersion": "1.2.3",
 "basecaller": {},
 "chipClass": "Sequel",
 "chipLayoutName": "SequEL_4.0_RTO3",
 "readout":"bases","metricsVerbosity":"high",
 "crosstalkFilter":[[ 0.0,1.0,2.0],[3.0,4,5],[6,7,8]]
}
)";
    Json::Value json = PacBio::IPC::ParseJSON(payload);

    Acquisition::Setup setup(json);

    EXPECT_FLOAT_EQ(0.01,setup.exposure);
    EXPECT_FLOAT_EQ(80,setup.expectedFrameRate);

    auto zmws = setup.GetSequencingZmwMap();
    EXPECT_EQ((128*1/2),zmws.size());
    zmws = setup.GetTraceFileZmwMap();
    EXPECT_EQ((96*3/2),zmws.size());


    SequelSparseROI seqRoiRef(SequelSensorROI::SequelAlpha());
    seqRoiRef.AddRectangle(0,32,1,128);

    ASSERT_EQ(PacBio::Primary::SequelROI::ROI_Type_e::Sparse,setup.SequencingROI().Type());
    EXPECT_EQ(seqRoiRef , setup.SequencingROI());

    SequelRectangularROI traceRoiRef(2,64,3,96,SequelSensorROI::SequelAlpha());

    ASSERT_EQ(PacBio::Primary::SequelROI::ROI_Type_e::Rectangular,setup.FileROI().Type());
    EXPECT_EQ(traceRoiRef, setup.FileROI());
    EXPECT_FALSE(setup.remapVroiOption);
}

TEST_F(Acquisition_SetupTest,BadValues1)
{
    LogSeverityContext logSeverity(LogLevel::ERROR);
    const char* payload = R"({"exposure":0.01,"expectedFrameRate":-1})";
    Json::Value json = PacBio::IPC::ParseJSON(payload);

    EXPECT_THROW(Acquisition::Setup setup(json),std::runtime_error);
}

TEST_F(Acquisition_SetupTest,BadValues2)
{
    LogSeverityContext logSeverity(LogLevel::ERROR);
    const char* payload = R"({"exposure":0.0125,"expectedFrameRate":100})";
    Json::Value json = PacBio::IPC::ParseJSON(payload);

    EXPECT_THROW(Acquisition::Setup setup(json),std::runtime_error);
}

TEST_F(Acquisition_SetupTest,BadValues3)
{
    LogSeverityContext logSeverity(LogLevel::ERROR);
    const char* payload = R"({"exposure":0.0,"expectedFrameRate":100})";
    Json::Value json = PacBio::IPC::ParseJSON(payload);

    EXPECT_THROW(Acquisition::Setup setup(json),std::runtime_error);
}

TEST_F(Acquisition_SetupTest,BadValues4)
{
    LogSeverityContext logSeverity(LogLevel::ERROR);
    const char* payload = R"({"refDwsSnr":0.0,"refSpectrum":null})";
    Json::Value json = PacBio::IPC::ParseJSON(payload);

//    Acquisition::Setup setup(json);
    EXPECT_THROW(Acquisition::Setup setup(json),std::runtime_error); // because refSpectrum is null, not 2 elements
}

TEST_F(Acquisition_SetupTest,BadValues4a)
{
    LogSeverityContext logSeverity(LogLevel::ERROR);
    const char* payload = R"({"refDwsSnr":0.0,"refSpectrum":[0.5,0.25,0.25]})";
    Json::Value json = PacBio::IPC::ParseJSON(payload);

//    Acquisition::Setup setup(json);
    EXPECT_THROW(Acquisition::Setup setup(json),std::runtime_error); // because refSpectrum is > 2 elements
}

TEST_F(Acquisition_SetupTest,NewROIParsing2)
{
    LogSeverityContext logSeverity(LogLevel::ERROR);

    const char* payload = R"(

{"token":"1234","hdf5output":"/data/pa/foo.trc.h5","cameraConfigKey":567,
"bazfile":"hello.baz",
 "instrumentName":"Barney","chipId":"exp5","numFrames":51200,"exposure":0.01,
 "roiMetaData":{
    "sequencingPixelROI": [[ 0, 32, 1, 128],[10,64,1,128]],
    "traceFilePixelROI": [[2, 64, 3, 96],[13,64,1,128]],
    "sensorPixelRowMin" : 0,
    "sensorPixelColMin" :0,
    "sensorPixelRowSize":1144,
    "sensorPixelColSize":2048,
    "remapVroiOption": true
}, "acquisitionXML":"<some>xml</some>",
"baseCallerVersion": "1.2.3",
 "basecaller": {},
 "chipClass": "Sequel",
 "chipLayoutName": "SequEL_4.0_RTO3",
 "readout":"bases","metricsVerbosity":"high",
 "crosstalkFilter":[[ 0.0,1.0,2.0],[3.0,4,5],[6,7,8]]
}
)";
  Json::Value json = PacBio::IPC::ParseJSON(payload);

  Acquisition::Setup setup(json);

  EXPECT_EQ("1234", setup.token());
  EXPECT_EQ("/data/pa/foo.trc.h5",setup.hdf5output());
  EXPECT_EQ(567,setup.cameraConfigKey());
  EXPECT_EQ("hello.baz", setup.bazfile());
  EXPECT_EQ("exp5", setup.chipId());
  EXPECT_EQ(51200, setup.numFrames());
  EXPECT_FLOAT_EQ(0.01,setup.exposure());
  EXPECT_EQ(PacBio::SmrtData::Readout::BASES, setup.readout());

  auto zmws = setup.GetSequencingZmwMap();
  EXPECT_EQ((128*1*2/2),zmws.size());
  zmws = setup.GetTraceFileZmwMap();
  EXPECT_EQ(((96*3 + 128*1)/2),zmws.size());


  SequelSparseROI seqRoiRef(SequelSensorROI::SequelAlpha());
  seqRoiRef.AddRectangle(0,32,1,128);
  seqRoiRef.AddRectangle(10,64,1,128);

  ASSERT_EQ(PacBio::Primary::SequelROI::ROI_Type_e::Sparse,setup.SequencingROI().Type());
  EXPECT_EQ(seqRoiRef , setup.SequencingROI());

  SequelSparseROI traceRoiRef(SequelSensorROI::SequelAlpha());
  traceRoiRef.AddRectangle(2,64,3,96);
  traceRoiRef.AddRectangle(13,64,1,128);

  ASSERT_EQ(PacBio::Primary::SequelROI::ROI_Type_e::Sparse,setup.FileROI().Type());
  EXPECT_EQ(traceRoiRef, setup.FileROI());
  EXPECT_TRUE(setup.remapVroiOption);

    ASSERT_EQ( 4,setup.analogs.size());
    EXPECT_FLOAT_EQ( 0.94405341, setup.analogs[0].SpectralAngle());
    EXPECT_FLOAT_EQ( 0.94405341, setup.analogs[1].SpectralAngle());
    EXPECT_FLOAT_EQ(1.507053,    setup.analogs[2].SpectralAngle());
    EXPECT_FLOAT_EQ(1.507053,    setup.analogs[3].SpectralAngle());
    EXPECT_FLOAT_EQ(0.85000002,  setup.analogs[0].RelativeAmplitude());
    EXPECT_FLOAT_EQ(0.54400003,  setup.analogs[1].RelativeAmplitude());
    EXPECT_FLOAT_EQ(1,           setup.analogs[2].RelativeAmplitude());
    EXPECT_FLOAT_EQ(0.5,         setup.analogs[3].RelativeAmplitude());

    EXPECT_FLOAT_EQ(1.0,setup.smrtSensor().PhotoelectronSensitivity());

    // Now test the "update" feature by loading in new JSON data.

    const char* payloadUpdate = R"(

    {"token":"12345","hdf5output":"/data/pa/bar.trc.h5","cameraConfigKey":765,
    "bazfile":"goodbye.baz",
     "instrumentName":"Barney","chipId":"exp6","numFrames":61200,"exposure":0.0125,
     "expectedFrameRate": 80,
     "roiMetaData":{
        "sequencingPixelROI": [[ 0, 32, 2, 128],[10,64,2,128]],
        "traceFilePixelROI": [[20, 64, 30, 96],[130,64,10,128]],
        "sensorPixelRowMin" : 0,
        "sensorPixelColMin" :0,
        "sensorPixelRowSize":1144,
        "sensorPixelColSize":2048,
        "remapVroiOption": true
    }, "acquisitionXML":"<some>xml</some>",
    "baseCallerVersion": "1.2.3",
     "basecaller": {},
     "chipClass": "Sequel",
     "readout":"PULSES","metricsVerbosity":"HIGH",
     "crosstalkFilter":[[ 0.0,1.0,2.0],[3.0,4,5],[6,7,8]],
        "analogs" :
        [
                {
                        "base" : "T",
                        "intraPulseXsnCV" : 0.0099999997764825821,
                        "ipdMeanSeconds" : 0.5,
                        "pulseWidthMeanSeconds" : 0.20000000298023224,
                        "relativeAmplitude" : 0.94999998807907104,
                        "spectrumValues" :
                        [
                                0.80000001192092896,
                                0.20000000298023224
                        ]
                },
                {
                        "base" : "G",
                        "intraPulseXsnCV" : 0.0099999997764825821,
                        "ipdMeanSeconds" : 0.5,
                        "pulseWidthMeanSeconds" : 0.20000000298023224,
                        "relativeAmplitude" : 0.28499999642372131,
                        "spectrumValues" :
                        [
                                0.80000001192092896,
                                0.20000000298023224
                        ]
                },
                {
                        "base" : "C",
                        "intraPulseXsnCV" : 0.0099999997764825821,
                        "ipdMeanSeconds" : 0.5,
                        "pulseWidthMeanSeconds" : 0.20000000298023224,
                        "relativeAmplitude" : 1,
                        "spectrumValues" :
                        [
                                0.05000000074505806,
                                0.94999998807907104
                        ]
                },
                {
                        "base" : "A",
                        "intraPulseXsnCV" : 0.0099999997764825821,
                        "ipdMeanSeconds" : 0.5,
                        "pulseWidthMeanSeconds" : 0.20000000298023224,
                        "relativeAmplitude" : 0.30000001192092896,
                        "spectrumValues" :
                        [
                                0.05000000074505806,
                                0.94999998807907104
                        ]
                }
        ]
    }
    )";
    Json::Value jsonUpdate = PacBio::IPC::ParseJSON(payloadUpdate);

    setup.Parse(jsonUpdate);

    EXPECT_EQ("12345", setup.token());
    EXPECT_EQ("/data/pa/bar.trc.h5",setup.hdf5output());
    EXPECT_EQ(765,setup.cameraConfigKey());
    EXPECT_EQ("goodbye.baz", setup.bazfile());
    EXPECT_EQ("exp6", setup.chipId());
    EXPECT_EQ(61200, setup.numFrames());
    EXPECT_FLOAT_EQ(0.0125,setup.exposure());
    EXPECT_EQ(PacBio::SmrtData::Readout::PULSES, setup.readout());

    auto zmwsUpdate = setup.GetSequencingZmwMap();
    EXPECT_EQ((128*2*2/2),zmwsUpdate.size());
    zmwsUpdate = setup.GetTraceFileZmwMap();
    EXPECT_EQ(((96*30 + 128*10)/2),zmwsUpdate.size());


    SequelSparseROI seqRoiRefUpdate(SequelSensorROI::SequelAlpha());
    seqRoiRefUpdate.AddRectangle(0,32,2,128);
    seqRoiRefUpdate.AddRectangle(10,64,2,128);

    ASSERT_EQ(PacBio::Primary::SequelROI::ROI_Type_e::Sparse,setup.SequencingROI().Type());
    EXPECT_EQ(seqRoiRefUpdate , setup.SequencingROI());

    SequelSparseROI traceRoiRefUpdate(SequelSensorROI::SequelAlpha());
    traceRoiRefUpdate.AddRectangle(20,64,30,96);
    traceRoiRefUpdate.AddRectangle(130,64,10,128);

    ASSERT_EQ(PacBio::Primary::SequelROI::ROI_Type_e::Sparse,setup.FileROI().Type());
    EXPECT_EQ(traceRoiRefUpdate, setup.FileROI());
    EXPECT_TRUE(setup.remapVroiOption);

#if 0
    ASSERT_EQ( 4,setup.analogs.size());
    EXPECT_FLOAT_EQ( 0.93, setup.analogs[0].SpectralAngle());
    EXPECT_FLOAT_EQ( 0.93, setup.analogs[1].SpectralAngle());
    EXPECT_FLOAT_EQ(1.51,    setup.analogs[2].SpectralAngle());
    EXPECT_FLOAT_EQ(1.51,    setup.analogs[3].SpectralAngle());
    EXPECT_FLOAT_EQ(0.85000002,  setup.analogs[0].RelativeAmplitude());
    EXPECT_FLOAT_EQ(0.54400003,  setup.analogs[1].RelativeAmplitude());
    EXPECT_FLOAT_EQ(1,           setup.analogs[2].RelativeAmplitude());
    EXPECT_FLOAT_EQ(0.5,         setup.analogs[3].RelativeAmplitude());

    EXPECT_FLOAT_EQ(1.0,setup.smrtSensor().PhotoelectronSensitivity());
#endif

}

TEST_F(Acquisition_SetupTest,SpiderParse)
{
    ResetPrimaryConfig();
    GetPrimaryConfig().platform = Platform::Spider;
    GetPrimaryConfig().chipClass = ChipClass::Spider;

    LogSeverityContext logSeverity(LogLevel::ERROR);
    const std::string setupString = R"(
{
    "acquisitionVersion" : "5.2.0.mlakata_feature_SPI_768_end_to_end_7a48a91",
    "acquisitionXML" : "\n<?xml version=\"1.0\" encoding=\"utf-8\"?>\n  <PacBioDataModel>\n    <ProjectContainer>\n      <Runs>\n        <Run>\n           <Collections>\n      <CollectionMetadata CreatedAt=\"0001-01-01T00:00:00\" ModifiedAt=\"0001-01-01T00:00:00\" Status=\"Ready\" InstrumentId=\"Inst1234\" InstrumentName=\"Inst1234\">\n      <AutomationParameters>\n      <AutomationParameter xsi:type=\"SequencingChemistry\" Name=\"2C2A\" CreatedAt=\"0001-01-01T00:00:00\" ModifiedAt=\"0001-01-01T00:00:00\">\n        <DyeSet Name=\"Super duper dye set\" CreatedAt=\"0001-01-01T00:00:00\" ModifiedAt=\"0001-01-01T00:00:00\">\n        <Analogs>\n        <Analog Name=\"AAnalog\" CreatedAt=\"0001-01-01T00:00:00\" ModifiedAt=\"0001-01-01T00:00:00\" Base=\"A\" SpectralAngle=\"1.4446\" RelativeAmplitude=\"0.9\" Wavelength=\"665\" />\n        <Analog Name=\"CAnalog\" CreatedAt=\"0001-01-01T00:00:00\" ModifiedAt=\"0001-01-01T00:00:00\" Base=\"C\" SpectralAngle=\"1.4446\" RelativeAmplitude=\"1.8\" Wavelength=\"665\" />\n        <Analog Name=\"GAnalog\" CreatedAt=\"0001-01-01T00:00:00\" ModifiedAt=\"0001-01-01T00:00:00\" Base=\"G\" SpectralAngle=\"0.8205\" RelativeAmplitude=\"1.3\" Wavelength=\"600\" />\n        <Analog Name=\"TAnalog\" CreatedAt=\"0001-01-01T00:00:00\" ModifiedAt=\"0001-01-01T00:00:00\" Base=\"T\" SpectralAngle=\"0.8205\" RelativeAmplitude=\"3.3\" Wavelength=\"600\" />\n        </Analogs>\n        </DyeSet>\n      </AutomationParameter></AutomationParameters>\n      </CollectionMetadata>\n      </Collections>\n      </Run>\n      </Runs>\n      </ProjectContainer>\n      </PacBioDataModel>\n      ",
    "analogs" :
    [
    {
        "base" : "C",
        "baseMap" : null,
        "intraPulseXsnCV" : 0.0099999997764825821,
        "ipdMeanSeconds" : 0.16249999403953552,
        "pulseWidthMeanSeconds" : 0.125,
        "relativeAmplitude" : 1,
        "spectrumValues" :
        [
        1.0
        ]
    },
    {
        "base" : "A",
        "baseMap" : null,
        "intraPulseXsnCV" : 0.0099999997764825821,
        "ipdMeanSeconds" : 0.22499999403953552,
        "pulseWidthMeanSeconds" : 0.087499998509883881,
        "relativeAmplitude" : 0.66666668653488159,
        "spectrumValues" :
        [
        1.0
        ]
    },
    {
        "base" : "T",
        "baseMap" : null,
        "intraPulseXsnCV" : 0.0099999997764825821,
        "ipdMeanSeconds" : 0.17499999701976776,
        "pulseWidthMeanSeconds" : 0.11249999701976776,
        "relativeAmplitude" : 0.3333333432674408,
        "spectrumValues" :
        [
        1.0
        ]
    },
    {
        "base" : "G",
        "baseMap" : null,
        "intraPulseXsnCV" : 0.0099999997764825821,
        "ipdMeanSeconds" : 0.17499999701976776,
        "pulseWidthMeanSeconds" : 0.11249999701976776,
        "relativeAmplitude" : 0.1666666716337204,
        "spectrumValues" :
        [
        1.0
        ]
    }
    ],
    "bazfile" : "/data/pa/e2e_spider.baz",
    "cameraConfigKey" : 0,
    "chipId" : "abc",
    "chipLayoutName" : "ChipLayoutSpider1",
    "chipClass": "Spider",
    "crosstalkFilter" :
    [
        [
            1
        ]
    ],
    "dryrun" : false,
    "expectedFrameRate" : 100,
    "exposure" : 0.01,
    "hd5output" : "",
    "instrumentName" : "unknown",
    "metricsVerbosity" : "minimal",
    "noCalibration" : true,
    "numFrames" : 1024,
    "numPixelLanes" :
    [
        250796
    ],
    "numZmwLanes" :
    [
        250796
    ],
    "photoelectronSensitivity" : 1,
    "readout" : "bases_without_qvs",
    "refDwsSnr" : 50,
    "roiMetaData" :
    {
        "remapVroiOption" : false,
        "sensorPixelColMin" : 0,
        "sensorPixelColSize" : 2912,
        "sensorPixelRowMin" : 0,
        "sensorPixelRowSize" : 2756,
        "sequencingPixelROI" :
        [
            [
            0, 0, 2756, 2912
            ]
        ]
    },
    "token" : ""
}
)";

    Json::Value json = PacBio::IPC::ParseJSON(setupString);

    Acquisition::Setup setup(json);

    auto features  = setup.GetUnitCellFeatureList();
    EXPECT_EQ(2756*2912,features.size());

    EXPECT_EQ(0,features[0].first);
    EXPECT_EQ(0x00000000,features[0].second.Number());
    EXPECT_EQ(0,features[1].first);
    EXPECT_EQ(0x00000001,features[1].second.Number());

    ASSERT_EQ(1,setup.smrtSensor().FilterMap().size());
    EXPECT_EQ(0,setup.smrtSensor().FilterMap()[0]);
    EXPECT_FLOAT_EQ(50.0, setup.smrtSensor().RefDwsSnr());
    ASSERT_EQ(1,setup.smrtSensor().RefSpectrum().size());
    EXPECT_FLOAT_EQ(1.0, setup.smrtSensor().RefSpectrum()[0]);
}

TEST_F(Acquisition_SetupTest,ChipLayoutName)
{
    Acquisition::Setup setup1(sequelLayout.get());
    EXPECT_EQ("SequEL_4.0_RTO3", setup1.chipLayoutName());

    Acquisition::Setup setup2(spiderLayout.get());
    EXPECT_EQ("Spider_1p0_NTO", setup2.chipLayoutName());
}

TEST_F(Acquisition_SetupTest,NormalChipLayout)
{
    LogSeverityContext logSeverity(LogLevel::WARN);
    {
        Acquisition::Setup setup(sequelLayout.get());
        EXPECT_EQ(ChipClass::Sequel, setup.chipClass());
        const auto layout1 = setup.GetChipLayout();
        EXPECT_EQ("SequEL_4.0_RTO3", layout1->Name());
        EXPECT_EQ(2, layout1->FilterMap().size());
    }

    {
        Acquisition::Setup setup(spiderLayout.get());
        EXPECT_EQ(ChipClass::Spider, setup.chipClass());
        const auto layout2 = setup.GetChipLayout();
        EXPECT_EQ("Spider_1p0_NTO", layout2->Name());
        EXPECT_EQ(1, layout2->FilterMap().size());
    }
}

TEST_F(Acquisition_SetupTest,Copy)
{
    LogSeverityContext logSeverity(LogLevel::ERROR);
    Acquisition::Setup orig(sequelLayout.get());
    orig.PostImport();
    orig.minSnr = 13.5;
    ASSERT_GE(orig.NumFilters() , 0);
    orig.smrtSensor().RefDwsSnr(12.75);
    orig.analogs[0].relAmplitude= 0.875;
    orig.crosstalkFilter(0,0) = 0.125;
    orig.psfs[0](0,0) = 1.5;
    orig.SetSensorROIAndResetOtherROIs( SequelSensorROI::SequelAlpha());
    orig.SetSequencingROI(SequelRectangularROI(1,0,3,32,SequelSensorROI::SequelAlpha()));
    orig.SetFileROI(SequelRectangularROI(4,0,7,32,SequelSensorROI::SequelAlpha()));

    {
        // test copy constructor
        Acquisition::Setup dup1(orig);
        EXPECT_EQ(ChipClass::Sequel,             dup1.chipClass());
        EXPECT_EQ(orig.minSnr(),                 dup1.minSnr());
        EXPECT_EQ(orig.NumFilters(),             dup1.NumFilters());
        EXPECT_EQ(orig.smrtSensor().RefDwsSnr(), dup1.smrtSensor().RefDwsSnr());
        EXPECT_EQ(orig.analogs[0].relAmplitude,  dup1.analogs[0].relAmplitude);
        EXPECT_EQ(orig.crosstalkFilter(0,0),     dup1.crosstalkFilter(0,0));
        EXPECT_EQ(orig.psfs[0](0,0),             dup1.psfs[0](0,0));
        EXPECT_EQ(orig.SensorROI(),              dup1.SensorROI());
        EXPECT_EQ(orig.SequencingROI(),          dup1.SequencingROI());
        EXPECT_EQ(orig.FileROI(),                dup1.FileROI());
        EXPECT_EQ(orig.Json(),                   dup1.Json());
    }
    {
        // test copy constructor
        Acquisition::Setup dup2(orig);

        EXPECT_EQ(ChipClass::Sequel,             dup2.chipClass());
        EXPECT_EQ(orig.minSnr(),                 dup2.minSnr());
        EXPECT_EQ(orig.NumFilters(),             dup2.NumFilters());
        EXPECT_EQ(orig.smrtSensor().RefDwsSnr(), dup2.smrtSensor().RefDwsSnr());
        EXPECT_EQ(orig.analogs[0].relAmplitude,  dup2.analogs[0].relAmplitude);
        EXPECT_EQ(orig.crosstalkFilter(0,0),     dup2.crosstalkFilter(0,0));
        EXPECT_EQ(orig.psfs[0](0,0),             dup2.psfs[0](0,0));
        EXPECT_EQ(orig.SensorROI(),              dup2.SensorROI());
        EXPECT_EQ(orig.SequencingROI(),          dup2.SequencingROI());
        EXPECT_EQ(orig.FileROI(),                dup2.FileROI());
        EXPECT_EQ(orig.Json(),                   dup2.Json());
        EXPECT_EQ(orig.chipLayoutName(),         dup2.chipLayoutName());
    }
}

TEST_F(Acquisition_SetupTest,PPA)
{
    Acquisition::Setup setup(sequelLayout.get());
    setup.readout = PacBio::SmrtData::Readout::PULSES;
    setup.metricsVerbosity = PacBio::SmrtData::MetricsVerbosity::NONE;
    EXPECT_EQ("PULSES",setup.Json()["readout"].asString());
    EXPECT_EQ("NONE",setup.Json()["metricsVerbosity"].asString());
}

TEST_F(Acquisition_SetupTest,Psfs)
{
    Acquisition::Setup setup(sequelLayout.get());

    auto Gjson = PacBio::IPC::ParseJSON("[[-0.001,0.0038,0.0067,0.0038,0.0019],"
                                        "[-0.001,0.0172,0.062,0.0153,0.0038],"
                                        "[0.001,0.0477,0.6692,0.0372,0.0076],"
                                        "[0.0029,0.0219,0.0629,0.0191,0.001],"
                                        "[0.0019,0.0048,0.0048,0.0038,0.0019]]");
    auto Rjson = PacBio::IPC::ParseJSON("[[0.003,0.004	,0.008	,0.005	,0.002],"
                                        "[0.005	,0.023	,0.0549	,0.025	,0.006],"
                                        "[0.01	,0.0579	,0.6004	,0.0579	,0.01],"
                                        "[0.006	,0.022	,0.0509	,0.024	,0.006],"
                                        "[0.002	,0.004	,0.007	,0.004	,0.002]]");
    Json::Value psfs;
    psfs.append(Gjson);
    psfs.append(Rjson);

    setup.SetPsfs(psfs);

    EXPECT_FLOAT_EQ(-0.001,setup.Json()["psfs"][0][0][0].asFloat());
    EXPECT_FLOAT_EQ(0.0019,setup.Json()["psfs"][0][0][4].asFloat());
    EXPECT_FLOAT_EQ(0.002,setup.Json()["psfs"][1][4][4].asFloat());
}

TEST_F(Acquisition_SetupTest,PhotoelectronSensitivity)
{
    Acquisition::Setup setup(sequelLayout.get());
    EXPECT_FLOAT_EQ(1.0,setup.photoelectronSensitivity());

    setup.photoelectronSensitivity = 0.5;
    EXPECT_FLOAT_EQ(0.5,setup.Json()["photoelectronSensitivity"].asFloat());
}

// unprotected protected members
class MySetup : public PacBio::Primary::Acquisition::Setup
{
public:
    using PacBio::Primary::Acquisition::Setup::Setup;
    using PacBio::Primary::Acquisition::Setup::PostImportCrosstalk;
};


TEST_F(Acquisition_SetupTest,CrosstalkFilter)
{
    PacBio::Logging::LogSeverityContext logSeverity{PacBio::Logging::LogLevel::WARN};

    GetPrimaryConfig().chipClass = ChipClass::Spider;

    MySetup setup(sequelLayout.get());
    Json::Value psfs = PacBio::IPC::ParseJSON(
            R"(
[
 [
  [0.0, 0.0,0.0],
  [0.05,0.8,0.1],
  [0.0, 0.0,0.0]
 ],
 [
  [0.0, 0.0,0.0],
  [0.15,0.8,0.1],
  [0.0 ,0.0,0.0]
 ]
]
            )"
    );
    setup.SetPsfs(psfs);
    setup.PostImportCrosstalk();
    Json::Value originalJson = setup.Json();

    //TEST_COUT << "before render:" << setup.crosstalkFilter << std::endl;
    setup.RenderCrosstalkFilter(originalJson);
    //TEST_COUT << "after render:" << setup.crosstalkFilter << std::endl;

    EXPECT_EQ(7, setup.crosstalkFilter.NumRows());
    EXPECT_DOUBLE_EQ(0.0, setup.crosstalkFilter(0,0));
    EXPECT_DOUBLE_EQ(0.0, setup.crosstalkFilter(3,0));
    EXPECT_DOUBLE_EQ(0.010050188126959,   setup.crosstalkFilter(3,1));
    EXPECT_DOUBLE_EQ(-0.1399488696679041, setup.crosstalkFilter(3,2));
    EXPECT_DOUBLE_EQ(1.2567948693789612,  setup.crosstalkFilter(3,3));
}
