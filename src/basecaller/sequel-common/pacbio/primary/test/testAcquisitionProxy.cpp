//
// Created by mlakata on 2/4/16.
//

#include <gtest/gtest.h>

#include <pacbio/primary/AcquisitionProxy.h>
#include <pacbio/primary/Acquisition_Setup.h>
#include <pacbio/ipc/JSON.h>
#include <pacbio/PBException.h>
#include <pacbio/dev/gtest-extras.h>
#include <pacbio/logging/Logger.h>

#include "testTraceFilePath.h"

using namespace PacBio::Primary;

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
    "readout":"bases_without_qvs","metricsVerbosity":"high",
    "crosstalkFilter":[[ 0.0,1.0,2.0],[3.0,4,5],[6,7,8]]
    }
    )";

#ifdef PB_MIC_COPROCESSOR
TEST(AcquisitionProxy,DISABLED_UpdateSetupWithOldTraceFile)
#else
TEST(AcquisitionProxy,UpdateSetupWithOldTraceFile)
#endif
{
    PacBio::Logging::LogSeverityContext noInfo(PacBio::Logging::LogLevel::ERROR);

    std::unique_ptr<AcquisitionProxy> acquisition( CreateAcquisitionProxy(16384, TraceFilePath("/dept/primary/sim/ffhmm/ffHmm_256Lx3C_102715_SNR-18.trc.h5")));

    Acquisition::Setup setup = acquisition->UpdateSetup();

    EXPECT_FLOAT_EQ(1.507053, setup.analogs[2].SpectralAngle());
    EXPECT_EQ("SequEL_4.0_RTO2", setup.chipLayoutName());
}

#ifdef PB_MIC_COPROCESSOR
TEST(AcquisitionProxy,DISABLED_UpdateSetupWithNewTraceFile)
#else
TEST(AcquisitionProxy,UpdateSetupWithNewTraceFile)
#endif
{
    PacBio::Logging::LogSeverityContext noInfo(PacBio::Logging::LogLevel::ERROR);

    std::unique_ptr<AcquisitionProxy> acquisition( CreateAcquisitionProxy(16384, TraceFilePath("/dept/primary/unitTestInput/libSequelCommon/file1.trc.h5")));

    // this (for some strange reason) disables RVO, and forces the copy constructor of Setup to be used.
    Acquisition::Setup setup = Acquisition::Setup(std::move(acquisition->UpdateSetup()));

    EXPECT_FLOAT_EQ(0.12586163, setup.analogs[0].dyeSpectrum[0]);
    EXPECT_FLOAT_EQ(0.68185556, setup.analogs[2].dyeSpectrum[0]);
    EXPECT_FLOAT_EQ(0.73148686, setup.analogs[3].dyeSpectrum[1]);

    // this seems to always use NRVO and RVO and elides the copy
    Acquisition::Setup setup2(acquisition->UpdateSetup());

    EXPECT_FLOAT_EQ(0.12586163, setup2.analogs[0].dyeSpectrum[0]);
    EXPECT_FLOAT_EQ(0.68185556, setup2.analogs[2].dyeSpectrum[0]);
    EXPECT_FLOAT_EQ(0.73148686, setup2.analogs[3].dyeSpectrum[1]);

}
