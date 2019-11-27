#include "gtest/gtest.h"

#include <pacbio/dev/gtest-extras.h>
#include <pacbio/dev/TemporaryDirectory.h>
#include <pacbio/logging/Logger.h>

#include <pacbio/primary/SequelCalibrationFile.h>

using namespace std;

using namespace PacBio::Primary;

TEST(SequelCalibrationFile,SequelCalibrationFileHDF5)
{
    PacBio::Dev::TemporaryDirectory td;
    std::string filename = td.DirName() + "cal.h5";
    {// create a dummy file
        SequelMovieConfig mc;
        mc.path = filename;
        SequelCalibrationFileHDF5 a(mc, 32, 32);

        SequelMovieFrame<float> frame(32,32);
        frame.SetPattern1(11);
        a.Write(SequelCalibrationFileHDF5::Sets::DarkNoiseSlope,frame);

        EXPECT_EQ("DarkFrameMean_0.125000",a.GetDarkFrameMeanDataSetName(0.125));
        EXPECT_EQ("DarkFrameSigma_0.125000",a.GetDarkFrameSigmaDataSetName(0.125));

        a.WriteDarkFrameMean(0.125, frame);
        a.WriteDarkFrameSigma(0.125,frame);

        a.WritePhotoelectronSensitivity(1.3);
        a.WriteFrameRate(80);
    }

    {
        // read the file back
        SequelCalibrationFileHDF5 b(filename);
        SequelMovieFrame<float> frame(32, 32);
        b.Read(SequelCalibrationFileHDF5::Sets::DarkNoiseSlope, frame);
        frame.ValidatePattern1(11);
        EXPECT_TRUE(b.ReadDarkFrameMean(0.125, frame));
        frame.ValidatePattern1(11);
        EXPECT_TRUE(b.ReadDarkFrameSigma(0.125, frame));
        frame.ValidatePattern1(11);

        EXPECT_DOUBLE_EQ(1.3, b.ReadPhotoelectronSensitivity());
        EXPECT_DOUBLE_EQ(80, b.ReadFrameRate());
    }
}

TEST(SequelCalibrationFile,SequelLoadingFileHDF5)
{
    PacBio::Dev::TemporaryDirectory td;
#if 0
    TEST_COUT << "Keeping files in  " << td.DirName() << std::endl;
    td.Keep();
#endif

    PacBio::Logging::LogSeverityContext context(PacBio::Logging::LogLevel::INFO);

    std::string filename = td.DirName() + "/test.cal.h5";
    {
        // create a dummy file
        SequelMovieConfig mc;
        mc.path = filename;
        SequelLoadingFileHDF5 a(mc,16,32);
        SequelMovieFrame<float> frame(16,32);
        frame.SetPattern1(11);
        frame.timestamp = 31415;
        frame.index = 999;
        frame.cameraConfiguration = 0x12345678;

        EXPECT_EQ("LoadingMean",a.GetLoadingFrameMeanDataSetName());
        EXPECT_EQ("LoadingVariance",a.GetLoadingFrameVarianceDataSetName());

        uint64_t numFrames = 16;
        float frameRate = 80.0;
        a.WriteMean(frame, numFrames, frameRate);
        a.WriteVariance(frame, numFrames, frameRate);
    }

    {
        // read the file back
        SequelLoadingFileHDF5 b(filename);
        SequelMovieFrame<float> frame;
        EXPECT_TRUE(b.ReadMean(frame));
        EXPECT_EQ(16,frame.NROW);
        EXPECT_EQ(32,frame.NCOL);
        frame.ValidatePattern1(11);
        EXPECT_TRUE(b.ReadMean(frame));
        frame.ValidatePattern1(11);
        EXPECT_TRUE(b.ReadVariance(frame));
        frame.ValidatePattern1(11);

        EXPECT_EQ(999, frame.index);
        EXPECT_FLOAT_EQ(31415, frame.timestamp);
        EXPECT_EQ(0x12345678, frame.cameraConfiguration);
        EXPECT_FLOAT_EQ(80.0, b.ReadFrameRate());
        EXPECT_EQ(16, b.ReadNumFrames());
    }
}

// you can reenable this "test" to generate a full size file.
TEST(SequelCalibrationFile,DISABLED_FullSizeSequelLoadingFileHDF5)
{
    PacBio::Dev::TemporaryDirectory td;
#if 1
    TEST_COUT << "Keeping files in  " << td.DirName() << std::endl;
    td.Keep();
#endif

    PacBio::Logging::LogSeverityContext context(PacBio::Logging::LogLevel::DEBUG);

    std::string filename = td.DirName() + "/test.cal.h5";
    {
        SequelMovieConfig mc;
        mc.path = filename;
        SequelLoadingFileHDF5 a(mc, Spider::maxPixelRows, Spider::maxPixelCols);
        SequelMovieFrame<float> frame(Spider::maxPixelRows, Spider::maxPixelCols);
        frame.SetPattern1(11);
        frame.timestamp = 31415;
        frame.index = 999;
        frame.cameraConfiguration = 0x12345678;

        EXPECT_EQ("LoadingMean", a.GetLoadingFrameMeanDataSetName());
        EXPECT_EQ("LoadingVariance", a.GetLoadingFrameVarianceDataSetName());

        uint64_t numFrames = 16;
        float frameRate = 80.0;
        a.WriteMean(frame, numFrames, frameRate);
        a.WriteVariance(frame, numFrames, frameRate);
    }
    td.Keep();
    TEST_COUT<< "file in " << filename;
}

TEST(SequelCalibrationFile,DISABLED_StuartFile)
{
    SequelLoadingFileHDF5 a("/home/UNIXHOME/mlakata/r54183_20180517_212818_1_A01_rf.dyn.h5");
    EXPECT_EQ(Sequel::maxPixelCols, a.NCOL());
    EXPECT_EQ(Sequel::maxPixelRows, a.NROW());
    SequelMovieFrame<float> mean;
    a.ReadMean(mean); // (Spider::maxPixelRows, Spider::maxPixelCols);
    SequelMovieFrame<float> var;
    a.ReadVariance(var); // (Spider::maxPixelRows, Spider::maxPixelCols);
}
