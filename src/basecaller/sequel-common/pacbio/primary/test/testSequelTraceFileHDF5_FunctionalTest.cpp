
#include <pacbio/primary/Tile.h>
#include <pacbio/primary/SequelTraceFile.h>
#include "gtest/gtest.h"
#include <pacbio/dev/AutoTimer.h>
#include <pacbio/dev/gtest-extras.h>
#include <pacbio/POSIX.h>

using namespace std;

#include <stdexcept>

using namespace PacBio::Primary;
using namespace PacBio;
using namespace PacBio::Dev;

#include "testSequelTraceFileHDF5.h"


void SequelTraceFileHDF5_BaseTest::Run32KTest(const int numChunks )
{
    SequelRectangularROI roi(RowPixels(0), ColPixels(0), RowPixels(32), ColPixels(2048), SequelSensorROI::SequelAlpha());
    const std::string traceFile = temps.GenerateTempFileName(".trc.h5");
    ChipLayoutRTO3 layout;
    PRINTF("roi: %s\n", roi.ToString().c_str());
    {
        QuietAutoTimer x( numChunks * Tile::NumFrames  );

        FillMovieFile(traceFile, roi, numChunks, layout);
        double rate = x.GetRate();
        EXPECT_GT(rate,100.0);
        RecordProperty("FrameRate", rate);
        TEST_COUT << "Filesize:" << PacBio::Utilities::Stat::FileSize(traceFile) << std::endl;
    }
    //    temps.Keep();
}

TEST_F(SequelTraceFileHDF5_FunctionalTest,SpeedTest_32x1024_ZMW)
{
    Run32KTest(120);
#if 0
    for(gzipCompression = 0; gzipCompression <=9 ; gzipCompression++)
    {
        TEST_COUT << "gzipcompress:" << gzipCompression << std::endl;
        Run32KTest(10);
    }
#endif
}

TEST_F(SequelTraceFileHDF5_FunctionalTest,DISABLED_SpeedTest_32x1024_ZMW_NewChunking)
{
    chunking.zmw = 32;
    chunking.channel = 2;
    chunking.frame = 4096;
    for(gzipCompression = 0; gzipCompression <=9 ; gzipCompression++)
    {
        TEST_COUT << "gzipcompress:" << gzipCompression << std::endl;
        Run32KTest(10);
    }
}

TEST_F(SequelTraceFileHDF5_FunctionalTest,SpeedTest_288x1024_ZMW)
{
    const int numChunks = 120;

    SequelRectangularROI roi(RowPixels(0), ColPixels(0), RowPixels(288), ColPixels(2048), SequelSensorROI::SequelAlpha());
    const std::string traceFile = temps.GenerateTempFileName(".trc.h5");
    ChipLayoutRTO3 layout;
    PRINTF("roi: %s\n", roi.ToString().c_str());
    {
        QuietAutoTimer x( numChunks * Tile::NumFrames  );

        FillMovieFile(traceFile, roi, numChunks, layout);
        double rate = x.GetRate();
        EXPECT_GT(rate,100.0);
        RecordProperty("FrameRate", rate);
    }
//    temps.Keep();
}

TEST_F(SequelTraceFileHDF5_FunctionalTest,SpeedTest_1144x1024_ZMW)
{
    const int numChunks = 12;

    SequelRectangularROI roi(RowPixels(0), ColPixels(0), RowPixels(1144), ColPixels(2048), SequelSensorROI::SequelAlpha());
    const std::string traceFile = temps.GenerateTempFileName(".trc.h5");
    ChipLayoutRTO3 layout;
    PRINTF("roi: %s\n", roi.ToString().c_str());
    {
        QuietAutoTimer x( numChunks * Tile::NumFrames  );

        FillMovieFile(traceFile, roi, numChunks, layout);
        double rate = x.GetRate();
        EXPECT_GT(rate,25.0);
        RecordProperty("FrameRate", rate);
    }
//    temps.Keep();
}

TEST_F(SequelTraceFileHDF5_FunctionalTest,SpeedTest_1144x1024_ZMW_withChunking)
{
    const int numChunks = 12;

    SequelRectangularROI roi(RowPixels(0), ColPixels(0), RowPixels(1144), ColPixels(2048), SequelSensorROI::SequelAlpha());
    const std::string traceFile = temps.GenerateTempFileName(".trc.h5");
    ChipLayoutRTO3 layout;
    PRINTF("roi: %s\n", roi.ToString().c_str());
    {
        QuietAutoTimer x( numChunks * Tile::NumFrames  );

        chunking.zmw = 32;
        chunking.channel = 2;
        chunking.frame = 4096;
        gzipCompression = 0;

        FillMovieFile(traceFile, roi, numChunks, layout);
        double rate = x.GetRate();
        EXPECT_GT(rate,25.0);
        RecordProperty("FrameRate", rate);
    }
//    temps.Keep();
}

TEST_F(SequelTraceFileHDF5_FunctionalTest,SpeedTest_1144x1024_ZMW_withChunkingAndGzip)
{
    const int numChunks = 12;

    SequelRectangularROI roi(RowPixels(0), ColPixels(0), RowPixels(1144), ColPixels(2048), SequelSensorROI::SequelAlpha());
    const std::string traceFile = temps.GenerateTempFileName(".trc.h5");
    ChipLayoutRTO3 layout;
    PRINTF("roi: %s\n", roi.ToString().c_str());
    {
        QuietAutoTimer x( numChunks * Tile::NumFrames  );

        chunking.zmw = 32;
        chunking.channel = 2;
        chunking.frame = 4096;
        gzipCompression = 1;

        FillMovieFile(traceFile, roi, numChunks, layout);
        double rate = x.GetRate();
        EXPECT_GT(rate,25.0);
        RecordProperty("FrameRate", rate);
    }
//    temps.Keep();
}


TEST_F(SequelTraceFileHDF5_FunctionalTest,FourHourTest)
{
    const int numChunks = 1440000/Tile::NumFrames;

    SequelRectangularROI roi(RowPixels(0), ColPixels(0), RowPixels(256), ColPixels(256), SequelSensorROI::SequelAlpha());
    const std::string traceFile = temps.GenerateTempFileName(".trc.h5");
    ChipLayoutRTO3 layout;
    PRINTF("roi: %s\n", roi.ToString().c_str());
    {
        QuietAutoTimer x( numChunks * Tile::NumFrames  );

        FillMovieFile(traceFile, roi, numChunks, layout);
        double rate = x.GetRate();
        EXPECT_GT(rate,25.0);
        RecordProperty("FrameRate", rate);
    }
//    temps.Keep();
}
