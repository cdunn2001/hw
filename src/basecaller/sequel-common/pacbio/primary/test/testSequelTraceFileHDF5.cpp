#include <boost/filesystem.hpp>

#include <pacbio/primary/Tile.h>
#include <pacbio/primary/SequelTraceFile.h>
#include "gtest/gtest.h"
#include <pacbio/dev/AutoTimer.h>
#include <pacbio/dev/gtest-extras.h>
#include <pacbio/ipc/JSON.h>
#include <pacbio/POSIX.h>
#include <pacbio/primary/ChipLayoutSpider1p0NTO.h>
#include <pacbio/primary/EventObject.h>

using namespace std;

#include <stdexcept>

using namespace PacBio::Primary;
using namespace PacBio;
using namespace PacBio::Dev;
using namespace boost::filesystem;

#include "testTraceFilePath.h"
#include "testSequelTraceFileHDF5.h"

bool floatFormat = false;

TEST_F(SequelTraceFileHDF5_UnitTest,SequelConstructor)
{
    const int numChunks = 1;

    SequelRectangularROI roi(RowPixels(0),ColPixels(0),RowPixels(128),ColPixels(256), SequelSensorROI::SequelAlpha());
    ASSERT_EQ(128*128,roi.CountHoles());
    const std::string traceFile = temps.GenerateTempFileName(".trc.h5");

    // PacBio::Logging::LogSeverityContext _(PacBio::Logging::LogLevel::TRACE);

    ChipLayoutRTO3 layout;

    FillMovieFile(traceFile, roi, numChunks, layout);

    {
        ifstream traceFileExists(traceFile);

        EXPECT_TRUE(static_cast<bool>(traceFileExists));
    }

    {
        SequelTraceFileHDF5 trace(traceFile);

        std::string formatVersion;
        trace.FormatVersion >> formatVersion;
        EXPECT_EQ("2.3",formatVersion);

        uint32_t platformId;
        trace.RI_PlatformId >> platformId;
        EXPECT_EQ(platformId,SequelTraceFileHDF5::PlatformId_SequelAlpha4);
        EXPECT_EQ(4,SequelTraceFileHDF5::PlatformId_SequelAlpha4);



        string basemap;
        trace.BaseMap >> basemap;
        EXPECT_EQ("TGCA",basemap);

        string softwareVersion;
        trace.SoftwareVersion >> softwareVersion;
        EXPECT_EQ("1.2.3",softwareVersion);

        string xml;
        trace.AcquisitionXML >> xml;
        EXPECT_EQ("<myxml>hello _ underscore</myxml>",xml);

        string instrument;
        trace.RI_InstrumentName >> instrument;
        EXPECT_NE("",instrument);

        string moviePath;
        trace.MoviePath >> moviePath;
        boost::filesystem::path p(traceFile);
        boost::filesystem::path q(moviePath);
        EXPECT_TRUE(boost::filesystem::equivalent(p, q)) << traceFile << " " << moviePath;

        string movieName;
        trace.MovieName >> movieName;
        // PRINTF("%s %s\n",movieName.c_str(), moviePath.c_str());
        EXPECT_EQ(boost::filesystem::path(traceFile).stem().stem().string(),movieName);
        EXPECT_EQ(traceFile,moviePath);




        string platformName;
        trace.RI_PlatformName >> platformName;
        EXPECT_EQ("SequelAlpha",platformName);

        string layoutName;
        trace.LayoutName >> layoutName;
        EXPECT_EQ("SequEL_4.0_RTO3",layoutName);

        EXPECT_EQ(trace.NFRAMES, numChunks * framesPerTile);

        EXPECT_EQ(trace.NUM_HOLES, roi.CountHoles());

        std::vector<int16_t> holeXY(2*roi.CountHoles());
        std::vector<float> holeXYPlot(2*roi.CountHoles());
        trace.HoleXY >> holeXY;
        trace.HoleXYPlot >> holeXYPlot;
        int index=0;
        for(uint32_t row=roi.AbsoluteRowPixelMin();row<roi.AbsoluteRowPixelMax();row++)
        {
            for (uint32_t col=roi.AbsoluteColPixelMin();col < roi.AbsoluteColPixelMax();col+=2 )
            {
                UnitX x = layout.ConvertAbsoluteRowPixelToUnitCellX(row);
                UnitY y = layout.ConvertAbsoluteColPixelToUnitCellY(col);
                EXPECT_EQ(holeXY[index+0],x.Value());
                EXPECT_EQ(holeXY[index+1],y.Value());
                EXPECT_EQ(holeXYPlot[index+0],x.Value());
                ASSERT_EQ(holeXYPlot[index+1],y.Value());
                index += 2;
            }
        }

        string codecName;
        trace.Codec_Name      >> codecName;
        EXPECT_EQ("FixedPoint",codecName);
        std::vector<std::string> configNames;
        trace.Codec_Config    >> configNames;
        EXPECT_EQ("BitDepth",configNames[0]);
        EXPECT_EQ("DynamicRange",configNames[1]);
        EXPECT_EQ("Bias",configNames[2]);
        string units;
        trace.Codec_Units     >> units;
        EXPECT_EQ("Counts",units);
        uint16_t bitDepth;
        trace.Codec_BitDepth  >> bitDepth;
        EXPECT_EQ(16,bitDepth);
        float dynamicRange;
        trace.Codec_DynamicRange >> dynamicRange;
        EXPECT_FLOAT_EQ(65535.0,dynamicRange);
        float bias;
        trace.Codec_Bias      >> bias;
        EXPECT_FLOAT_EQ(0.0,bias);


        std::vector<uint32_t> holeNumber(roi.CountHoles());
        trace.HoleNumber >> holeNumber;
        EXPECT_EQ(holeNumber[0],0x00200020); // first row
        EXPECT_EQ(holeNumber[1],0x00200021);
        EXPECT_EQ(holeNumber[127],0x0020009F);
        EXPECT_EQ(holeNumber[128],0x00210020); // second row
        EXPECT_EQ(holeNumber[129],0x00210021);
        EXPECT_EQ(holeNumber[255],0x0021009F);
        EXPECT_EQ(holeNumber[roi.CountHoles()-1],0x009F009F);

        std::vector<uint8_t> holeStatus(roi.CountHoles());
        trace.HoleStatus >> holeStatus;
        EXPECT_EQ(1, holeStatus[0]); // first row
        EXPECT_EQ(0, holeStatus[0x001020]);
        EXPECT_EQ(1, holeStatus[roi.CountHoles()-1]);

        std::vector<uint8_t> holeType(roi.CountHoles());
        trace.HoleType >> holeType;
        EXPECT_EQ(holeType[0],3); // first row
        EXPECT_EQ(holeType[0x001020],1);
        EXPECT_EQ(holeType[roi.CountHoles()-1],4);

        std::vector<int16_t> traces(roi.CountHoles()*2*Tile::NumFrames);
        trace.Traces >> traces;
        EXPECT_EQ(0,traces[0]); // first zmw, red
        EXPECT_EQ(0,traces[1]);
        EXPECT_EQ(0,traces[2]);
        EXPECT_EQ(0,traces[Tile::NumFrames]); // first zmw, green
        EXPECT_EQ(0,traces[Tile::NumFrames*2]); // second zmw, red
        EXPECT_EQ(0,traces[Tile::NumFrames*4-1]); // second zmw, red

        EXPECT_EQ(0x0101,traces[Tile::NumFrames*2*16]); // 17th zmw, red, first tile
        EXPECT_EQ(0x0202,traces[Tile::NumFrames*2*32]);  // 33th zmw, red, first tile
        EXPECT_EQ(0x0202,traces[Tile::NumFrames*2*32]);  // 33th zmw, red, first tile

        EXPECT_EQ(0x4040,traces[Tile::NumFrames*2*128]);  // 129th zmw, red, first tile, 2nd row
        EXPECT_EQ(0x8080 - 65536,traces[Tile::NumFrames*2*256]);  // 257th zmw, red, first tile, 3rd row

        EXPECT_EQ(0xC7C7 - 65536,traces[2*Tile::NumFrames * roi.CountHoles()-1]); // last pixel.

        double rate;
        trace.FrameRate >> rate;
        EXPECT_EQ(0.0, rate);

        uint32_t numFrames;
        trace.NumFrames >> numFrames;
        EXPECT_EQ(numChunks * framesPerTile,numFrames);
        EXPECT_EQ(numChunks * framesPerTile,trace.NFRAMES);

        uint16_t numColors;
        trace.NumColors >> numColors;
        EXPECT_EQ(2,numColors);

        std::vector<float> spectra(2 * 4 * trace.NUM_HOLES);
        trace.Spectra >> spectra;


        EXPECT_FLOAT_EQ(1      ,spectra[0* trace.NUM_HOLES *2+0]);
        EXPECT_FLOAT_EQ(0      ,spectra[0* trace.NUM_HOLES *2+1]);
        EXPECT_FLOAT_EQ(1      ,spectra[0* trace.NUM_HOLES *2+2]);
        EXPECT_FLOAT_EQ(0      ,spectra[0* trace.NUM_HOLES *2+3]);

        EXPECT_FLOAT_EQ(1,      spectra[1* trace.NUM_HOLES *2+0]);
        EXPECT_FLOAT_EQ(0,      spectra[1* trace.NUM_HOLES *2+1]);
        EXPECT_FLOAT_EQ(1,      spectra[1* trace.NUM_HOLES *2+2]);
        EXPECT_FLOAT_EQ(0,      spectra[1* trace.NUM_HOLES *2+3]);

        EXPECT_FLOAT_EQ(0.1    ,spectra[2* trace.NUM_HOLES *2+0]);
        EXPECT_FLOAT_EQ(0.9    ,spectra[2* trace.NUM_HOLES *2+1]);
        EXPECT_FLOAT_EQ(0.1    ,spectra[2* trace.NUM_HOLES *2+2]);
        EXPECT_FLOAT_EQ(0.9    ,spectra[2* trace.NUM_HOLES *2+3]);

        EXPECT_FLOAT_EQ(0.1    ,spectra[3* trace.NUM_HOLES *2+0]);
        EXPECT_FLOAT_EQ(0.9    ,spectra[3* trace.NUM_HOLES *2+1]);
        EXPECT_FLOAT_EQ(0.1    ,spectra[3* trace.NUM_HOLES *2+2]);
        EXPECT_FLOAT_EQ(0.9    ,spectra[3* trace.NUM_HOLES *2+3]);

        std::vector<float> variance;
        trace.Variance >> variance;
        EXPECT_FLOAT_EQ(1.0,variance[0]);

        std::vector<float> readVariance;
        trace.ReadVariance >> readVariance;
        EXPECT_FLOAT_EQ(0.0,readVariance[0]);

        SequelMovieConfig::TraceChunking chunking1 = trace.GetChunking();
        EXPECT_EQ(2, chunking1.channel);
    }
}

TEST_F(SequelTraceFileHDF5_UnitTest,SequelConstructorSimplified)
{
    const int numChunks = 1;

    SequelRectangularROI roi(RowPixels(0),ColPixels(0),RowPixels(1),ColPixels(64), SequelSensorROI::SequelAlpha());
    ASSERT_EQ(1*32,roi.CountHoles());
    const std::string traceFile = temps.GenerateTempFileName(".trc.h5");

    // PacBio::Logging::LogSeverityContext _(PacBio::Logging::LogLevel::TRACE);

    ChipLayoutRTO3 layout;

    FillMovieFile(traceFile, roi, numChunks, layout);

    {
        ifstream traceFileExists(traceFile);

        EXPECT_TRUE(static_cast<bool>(traceFileExists));
    }

    {
        SequelTraceFileHDF5 trace(traceFile);

        std::string formatVersion;
        trace.FormatVersion >> formatVersion;
        EXPECT_EQ("2.3",formatVersion);

        uint32_t platformId;
        trace.RI_PlatformId >> platformId;
        EXPECT_EQ(platformId,SequelTraceFileHDF5::PlatformId_SequelAlpha4);
        EXPECT_EQ(4,SequelTraceFileHDF5::PlatformId_SequelAlpha4);



        string basemap;
        trace.BaseMap >> basemap;
        EXPECT_EQ("TGCA",basemap);

        string softwareVersion;
        trace.SoftwareVersion >> softwareVersion;
        EXPECT_EQ("1.2.3",softwareVersion);

        string xml;
        trace.AcquisitionXML >> xml;
        EXPECT_EQ("<myxml>hello _ underscore</myxml>",xml);

        string instrument;
        trace.RI_InstrumentName >> instrument;
        EXPECT_NE("",instrument);

        string moviePath;
        trace.MoviePath >> moviePath;
        boost::filesystem::path p(traceFile);
        boost::filesystem::path q(moviePath);
        EXPECT_TRUE(boost::filesystem::equivalent(p, q)) << traceFile << " " << moviePath;

        string movieName;
        trace.MovieName >> movieName;
        // PRINTF("%s %s\n",movieName.c_str(), moviePath.c_str());
        EXPECT_EQ(boost::filesystem::path(traceFile).stem().stem().string(),movieName);
        EXPECT_EQ(traceFile,moviePath);




        string platformName;
        trace.RI_PlatformName >> platformName;
        EXPECT_EQ("SequelAlpha",platformName);

        string layoutName;
        trace.LayoutName >> layoutName;
        EXPECT_EQ("SequEL_4.0_RTO3",layoutName);

        EXPECT_EQ(trace.NFRAMES, numChunks * framesPerTile);

        EXPECT_EQ(trace.NUM_HOLES, roi.CountHoles());

        std::vector<int16_t> holeXY(2*roi.CountHoles());
        std::vector<float> holeXYPlot(2*roi.CountHoles());
        trace.HoleXY >> holeXY;
        trace.HoleXYPlot >> holeXYPlot;
        int index=0;
        for(uint32_t row=roi.AbsoluteRowPixelMin();row<roi.AbsoluteRowPixelMax();row++)
        {
            for (uint32_t col=roi.AbsoluteColPixelMin();col < roi.AbsoluteColPixelMax();col+=2 )
            {
                UnitX x = layout.ConvertAbsoluteRowPixelToUnitCellX(row);
                UnitY y = layout.ConvertAbsoluteColPixelToUnitCellY(col);
                EXPECT_EQ(holeXY[index+0],x.Value());
                EXPECT_EQ(holeXY[index+1],y.Value());
                EXPECT_EQ(holeXYPlot[index+0],x.Value());
                ASSERT_EQ(holeXYPlot[index+1],y.Value());
                index += 2;
            }
        }

        string codecName;
        trace.Codec_Name      >> codecName;
        EXPECT_EQ("FixedPoint",codecName);
        std::vector<std::string> configNames;
        trace.Codec_Config    >> configNames;
        EXPECT_EQ("BitDepth",configNames[0]);
        EXPECT_EQ("DynamicRange",configNames[1]);
        EXPECT_EQ("Bias",configNames[2]);
        string units;
        trace.Codec_Units     >> units;
        EXPECT_EQ("Counts",units);
        uint16_t bitDepth;
        trace.Codec_BitDepth  >> bitDepth;
        EXPECT_EQ(16,bitDepth);
        float dynamicRange;
        trace.Codec_DynamicRange >> dynamicRange;
        EXPECT_FLOAT_EQ(65535.0,dynamicRange);
        float bias;
        trace.Codec_Bias      >> bias;
        EXPECT_FLOAT_EQ(0.0,bias);


        std::vector<uint32_t> holeNumber(roi.CountHoles());
        trace.HoleNumber >> holeNumber;
        EXPECT_EQ(holeNumber[0],0x00200020); // first row
        EXPECT_EQ(holeNumber[1],0x00200021);
        EXPECT_EQ(holeNumber[31],0x0020003F);

        std::vector<uint8_t> holeStatus(roi.CountHoles());
        trace.HoleStatus >> holeStatus;
        EXPECT_EQ(1, holeStatus[0]); // first row
//        EXPECT_EQ(0, holeStatus[0x001020]);
        EXPECT_EQ(1, holeStatus[roi.CountHoles()-1]);

        std::vector<uint8_t> holeType(roi.CountHoles());
        trace.HoleType >> holeType;
        EXPECT_EQ(holeType[0],3); // first row
        EXPECT_EQ(holeType[roi.CountHoles()-1],13);

        std::vector<int16_t> traces(roi.CountHoles()*2*Tile::NumFrames);
        trace.Traces >> traces;
        EXPECT_EQ(0,traces[0]); // first zmw, red
        EXPECT_EQ(0,traces[1]);
        EXPECT_EQ(0,traces[2]);
        EXPECT_EQ(0,traces[Tile::NumFrames]); // first zmw, green
        EXPECT_EQ(0,traces[Tile::NumFrames*2]); // second zmw, red
        EXPECT_EQ(0,traces[Tile::NumFrames*4-1]); // second zmw, red

#if 0
        std::stringstream ss;
        for(int i=0;i<traces.size();i++)
        {
            ss << "[" << i << "]:" << std::hex << traces[i] << std::dec << " ";
            if (i % 16 == 15)
            {
                TEST_COUT << ss.str() << std::endl;
                ss.str("");
            }
        }
#endif
        EXPECT_EQ(0x0000,traces[Tile::NumFrames*2*16-1]); // 16th zmw, red, first tile, last sample
        EXPECT_EQ(0x0101,traces[Tile::NumFrames*2*16]); // 17th zmw, red, first tile, first sample

        EXPECT_EQ(0x0101,traces[2*Tile::NumFrames * roi.CountHoles()-1]); // last pixel.

        double rate;
        trace.FrameRate >> rate;
        EXPECT_EQ(0.0, rate);

        uint32_t numFrames;
        trace.NumFrames >> numFrames;
        EXPECT_EQ(numChunks * framesPerTile,numFrames);
        EXPECT_EQ(numChunks * framesPerTile,trace.NFRAMES);

        uint16_t numColors;
        trace.NumColors >> numColors;
        EXPECT_EQ(2,numColors);

        std::vector<float> spectra(2 * 4 * trace.NUM_HOLES);
        trace.Spectra >> spectra;


        EXPECT_FLOAT_EQ(1      ,spectra[0* trace.NUM_HOLES *2+0]);
        EXPECT_FLOAT_EQ(0      ,spectra[0* trace.NUM_HOLES *2+1]);
        EXPECT_FLOAT_EQ(1      ,spectra[0* trace.NUM_HOLES *2+2]);
        EXPECT_FLOAT_EQ(0      ,spectra[0* trace.NUM_HOLES *2+3]);

        EXPECT_FLOAT_EQ(1,      spectra[1* trace.NUM_HOLES *2+0]);
        EXPECT_FLOAT_EQ(0,      spectra[1* trace.NUM_HOLES *2+1]);
        EXPECT_FLOAT_EQ(1,      spectra[1* trace.NUM_HOLES *2+2]);
        EXPECT_FLOAT_EQ(0,      spectra[1* trace.NUM_HOLES *2+3]);

        EXPECT_FLOAT_EQ(0.1    ,spectra[2* trace.NUM_HOLES *2+0]);
        EXPECT_FLOAT_EQ(0.9    ,spectra[2* trace.NUM_HOLES *2+1]);
        EXPECT_FLOAT_EQ(0.1    ,spectra[2* trace.NUM_HOLES *2+2]);
        EXPECT_FLOAT_EQ(0.9    ,spectra[2* trace.NUM_HOLES *2+3]);

        EXPECT_FLOAT_EQ(0.1    ,spectra[3* trace.NUM_HOLES *2+0]);
        EXPECT_FLOAT_EQ(0.9    ,spectra[3* trace.NUM_HOLES *2+1]);
        EXPECT_FLOAT_EQ(0.1    ,spectra[3* trace.NUM_HOLES *2+2]);
        EXPECT_FLOAT_EQ(0.9    ,spectra[3* trace.NUM_HOLES *2+3]);

        std::vector<float> variance;
        trace.Variance >> variance;
        EXPECT_FLOAT_EQ(1.0,variance[0]);

        std::vector<float> readVariance;
        trace.ReadVariance >> readVariance;
        EXPECT_FLOAT_EQ(0.0,readVariance[0]);

        SequelMovieConfig::TraceChunking chunking1 = trace.GetChunking();
        EXPECT_EQ(2, chunking1.channel);
    }
}

TEST_F(SequelTraceFileHDF5_UnitTest, SequelConstructorSparse)
{
    const int numChunks = 1;

    SequelSparseROI roi(SequelSensorROI::SequelAlpha());
    roi.AddRectangle(RowPixels(0), ColPixels(0), RowPixels(1), ColPixels(64));
    roi.AddRectangle(RowPixels(1), ColPixels(64), RowPixels(1), ColPixels(64));

    ASSERT_EQ(2*32,roi.CountHoles());
    const std::string traceFile = temps.GenerateTempFileName(".trc.h5");

    // PacBio::Logging::LogSeverityContext _(PacBio::Logging::LogLevel::TRACE);

    ChipLayoutRTO3 layout;

    FillMovieFile(traceFile, roi, numChunks, layout);

    {
        ifstream traceFileExists(traceFile);

        EXPECT_TRUE(static_cast<bool>(traceFileExists));
    }

    {
        SequelTraceFileHDF5 trace(traceFile);

        std::string formatVersion;
        trace.FormatVersion >>formatVersion;
        EXPECT_EQ("2.3",formatVersion);

        uint32_t platformId;
        trace.RI_PlatformId >>platformId;
        EXPECT_EQ(platformId, SequelTraceFileHDF5::PlatformId_SequelAlpha4);
        EXPECT_EQ(4,SequelTraceFileHDF5::PlatformId_SequelAlpha4);


        string basemap;
        trace.BaseMap >>basemap;
        EXPECT_EQ("TGCA",basemap);

        string softwareVersion;
        trace.SoftwareVersion >>softwareVersion;
        EXPECT_EQ("1.2.3",softwareVersion);

        string xml;
        trace.AcquisitionXML >>xml;
        EXPECT_EQ("<myxml>hello _ underscore</myxml>",xml);

        string instrument;
        trace.RI_InstrumentName >>instrument;
        EXPECT_NE("",instrument);

        string moviePath;
        trace.MoviePath >>moviePath;
        boost::filesystem::path p(traceFile);
        boost::filesystem::path q(moviePath);
        EXPECT_TRUE(boost::filesystem::equivalent(p, q)) << traceFile << " " <<moviePath;

        string movieName;
        trace.MovieName >>movieName;
        // PRINTF("%s %s\n",movieName.c_str(), moviePath.c_str());
        EXPECT_EQ(boost::filesystem::path(traceFile).stem().stem().string(), movieName);
        EXPECT_EQ(traceFile, moviePath);


        string platformName;
        trace.RI_PlatformName >>platformName;
        EXPECT_EQ("SequelAlpha",platformName);

        string layoutName;
        trace.LayoutName >>layoutName;
        EXPECT_EQ("SequEL_4.0_RTO3",layoutName);

        EXPECT_EQ(trace.NFRAMES,numChunks* framesPerTile);

        EXPECT_EQ(trace.NUM_HOLES, roi.CountHoles());

        std::vector<int16_t> holeXY(2 * roi.CountHoles());
        std::vector<float> holeXYPlot(2 * roi.CountHoles());
        trace.HoleXY >>holeXY;
        trace.HoleXYPlot >>holeXYPlot;
        #if 0
        int index = 0;
        for(uint32_t row = roi.AbsoluteRowPixelMin();row<roi.AbsoluteRowPixelMax();row++)
        {
        for (uint32_t col = roi.AbsoluteColPixelMin();col<roi.AbsoluteColPixelMax();col+=2 )
        {
        UnitX x = layout.ConvertAbsoluteRowPixelToUnitCellX(row);
        UnitY y = layout.ConvertAbsoluteColPixelToUnitCellY(col);
        EXPECT_EQ(holeXY[index + 0], x.Value());
        EXPECT_EQ(holeXY[index + 1], y.Value());
        EXPECT_EQ(holeXYPlot[index + 0], x.Value());
        ASSERT_EQ(holeXYPlot[index + 1], y.Value());
        index += 2;
        }
        }
        #endif

        string codecName;
        trace.Codec_Name      >>codecName;
        EXPECT_EQ("FixedPoint",codecName);
        std::vector<std::string> configNames;
        trace.Codec_Config    >>configNames;
        EXPECT_EQ("BitDepth",configNames[0]);
        EXPECT_EQ("DynamicRange",configNames[1]);
        EXPECT_EQ("Bias",configNames[2]);
        string units;
        trace.Codec_Units     >>units;
        EXPECT_EQ("Counts",units);
        uint16_t bitDepth;
        trace.Codec_BitDepth  >>bitDepth;
        EXPECT_EQ(16,bitDepth);
        float dynamicRange;
        trace.Codec_DynamicRange >>dynamicRange;
        EXPECT_FLOAT_EQ(65535.0,dynamicRange);
        float bias;
        trace.Codec_Bias      >>bias;
        EXPECT_FLOAT_EQ(0.0,bias);


        std::vector<uint32_t> holeNumber(roi.CountHoles());
        trace.HoleNumber >>holeNumber;
        EXPECT_EQ(holeNumber[0],0x00200020); // first row
        EXPECT_EQ(holeNumber[1],0x00200021);
        EXPECT_EQ(holeNumber[31],0x0020003F);

        std::vector<uint8_t> holeStatus(roi.CountHoles());
        trace.HoleStatus >>holeStatus;
        EXPECT_EQ(1, holeStatus[0]); // first row
        //        EXPECT_EQ(0, holeStatus[0x001020]);
        EXPECT_EQ(1, holeStatus[roi.CountHoles()-1]);

        std::vector<uint8_t> holeType(roi.CountHoles());
        trace.HoleType >>holeType;
        EXPECT_EQ(holeType[0],3); // first row
        EXPECT_EQ(holeType[31],13);
        EXPECT_EQ(holeType[63],22);

        std::vector<int16_t> traces(roi.CountHoles() * 2 * Tile::NumFrames);
        trace.Traces >>traces;
        EXPECT_EQ(0,traces[0]); // first zmw, red
        EXPECT_EQ(0,traces[1]);
        EXPECT_EQ(0,traces[2]);
        EXPECT_EQ(0,traces[Tile::NumFrames]); // first zmw, green
        EXPECT_EQ(0,traces[Tile::NumFrames*2]); // second zmw, red
        EXPECT_EQ(0,traces[Tile::NumFrames*4-1]); // second zmw, red

        #if 0
        std::stringstream ss;
        for(int i=0;i<traces.size();i++)
        {
            ss << "[" << i << "]:" << std::hex << traces[i] << std::dec << " ";
            if (i % 16 == 15)
            {
                TEST_COUT << ss.str() << std::endl;
                ss.str("");
            }
        }
        #endif
        EXPECT_EQ(0x0000,traces[Tile::NumFrames*2*16-1]); // 16th zmw, red, first tile, last sample
        EXPECT_EQ(0x0101,traces[Tile::NumFrames*2*16]); // 17th zmw, red, first tile, first sample

        EXPECT_EQ(0x4343,traces[2*Tile::NumFrames* roi.CountHoles()-1]); // last pixel.

        double rate;
        trace.FrameRate >>rate;
        EXPECT_EQ(0.0, rate);

        uint32_t numFrames;
        trace.NumFrames >>numFrames;
        EXPECT_EQ(numChunks* framesPerTile,numFrames);
        EXPECT_EQ(numChunks* framesPerTile,trace.NFRAMES);

        uint16_t numColors;
        trace.NumColors >>numColors;
        EXPECT_EQ(2,numColors);

        std::vector<float> spectra(2 * 4 * trace.NUM_HOLES);
        trace.Spectra >>spectra;


        EXPECT_FLOAT_EQ(1      ,spectra[0* trace.NUM_HOLES *2+0]);
        EXPECT_FLOAT_EQ(0      ,spectra[0* trace.NUM_HOLES *2+1]);
        EXPECT_FLOAT_EQ(1      ,spectra[0* trace.NUM_HOLES *2+2]);
        EXPECT_FLOAT_EQ(0      ,spectra[0* trace.NUM_HOLES *2+3]);

        EXPECT_FLOAT_EQ(1,      spectra[1* trace.NUM_HOLES *2+0]);
        EXPECT_FLOAT_EQ(0,      spectra[1* trace.NUM_HOLES *2+1]);
        EXPECT_FLOAT_EQ(1,      spectra[1* trace.NUM_HOLES *2+2]);
        EXPECT_FLOAT_EQ(0,      spectra[1* trace.NUM_HOLES *2+3]);

        EXPECT_FLOAT_EQ(0.1    ,spectra[2* trace.NUM_HOLES *2+0]);
        EXPECT_FLOAT_EQ(0.9    ,spectra[2* trace.NUM_HOLES *2+1]);
        EXPECT_FLOAT_EQ(0.1    ,spectra[2* trace.NUM_HOLES *2+2]);
        EXPECT_FLOAT_EQ(0.9    ,spectra[2* trace.NUM_HOLES *2+3]);

        EXPECT_FLOAT_EQ(0.1    ,spectra[3* trace.NUM_HOLES *2+0]);
        EXPECT_FLOAT_EQ(0.9    ,spectra[3* trace.NUM_HOLES *2+1]);
        EXPECT_FLOAT_EQ(0.1    ,spectra[3* trace.NUM_HOLES *2+2]);
        EXPECT_FLOAT_EQ(0.9    ,spectra[3* trace.NUM_HOLES *2+3]);

        std::vector<float> variance;
        trace.Variance >>variance;
        EXPECT_FLOAT_EQ(1.0,variance[0]);

        std::vector<float> readVariance;
        trace.ReadVariance >>readVariance;
        EXPECT_FLOAT_EQ(0.0,readVariance[0]);

        SequelMovieConfig::TraceChunking chunking1 = trace.GetChunking();
        EXPECT_EQ(2, chunking1.channel);
    }
}

TEST_F(SequelTraceFileHDF5_UnitTest,SpiderConstructor)
{
    const int numChunks = 1;

    SequelRectangularROI roi(RowPixels(0),ColPixels(0),RowPixels(128),ColPixels(128), SequelSensorROI::Spider());
    ASSERT_EQ(128*128,roi.CountHoles());
    const std::string traceFile = temps.GenerateTempFileName(".trc.h5");

    // PacBio::Logging::LogSeverityContext _(PacBio::Logging::LogLevel::TRACE);

    ChipLayoutSpider1p0NTO layout;
    ASSERT_EQ(ChipClass::Spider, layout.GetChipClass());

    FillMovieFile(traceFile, roi, numChunks, layout);

    {
        ifstream traceFileExists(traceFile);

        EXPECT_TRUE(static_cast<bool>(traceFileExists));
    }

    {
        SequelTraceFileHDF5 trace(traceFile);

        std::string formatVersion;
        trace.FormatVersion >> formatVersion;
        EXPECT_EQ("2.3",formatVersion);

        uint32_t platformId;
        trace.RI_PlatformId >> platformId;
        EXPECT_EQ(SequelTraceFileHDF5::PlatformId_Spider5,platformId);
        EXPECT_EQ(5,SequelTraceFileHDF5::PlatformId_Spider5);

        string basemap;
        trace.BaseMap >> basemap;
        EXPECT_EQ("TGCA",basemap);

        string softwareVersion;
        trace.SoftwareVersion >> softwareVersion;
        EXPECT_EQ("1.2.3",softwareVersion);

        string xml;
        trace.AcquisitionXML >> xml;
        EXPECT_EQ("<myxml>hello _ underscore</myxml>",xml);

        string instrument;
        trace.RI_InstrumentName >> instrument;
        EXPECT_NE("",instrument);

        string moviePath;
        trace.MoviePath >> moviePath;
        boost::filesystem::path p(traceFile);
        boost::filesystem::path q(moviePath);
        EXPECT_TRUE(boost::filesystem::equivalent(p, q)) << traceFile << " " << moviePath;

        string movieName;
        trace.MovieName >> movieName;
        // PRINTF("%s %s\n",movieName.c_str(), moviePath.c_str());
        EXPECT_EQ(boost::filesystem::path(traceFile).stem().stem().string(),movieName);
        EXPECT_EQ(traceFile,moviePath);

        string platformName;
        trace.RI_PlatformName >> platformName;
        EXPECT_EQ("Spider",platformName);

        string layoutName;
        trace.LayoutName >> layoutName;
        EXPECT_EQ("Spider_1p0_NTO",layoutName);

        EXPECT_EQ(trace.NFRAMES, numChunks * framesPerTile);

        EXPECT_EQ(trace.NUM_HOLES, roi.CountHoles());

        const int spatialDimensions = 2;
        std::vector<int16_t> holeXY( spatialDimensions*roi.CountHoles());
        std::vector<float> holeXYPlot(spatialDimensions*roi.CountHoles());
        trace.HoleXY >> holeXY;
        trace.HoleXYPlot >> holeXYPlot;
        int index=0;

        for(uint32_t row=roi.AbsoluteRowPixelMin();row<roi.AbsoluteRowPixelMax();row += layout.RowPixelsPerZmw())
        {
            for (uint32_t col=roi.AbsoluteColPixelMin();col < roi.AbsoluteColPixelMax();col += layout.ColPixelsPerZmw() )
            {
                UnitX x = layout.ConvertAbsoluteRowPixelToUnitCellX(row);
                UnitY y = layout.ConvertAbsoluteColPixelToUnitCellY(col);
                EXPECT_EQ(holeXY[index+0],x.Value()) << " index:" << index << " row:" << row << " col:" << col;
                EXPECT_EQ(holeXY[index+1],y.Value()) << " index:" << index << " row:" << row << " col:" << col;
                EXPECT_EQ(holeXYPlot[index+0],x.Value()) << " index:" << index << " row:" << row << " col:" << col;
                ASSERT_EQ(holeXYPlot[index+1],y.Value()) << " index:" << index << " row:" << row << " col:" << col;
                index += 2;
            }
        }

        string codecName;
        trace.Codec_Name      >> codecName;
        EXPECT_EQ("FixedPoint",codecName);
        std::vector<std::string> configNames;
        trace.Codec_Config    >> configNames;
        EXPECT_EQ("BitDepth",configNames[0]);
        EXPECT_EQ("DynamicRange",configNames[1]);
        EXPECT_EQ("Bias",configNames[2]);
        string units;
        trace.Codec_Units     >> units;
        EXPECT_EQ("Counts",units);
        uint16_t bitDepth;
        trace.Codec_BitDepth  >> bitDepth;
        EXPECT_EQ(16,bitDepth);
        float dynamicRange;
        trace.Codec_DynamicRange >> dynamicRange;
        EXPECT_FLOAT_EQ(65535.0,dynamicRange);
        float bias;
        trace.Codec_Bias      >> bias;
        EXPECT_FLOAT_EQ(0.0,bias);


        std::vector<uint32_t> holeNumber(roi.CountHoles());
        trace.HoleNumber >> holeNumber;
        EXPECT_EQ((0 << 16)    ,   holeNumber[0] ); // first row from botom
        EXPECT_EQ((0 << 16) + 1,   holeNumber[1] );
        EXPECT_EQ((0 << 16) + 127, holeNumber[127]);
        EXPECT_EQ((1 << 16)    ,   holeNumber[128]); // second row from bottom
        EXPECT_EQ((1 << 16) + 1,   holeNumber[129]);
        EXPECT_EQ((1 << 16) + 127, holeNumber[255]);
        EXPECT_EQ((127 << 16) + 127, holeNumber.back());

        std::vector<uint8_t> holeStatus(roi.CountHoles());
        trace.HoleStatus >> holeStatus;
        EXPECT_EQ(0  , holeStatus[0]);     // 0 = zmw
        EXPECT_EQ(0  , holeStatus.back()); // 1 = not zmw

        std::vector<uint8_t> holeType(roi.CountHoles());
        trace.HoleType >> holeType;
        EXPECT_EQ(static_cast<uint8_t>(ChipLayoutSpider1p0NTO::UnitCellType::Sequencing), holeType[0]);
        EXPECT_EQ(static_cast<uint8_t>(ChipLayoutSpider1p0NTO::UnitCellType::Sequencing), holeType.back());

        auto dims = GetDims(trace.Traces);
        ASSERT_EQ(3,dims.size());
        ASSERT_EQ(roi.CountHoles(),dims[0]);
        ASSERT_EQ(1,dims[1]);
        ASSERT_EQ((int)Tile::NumFrames,dims[2]);

        std::vector<int16_t> traces;
        trace.Traces >> traces;
        EXPECT_EQ(roi.CountHoles()*Tile::NumFrames, traces.size());

        // the first tile is all 0x00. The second tile is all 0x01, etc.
        EXPECT_EQ(0,traces[0]); // first zmwd
        EXPECT_EQ(0,traces[1]);
        EXPECT_EQ(0,traces[2]);
        EXPECT_EQ(0,traces[Tile::NumFrames]); // second zmw
        EXPECT_EQ(0,traces[Tile::NumFrames*2]); // third zmw
        EXPECT_EQ(0,traces[Tile::NumFrames*15]); // 16th zmw
        EXPECT_EQ(0,traces[Tile::NumFrames*31]); // 32nd zmw

        EXPECT_EQ(0x0101, traces[Tile::NumFrames*Tile::NumPixels]);  // 33th zmw,  second tile
        EXPECT_EQ(0x0101, traces[Tile::NumFrames*Tile::NumPixels + Tile::NumFrames*31]);  // 64th zmw,  second tile

#if 0
        EXPECT_EQ(0x4040,traces[Tile::NumFrames*128]);  // 129th zmw, red, first tile, 2nd row
        EXPECT_EQ(0x8080 - 65536,traces[Tile::NumFrames*256]);  // 257th zmw, red, first tile, 3rd row

        EXPECT_EQ(0xC7C7 - 65536,traces[Tile::NumFrames * roi.CountHoles()-1]); // last pixel.
#endif

        double rate;
        trace.FrameRate >> rate;
        EXPECT_EQ(0.0, rate);

        uint32_t numFrames;
        trace.NumFrames >> numFrames;
        EXPECT_EQ(numChunks * framesPerTile,numFrames);
        EXPECT_EQ(numChunks * framesPerTile,trace.NFRAMES);

        uint16_t numColors;
        trace.NumColors >> numColors;
        EXPECT_EQ(1,numColors);

        std::vector<float> spectra(4 * trace.NUM_HOLES);
        trace.Spectra >> spectra;

#if 0
        // fix me. enable these tests
        EXPECT_FLOAT_EQ(1      ,spectra[0* trace.NUM_HOLES *2+0]);
        EXPECT_FLOAT_EQ(0      ,spectra[0* trace.NUM_HOLES *2+1]);
        EXPECT_FLOAT_EQ(1      ,spectra[0* trace.NUM_HOLES *2+2]);
        EXPECT_FLOAT_EQ(0      ,spectra[0* trace.NUM_HOLES *2+3]);

        EXPECT_FLOAT_EQ(1,      spectra[1* trace.NUM_HOLES *2+0]);
        EXPECT_FLOAT_EQ(0,      spectra[1* trace.NUM_HOLES *2+1]);
        EXPECT_FLOAT_EQ(1,      spectra[1* trace.NUM_HOLES *2+2]);
        EXPECT_FLOAT_EQ(0,      spectra[1* trace.NUM_HOLES *2+3]);

        EXPECT_FLOAT_EQ(0.1    ,spectra[2* trace.NUM_HOLES *2+0]);
        EXPECT_FLOAT_EQ(0.9    ,spectra[2* trace.NUM_HOLES *2+1]);
        EXPECT_FLOAT_EQ(0.1    ,spectra[2* trace.NUM_HOLES *2+2]);
        EXPECT_FLOAT_EQ(0.9    ,spectra[2* trace.NUM_HOLES *2+3]);

        EXPECT_FLOAT_EQ(0.1    ,spectra[3* trace.NUM_HOLES *2+0]);
        EXPECT_FLOAT_EQ(0.9    ,spectra[3* trace.NUM_HOLES *2+1]);
        EXPECT_FLOAT_EQ(0.1    ,spectra[3* trace.NUM_HOLES *2+2]);
        EXPECT_FLOAT_EQ(0.9    ,spectra[3* trace.NUM_HOLES *2+3]);
#endif


        std::vector<float> variance;
        trace.Variance >> variance;
        EXPECT_FLOAT_EQ(1.0,variance[0]);

        std::vector<float> readVariance;
        trace.ReadVariance >> readVariance;
        EXPECT_FLOAT_EQ(0.0,readVariance[0]);

        SequelMovieConfig::TraceChunking chunking1 = trace.GetChunking();
        EXPECT_EQ(1, chunking1.channel);
    }
}

AnalogSet CreateSequelAnalogs1()
{
    const std::string sequelAnalogs = R"(
        [
            {"base":"T", "spectrumValues":[1.0,0.0], "relativeAmplitude": 0.8, "intraPulseXsnCV": 0.01, "ipdMeanSeconds": 15, "pulseWidthMeanSeconds": 5.0,
             "pw2SlowStepRatio": 0.11, "ipd2SlowStepRatio":0.21},
            {"base":"G", "spectrumValues":[1.0,0.0], "relativeAmplitude": 0.5, "intraPulseXsnCV": 0.02, "ipdMeanSeconds": 16, "pulseWidthMeanSeconds": 5.5,
             "pw2SlowStepRatio": 0.12, "ipd2SlowStepRatio":0.22},
            {"base":"C", "spectrumValues":[0.1,0.9], "relativeAmplitude": 1.0, "intraPulseXsnCV": 0.03, "ipdMeanSeconds": 18, "pulseWidthMeanSeconds": 6.0,
             "pw2SlowStepRatio": 0.13, "ipd2SlowStepRatio":0.23},
            {"base":"A", "spectrumValues":[0.1,0.9], "relativeAmplitude": 0.6, "intraPulseXsnCV": 0.04, "ipdMeanSeconds": 42, "pulseWidthMeanSeconds": 6.5,
             "pw2SlowStepRatio": 0.14, "ipd2SlowStepRatio":0.24}
        ]
    )";
    Json::Value analogsJson=PacBio::IPC::ParseJSON(sequelAnalogs);
    return ParseAnalogSet(analogsJson);
}

AnalogSet CreateSpiderAnalogs1()
{
    const std::string spiderAnalogs = R"(
        [
            {"base":"T", "spectrumValues":[1.0], "relativeAmplitude": 0.8, "intraPulseXsnCV": 0.01, "ipdMeanSeconds": 15, "pulseWidthMeanSeconds": 5.0},
            {"base":"G", "spectrumValues":[1.0], "relativeAmplitude": 0.5, "intraPulseXsnCV": 0.02, "ipdMeanSeconds": 16, "pulseWidthMeanSeconds": 5.5},
            {"base":"C", "spectrumValues":[1.0], "relativeAmplitude": 1.0, "intraPulseXsnCV": 0.03, "ipdMeanSeconds": 18, "pulseWidthMeanSeconds": 6.0},
            {"base":"A", "spectrumValues":[1.0], "relativeAmplitude": 0.6, "intraPulseXsnCV": 0.04, "ipdMeanSeconds": 42, "pulseWidthMeanSeconds": 6.5}
        ]
    )";
    Json::Value analogsJson=PacBio::IPC::ParseJSON(spiderAnalogs);
    return ParseAnalogSet(analogsJson);
}

void CreateOddFile(const std::string traceFile, int size)
{

    SequelRectangularROI roi(RowPixels(0), ColPixels(0), RowPixels(128), ColPixels(256), SequelSensorROI::SequelAlpha());

    const size_t TilesPerChunk = framesPerTile * roi.SensorROI().TotalPixels() * sizeof(int16_t) / sizeof(Tile);
    POSIX::Unlink(traceFile.c_str());

    ChipLayoutRTO3 layout;
    auto chipClass = layout.GetChipClass();
    {
        SequelMovieConfig mc;
        mc.path = traceFile;
        mc.roi.reset(roi.Clone());
        mc.chipClass = chipClass;
        mc.numFrames = size;
        SequelTraceFileHDF5 trace(mc);

        auto analogs = CreateSequelAnalogs1();
        trace.SetAnalogs(analogs, layout);

        trace.PrepareChunkBuffers();

        int framesToGo = size;
        while (framesToGo > 0)
        {
            int thisTime = framesToGo;
            if (thisTime > (int)Tile::NumFrames) thisTime = Tile::NumFrames;

            std::unique_ptr<Tile> header(Tile::makeHeader());
            trace.AddChunk(header.get(), thisTime);

            std::unique_ptr<Tile> tile(Tile::make());
            uint32_t tileOffset = 0;
            for (uint32_t i = 0; i < TilesPerChunk; i++)
            {
                tile->SetPattern2(i * 32);
                trace.AddTile(tile.get(), tileOffset++);
            }
            framesToGo -= thisTime;
            trace.FlushChunk();
        }
    }
    {
        SequelTraceFileHDF5 trace(traceFile);

        size_t numPixels = roi.CountHoles() * 2;
        std::vector<int16_t> traces(numPixels* size);
        trace.Traces >> traces;
        int j=0;
        for(uint32_t pixel=0;pixel< numPixels ;pixel++)
        {
            int physicalPixel = ((pixel & ~0xFF) << 3 ) | (pixel & 0xFF); // the shift is because the selected ROI is 256 columns from the sensor ROI of 2048 columns

            for(int frame = 0;frame < size; frame++)
            {
                int16_t v = static_cast<int16_t>((physicalPixel ^ 1) + (frame % Tile::NumFrames) * 10);
                ASSERT_EQ(v,traces[j]) << "pixel:" << pixel << " frame:" << frame;
                j++;
            }
        }
    }
}

TEST_F(SequelTraceFileHDF5_UnitTest,768Frames)
{
    const std::string traceFile = temps.GenerateTempFileName(".trc.h5");
    CreateOddFile(traceFile,768);
}

TEST_F(SequelTraceFileHDF5_UnitTest,256Frames)
{
    const std::string traceFile = temps.GenerateTempFileName(".trc.h5");
    CreateOddFile(traceFile,256);
}


TEST_F(SequelTraceFileHDF5_UnitTest,BadROI)
{
    PacBio::Logging::LogSeverityContext x(PacBio::Logging::LogLevel::FATAL);
    const int numChunks = 1;
    ChipLayoutRTO3 layout;

    for(int cols=1;cols<32;cols++)
    {
        try
        {
            SequelRectangularROI roi(RowPixels(0), ColPixels(cols), RowPixels(128), ColPixels(256),SequelSensorROI::SequelAlpha());
            const std::string traceFile  = temps.GenerateTempFileName(".trc.h5");
            FillMovieFile(traceFile, roi, numChunks, layout);
            ASSERT_TRUE(false) << "Cols = " << cols; // we should not get here. there should be a throw above.
        }
        catch(const std::exception&)
        {
            // ok. we want something to throw above. Either the ROI constructor, or the trace file.
        }
    }
}


TEST_F(SequelTraceFileHDF5_UnitTest,ShiftedROI)
{
    const int numChunks = 1;

    SequelRectangularROI roi(RowPixels(0), ColPixels(32), RowPixels(128), ColPixels(256),SequelSensorROI::SequelAlpha());
    const std::string traceFile = temps.GenerateTempFileName(".trc.h5");

    ChipLayoutRTO3 layout;
    FillMovieFile(traceFile, roi, numChunks, layout, 3);
    SequelTraceFileHDF5 trace(traceFile);

    std::vector<int16_t> traces(roi.CountHoles()*2*Tile::NumFrames);
    trace.Traces >> traces;
    uint32_t ipixel = 32 + 1; // plus one to skip first red pixel, to get green pixel
    uint32_t iframe = 0;
    EXPECT_EQ(Tile::GetPattern3Pixel(ipixel,0),traces[0]); // first zmw, red
    EXPECT_EQ(Tile::GetPattern3Pixel(ipixel,1),traces[1]);
    EXPECT_EQ(Tile::GetPattern3Pixel(ipixel,2),traces[2]);

    std::srand(0);
    for(int i=0;i<100;i++)
    {
        uint32_t irow = std::rand() % roi.NumPixelRows();
        uint32_t row = irow + roi.AbsoluteRowPixelMin();
        uint32_t icol = std::rand() % roi.NumPixelCols();
        uint32_t col = icol + roi.AbsoluteColPixelMin();
        uint32_t pixel = row * roi.SensorROI().PhysicalCols() + col;
        ipixel = irow * roi.NumPixelCols() + icol;
        iframe = std::rand() % (512 * numChunks);
        uint32_t icolor = 1 - ipixel %2;
        uint32_t izmw = ipixel / 2;
        ASSERT_EQ(Tile::GetPattern3Pixel(pixel,iframe),traces[iframe + trace.NFRAMES * (icolor  + izmw * 2)])
         << "#" << i << " ipixel:" << ipixel << " iframe:" << iframe << " icolor:"<<icolor<< " izmw:" <<izmw;
    }
}

TEST_F(SequelTraceFileHDF5_UnitTest,FrameRates)
{
    const std::string traceFile = temps.GenerateTempFileName(".h5");
    const float frameRate = 42.0f;

    ChipClass chipClass = ChipClass::Sequel;
    SequelRectangularROI roi(RowPixels(32), ColPixels(32), RowPixels(128), ColPixels(256), SequelSensorROI::SequelAlpha());
    {
        SequelMovieConfig mc;
        mc.path = traceFile;
        mc.roi.reset(roi.Clone());
        mc.chipClass = chipClass;
        mc.numFrames = 1 * framesPerTile;
        SequelTraceFileHDF5 trace(mc);
        ChipLayoutRTO3 layout;
        auto analogs = CreateSequelAnalogs1();
        trace.SetAnalogs(analogs, layout);
        trace.FrameRate << frameRate;
    }

    {
        SequelTraceFileHDF5 readback(traceFile);
        float rate;
        readback.FrameRate >> rate;
        EXPECT_EQ(frameRate, rate);
        EXPECT_EQ(512,readback.NFRAMES);

        uint32_t numFrames;
        readback.NumFrames >> numFrames;
        EXPECT_EQ(512,numFrames);
        EXPECT_EQ(512,readback.NFRAMES);
    }
}

TEST_F(SequelTraceFileHDF5_UnitTest,DumpSummary)
{
    const int numChunks = 1;

    SequelRectangularROI roi(RowPixels(32), ColPixels(32), RowPixels(128), ColPixels(256), SequelSensorROI::SequelAlpha());
    const std::string traceFile = temps.GenerateTempFileName(".trc.h5");

    ChipLayoutRTO3 layout;
    FillMovieFile(traceFile, roi, numChunks, layout);
    SequelTraceFileHDF5 trace(traceFile);
    // trace.DumpSummary(std::cout);
    {
        stringstream ss;

        ss.precision(100);
        ss << trace.NumAnalog;
        EXPECT_EQ("NumAnalog:\t4 (0x4)\n", ss.str());
    }
    {
        stringstream ss;
        ss.precision(0);
        ss << trace.NumAnalog;
        EXPECT_EQ("4", ss.str());
    }
}

TEST_F(SequelTraceFileHDF5_UnitTest,ReadTileReadTranche)
{
    const int numChunks = 2;

    SequelRectangularROI roi(RowPixels(32), ColPixels(32), RowPixels(128), ColPixels(256), SequelSensorROI::SequelAlpha());
    const std::string traceFile = temps.GenerateTempFileName(".trc.h5");

    ChipLayoutRTO3 layout;
    FillMovieFile(traceFile, roi, numChunks, layout);
    SequelTraceFileHDF5 trace(traceFile);

    std::unique_ptr<Tile> tile(Tile::make());
    int16_t* ptr = reinterpret_cast<int16_t*>(tile.get());

    trace.ReadTile(0,0,tile.get());
    for(int i=0;i<16384;i++) ASSERT_EQ(0x0101,ptr[i]) << "tile i:" << i;

    trace.ReadTile(0,512,tile.get());
    for(int i=0;i<16384;i++) ASSERT_EQ(0x0101,ptr[i]) << "tile i:" << i;

    trace.ReadTile(16,0,tile.get());
    for(int i=0;i<16384;i++) ASSERT_EQ(0x0202,ptr[i]) << "tile i:" << i;
}


TEST_F(SequelTraceFileHDF5_UnitTest,GetROI)
{
    const int numChunks = 0;

    SequelRectangularROI roi(RowPixels(0), ColPixels(0), RowPixels(3), ColPixels(32), SequelSensorROI::SequelAlpha());
    const std::string traceFile = temps.GenerateTempFileName(".trc.h5");

    ChipLayoutRTO3 layout;
    FillMovieFile(traceFile, roi, numChunks, layout);
    SequelTraceFileHDF5 trace(traceFile);
    {
        auto roiT = trace.GetROI();
       // std::cout << *roi;
        EXPECT_EQ(96, roiT->TotalPixels());
        EXPECT_EQ(0, roiT->AbsoluteRowPixelMin());
        EXPECT_EQ(0, roiT->AbsoluteColPixelMin());
        EXPECT_TRUE(roiT->ContainsPixel(2,0));
        EXPECT_FALSE(roiT->ContainsPixel(3,0));
        EXPECT_FALSE(roiT->ContainsPixel(0,32));
    }
}

#ifdef PB_MIC_COPROCESSOR
TEST_F(SequelTraceFileHDF5_UnitTest,DISABLED_MatLabGen)
#else
TEST_F(SequelTraceFileHDF5_UnitTest,MatLabGen)
#endif
{
    //SequelTraceFileHDF5 readback("/pbi/collections/sequelmilestone/3100114/0001/m54006_151015_231014_pochits.trc.h5");
    SequelTraceFileHDF5 readback(TraceFilePath("/dept/primary/testdata/3100114/0001/m54006_151015_231014_pochits.trc.h5"));

    std::string v;
    readback.Version >> v;
    // std::cout << "version:" << v;
    EXPECT_EQ("SequelPOC MovieToTrace 2.2.0",v);
}

TEST_F(SequelTraceFileHDF5_UnitTest,TempNaming)
{
    SequelRectangularROI roi(RowPixels(0), ColPixels(0), RowPixels(3), ColPixels(32), SequelSensorROI::SequelAlpha());

    {
        SequelMovieConfig mc;
        mc.path = "/dev/null";
        mc.roi.reset(roi.Clone());
        mc.chipClass = ChipClass::Sequel;
        mc.numFrames = 100;

        SequelTraceFileHDF5 nulltracefile(mc );
    }

    const std::string traceFile = temps.GenerateTempFileName(".trc.h5");
    {
        SequelMovieConfig mc;
        mc.path = traceFile;
        mc.roi.reset(roi.Clone());
        mc.chipClass = ChipClass::Sequel;
        mc.numFrames = 100;

        SequelTraceFileHDF5 inProgressFile(mc);
        EXPECT_FALSE(PacBio::POSIX::IsFile(traceFile));
        EXPECT_TRUE(PacBio::POSIX::IsFile(traceFile + ".tmp"));
    }
    EXPECT_TRUE(PacBio::POSIX::IsFile(traceFile));

    SequelMovieConfig mc;
    mc.roi.reset(roi.Clone());
    mc.chipClass = ChipClass::Sequel;
    mc.numFrames = 100;
    {
        mc.path = "/tmp";
        EXPECT_THROW(SequelTraceFileHDF5 readback(mc), std::exception);
    }
    {
        mc.path = "/var/log/messages";
        EXPECT_THROW(SequelTraceFileHDF5 readback(mc), std::exception);
    }
}

TEST_F(SequelTraceFileHDF5_UnitTest,ChipInfo)
{
    SequelRectangularROI roi(RowPixels(0), ColPixels(0), RowPixels(3), ColPixels(32), SequelSensorROI::SequelAlpha());

    const std::string traceFile = temps.GenerateTempFileName(".trc.h5");
    {
        ChipLayoutRTO3 layout;
        SequelMovieConfig mc;
        mc.path = traceFile;
        mc.roi.reset(roi.Clone());
        mc.chipClass = layout.GetChipClass();
        mc.numFrames = 100;
        SequelTraceFileHDF5 writer(mc);
        auto analogs = CreateSequelAnalogs1();
        writer.SetAnalogs(analogs, layout);

        std::vector<uint16_t> filterMap({1,0});
        writer.FilterMap << filterMap;

        boost::multi_array<float,3> imagePsf(boost::extents[2][5][5]);
        imagePsf[0][0][0] = 0.1;
        writer.ImagePsf << imagePsf;

        boost::multi_array<float,2> xtalkCorrection(boost::extents[7][7]);
//        xtalkCorrection.at(0,0) = 0.1;
        xtalkCorrection[0][0] = 0.1;
        writer.XtalkCorrection << xtalkCorrection;

        std::vector<float> analogRefSpectrum({0.1,0.2});
        writer.AnalogRefSpectrum << analogRefSpectrum;

        float analogRefSnr;
        analogRefSnr = 0.1;
        writer.AnalogRefSnr << analogRefSnr;
    }
    {
        SequelTraceFileHDF5 readback(traceFile);

        std::vector<uint16_t> filterMap;
        readback.FilterMap >> filterMap;
        EXPECT_EQ(0,filterMap[1]);
        EXPECT_EQ(1,filterMap[0]);

        boost::multi_array<double,3> imagePsf;
        readback.ImagePsf >> imagePsf;
        EXPECT_EQ(2,imagePsf.shape()[0]);
        EXPECT_EQ(5,imagePsf.shape()[1]);
        EXPECT_EQ(5,imagePsf.shape()[2]);
        EXPECT_FLOAT_EQ(0.1,imagePsf[0][0][0]);

        boost::multi_array<double,2> xtalkCorrection;
        readback.XtalkCorrection >> xtalkCorrection;
        EXPECT_EQ(7,xtalkCorrection.shape()[0]);
        EXPECT_EQ(7,xtalkCorrection.shape()[1]);
        EXPECT_FLOAT_EQ(0.1,xtalkCorrection[0][0]);

        std::vector<double> analogRefSpectrum;
        readback.AnalogRefSpectrum >> analogRefSpectrum;
        EXPECT_FLOAT_EQ(0.1,analogRefSpectrum[0] );
        EXPECT_FLOAT_EQ(0.2,analogRefSpectrum[1] );

        float analogRefSnr;
        readback.AnalogRefSnr >> analogRefSnr;
        EXPECT_FLOAT_EQ(0.1,analogRefSnr );


        boost::multi_array<double,2> analogSpectra;
        readback.AnalogSpectra >> analogSpectra;
        EXPECT_EQ(4,analogSpectra.shape()[0]);
        EXPECT_EQ(2,analogSpectra.shape()[1]);

        // analogs are sorted in wavelength order, then decreasing amplitude
        EXPECT_FLOAT_EQ(1.0f, analogSpectra[0][0] );
        EXPECT_FLOAT_EQ(0.0f, analogSpectra[0][1] );
        EXPECT_FLOAT_EQ(1.0f, analogSpectra[1][0] );
        EXPECT_FLOAT_EQ(0.0f, analogSpectra[1][1] );
        EXPECT_FLOAT_EQ(0.1f, analogSpectra[2][0] );
        EXPECT_FLOAT_EQ(0.9f, analogSpectra[2][1] );
        EXPECT_FLOAT_EQ(0.1f, analogSpectra[3][0] );
        EXPECT_FLOAT_EQ(0.9f, analogSpectra[3][1] );

        string basemap;
        readback.BaseMap >> basemap;
        EXPECT_EQ("TGCA",basemap);

        std::vector<float> relativeAmp;
        readback.RelativeAmp >> relativeAmp;
        EXPECT_FLOAT_EQ(0.8,relativeAmp[0]);
        EXPECT_FLOAT_EQ(0.5,relativeAmp[1]);
        EXPECT_FLOAT_EQ(1.0,relativeAmp[2]);
        EXPECT_FLOAT_EQ(0.6,relativeAmp[3]);

        std::vector<float> excessNoiseCV;
        readback.ExcessNoiseCV >> excessNoiseCV;
        EXPECT_FLOAT_EQ(0.01,excessNoiseCV[0]);
        EXPECT_FLOAT_EQ(0.02,excessNoiseCV[1]);
        EXPECT_FLOAT_EQ(0.03,excessNoiseCV[2]);
        EXPECT_FLOAT_EQ(0.04,excessNoiseCV[3]);

        std::vector<float> pwMean;
        readback.PulseWidthMean >> pwMean;
        EXPECT_EQ(5.0f, pwMean[0]);
        EXPECT_EQ(5.5f, pwMean[1]);
        EXPECT_EQ(6.0f, pwMean[2]);
        EXPECT_EQ(6.5f, pwMean[3]);

        std::vector<float> ipdMean;
        readback.IpdMean >> ipdMean;
        EXPECT_EQ(15.0f, ipdMean[0]);
        EXPECT_EQ(16.0f, ipdMean[1]);
        EXPECT_EQ(18.0f, ipdMean[2]);
        EXPECT_EQ(42.0f, ipdMean[3]);

        std::vector<float> pw2SlowStepRatio ;
        readback.Pw2SlowStepRatio >> pw2SlowStepRatio;
        EXPECT_FLOAT_EQ(0.11f, pw2SlowStepRatio[0]);
        EXPECT_FLOAT_EQ(0.12f, pw2SlowStepRatio[1]);
        EXPECT_FLOAT_EQ(0.13f, pw2SlowStepRatio[2]);
        EXPECT_FLOAT_EQ(0.14f, pw2SlowStepRatio[3]);

        std::vector<float> ipd2SlowStepRatio;
        readback.Ipd2SlowStepRatio >> ipd2SlowStepRatio ;
        EXPECT_FLOAT_EQ(0.21f, ipd2SlowStepRatio[0]);
        EXPECT_FLOAT_EQ(0.22f, ipd2SlowStepRatio[1]);
        EXPECT_FLOAT_EQ(0.23f, ipd2SlowStepRatio[2]);
        EXPECT_FLOAT_EQ(0.24f, ipd2SlowStepRatio[3]);

    }
}

TEST_F(SequelTraceFileHDF5_UnitTest,SequelChipLayout)
{
    const int numChunks = 0;

    SequelRectangularROI roi(RowPixels(0), ColPixels(0), RowPixels(3), ColPixels(32), SequelSensorROI::SequelAlpha());
    const std::string traceFile = temps.GenerateTempFileName(".trc.h5");

    ChipLayoutRTO3 layout;
    FillMovieFile(traceFile, roi, numChunks, layout);
    SequelTraceFileHDF5 trace(traceFile);
    {
        auto& layoutT = trace.GetChipLayout();
        EXPECT_EQ("SequEL_4.0_RTO3",layoutT.Name());
    }
}

TEST_F(SequelTraceFileHDF5_UnitTest,SpiderChipLayout)
{
    const int numChunks = 0;

    SequelRectangularROI roi(RowPixels(0), ColPixels(0), RowPixels(3), ColPixels(32), SequelSensorROI::Spider());
    const std::string traceFile = temps.GenerateTempFileName(".trc.h5");

    ChipLayoutSpider1p0NTO layout;
    FillMovieFile(traceFile, roi, numChunks, layout);
    SequelTraceFileHDF5 trace(traceFile);
    {
        auto& layoutT = trace.GetChipLayout();
        EXPECT_EQ("Spider_1p0_NTO",layoutT.Name());
    }
}

void VerifyFrame3(uint32_t iframe, SequelMovieFrame<int16_t>& frame, SequelTraceFileHDF5& trace, SequelRectangularROI& roi)
{
    frame.SetDefaultValue(0);
    trace.ReadFrame(iframe,frame);
    ASSERT_EQ(roi.SensorROI().PhysicalRows(),frame.NROW);
    ASSERT_EQ(roi.SensorROI().PhysicalCols(),frame.NCOL);

    uint32_t row = 0;
    uint32_t col = 0;
    EXPECT_EQ(Tile::GetPattern3Pixel(row*2048+col,iframe), frame.GetPixel(row,col))
                        << "Frame:" << iframe << " row,col=" << row << "," << col;

    row = 0;
    col = 1;
    EXPECT_EQ(Tile::GetPattern3Pixel(row*2048+col,iframe), frame.GetPixel(row,col))
                        << "Frame:" << iframe << " row,col=" << row << "," << col;

    row = 0;
    col = 2;
    EXPECT_EQ(Tile::GetPattern3Pixel(row*2048+col,iframe), frame.GetPixel(row,col))
                << "Frame:" << iframe << " row,col=" << row << "," << col;

    row = 1;
    col = 0;
    EXPECT_EQ(Tile::GetPattern3Pixel(row*2048+col,iframe), frame.GetPixel(row,col))
                        << "Frame:" << iframe << " row,col=" << row << "," << col;

    row = 1;
    col = 1;
    EXPECT_EQ(Tile::GetPattern3Pixel(row*2048+col,iframe), frame.GetPixel(row,col))
                << "Frame:" << iframe << " row,col=" << row << "," << col;

    for(int i=0;i<100;i++)
    {
        uint32_t rowT = rand() % roi.NumPixelRows() + roi.AbsoluteRowPixelMin();
        uint32_t colT = rand() % roi.NumPixelCols() + roi.AbsoluteColPixelMin();
        ASSERT_EQ(Tile::GetPattern3Pixel(rowT*2048+colT,iframe), frame.GetPixel(rowT,colT))
                    << "Frame:" << iframe << " point#" << i << " row,col=" << rowT << "," << colT;
    }
}

TEST_F(SequelTraceFileHDF5_UnitTest,ReadFrame)
{
    const int numChunks = 2;

    SequelRectangularROI roi(RowPixels(0), ColPixels(0), RowPixels(3), ColPixels(32), SequelSensorROI::SequelAlpha());
    const std::string traceFile = temps.GenerateTempFileName(".trc.h5");

    ChipLayoutRTO3 layout;
    FillMovieFile(traceFile, roi, numChunks, layout, 3);
    std::srand(0);
    SequelTraceFileHDF5 trace(traceFile);
    {
        ASSERT_EQ(SequelSensorROI::SequelAlpha().PhysicalCols(), trace.GetChipLayout().GetSensorROI().PhysicalCols());
        ASSERT_EQ(SequelSensorROI::SequelAlpha().PhysicalRows(), trace.GetChipLayout().GetSensorROI().PhysicalRows());
        SequelMovieFrame<int16_t> frame;
        frame.Resize(SequelSensorROI::SequelAlpha().PhysicalRows(), SequelSensorROI::SequelAlpha().PhysicalCols());

        VerifyFrame3(0,frame, trace,roi);
        VerifyFrame3(1,frame, trace,roi);
        VerifyFrame3(100,frame, trace,roi);
        VerifyFrame3(511,frame, trace,roi);
        VerifyFrame3(512,frame, trace,roi);
        VerifyFrame3(612,frame, trace,roi);
        VerifyFrame3(1023,frame, trace,roi);
    }
}

#ifdef PB_MIC_COPROCESSOR
TEST_F(SequelTraceFileHDF5_UnitTest,DISABLED_ReadFrame2)
#else
TEST_F(SequelTraceFileHDF5_UnitTest,ReadFrame2)
#endif
{
    PacBio::Logging::LogSeverityContext context(PacBio::Logging::LogLevel::INFO);

    std::string traceFile = TraceFilePath("/dept/secondary/siv/mdsmith/Analyses/matlabSim/designer64800LcleanT1_SNR-100.trc.h5");

    SequelTraceFileHDF5 trace(traceFile);

    const auto& roi = trace.GetROI();

    SequelMovieFrame<int16_t> frame;
    trace.ReadFrame(0,frame);

    uint32_t rowOffset = roi->AbsoluteRowPixelMin();
    uint32_t colOffset = roi->AbsoluteColPixelMin();

    UnitX unitX = trace.GetChipLayout().ConvertAbsoluteRowPixelToUnitCellX(0);
    EXPECT_EQ(32,unitX.Value());
    EXPECT_EQ(32,rowOffset);
    EXPECT_EQ(64,colOffset);
    EXPECT_EQ((int16_t)462,frame.GetPixel(32,64));
    EXPECT_EQ((int16_t)251,frame.GetPixel(32,65));
    EXPECT_EQ((int16_t)517,frame.GetPixel(32,66));
    EXPECT_EQ((int16_t)271,frame.GetPixel(32,67));

    trace.ReadFrame(1,frame);
    EXPECT_EQ((int16_t)500,frame.GetPixel(32,64));
    EXPECT_EQ((int16_t)272,frame.GetPixel(32,65));

    trace.ReadFrame(415,frame);
    EXPECT_EQ((int16_t)193,frame.GetPixel(32,64));
    EXPECT_EQ((int16_t)200,frame.GetPixel(32,65));
    EXPECT_EQ((int16_t)201,frame.GetPixel(32,66));
    EXPECT_EQ((int16_t)200,frame.GetPixel(32,67));
    EXPECT_EQ((int16_t)207,frame.GetPixel(32 + 1079,64 + 1918));
    EXPECT_EQ((int16_t)187,frame.GetPixel(32 + 1079,65 + 1918));

}

#ifdef PB_MIC_COPROCESSOR
TEST_F(SequelTraceFileHDF5_UnitTest,DISABLED_ReadFrameCondensed)
#else
TEST_F(SequelTraceFileHDF5_UnitTest,ReadFrameCondensed)
#endif
{
    PacBio::Logging::LogSeverityContext context(PacBio::Logging::LogLevel::INFO);

    std::string traceFile = TraceFilePath("/dept/primary/sim/ffhmm/ffHmm_256Lx3C_102715_SNR-15.trc.h5");

    SequelTraceFileHDF5 trace(traceFile);

    const auto& roi = trace.GetROI();
    uint32_t rowOffset = roi->AbsoluteRowPixelMin();
    uint32_t colOffset = roi->AbsoluteColPixelMin();
    EXPECT_EQ(135,rowOffset);
    EXPECT_EQ(288,colOffset);

    SequelMovieFrame<int16_t> frame(64,128); // 4096 ZMWs
    trace.ReadFrame(0,frame, SequelTraceFileHDF5::ReadFrame_Condensed);

    EXPECT_EQ((int16_t)203,frame.GetPixel(0,0));
    EXPECT_EQ((int16_t)203,frame.GetPixel(0,1));
    EXPECT_EQ((int16_t)207,frame.GetPixel(0,2));
    EXPECT_EQ((int16_t)207,frame.GetPixel(0,3));
    EXPECT_EQ((int16_t)198,frame.GetPixel(0,4));
    EXPECT_EQ((int16_t)205,frame.GetPixel(0,5));
    EXPECT_EQ((int16_t)204,frame.GetPixel(0,126));
    EXPECT_EQ((int16_t)197,frame.GetPixel(0,127));
    EXPECT_EQ((int16_t)205,frame.GetPixel(1,0));
    EXPECT_EQ((int16_t)208,frame.GetPixel(1,1));
    EXPECT_EQ((int16_t)206,frame.GetPixel(1,2));
    EXPECT_EQ((int16_t)208,frame.GetPixel(1,3));

    trace.ReadFrame(1,frame, SequelTraceFileHDF5::ReadFrame_Condensed);
    EXPECT_EQ((int16_t)236,frame.GetPixel(0,0));
    EXPECT_EQ((int16_t)205,frame.GetPixel(0,1));

    trace.ReadFrame(511,frame, SequelTraceFileHDF5::ReadFrame_Condensed);
    EXPECT_EQ((int16_t)198,frame.GetPixel(0,0));
    EXPECT_EQ((int16_t)206,frame.GetPixel(0,1));
    trace.ReadFrame(512,frame, SequelTraceFileHDF5::ReadFrame_Condensed);
    EXPECT_EQ((int16_t)206,frame.GetPixel(0,0));
    EXPECT_EQ((int16_t)204,frame.GetPixel(0,1));

}

#if 0
TEST(SequelTraceFileHDF5,Lane216)
{
    std::string srcFilename = "/pbi/collections/313/3130023/r54008_20160424_152422/m54008_160424_230725.trc.h5";
    std::string dstFilename = TraceFilePath("/dept/primary/collections/mtl/m54008_160424_230725_lane216x1024.trc.h5");

    size_t numFrames;
    std::vector<std::pair<int16_t, int16_t> > traceData;
    {
        SequelTraceFileHDF5 trace(srcFilename);
        numFrames = trace.NFRAMES;
        traceData.resize(numFrames);

        uint32_t laneOffset  =  216;
        uint32_t zmwOffset   = laneOffset*16 + 4;
        uint64_t frameOffset = 0;
        trace.ReadTrace(zmwOffset, frameOffset, traceData);
    }

    {
        size_t numChunks = numFrames / framesPerTile;
        uint32_t numLanes = 1024;
        SequelRectangularROI roi(0, 0, numLanes, 32, SequelSensorROI::SequelAlpha()); // one tile
        POSIX::Unlink(dstFilename.c_str());

        ChipLayoutRTO3 layout;

        const size_t TilesPerChunk = framesPerTile * maxPixelRows * maxPixelCols * sizeof(uint16_t) / sizeof(Tile);
        SequelTraceFileHDF5 trace(dstFilename, roi, numChunks * framesPerTile);

        AnalogSet analogs = CreateAnalogs();
        trace.SetAnalogs(analogs, layout);

        trace.DetermineAntiZMWs(roi, layout);

        EXPECT_EQ(trace.TypeName(), "Trace");
        trace.PrepareChunkBuffers();
        trace.RI_PlatformId << SequelTraceFileHDF5::PlatformId_SequelAlpha4;
        trace.SoftwareVersion << "1.2.3";
        trace.AcquisitionXML << "<myxml>hello _ underscore</myxml>";
        trace.RI_InstrumentName << GtestCurrentTestName();
        trace.RI_PlatformName << "SequelAlpha";
        trace.LayoutName << layout.Name();

        trace.AduGain << defaultPhotoelectronSensitivity;
        trace.CameraGain << 1.0;
        trace.FilterMap << layout.FilterMap();
        //traceWriter->ImagePsf << imagePsf;
        //traceWriter->XtalkCorrection << xtalkCorrection;

        // ref spectrum and SNR
        const std::vector<float> analogRefSpectrum{{ 0.06f, 0.94f }};
        float analogRefSnr = 11.0f;
        trace.AnalogRefSpectrum << analogRefSpectrum;
        trace.AnalogRefSnr << analogRefSnr;

        uint64_t frameOffset = 0;

        for (uint32_t zmwOffset = 0; zmwOffset < 16 * numLanes; zmwOffset++)
        {
            trace.WriteTrace(zmwOffset, frameOffset, traceData);
        }


#if 0
int numChannels = setup.analogs[0].NumFilters;
boost::multi_array<float, 3> imagePsf(boost::extents[numChannels][5][5]);
boost::multi_array<double, 2> xtalkCorrection(boost::extents[7][7]);
{
// set up the PSF multi_array
const int numPsfChannels = setup.psfs.size();
PBLOG_INFO << "Copying " << numPsfChannels << " PSFs";
if (numPsfChannels != numChannels)
{
throw PBException("Wrong number of PSFs specified");
}
for (int j = 0; j < numChannels; j++)
{
imagePsf[j] = setup.psfs[j].AsMultiArray();
}
}

{
// set up the crosstalk multi_array. May need to center it.
int32_t rowShift = (xtalkCorrection.shape()[0] - setup.crosstalkFilter.NumRows()) / 2;
int32_t colShift = (xtalkCorrection.shape()[1] - setup.crosstalkFilter.NumCols()) / 2;
if (rowShift < 0 || colShift < 0)
throw PBException("crosstalk filter is too large to be stored in trace file");

for (int row = 0; row < setup.crosstalkFilter.NumRows(); row++)
{
for (int col = 0; col < setup.crosstalkFilter.NumCols(); col++)
{
xtalkCorrection[row + rowShift][col + colShift] = setup.crosstalkFilter(row, col);
}
}
}
#endif


    }
    {
        SequelTraceFileHDF5 trace(dstFilename);
        EXPECT_EQ(numFrames, trace.NFRAMES);

        std::vector<std::pair<int16_t, int16_t> > traceReadData;
        trace.ReadTrace(0, 0, traceReadData);
    }
}

#endif

#if 0
TEST(SequelTraceFileHDF5,GenerateNoise)
{
    std::string filename = "/pbi/dept/primary/collections/mtl/m54008_160424_230725_noise.trc.h5";
    size_t numFrames = 102400;
    {
        size_t numChunks = numFrames / NumFramesPerTile;
        SequelRectangularROI roi(0, 0, 1, 32, SequelSensorROI::SequelAlpha()); // one tile
        POSIX::Unlink(filename.c_str());

        ChipLayoutRTO3 layout;

        const size_t TilesPerChunk = NumFramesPerTile * maxPixelRows * maxPixelCols * sizeof(uint16_t) / sizeof(Tile);
        SequelTraceFileHDF5 trace(filename, roi, numChunks * NumFramesPerTile);

        AnalogSet analogs = CreateAnalogs();
        trace.SetAnalogs(analogs, layout);

        trace.DetermineAntiZMWs(roi, layout);

        EXPECT_EQ(trace.TypeName(), "Trace");
        trace.PrepareChunkBuffers();
        trace.RI_PlatformId << SequelTraceFileHDF5::PlatformId_SequelAlpha4;
        trace.SoftwareVersion << "1.2.3";
        trace.AcquisitionXML << "<myxml>hello _ underscore</myxml>";
        trace.RI_InstrumentName << GtestCurrentTestName();
        trace.RI_PlatformName << "SequelAlpha";
        trace.LayoutName << layout.Name();


        uint64_t frameOffset = 0;
        std::vector<std::pair<int16_t, int16_t> > traceData(numFrames);

        ifstream csv("/home/UNIXHOME/mlakata/bug32584.txt");
        int i = 0;
        while (csv)
        {
            csv >> traceData[i].first >> traceData[i].second;
            i++;
        }

        for (uint32_t zmwOffset = 0; zmwOffset < 16; zmwOffset++)
        {
            trace.WriteTrace(zmwOffset, frameOffset, traceData);
        }
    }
    {
        SequelTraceFileHDF5 trace(filename);
        EXPECT_EQ(numFrames, trace.NFRAMES);
    }
}

#endif

TEST_F(SequelTraceFileHDF5_UnitTest,CompareTraces1)
{
    const int numChunks = 1;

    SequelRectangularROI roi(RowPixels(0),ColPixels(0),RowPixels(128),ColPixels(256), SequelSensorROI::SequelAlpha());
    ASSERT_EQ(128*128,roi.CountHoles());
    const std::string traceFileA = temps.GenerateTempFileName("_a.trc.h5");
    const std::string traceFileB = temps.GenerateTempFileName("_b.trc.h5");

    // same as roi, but missing first row
    SequelRectangularROI roiC(RowPixels(1),ColPixels(0),RowPixels(127),ColPixels(256), SequelSensorROI::SequelAlpha());
    const std::string traceFileC = temps.GenerateTempFileName("_c.trc.h5");

#if 0
    PacBio::Logging::LogSeverityContext _(PacBio::Logging::LogLevel::WARN);
#endif

    ChipLayoutRTO3 layout;

    FillMovieFile(traceFileA, roi,  numChunks, layout);
    FillMovieFile(traceFileB, roi,  numChunks, layout);
    FillMovieFile(traceFileC, roiC, numChunks, layout);

    {
        ifstream traceFileExistsA(traceFileA);
        EXPECT_TRUE(static_cast<bool>(traceFileExistsA));
        ifstream traceFileExistsB(traceFileB);
        EXPECT_TRUE(static_cast<bool>(traceFileExistsB));
        ifstream traceFileExistsC(traceFileC);
        EXPECT_TRUE(static_cast<bool>(traceFileExistsC));
    }

    {
        SequelTraceFileHDF5 traceA(traceFileA);
        SequelTraceFileHDF5 traceB(traceFileB);
        SequelTraceFileHDF5 traceC(traceFileC);

        // should be identical, no errors
        EXPECT_EQ(0,traceA.CompareTraces(traceB));

        // should not be identical, different number of ZMWs
        EXPECT_EQ(1,traceA.CompareTraces(traceC,false));

        // C is a subset of A, so it should have no errors with subsetFlag=true option
        EXPECT_EQ(0,traceC.CompareTraces(traceA,true));

        // A is a superset of C, so it should have errors with subsetFlag=true option
        EXPECT_LE(128,traceA.CompareTraces(traceC,true));
    }
    // temps.Keep();
}

TEST_F(SequelTraceFileHDF5_UnitTest,EventObject)
{
    const std::string traceFileName = temps.GenerateTempFileName(".h5");
    {
        // create file
        uint64_t nFrames = 512;
        SequelRectangularROI roi(0,0,1144,2048,SequelSensorROI::SequelAlpha());

        SequelMovieConfig mc;
        mc.path = traceFileName;
        mc.roi.reset(roi.Clone());
        mc.chipClass = ChipClass::Sequel;
        mc.numFrames = nFrames;

        SequelTraceFileHDF5 traceFile(mc);
        AnalogSet analogs = CreateSequelAnalogs1();
        ChipLayoutRTO3 layout;
        traceFile.SetAnalogs(analogs, layout);


        EventObject eo;
        eo.eventType = EventObject::EventType::laserpower;
        eo.timeStamp = "2018-11-11T12:00:00Z";
        eo.startFrame = 123;
        eo.stopFrame = 456;
        eo.lasers[0].name = LaserPowerObject::LaserName::bottomLaser;
        eo.lasers[0].startPower_mW = 123.0;
        eo.lasers[0].stopPower_mW = 145.0;
        // leave out laser[1] for now
        traceFile.events.laserPowerChanges.SetFirstFrame(100);
        traceFile.events.laserPowerChanges.AddEvent(eo);
    }

    {
        //verify file
        SequelTraceFileHDF5 traceFile(traceFileName);

        ASSERT_EQ(1,traceFile.events.laserPowerChanges.nLPC);
        auto events = traceFile.events.laserPowerChanges.ReadAllEvents();
        ASSERT_EQ(1, events.size());
        EXPECT_EQ(EventObject::EventType::laserpower, events[0].eventType());
        EXPECT_EQ("2018-11-11T12:00:00Z", events[0].timeStamp());
        EXPECT_EQ(23, events[0].startFrame());
        EXPECT_EQ(356, events[0].stopFrame());
        ASSERT_EQ(2, events[0].lasers.size());
        EXPECT_FLOAT_EQ(-1.0, events[0].lasers[0].startPower_mW());
        EXPECT_FLOAT_EQ(-1.0, events[0].lasers[0].stopPower_mW());
        EXPECT_FLOAT_EQ(123.0, events[0].lasers[1].startPower_mW());
        EXPECT_FLOAT_EQ(145.0, events[0].lasers[1].stopPower_mW());
    }

//    temps.Keep();
}
