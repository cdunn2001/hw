//
// Created by mlakata on 4/7/15.
//

#ifndef PA_ACQUISITION_TESTSEQUELTRACEFILEHDF5_H
#define PA_ACQUISITION_TESTSEQUELTRACEFILEHDF5_H

#include <cmath>
#include <array>
#include <pacbio/POSIX.h>
#include <pacbio/logging/Logger.h>
#include <pacbio/primary/ChipLayoutRTO3.h>
#include <pacbio/primary/AnalogMode.h>
#include <pacbio/primary/ChipClass.h>

using namespace PacBio::Primary; // this is usually discouraged in header files, but this header file is for testing only

AnalogSet CreateSequelAnalogs1();
AnalogSet CreateSpiderAnalogs1();

class SequelTraceFileHDF5_BaseTest : public ::testing::Test
{
public:
    TempFileManager temps; //("/data/pa","tracetemp");
    PacBio::Logging::LogSeverityContext loggerContext;
    SequelTraceFileHDF5_BaseTest(const std::string& root) : temps(root,"trace"), loggerContext(PacBio::Logging::LogLevel::FATAL)
    {

    }

    virtual ~SequelTraceFileHDF5_BaseTest()
    {
        if (HasFailure()) temps.Keep();
    }

    SequelMovieConfig::TraceChunking chunking;
    int gzipCompression = 0;

    void Run32KTest(const int numChunks );

    void FillMovieFile(const std::string filename, SequelROI& roi, int numChunks, ChipLayout& layout, int pattern=1)
    {
        PacBio::POSIX::Unlink(filename.c_str());

        QuietAutoTimer x( Tile::NumFrames );
        ChipClass chipClass = layout.GetChipClass();

        //const size_t TilesPerChunk = roi.TotalPixels()/32; 
        const size_t TilesPerChunk = framesPerTile * Sequel::maxPixelRows * Sequel::maxPixelCols * sizeof(uint16_t) / sizeof(Tile);
//        TEST_COUT << "TilesPerChunk:" << TilesPerChunk << std::endl;
        {
//            AutoTimer _("File Open", numChunks * TilesPerChunk, "tiles");
            {
                SequelMovieConfig mc;
                mc.path = filename;
                mc.roi.reset(roi.Clone());
                mc.chipClass = chipClass;
                mc.numFrames = numChunks * framesPerTile;
                mc.trace.chunking = chunking;
                mc.trace.compression = gzipCompression;
                SequelTraceFileHDF5 trace(mc);

                switch(chipClass)
                {
                case ChipClass::Sequel:
                    {
                        AnalogSet analogs = CreateSequelAnalogs1();
                        trace.SetAnalogs(analogs, layout);
                    }
                    trace.RI_PlatformId << SequelTraceFileHDF5::PlatformId_SequelAlpha4;
                    trace.RI_PlatformName << "SequelAlpha";
                    break;
                case ChipClass::Spider:
                    {
                        AnalogSet analogs = CreateSpiderAnalogs1();
                        trace.SetAnalogs(analogs, layout);
                    }
                    trace.RI_PlatformId << SequelTraceFileHDF5::PlatformId_Spider5;
                    trace.RI_PlatformName << "Spider";
                    break;
                default:
                    throw PBException("Not supported!");
                }
                trace.DetermineAntiZMWs(roi, layout);

                EXPECT_EQ(trace.TypeName(), "Trace");
                trace.PrepareChunkBuffers();

                trace.LayoutName << layout.Name();
                trace.SoftwareVersion << "1.2.3";
                trace.AcquisitionXML << "<myxml>hello _ underscore</myxml>";
                trace.RI_InstrumentName << GtestCurrentTestName();

                for (int chunks = 0; chunks < numChunks; chunks++)
                {
                    QuietAutoTimer y( Tile::NumFrames  );
                    {
//                        AutoTimer _("memmove", TilesPerChunk, "tiles");
                        std::unique_ptr<Tile> header(Tile::makeHeader());
                        trace.AddChunk(header.get(),Tile::NumFrames);

                        std::unique_ptr<Tile> tile(Tile::make());
                        uint32_t tileOffset = 0;
                        for (uint32_t i = 0; i < TilesPerChunk; i++)
                        {
                            if (pattern == 1) tile->SetPattern(i);
                            else if (pattern == 2) tile->SetPattern2(i);
                            else if (pattern == 3) tile->SetPattern3(i*Tile::NumPixels, chunks * Tile::NumFrames);
                            else throw PBException("not supported");
                            trace.AddTile(tile.get(), tileOffset++);
                        }
                    }
                    {
//                        AutoTimer _("fileIO", TilesPerChunk, "tiles");
                        trace.FlushChunk();
                    }
#if 0
                    if (numChunks > 1)
                    {
                        double rateNet    = x.GetRate() * chunks ;
                        double rateCurrent= y.GetRate();
                        PRINTF("Chunk %d (of %d) Frame Rate: %.1f f/s (net: %.1f) to %s\n", chunks , numChunks ,  rateCurrent , rateNet ,filename.c_str());
                    }
#endif
                }
            }
        }
    }
};

class SequelTraceFileHDF5_FunctionalTest : public SequelTraceFileHDF5_BaseTest
{
public:
    SequelTraceFileHDF5_FunctionalTest() : SequelTraceFileHDF5_BaseTest("/data/pa") {}
};

class SequelTraceFileHDF5_UnitTest : public SequelTraceFileHDF5_BaseTest
{
public:
    SequelTraceFileHDF5_UnitTest() : SequelTraceFileHDF5_BaseTest(".") {}
};

#endif //PA_ACQUISITION_TESTSEQUELTRACEFILEHDF5_H
