#include <stdexcept>
#include <gtest/gtest.h>

#include <pacbio/dev/AutoTimer.h>
#include <pacbio/dev/gtest-extras.h>
#include <pacbio/logging/Logger.h>

#include <pacbio/primary/Tile.h>
#include <pacbio/primary/EventObject.h>
#include <pacbio/primary/SequelMovieFileHDF5.h>
#include <pacbio/primary/SequelMovieEventsHDF5.h>

using namespace std;
using namespace PacBio::Primary;
using namespace PacBio::Dev;


SequelRectangularROI roi(0,0,1144,2048,SequelSensorROI::SequelAlpha());

class SequelMovieFileHDF5_Test : public ::testing::Test
{
public:
    TempFileManager tempFileManager;
    PacBio::Logging::LogSeverityContext loggerContext;

    SequelMovieFileHDF5_Test()
            : loggerContext(PacBio::Logging::LogLevel::FATAL)
    {
    }

    virtual ~SequelMovieFileHDF5_Test()
    {
        if (HasFailure()) tempFileManager.Keep();
    }
};

static std::vector<ChipClass> chipClasses = {ChipClass::Sequel, ChipClass::Spider};

TEST_F(SequelMovieFileHDF5_Test,Frames)
{
    for(auto chipClass : chipClasses)
    {
        const std::string uncompressedMovieFile = tempFileManager.GenerateTempFileName(".h5");
        const std::string compressedMovieFile   = tempFileManager.GenerateTempFileName(".h5");

        SequelMovieConfig mc;
        mc.path = uncompressedMovieFile;
        mc.roi.reset(roi.Clone());
        mc.chipClass = chipClass;
        {
            size_t uncompressedSize = 0;
            size_t compressedSize = 0;
            {
//            AutoTimer _("File Open", 1, "?");
                mc.path = uncompressedMovieFile;
                mc.movie.compression = 0;
                SequelMovieFileHDF5 movie(mc);
                EXPECT_EQ(movie.TypeName(), "Frame");

                SequelMovieFrame<int16_t> frame(roi);
                frame.SetPattern1(10);
                frame.index = 123;
                frame.cameraConfiguration = 0xDEADBEEF;
                movie.AddFrame(frame);
            }
            {
                ifstream test(uncompressedMovieFile);
                EXPECT_TRUE(static_cast<bool>(test));
                uncompressedSize = PacBio::Utilities::Stat::FileSize(uncompressedMovieFile);
            }
            {
//            AutoTimer _("File Open", 1, "?");
                mc.path = compressedMovieFile;
                mc.movie.compression = 3;
                SequelMovieFileHDF5 movie(mc);
                EXPECT_EQ(movie.TypeName(), "Frame");

                SequelMovieFrame<int16_t> frame(roi);
                frame.SetPattern1(10);
                frame.index = 123;
                frame.cameraConfiguration = 0xDEADBEEF;
                movie.AddFrame(frame);
            }
            {
                ifstream test(compressedMovieFile);
                EXPECT_TRUE(static_cast<bool>(test));
                compressedSize = PacBio::Utilities::Stat::FileSize(compressedMovieFile);
                EXPECT_LT(compressedSize,uncompressedSize)
                    << uncompressedMovieFile << " " << compressedMovieFile;
                TEST_COUT << chipClass.toString() << " compression ratio:"
                    << compressedSize * 100.0 / uncompressedSize << "%\n";
            }
            {
                SequelMovieFileHDF5 movie(compressedMovieFile);

                std::string s;
                movie.FormatVersion >> s;
                EXPECT_EQ(s, "1.0");
                EXPECT_EQ(movie.NFRAMES, 1);

                std::stringstream ss;
                SequelMovieFrame<int16_t> frame(roi);
                movie.ReadFrame(0, frame);

                EXPECT_NO_THROW(frame.ValidatePattern1(10,&ss));
                EXPECT_EQ(frame.index, 123);
                EXPECT_EQ(frame.cameraConfiguration , 0xDEADBEEF);

                // test resizing. Here, default resizing (frame has no size)
                SequelMovieFrame<int16_t> smallFrame1;
                movie.ReadFrame(0,smallFrame1);
                EXPECT_EQ(1144,smallFrame1.NROW);
                EXPECT_EQ(2048,smallFrame1.NCOL);
                EXPECT_NO_THROW(smallFrame1.ValidatePattern1(10,&ss));
                EXPECT_EQ(smallFrame1.index, 123);
                EXPECT_EQ(smallFrame1.cameraConfiguration , 0xDEADBEEF);

                SequelMovieFrame<int16_t> smallFrame2(5,5);
                movie.ReadFrame(0,smallFrame2,SequelMovieFileHDF5::ReadFrame_DontResize);
                EXPECT_EQ(5,smallFrame2.NROW);
                EXPECT_EQ(5,smallFrame2.NCOL);
                EXPECT_EQ(smallFrame2.index, 123);
                EXPECT_EQ(smallFrame2.cameraConfiguration , 0xDEADBEEF);
                EXPECT_EQ(smallFrame1.GetPixel(0,0), smallFrame2.GetPixel(0,0));
                EXPECT_EQ(smallFrame1.GetPixel(0,1), smallFrame2.GetPixel(0,1));
                EXPECT_EQ(smallFrame1.GetPixel(1,0), smallFrame2.GetPixel(1,0));
                EXPECT_EQ(smallFrame1.GetPixel(4,4), smallFrame2.GetPixel(4,4));

                SequelMovieFrame<int16_t> smallFrame3(5,5);
                EXPECT_THROW(movie.ReadFrame(0,smallFrame3), std::exception);
            }
        }
    }
}

void FillMovie(SequelMovieFileBase& movie, int numChunks, uint32_t numFrames = Tile::NumFrames)
{
    for(int chunk=0;chunk<numChunks;chunk++)
    {
        PBLOG_DEBUG << "FillMovie, chunks=" << chunk << " of " << numChunks << ",numFrames:" << numFrames;

        std::unique_ptr<Tile> header(Tile::makeHeader());
        PBLOG_DEBUG << "td::unique_ptr<Tile> header(Tile::makeHeader(), numFrames:" << numFrames;
        //header->NumFramesInHeader(numFrames);
        PBLOG_DEBUG << "header->NumFramesInHeader, header.get():" << (void*) header.get() << ", numFrames:" << numFrames;
        numFrames  = movie.AddChunk(header.get(), numFrames);

        PBLOG_DEBUG << "AddChunk returned numFrames:" << numFrames;

        uint32_t tileOffset = 0;
        std::unique_ptr<Tile> tile(Tile::make());
        for (uint32_t i = 0; i < movie.ChunkBufferNumTiles(); i++)
        {
            tile->SetPattern(i + chunk);
            movie.AddTile(tile.get(), tileOffset++);
            //cout  << "Adding chunk:tile " << chunks << ":" << i << endl;
        }
        PBLOG_DEBUG << "will call FlushChunk with numFrames:" << numFrames;

        movie.FlushChunk(numFrames);
    }
}

TEST_F(SequelMovieFileHDF5_Test,Chunks)
{
    loggerContext.ChangeSeverity(PacBio::Logging::LogLevel::WARN);

    const int numChunks = 1;
    const std::string movieFile = tempFileManager.GenerateTempFileName(".h5");
    SequelMovieConfig mc;
    mc.path = movieFile;
    mc.roi.reset(roi.Clone());
    mc.chipClass = ChipClass::Sequel;

    {
        {
//            AutoTimer _("File Open", 1, "?");
            SequelMovieFileHDF5 movie(mc);
            movie.PrepareChunkBuffers();
            FillMovie(movie,numChunks);
        }
        {
            SequelMovieFileHDF5 movie(movieFile);

            std::string s;
            movie.FormatVersion >> s;
            EXPECT_EQ(s, "1.0");
            EXPECT_EQ(movie.NFRAMES, framesPerTile * numChunks);
            movie.LimitFileFrames(23);
            EXPECT_EQ(23,movie.LimitedNumFrames()); // actually limited
            movie.LimitFileFrames(10000000000ULL);
            EXPECT_EQ(movie.NFRAMES,movie.LimitedNumFrames()); // effectively not limited

            SequelMovieFrame<int16_t> frame(roi);
            movie.ReadFrame(0,frame);
            EXPECT_EQ(0,frame.data[0]);
            EXPECT_EQ(0,frame.data[31]);
            EXPECT_EQ(0x0101,frame.data[32]);
            EXPECT_EQ(0x0101,frame.data[63]);
            EXPECT_EQ(0x0202,frame.data[64]);
            EXPECT_EQ((int16_t)0xFFFF, frame.data[roi.TotalPixels()-1]);

            movie.ReadFrame(1,frame);
            EXPECT_EQ(0,frame.data[0]);
            EXPECT_EQ(0,frame.data[31]);
            EXPECT_EQ(0x0101,frame.data[32]);
            EXPECT_EQ((int16_t)0xFFFF, frame.data[roi.TotalPixels()-1]);

            movie.ReadFrame(511,frame);
            EXPECT_EQ(0,frame.data[0]);
            EXPECT_EQ(0,frame.data[31]);
            EXPECT_EQ(0x0101,frame.data[32]);
            EXPECT_EQ((int16_t)0xFFFF, frame.data[roi.TotalPixels()-1]);
        }
    }
}

TEST_F(SequelMovieFileHDF5_Test,ROI)
{
    const int numChunks = 1;
    const int rows = 1144;
    const int cols = 2048;
    const size_t TilesPerChunk = framesPerTile * rows * cols * sizeof(uint16_t) / sizeof(Tile);

    for (auto chipClass : chipClasses)
    {
        const std::string movieFile = tempFileManager.GenerateTempFileName(".h5");



//        SequelRectangularROI roi = SequelROI::SequelAlphaFull();
        SequelRectangularROI testRoi(RowPixels(0), ColPixels(0), RowPixels(32), ColPixels(32),
                                     SequelSensorROI::SequelAlpha());

        SequelMovieConfig mc;
        mc.path = movieFile;
        mc.roi.reset(testRoi.Clone());
        mc.chipClass = chipClass;

        {
//            AutoTimer _("File Open", 1, "?");
            SequelMovieFileHDF5 movie(mc);
            movie.PrepareChunkBuffers();

            for (int chunks = 0; chunks < numChunks; chunks++)
            {
                {
//                    AutoTimer _("memmove", TilesPerChunk, "tiles");
                    std::unique_ptr<Tile> header(Tile::makeHeader());
                    header->CreateHeaderTile(100, 0, 0);
                    header->data[1024] = 5; // flag 2 errors at offset 0 and 2
                    uint32_t numFrames = movie.AddChunk(header.get(), Tile::NumFrames);

                    uint32_t tileOffset = 0;
                    std::unique_ptr<Tile> tile(Tile::make());
                    for (uint32_t i = 0; i < TilesPerChunk; i++)
                    {
                        tile->SetPattern(i);
                        movie.AddTile(tile.get(), tileOffset++);
                        //cout  << "Adding chunk:tile " << chunks << ":" << i << endl;
                    }
                    //                  AutoTimer _("fileIO", TilesPerChunk, "tiles");
                    movie.FlushChunk(numFrames);
                }
            }
        }
        {
            SequelMovieFileHDF5 movie(movieFile);

            std::string s;
            movie.FormatVersion >> s;
            EXPECT_EQ(s, "1.0");
            EXPECT_EQ(testRoi.NumPixelRows(), movie.NROW);
            EXPECT_EQ(testRoi.NumPixelCols(), movie.NCOL);
            EXPECT_EQ(movie.NFRAMES, framesPerTile * numChunks);

            //SequelMovieFrame<int16_t> frame(32,32);
            SequelMovieFrame<int16_t> frame; //(roi.NumPixelRows, roi.NumPixelCols);
            movie.ReadFrame(0, frame);
            EXPECT_EQ(0, frame.data[0]);
            EXPECT_EQ(0, frame.data[31]);
            EXPECT_EQ(0x4040, frame.data[32]);
            EXPECT_EQ((int16_t) 0xC0C0, frame.data[32 * 32 - 1]);

            movie.ReadFrame(1, frame);
            EXPECT_EQ(0, frame.data[0]);
            EXPECT_EQ(0, frame.data[31]);
            EXPECT_EQ(0x4040, frame.data[32]);
            EXPECT_EQ((int16_t) 0xC0C0, frame.data[32 * 32 - 1]);

            movie.ReadFrame(511, frame);
            EXPECT_EQ(0, frame.data[0]);
            EXPECT_EQ(0, frame.data[31]);
            EXPECT_EQ(0x4040, frame.data[32]);
            EXPECT_EQ((int16_t) 0xC0C0, frame.data[32 * 32 - 1]);

            std::vector<uint64_t> errors;
            movie.ErrorFrames >> errors;
            EXPECT_EQ(2 * numChunks, errors.size());
            EXPECT_EQ(100, errors[0]);
            EXPECT_EQ(102, errors[1]);
        }
    }
}

TEST_F(SequelMovieFileHDF5_Test,ChunksSelectedFrames)
{
    const int numChunks = 3;
    for(auto chipClass : chipClasses)
    {
        const std::string movieFile = tempFileManager.GenerateTempFileName(".h5");
        PBLOG_DEBUG << "opening file" << movieFile;
        SequelMovieConfig mc;
        mc.path = movieFile;
        mc.roi.reset(roi.Clone());
        mc.chipClass = chipClass;

        {
//            AutoTimer _("File Open", 1, "?");
            SequelMovieFileHDF5 movie(mc);
            movie.AddSelectedFrame(0);
            movie.AddSelectedFrame(510);
            movie.AddSelectedFrame(-1);

            movie.PrepareChunkBuffers();
            uint32_t numFrames = movie.CountSelectedFrames();
            FillMovie(movie,numChunks, numFrames);
        }
        PBLOG_DEBUG << "closing and reopening file" << movieFile;
        {
            SequelMovieFileHDF5 movie(movieFile);

            std::string s;
            movie.FormatVersion >> s;
            EXPECT_EQ(s, "1.0");
            EXPECT_EQ(movie.NFRAMES, numChunks * 3);

            SequelMovieFrame<int16_t> frame(roi);
            // first frame of first chunk
            movie.ReadFrame(0,frame);
            EXPECT_EQ(0,frame.data[0])<< frame;
            EXPECT_EQ(0,frame.data[31])<< frame;
            ASSERT_EQ(0x0101,frame.data[32]) << frame << " chipclass:" << chipClass.toString();

            EXPECT_EQ((int16_t)0xFFFF, frame.data[roi.TotalPixels()-1])<< frame;

            // last-1 frame of first chunk, should be same as first frame,
            movie.ReadFrame(1,frame);
            EXPECT_EQ(0,frame.data[0])<< frame;
            EXPECT_EQ(0,frame.data[31])<< frame;
            EXPECT_EQ(0x0101,frame.data[32])<< frame;
            EXPECT_EQ((int16_t)0xFFFF, frame.data[roi.TotalPixels()-1])<< frame;

            // last frame of first chunk, should be same as first frame,
            movie.ReadFrame(2,frame);
            EXPECT_EQ(0,frame.data[0])<< frame;
            EXPECT_EQ(0,frame.data[31])<< frame;
            EXPECT_EQ(0x0101,frame.data[32])<< frame;
            EXPECT_EQ((int16_t)0xFFFF, frame.data[roi.TotalPixels()-1])<< frame;

            movie.ReadFrame(3,frame);
            EXPECT_EQ(0x0101,frame.data[0]);
            EXPECT_EQ(0x0101,frame.data[31]);
            EXPECT_EQ(0x0202,frame.data[32]);
            EXPECT_EQ(0x0000, frame.data[roi.TotalPixels()-1]);

        }

    }
}

int roundUp64(int x)
{
    return (x+63)/64*64;
}

#ifdef PB_MIC_COPROCESSOR
//#define ALIGN_ME_64 __attribute__((aligned(64)));
#define ALIGN_ME_64 /* fixme above doesn't work */
#else
#define ALIGN_ME_64
#endif


TEST(SequelMovieFrame,Test1)
{
    SequelMovieFrame<int8_t> frame(10,10);
    EXPECT_EQ(100,frame.NumPixels());
    EXPECT_EQ(100,frame.DataSize());

    frame.SetDefaultValue(123);
    EXPECT_EQ(123,frame.GetPixel(0,0));
    frame.SetPixel(1,1,34);
    EXPECT_EQ(123,frame.GetPixel(0,0));
    EXPECT_EQ(34,frame.GetPixel(1,1));
    EXPECT_EQ(34,frame.GetPixel(1,1));

    SequelMovieFrame<int8_t> copiedFrame(frame);
    EXPECT_EQ(100,copiedFrame.NumPixels());
    EXPECT_EQ(123,copiedFrame.GetPixel(0,0));
    EXPECT_EQ(34,copiedFrame.GetPixel(1,1));

    EXPECT_EQ(0, frame.Compare(copiedFrame));

    const int bits = 7;
    frame.SetPattern1(bits);
    EXPECT_NO_THROW(frame.ValidatePattern1(bits));
    frame.SetPixel(1,1,34); // purposely mess up the pattern
    EXPECT_THROW(frame.ValidatePattern1(bits),std::exception);
}

#ifdef PB_MIC_COPROCESSOR
TEST(SequelMovieFrame,DISABLED_SequelBit10)
// segmentation faults for some reason... probably because "frame" is not 512bit aligned
#else
TEST(SequelMovieFrame,SequelBit10)
#endif
{
    vector<uint8_t> streamBuffer(10000000) ALIGN_ME_64;

    SequelMovieFrame<int16_t> frame(1024,1024) ALIGN_ME_64;
    frame.CreateRandomPattern(10);
    size_t packedSize = frame.Pack(streamBuffer.data(),streamBuffer.size(), 10,FrameClass::Format2C2A);
    EXPECT_EQ(roundUp64(24) + frame.NROW * roundUp64(8 + (frame.NCOL * 10/8)),packedSize);
    SequelMovieFrame<int16_t> b(1024,1024) ALIGN_ME_64;
    size_t unpackedSize = b.Unpack(streamBuffer.data(),1024,streamBuffer.size());
    EXPECT_EQ(unpackedSize,packedSize);
    EXPECT_EQ(0,frame.Compare(b)) << "num differences";
}

#ifdef PB_MIC_COPROCESSOR
TEST(SequelMovieFrame,DISABLED_SequelBit12)
// segmentation faults for some reason... probably because "frame" is not 512bit aligned
#else
TEST(SequelMovieFrame,SequelBit12)
#endif
{
    vector<uint8_t> streamBuffer(10000000) ALIGN_ME_64;

    SequelMovieFrame<int16_t> frame(1024,1024) ALIGN_ME_64;
    frame.CreateRandomPattern(12);
    size_t packedSize = frame.Pack(streamBuffer.data(),streamBuffer.size(), 12, FrameClass::Format2C2A);
    EXPECT_EQ(roundUp64(24) + frame.NROW * roundUp64(frame.NCOL * 12/8 + 8) ,packedSize);
    SequelMovieFrame<int16_t> b(1024,1024) ALIGN_ME_64;
    size_t unpackedSize = b.Unpack(streamBuffer.data(),1024,streamBuffer.size());
    EXPECT_EQ(unpackedSize,packedSize);
    EXPECT_EQ(0,frame.Compare(b)) << "num differences";
}

#ifdef PB_MIC_COPROCESSOR
TEST(SequelMovieFrame,DISABLED_SpiderBit12)
// segmentation faults for some reason... probably because "frame" is not 512bit aligned
#else
TEST(SequelMovieFrame,SpiderBit12)
#endif
{
    vector<uint8_t> streamBuffer(10000000) ALIGN_ME_64;

    SequelMovieFrame<int16_t> frame(1024,1024) ALIGN_ME_64;
    frame.CreateRandomPattern(12);
    size_t packedSize = frame.Pack(streamBuffer.data(),streamBuffer.size(),12, FrameClass::Format1C4A);
    EXPECT_EQ(roundUp64(24) + frame.NROW * roundUp64(frame.NCOL * 12/8 + 8) ,packedSize);
    SequelMovieFrame<int16_t> b(1024,1024) ALIGN_ME_64;
    size_t unpackedSize = b.Unpack(streamBuffer.data(),1024,streamBuffer.size());
    EXPECT_EQ(unpackedSize,packedSize);
    EXPECT_EQ(0,frame.Compare(b)) << "num differences";
}

TEST(SequelMovieFrame,Saturation)
{
    SequelMovieFrame<int16_t> x(128,128);
    x.SetDefaultValue(1.0);
    EXPECT_EQ(1.0,x.data[0]);
    x.Add(-2, -32767, 32767);
    EXPECT_EQ(-1,x.data[0]);
    x.Add(-32767, -32767, 32767);
    EXPECT_EQ(-32767,x.data[0]);
    x.Add(-1, -32767, 32767);
    EXPECT_EQ(-32767,x.data[0]);
    x.Add(+32767, -32767, 32767);
    EXPECT_EQ(0,x.data[0]);
    x.Add(32767, -32767, 32767);
    EXPECT_EQ(32767,x.data[0]);
    x.Add(+1, -32767, 32767);
    EXPECT_EQ(32767,x.data[0]);

    x.Scale(1.25, -32767, 32767);
    EXPECT_EQ(32767,x.data[0]);
    x.SetDefaultValue(-32767);
    x.Scale(1.25, -32767, 32767);
    EXPECT_EQ(-32767,x.data[0]);
}

TEST(SequelMovieFrame,Square)
{
    SequelMovieFrame<int16_t> x(128,128);
    x.SetDefaultValue(2.0);
    EXPECT_EQ(2.0,x.data[0]);
    x.Square();
    EXPECT_EQ(4.0,x.data[0]);
}

TEST(SequelMovieFrame,SaturatedSubtraction)
{
    SequelMovieFrame<int16_t> x(128, 128);
    x.SetDefaultValue(1.0);
    EXPECT_EQ(1.0, x.GetPixel(0, 0));
    x.SaturatedSubtract(2);
    EXPECT_EQ(-1.0, x.GetPixel(0, 0));
    // check lower bound
    x.SaturatedSubtract(32767);
    EXPECT_EQ(-32768, x.GetPixel(0, 0));
    //  check upperbound (by subtracting negative large magnitude twice)
    x.SaturatedSubtract(-32768);
    EXPECT_EQ(0, x.GetPixel(0, 0));
    x.SaturatedSubtract(-32768);
    EXPECT_EQ(32767, x.GetPixel(0, 0));
}


TEST(SequelMovieFrame,CopyConstructors)
{
    SequelMovieFrame<int16_t> sumFrame(5,5);
    sumFrame.CreateRandomPattern(5);

    SequelMovieFrame<int16_t> intFrame1 ( sumFrame );
    EXPECT_EQ(intFrame1.GetPixel(0,0),sumFrame.GetPixel(0,0));

    SequelMovieFrame<int16_t> intFrame2;
    intFrame2 = sumFrame;
    EXPECT_EQ(intFrame2.GetPixel(0,0),sumFrame.GetPixel(0,0));

    SequelMovieFrame<float> fltFrame1 ( sumFrame );
    EXPECT_FLOAT_EQ(fltFrame1.GetPixel(0,0),sumFrame.GetPixel(0,0));

    SequelMovieFrame<float> fltFrame2;
    fltFrame2 = sumFrame;
    EXPECT_FLOAT_EQ(fltFrame2.GetPixel(0,0),sumFrame.GetPixel(0,0));

}

TEST(SequelMovieFrame,BitComparison)
{
    EXPECT_EQ(1,SequelMovieFrame<float>::CountBitMismatches(15,14));
    EXPECT_EQ(1,SequelMovieFrame<double>::CountBitMismatches(15,14));
    EXPECT_EQ(1,SequelMovieFrame<int16_t>::CountBitMismatches(15,14));
    EXPECT_EQ(4,SequelMovieFrame<int16_t>::CountBitMismatches(7,8));

    EXPECT_EQ(0,SequelMovieFrame<float>::CountBitMismatches(1.5f,1.5f));
    EXPECT_EQ(0,SequelMovieFrame<double>::CountBitMismatches(1.5,1.5));
    EXPECT_EQ(0,SequelMovieFrame<int16_t>::CountBitMismatches((int16_t)1,(int16_t)1));

    EXPECT_EQ(1,SequelMovieFrame<int16_t>::CountBitMismatches((int16_t)0x07CF,(int16_t)0x47CF));
    EXPECT_EQ(1,SequelMovieFrame<int16_t>::CountBitMismatches((int16_t)0x07CF,(int16_t)0x87CF));
}

TEST_F(SequelMovieFileHDF5_Test,SequelMovieEventsHDF5)
{
    const std::string movieFile = tempFileManager.GenerateTempFileName(".h5");

    //TEST_COUT << "movieFile:" << movieFile << std::endl;
    //tempFileManager.Keep();


    {
        SequelMovieEventsHDF5 events;

        std::vector<std::string> lasers;
        lasers.push_back("topLaser");
        lasers.push_back("bottomLaser");

        H5::H5File hdf5file = H5::H5File(movieFile, H5F_ACC_TRUNC);
        events.CreateForWrite(hdf5file, lasers);
        SequelMovieLaserPowerEventsHDF5& laserPowerChanges(events.laserPowerChanges);

        laserPowerChanges.SetFirstFrame(0);

        EventObject eo;
        eo.Load( R"(
        {
            "eventType" : "laserpower","timeStamp" : "2018-11-13T15:12:06.203Z",
            "lasers" :[
              { "name": "topLaser",   "startPower_mW":   0.0, "stopPower_mW": 42.0 },
              { "name": "bottomLaser","startPower_mW":   0.0, "stopPower_mW": 24.0 }
            ], "startFrame" : 0, "stopFrame"  : 42, "token": "abc"
        }
        )");
        laserPowerChanges.AddEvent(eo);

        eo.Load( R"(
        {
            "eventType" : "laserpower","timeStamp" : "2018-11-13T15:21:41.961Z",
            "lasers" :[
              { "name": "topLaser",   "startPower_mW": 42.0, "stopPower_mW": 42.42 },
              { "name": "bottomLaser","startPower_mW": -1.0, "stopPower_mW": -1.0 }
            ], "startFrame" : 1032, "stopFrame"  : 1057, "token": "abc"
        }
        )");
        laserPowerChanges.AddEvent(eo);

        eo.Load( R"(
        {
            "eventType" : "laserpower","timeStamp" : "2018-11-13T15:26:38.985Z",
            "lasers" :[
              { "name": "topLaser",   "startPower_mW": -1.0, "stopPower_mW": -1.0 },
              { "name": "bottomLaser","startPower_mW": 24.0, "stopPower_mW": 24.24 }
            ], "startFrame" : 1000, "stopFrame"  : 1042, "token": "abc"
        }
        )");
        laserPowerChanges.AddEvent(eo);
        hdf5file.close();
    }

    {
        SequelMovieEventsHDF5 events;
        H5::H5File hdf5file = H5::H5File(movieFile, H5F_ACC_RDONLY);
        events.OpenForRead(hdf5file);

        SequelMovieLaserPowerEventsHDF5& laserPowerChanges(events.laserPowerChanges);
        EXPECT_EQ(2, laserPowerChanges.nL);
        EXPECT_EQ(3, laserPowerChanges.nLPC);

        std::vector<std::string> timestamps;
        laserPowerChanges.dsTimeStamp >> timestamps;
#if 0
        for(const auto& l : timestamps)
        {
            TEST_COUT << " timestamp: " <<l << std::endl;
        }
#endif

        auto eventList = laserPowerChanges.ReadAllEvents();
        ASSERT_EQ(3, eventList.size());

        {
            EventObject& eo1(eventList[0]);
            EXPECT_EQ(EventObject::EventType::laserpower, eo1.eventType());
            EXPECT_EQ("2018-11-13T15:12:06.203Z", eo1.timeStamp());
            EXPECT_EQ(0, eo1.startFrame());
            EXPECT_EQ(42, eo1.stopFrame());
            ASSERT_EQ(2, eo1.lasers.size());
            EXPECT_EQ(LaserPowerObject::LaserName::topLaser, eo1.lasers[0].name());
            EXPECT_EQ(LaserPowerObject::LaserName::bottomLaser, eo1.lasers[1].name());
            EXPECT_FLOAT_EQ(0.0, eo1.lasers[0].startPower_mW());
            EXPECT_FLOAT_EQ(42.0, eo1.lasers[0].stopPower_mW());
            EXPECT_FLOAT_EQ(0.0, eo1.lasers[1].startPower_mW());
            EXPECT_FLOAT_EQ(24.0, eo1.lasers[1].stopPower_mW());
        }

        {
            EventObject& eo3(eventList[2]);
            EXPECT_EQ(EventObject::EventType::laserpower, eo3.eventType());
            EXPECT_EQ("2018-11-13T15:26:38.985Z", eo3.timeStamp());
            EXPECT_EQ(1000, eo3.startFrame());
            EXPECT_EQ(1042, eo3.stopFrame());
            ASSERT_EQ(2, eo3.lasers.size());
            EXPECT_EQ(LaserPowerObject::LaserName::topLaser, eo3.lasers[0].name());
            EXPECT_EQ(LaserPowerObject::LaserName::bottomLaser, eo3.lasers[1].name());
            EXPECT_FLOAT_EQ(-1.0, eo3.lasers[0].startPower_mW());
            EXPECT_FLOAT_EQ(-1.0, eo3.lasers[0].stopPower_mW());
            EXPECT_FLOAT_EQ(24.0, eo3.lasers[1].startPower_mW());
            EXPECT_FLOAT_EQ(24.24, eo3.lasers[1].stopPower_mW());
        }

        hdf5file.close();
    }
}


TEST_F(SequelMovieFileHDF5_Test,EventObject)
{
    const std::string movieFile = tempFileManager.GenerateTempFileName(".h5");
    {
        // create file
        SequelMovieConfig mc;
        mc.path = movieFile;
        mc.roi.reset(roi.Clone());
        mc.chipClass = ChipClass::Sequel;

        SequelMovieFileHDF5 movie(mc);

        EventObject eo;
        eo.eventType = EventObject::EventType::laserpower;
        eo.timeStamp = "2018-11-11T12:00:00Z";
        eo.startFrame = 123;
        eo.stopFrame = 456;
        eo.lasers[0].name = LaserPowerObject::LaserName::bottomLaser;
        eo.lasers[0].startPower_mW = 123.0;
        eo.lasers[0].stopPower_mW = 145.0;
        // leave out laser[1] for now
        movie.events.laserPowerChanges.SetFirstFrame(100);
        movie.events.laserPowerChanges.AddEvent(eo);
    }

    {
        //verify file
        SequelMovieFileHDF5 movie(movieFile);

        ASSERT_EQ(1,movie.events.laserPowerChanges.nLPC);
        auto events = movie.events.laserPowerChanges.ReadAllEvents();
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

    // tempFileManager.Keep();
}


