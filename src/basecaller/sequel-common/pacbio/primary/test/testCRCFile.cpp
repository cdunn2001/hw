#include <pacbio/primary/Tile.h>
#include <pacbio/primary/SequelMovieFileCRC.h>
#include "gtest/gtest.h"
#include <pacbio/dev/AutoTimer.h>
#include <pacbio/POSIX.h>
#include <pacbio/logging/Logger.h>
#include <pacbio/dev/gtest-extras.h>

#include "SequelTestROIs.h"

using namespace std;

#include <stdexcept>

using namespace PacBio::Primary;
using namespace PacBio::Dev;
using namespace PacBio;


class CRCFileTest : public ::testing::Test
{
public:
    PacBio::Logging::LogSeverityContext loggerContext;
    TempFileManager temp;

    CRCFileTest()
            : loggerContext(PacBio::Logging::LogLevel::FATAL)
    {

    }

    virtual ~CRCFileTest()
    {
        if (HasFailure()) temp.Keep();
    }

    void SetUp()
    {
    }


};

TEST_F(CRCFileTest,Frames)
{
    const std::string movieFile = temp.GenerateTempFileName(".crc");
    SequelRectangularROI roi = SequelROI_SequelAlphaFull();
    {
        {
//            AutoTimer _("File Open", 1, "?");
            SequelMovieFileCRC movie(movieFile, roi);
            EXPECT_EQ(movie.TypeName(),"CRC");

            SequelMovieFrame<int16_t> frame(roi);
            frame.SetDefaultValue(0);
            movie.AddFrame(frame);
        }
        {
            ifstream test(movieFile);
            EXPECT_TRUE(static_cast<bool>(test)) << "Can't read " << movieFile;
            string line;
	        getline(test,line);
            EXPECT_EQ("2407922888",line);
        }
    }
}

TEST_F(CRCFileTest,Chunks)
{
    const int numChunks = 1;
    const std::string movieFile = temp.GenerateTempFileName(".crc");
    SequelRectangularROI roi = SequelROI_SequelAlphaFull();
    const size_t TilesPerChunk = framesPerTile * roi.TotalPixels() * sizeof(uint16_t) / sizeof(Tile);
    {
        {
//            AutoTimer _("File Open", 1, "?");
            SequelMovieFileCRC movie(movieFile, roi);
            movie.PrepareChunkBuffers();

            for(int chunk=0;chunk<numChunks;chunk++)
            {
                {
//                    AutoTimer _("memmove", TilesPerChunk, "tiles");
                    std::unique_ptr<Tile> header(Tile::makeHeader());
                    movie.AddChunk(header.get(),Tile::NumFrames);

                    uint32_t tileOffset = 0;
                    std::unique_ptr<Tile> tile(Tile::make());
                    for (uint32_t i = 0; i < TilesPerChunk; i++)
                    {
                        tile->SetPattern(i);
                        movie.AddTile(tile.get(), tileOffset++);
                        //cout  << "Adding chunk:tile " << chunk << ":" << i << endl;
                    }
                }
                {
  //                  AutoTimer _("fileIO", TilesPerChunk, "tiles");
                    movie.FlushChunk();
                }
            }
        }
        {
            ifstream test(movieFile);
            EXPECT_TRUE(static_cast<bool>(test));
            string line;
	        getline(test,line);
            EXPECT_EQ("2984128794",line);
        }

    }
}

TEST_F(CRCFileTest,ChunkError)
{
    const int numChunks = 1;
    const std::string movieFile = temp.GenerateTempFileName(".crc");
    SequelRectangularROI roi = SequelROI_SequelAlphaFull();

    const size_t TilesPerChunk = framesPerTile * roi.TotalPixels() * sizeof(uint16_t) / sizeof(Tile);
    {
        {
//            AutoTimer _("File Open", 1, "?");
            SequelMovieFileCRC movie(movieFile, roi);
            movie.PrepareChunkBuffers();

            for(int chunk=0;chunk<numChunks;chunk++)
            {
                {
//                    AutoTimer _("memmove", TilesPerChunk, "tiles");
                    std::unique_ptr<Tile> header(Tile::makeHeader());
                    header->data[1024] = 3;
                    EXPECT_EQ(2,header->ErroredFrames().size());
                    movie.AddChunk(header.get(),Tile::NumFrames);

                    uint32_t tileOffset = 0;
                    std::unique_ptr<Tile> tile(Tile::make());
                    for (uint32_t i = 0; i < TilesPerChunk; i++)
                    {
                        tile->SetPattern(i);
                        movie.AddTile(tile.get(), tileOffset++);
                        //cout  << "Adding chunk:tile " << chunk << ":" << i << endl;
                    }
                    //                  AutoTimer _("fileIO", TilesPerChunk, "tiles");
                    movie.FlushChunk();
                }
            }
        }
        {
            ifstream test(movieFile);
            EXPECT_TRUE(static_cast<bool>(test));
            string line;
            getline(test,line);
            EXPECT_EQ("2984128794 !",line);
            getline(test,line);
            EXPECT_EQ("2984128794 !",line);
            getline(test,line);
            EXPECT_EQ("2984128794",line);
        }

    }
}
