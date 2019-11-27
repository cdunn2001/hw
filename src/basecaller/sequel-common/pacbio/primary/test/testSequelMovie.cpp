#include <stdexcept>
#include <memory>

#include "gtest/gtest.h"

#include <pacbio/primary/Tile.h>
#include <pacbio/primary/SequelTraceFile.h>
#include <pacbio/dev/AutoTimer.h>
#include <pacbio/dev/gtest-extras.h>
#include <pacbio/dev/TemporaryDirectory.h>
#include <pacbio/ipc/JSON.h>
#include <pacbio/POSIX.h>
#include <pacbio/primary/ChipLayoutSpider1p0NTO.h>

#include "testTraceFilePath.h"
#include "testSequelTraceFileHDF5.h"

using namespace PacBio::Primary;
using namespace PacBio;
using namespace PacBio::Dev;


TEST(SequelMovie, ChunkBufferFrameOrientedConstructor)
{
    size_t s = 1000000;
    std::vector<uint8_t> someMemory(s);
    ChunkBufferFrameOriented bb(1000000,someMemory.data());
    ASSERT_NE(nullptr,bb.Data(0));
    ASSERT_NE(nullptr,bb.HeaderTile());
    EXPECT_EQ(bb.WritableData(0),bb.Data(0));
    EXPECT_EQ(0,bb.Data(0)[0]);
    EXPECT_EQ(0,bb.Data(0)[s-1]);
    EXPECT_TRUE(bb.HeaderTile()->IsHeader());
    EXPECT_EQ(s,bb.BufferByteSize());
    EXPECT_EQ(0,bb.FramesToWrite());
    std::unique_ptr<Tile> header(Tile::makeHeader());
    ASSERT_TRUE(static_cast<bool>(header));
    header->NumFramesInHeader(13);
    bb.SetHeader(header.get());
    bb.SetFramesToWrite(10);
    EXPECT_TRUE(bb.HeaderTile()->IsHeader());
    EXPECT_EQ(10,bb.FramesToWrite());
    EXPECT_EQ(13,bb.HeaderTile()->NumFramesInHeader());
}

TEST(SequelMovie, SequelMovieBuffer)
{
    SequelRectangularROI roi(RowPixels(0),ColPixels(0),RowPixels(31),ColPixels(32),SequelSensorROI::SequelAlpha());
    for(int n=1;n<=5;n++)
    {
      SequelMovieBuffer b(roi,false);
//      EXPECT_EQ(31,b.Rows());
//      EXPECT_EQ(32,b.Cols());

      ASSERT_FALSE(b.QueueReady());
      ASSERT_FALSE(b.BufferReady());
      b.PrepareChunkBuffers(n);

      ASSERT_EQ(31*Tile::NumPixels,roi.TotalPixels());
      ASSERT_EQ(73216,b.NumTiles());

      ASSERT_TRUE(b.QueueReady());
      ASSERT_FALSE(b.BufferReady());
      b.PrepareNextChunk();
      ASSERT_TRUE(b.BufferReady());

      if (n >= 2)
      {
          auto bigBuffer0 = b.CurrentChunkBuffer()->Data(0);
          b.FlushChunk();
          b.PrepareNextChunk();
          ASSERT_TRUE(b.BufferReady());
          auto bigBuffer1 = b.CurrentChunkBuffer()->Data(0);
          EXPECT_NE(bigBuffer0,bigBuffer1) << "n=" << n;
          ASSERT_TRUE(b.QueueReady()) << "n=" << n;
      }
    }
}

TEST(SequelMovie, ChunkBufferTraceOrientedConstructor)
{
    int numTiles = 15;
    size_t s = sizeof(Tile)*numTiles;
    std::vector<uint8_t> someMemory(s);

    ChunkBufferTraceOriented bb(someMemory.size(),someMemory.data());
    for (int itile=0; itile<numTiles;itile++)
    {
        ASSERT_NE(nullptr, bb.Data(itile));
        EXPECT_EQ(bb.WritableData(itile),bb.Data(itile));
        EXPECT_EQ(0,bb.Data(itile)[0]);
        EXPECT_EQ(0,bb.Data(itile)[sizeof(Tile)-1]);
    }
    ASSERT_NE(nullptr,bb.HeaderTile());
    EXPECT_TRUE(bb.HeaderTile()->IsHeader());
    EXPECT_EQ(s,bb.BufferByteSize());
    EXPECT_EQ(0,bb.FramesToWrite());
    std::unique_ptr<Tile> header(Tile::makeHeader());
    ASSERT_TRUE(static_cast<bool>(header));
    header->NumFramesInHeader(13);
    bb.SetHeader(header.get());
    bb.SetFramesToWrite(10);
    EXPECT_TRUE(bb.HeaderTile()->IsHeader());
    EXPECT_EQ(10,bb.FramesToWrite());
    EXPECT_EQ(13,bb.HeaderTile()->NumFramesInHeader());
}

TEST(SequelMovie,ChunkBufferTraceOriented_TestA)
{
    SequelSparseROI roi(SequelSensorROI::SequelAlpha());
    roi.AddRectangle(0,0,1,32); // one tile
    roi.AddRectangle(1,32,1,32); // another tile
    uint32_t numHoles = roi.CountZMWs();
    uint32_t numUnitCellsPerTile = 16; // fixme

    int numTiles = roi.TotalPixels() / Tile::NumPixels; // s;numHoles / numUnitCellsPerTile;
    ASSERT_EQ(2,numTiles);
    size_t s = sizeof(Tile)*numTiles;
    std::vector<uint8_t> someMemory(s);
    ASSERT_NE(nullptr, someMemory.data());

    ChunkBufferTraceOriented bb(someMemory.size(),someMemory.data());

    ASSERT_EQ(someMemory.data()             , bb.WritableData(0));
    ASSERT_EQ(someMemory.data()+sizeof(Tile), bb.WritableData(1));

    //TEST_COUT << "setting tile 0 pattern" << (void*) bb.WritableData(0) << "\n";
    reinterpret_cast<Tile*>(bb.WritableData(0))->SetPattern(0);
    //TEST_COUT << "setting tile 1 pattern" << (void*) bb.WritableData(1) << "\n";
    reinterpret_cast<Tile*>(bb.WritableData(1))->SetPattern(1);

    for(SequelROI::Enumerator e(roi,0,numHoles);e;e+=numUnitCellsPerTile)
    {
        uint32_t zmw = e.Index();
        uint32_t tileOffset = zmw/numUnitCellsPerTile;
        EXPECT_NO_THROW(reinterpret_cast<const Tile*>(bb.Data(tileOffset))->CheckPattern(tileOffset)) <<
             "ZMW:"<< zmw << " tileOffset:" << tileOffset;
    }
    //void SequelMovieBuffer::AddTile
}

TEST(SequelMovie,StreamingOps)
{
    PacBio::Dev::TemporaryDirectory temp;
    const std::string movieFile = temp.DirName() + "/streaming.h5";

    //TEST_COUT << "movieFile:" << movieFile << std::endl;
    //temp.Keep();

    {
        std::vector<std::string> lasers;
        lasers.push_back("topLaser");
        lasers.push_back("bottomLaser");

        H5::H5File hdf5file = H5::H5File(movieFile, H5F_ACC_TRUNC);
        hsize_t hspace1_start[3];
        hspace1_start[0] = 0;
        hspace1_start[1] = 0;
        hspace1_start[2] = 0;
        hsize_t hspace1_max[3];
        hspace1_max[0] = H5S_UNLIMITED;
        hspace1_max[1] = 0;
        hspace1_max[2] = 0;
        H5::DataSpace hspace_nLPC(1, hspace1_start, hspace1_max);
        H5::DSetCreatPropList propList1;
        hsize_t chunk_dims[3];
        chunk_dims[0] = 1;
        chunk_dims[1] = 1;
        chunk_dims[2] = 1;
        propList1.setChunk(1, chunk_dims);

        H5::DataSet dsTimeStamp = hdf5file.createDataSet("TimeStamp", SeqH5string(), hspace_nLPC         , propList1);

        PBHDF5::Append(dsTimeStamp,lasers[0]);
        PBHDF5::Append(dsTimeStamp,lasers[1]);
        hdf5file.close();
    }
    {
        H5::H5File hdf5file = H5::H5File(movieFile, H5F_ACC_RDONLY);
        H5::DataSet dsTimeStamp = hdf5file.openDataSet("TimeStamp");
        std::vector<std::string> lasers;
        dsTimeStamp >> lasers;
        ASSERT_EQ(2,lasers.size());
        EXPECT_EQ("topLaser",lasers[0]);
        EXPECT_EQ("bottomLaser",lasers[1]);
    }
}

TEST(SequelMovieBuffer,AddFrame)
{
    SequelMovieBuffer buf(4,32); // 4*32 pixels = 128
    buf.PrepareChunkBuffers(1);
    buf.PrepareNextChunk(); // this needs to be done just once.
    EXPECT_FALSE(buf.PixelFull());
    EXPECT_EQ(0, buf.NumPixels());

    {
        std::unique_ptr<Tile> header(Tile::makeHeader());
        buf.SetChunkHeader(header.get());
        EXPECT_EQ(0, buf.NumPixels());
        EXPECT_FALSE(buf.PixelFull());

        std::unique_ptr<Tile> tile(Tile::makeHeader());
        tile->SetPattern(0);
        buf.AddTileAsFrames(tile.get(), 0, 0, 512);
        EXPECT_EQ(32 * 512, buf.NumPixels());
        EXPECT_FALSE(buf.PixelFull());

        for (int itile = 1; itile < 4; itile++)
        {
            buf.AddTileAsFrames(tile.get(), itile, 0, 512);
            EXPECT_EQ((itile+1) * 32 * 512, buf.NumPixels());
        }
        EXPECT_TRUE(buf.PixelFull());
    }

    {
        std::unique_ptr<Tile> header(Tile::makeHeader());
        buf.SetChunkHeader(header.get());
        EXPECT_EQ(0, buf.NumPixels());
        EXPECT_FALSE(buf.PixelFull());

        std::unique_ptr<Tile> tile(Tile::makeHeader());
        tile->SetPattern(0);

        for (int itile = 0; itile < 4; itile++)
        {
            buf.AddTileAsFrames(tile.get(), itile, 100, 101);
            buf.AddTileAsFrames(tile.get(), itile, 200, 201);
            EXPECT_EQ((itile+1) * 32 * 2, buf.NumPixels());
        }
        EXPECT_FALSE(buf.PixelFull());
    }

    {
        std::unique_ptr<Tile> header(Tile::makeHeader());
        buf.SetChunkHeader(header.get());
        EXPECT_EQ(0, buf.NumPixels());
    }
}
