//
// Created by mlakata on 5/4/15.
//

#include <sstream>
#include <gtest/gtest.h>
#include <pacbio/primary/SequelROI.h>
#include <pacbio/primary/Tile.h>
#include <pacbio/ipc/JSON.h>

using namespace PacBio::Primary;
using namespace std;

TEST(SequelRectangularROI,OneRectangle)
{
   SequelSensorROI sensorROI(288,0,288,2048,1,2);

   EXPECT_EQ(0, sensorROI.PhysicalColOffset());
   EXPECT_EQ(2048,sensorROI.PhysicalCols());
   EXPECT_EQ(288,sensorROI.PhysicalRowOffset());
   EXPECT_EQ(288,sensorROI.PhysicalRows());

   SequelRectangularROI roi(289,256,100,256,sensorROI);
   EXPECT_EQ(512,roi.AbsoluteColPixelMax());
   EXPECT_EQ(256,roi.AbsoluteColPixelMin());
   EXPECT_EQ(389,roi.AbsoluteRowPixelMax());
   EXPECT_EQ(289,roi.AbsoluteRowPixelMin());
   EXPECT_EQ(512,roi.RelativeColPixelMax());
   EXPECT_EQ(256,roi.RelativeColPixelMin());
   EXPECT_EQ(101,roi.RelativeRowPixelMax());
   EXPECT_EQ(1  ,roi.RelativeRowPixelMin());
   EXPECT_EQ(100*256/2,roi.CountHoles());
   EXPECT_FALSE(roi.Everything());
   EXPECT_EQ(256,roi.NumPixelCols());
   EXPECT_EQ(100,roi.NumPixelRows());
   EXPECT_EQ(100*256, roi.TotalPixels());
   const uint32_t numTiles = roi.TotalPixels()/Tile::NumPixels;
   const uint32_t firstTile = (
           roi.RelativeRowPixelMin()*2048 + 256)/Tile::NumPixels;
   const uint32_t roiWidth = roi.NumPixelCols()/Tile::NumPixels;
   const uint32_t lastTileRow = firstTile + 99 * (2048/Tile::NumPixels);

   EXPECT_EQ((2048+256)/32,firstTile);

   uint32_t internalOffset;
   EXPECT_FALSE(roi.ContainsTileOffset(0));
   EXPECT_FALSE(roi.ContainsTileOffset(firstTile-1));
   EXPECT_TRUE(roi.ContainsTileOffset(firstTile,&internalOffset));
   EXPECT_EQ(0,internalOffset);
   EXPECT_TRUE(roi.ContainsTileOffset(firstTile+1,&internalOffset));
   EXPECT_EQ(1,internalOffset);
   EXPECT_TRUE(roi.ContainsTileOffset(firstTile+2,&internalOffset));
   EXPECT_EQ(2,internalOffset);
   EXPECT_TRUE(roi.ContainsTileOffset(firstTile+roiWidth-1,&internalOffset));
   EXPECT_EQ(roiWidth-1,internalOffset);
   EXPECT_FALSE(roi.ContainsTileOffset(firstTile+roiWidth));

   EXPECT_FALSE(roi.ContainsTileOffset(lastTileRow-1));
   EXPECT_TRUE(roi.ContainsTileOffset(lastTileRow,&internalOffset));
   EXPECT_EQ(numTiles-roiWidth,internalOffset);
   EXPECT_TRUE(roi.ContainsTileOffset(lastTileRow+roiWidth-1,&internalOffset));
   EXPECT_EQ(numTiles-1,internalOffset);
   EXPECT_FALSE(roi.ContainsTileOffset(lastTileRow+roiWidth));


   std::stringstream s;
   s << roi;
   EXPECT_EQ("region:(289pix,256pix) - (389pix,512pix) size:(100pix,256pix) (Sensor ROI:region:(288,0) - (576,2048) size:(288,2048) unitcell:(1 x 2))",s.str());
};


TEST(SequelRectangularROI,Enumerator)
{
    SequelRectangularROI roi(RowPixels(0), ColPixels(32), RowPixels(128), ColPixels(256),SequelSensorROI::SequelAlpha());

    SequelROI::Enumerator e(roi,0,roi.CountZMWs());
    std::pair<RowPixels, ColPixels> pix(0,0);

    EXPECT_EQ(0,e.Index());
    pix = e.UnitCellPixelOrigin();
    EXPECT_EQ(0,pix.first.Value());
    EXPECT_EQ(32,pix.second.Value());

}
TEST(SequelSparseROI,OneRectangle)
{
    SequelSensorROI sensorROI(288,0,288,2048,1,2);
//   SequelROI::SetPhysicalOffset(288,0);
//   SequelROI::SetPhysicalSize(288,2048);

    EXPECT_EQ(0, sensorROI.PhysicalColOffset());
    EXPECT_EQ(2048,sensorROI.PhysicalCols());
    EXPECT_EQ(288,sensorROI.PhysicalRowOffset());
    EXPECT_EQ(288,sensorROI.PhysicalRows());

    SequelSparseROI roi(sensorROI);
    EXPECT_NO_THROW(roi.AddRectangle(289,256,100,256));

    EXPECT_EQ(100*256/2,roi.CountHoles());
    EXPECT_FALSE(roi.Everything());
    EXPECT_EQ(100*256, roi.TotalPixels());

    const uint32_t firstTile = (2048 + 256)/Tile::NumPixels;
    const uint32_t roiWidth = 256/Tile::NumPixels;
    const uint32_t lastTileRow = firstTile + 99 * (2048/Tile::NumPixels);
    const uint32_t numRoiTiles = roi.TotalPixels()/Tile::NumPixels;

    uint32_t internalOffset;
    EXPECT_FALSE(roi.ContainsTileOffset(0));
    EXPECT_FALSE(roi.ContainsTileOffset(firstTile-1));
    EXPECT_NO_THROW(roi.ContainsTileOffset(firstTile,nullptr));
    EXPECT_TRUE(roi.ContainsTileOffset(firstTile,&internalOffset));
    EXPECT_EQ(0,internalOffset);
    EXPECT_TRUE(roi.ContainsTileOffset(firstTile+roiWidth-1,&internalOffset));
    EXPECT_EQ(roiWidth-1,internalOffset);
    EXPECT_FALSE(roi.ContainsTileOffset(firstTile+roiWidth));

    EXPECT_FALSE(roi.ContainsTileOffset(lastTileRow-1));
    EXPECT_TRUE(roi.ContainsTileOffset(lastTileRow,&internalOffset));
    EXPECT_EQ(numRoiTiles-roiWidth,internalOffset);
    EXPECT_TRUE(roi.ContainsTileOffset(lastTileRow+roiWidth-1,&internalOffset));
    EXPECT_EQ(numRoiTiles-1,internalOffset);
    EXPECT_FALSE(roi.ContainsTileOffset(lastTileRow+roiWidth));

    stringstream s;
    s << roi;
    EXPECT_EQ("SequelSparseROI, totalPixels:25600 (Sensor ROI:region:(288,0) - (576,2048) size:(288,2048) unitcell:(1 x 2))\n[[1,256,100,256]]",s.str());

    // now throw in another rectangle (in the upper left corner, so it is the first rectangle
    // and make sure the internalOffsets all change
    EXPECT_NO_THROW(roi.AddRectangle(288,0,1,32));
    const uint32_t firstTile2 = 0;
    EXPECT_NE(firstTile,firstTile2);
    EXPECT_TRUE(roi.ContainsTileOffset(firstTile2,&internalOffset));
    EXPECT_EQ(0,internalOffset);
    EXPECT_TRUE(roi.ContainsTileOffset(firstTile,&internalOffset));
    EXPECT_EQ(1,internalOffset);

};

TEST(SequelSparseROI,BadRectangles)
{
    SequelSensorROI sensorROI(288,32,288,2016,1,2);

    SequelSparseROI roi(sensorROI);
    EXPECT_THROW(roi.AddRectangle(0,32,10,32),std::runtime_error); // row=0 is less than physical row offset of 288
    EXPECT_THROW(roi.AddRectangle(287,32,10,32),std::runtime_error); // row=0 is less than physical row offset of 288
    EXPECT_THROW(roi.AddRectangle(288,33,10,32),std::runtime_error); // column is not multiple of 32
    EXPECT_THROW(roi.AddRectangle(288,32,10,33),std::runtime_error); // num cols is not multiple of 32
    EXPECT_THROW(roi.AddRectangle(288,32,289,32),std::runtime_error); // max row is more than sensor ROI
    EXPECT_THROW(roi.AddRectangle(288,0,10,32),std::runtime_error); // col=0 is less than physical row offset of 288
    EXPECT_THROW(roi.AddRectangle(288,32,10,2048),std::runtime_error); // max col is more than sensor ROI

}

TEST(SequelSparseROI,Everything)
{
    SequelSparseROI roi(SequelSensorROI::SequelAlpha());
    roi.AddRectangle(0,0,1144,2048);
    EXPECT_EQ(1144*2048/2,roi.CountHoles());
    EXPECT_TRUE(roi.Everything());
    EXPECT_EQ(1144*2048, roi.TotalPixels());

    std::pair<RowPixels, ColPixels> pix(0,0);

    SequelROI::Enumerator e(roi,0,roi.CountZMWs());

    EXPECT_TRUE(e);
    EXPECT_EQ(0,e.Index());
    pix = e.UnitCellPixelOrigin();
    EXPECT_EQ(0,pix.first.Value());
    EXPECT_EQ(0,pix.second.Value());

    e++;
    EXPECT_EQ(1,e.Index());
    pix = e.UnitCellPixelOrigin();
    EXPECT_EQ(0,pix.first.Value());
    EXPECT_EQ(2,pix.second.Value());

    e+=1022;
    EXPECT_EQ(1023,e.Index());
    pix = e.UnitCellPixelOrigin();
    EXPECT_EQ(0,pix.first.Value());
    EXPECT_EQ(2046,pix.second.Value());

    e++;
    EXPECT_EQ(1024,e.Index());
    EXPECT_TRUE(e);
    pix = e.UnitCellPixelOrigin();
    EXPECT_EQ(1,pix.first.Value());
    EXPECT_EQ(0,pix.second.Value());

    e+=1024*1143-1;
    EXPECT_EQ(1024*1144-1,e.Index());
    EXPECT_TRUE(e);
    pix = e.UnitCellPixelOrigin();
    EXPECT_EQ(1143,pix.first.Value());
    EXPECT_EQ(2046,pix.second.Value());

    e++;
    EXPECT_EQ(1024*1144,e.Index());
    EXPECT_FALSE(e);
}

TEST(SequelSparseROI,NonOverlap)
{
    SequelSparseROI roi(SequelSensorROI::SequelAlpha());
    roi.AddRectangle(0,0,10,64);
    EXPECT_FALSE(roi.Everything());
    EXPECT_EQ(10*64, roi.TotalPixels());

    roi.AddRectangle(100,0,10,64);
    EXPECT_FALSE(roi.Everything());
    EXPECT_EQ(10*64*2, roi.TotalPixels());
}

TEST(SequelSparseROI,Overlap)
{
    SequelSparseROI roi(SequelSensorROI::SequelAlpha());
    roi.AddRectangle(0,0,10,64);
    EXPECT_FALSE(roi.Everything());
    EXPECT_EQ(10*64, roi.TotalPixels());
    EXPECT_TRUE(roi.ContainsPixel(0,0));
    EXPECT_TRUE(roi.ContainsPixel(9,0));
    EXPECT_FALSE(roi.ContainsPixel(10,0));
    EXPECT_TRUE(roi.ContainsPixel(0,63));
    EXPECT_FALSE(roi.ContainsPixel(0,64));
    EXPECT_TRUE(roi.ContainsPixel(9,63));
    EXPECT_FALSE(roi.ContainsPixel(9,64));
    EXPECT_FALSE(roi.ContainsPixel(10,63));
    EXPECT_FALSE(roi.ContainsPixel(10,64));

    roi.AddRectangle(0,0,10,64);
    EXPECT_FALSE(roi.Everything());
    EXPECT_EQ(10*64, roi.TotalPixels());
    EXPECT_TRUE(roi.ContainsPixel(0,0));
    EXPECT_TRUE(roi.ContainsPixel(9,0));
    EXPECT_FALSE(roi.ContainsPixel(10,0));
    EXPECT_TRUE(roi.ContainsPixel(0,63));
    EXPECT_FALSE(roi.ContainsPixel(0,64));

    roi.AddRectangle(5,32,10,64);
    EXPECT_FALSE(roi.Everything());
    EXPECT_EQ(10*64*2-5*32, roi.TotalPixels());
    EXPECT_TRUE(roi.ContainsPixel(0,0));
    EXPECT_TRUE(roi.ContainsPixel(9,0));
    EXPECT_FALSE(roi.ContainsPixel(10,0));
    EXPECT_TRUE(roi.ContainsPixel(0,63));
    EXPECT_FALSE(roi.ContainsPixel(0,64));
    EXPECT_TRUE(roi.ContainsPixel(9,63));
    EXPECT_TRUE(roi.ContainsPixel(9,64));
    EXPECT_TRUE(roi.ContainsPixel(10,63));
    EXPECT_TRUE(roi.ContainsPixel(10,64));

    EXPECT_TRUE(roi.ContainsPixel(14,95));
    EXPECT_FALSE(roi.ContainsPixel(14,96));
    EXPECT_FALSE(roi.ContainsPixel(15,95));
    EXPECT_FALSE(roi.ContainsPixel(15,96));


    SequelROI::Enumerator e(roi,0,roi.CountZMWs());
    std::pair<RowPixels, ColPixels> pix(0,0);

    EXPECT_EQ(0,e.Index());
    pix = e.UnitCellPixelOrigin();
    EXPECT_EQ(0,pix.first.Value());
    EXPECT_EQ(0,pix.second.Value());

    e+=31;
    EXPECT_EQ(31,e.Index());
    pix = e.UnitCellPixelOrigin();
    EXPECT_EQ(0,pix.first.Value());
    EXPECT_EQ(62,pix.second.Value());

    e++;
    EXPECT_EQ(32,e.Index());
    pix = e.UnitCellPixelOrigin();
    EXPECT_EQ(1,pix.first.Value());
    EXPECT_EQ(0,pix.second.Value());

    e+=4*32-1;
    EXPECT_EQ(32*5-1,e.Index());
    pix = e.UnitCellPixelOrigin();
    EXPECT_EQ(4,pix.first.Value());
    EXPECT_EQ(62,pix.second.Value());

    e++;
    EXPECT_EQ(32*5,e.Index());
    pix = e.UnitCellPixelOrigin();
    EXPECT_EQ(5,pix.first.Value());
    EXPECT_EQ(0,pix.second.Value());

    e+=32;
    EXPECT_EQ(32*6,e.Index());
    pix = e.UnitCellPixelOrigin();
    EXPECT_EQ(5,pix.first.Value());
    EXPECT_EQ(64,pix.second.Value());

    e+=15;
    EXPECT_EQ(32*6+15,e.Index());
    pix = e.UnitCellPixelOrigin();
    EXPECT_EQ(5,pix.first.Value());
    EXPECT_EQ(94,pix.second.Value());

    e++;
    EXPECT_EQ(32*6+16,e.Index());
    pix = e.UnitCellPixelOrigin();
    EXPECT_EQ(6,pix.first.Value());
    EXPECT_EQ(0,pix.second.Value());

    e+=48*4;
    EXPECT_EQ(32*5+48*5,e.Index());
    pix = e.UnitCellPixelOrigin();
    EXPECT_EQ(10,pix.first.Value());
    EXPECT_EQ(32,pix.second.Value());

    e+=32*5-1;
    EXPECT_TRUE(e);
    pix = e.UnitCellPixelOrigin();
    EXPECT_EQ(14,pix.first.Value());
    EXPECT_EQ(94,pix.second.Value());

    e++;
    EXPECT_FALSE(e);
}

TEST(SequelSparseROI,LotsOfRectangles)
{
    SequelSparseROI roi1(SequelSensorROI::SequelAlpha());
    for(uint32_t row=0;row<roi1.SensorROI().PhysicalRowMax(); row++)
    {
        for(uint32_t col=0;col<roi1.SensorROI().PhysicalColMax(); col+= 32)
        {
            roi1.AddRectangle(row, col, 1, 32);
        }
    }
    EXPECT_EQ(SequelSensorROI::SequelAlpha().TotalPixels(), roi1.TotalPixels());
    EXPECT_TRUE(roi1.Everything());
}

TEST(SequelSparseROI,JsonSpecification)
{
    SequelSparseROI roi1(SequelSensorROI::SequelAlpha());
    roi1.AddRectangle(0, 0, 10, 64);
    roi1.AddRectangle(0, 0, 10, 64);
    roi1.AddRectangle(5, 32, 10, 64);

    std::string config = "[[0, 0, 10, 64],[0, 0, 10, 64],[5, 32, 10, 64]]";
    auto json = PacBio::IPC::ParseJSON(config);
    SequelSparseROI roi2(json,SequelSensorROI::SequelAlpha());

    EXPECT_EQ(roi1, roi2);


    config = "[]";
    json = PacBio::IPC::ParseJSON(config);
    SequelSparseROI roi3(json,SequelSensorROI::SequelAlpha());
    EXPECT_EQ(0,roi3.CountZMWs());

    {
        const char* payload1 = R"([[ 0, 32, 1, 128]])";
        Json::Value json1 = PacBio::IPC::ParseJSON(payload1);
        SequelSparseROI seqRoiRef(SequelSensorROI::SequelAlpha());
        seqRoiRef.AddRectangle(0, 32, 1, 128);
        SequelSparseROI seqRoi(json1, SequelSensorROI::SequelAlpha());
        EXPECT_EQ(seqRoi, seqRoiRef);
    }

    {
        const char* payload2 = R"([2, 64, 3, 96])";
        Json::Value json2 = PacBio::IPC::ParseJSON(payload2);
        SequelSparseROI traceRoiRef(SequelSensorROI::SequelAlpha());
        traceRoiRef.AddRectangle(2, 64, 3, 96);
        SequelSparseROI traceRoi(json2, SequelSensorROI::SequelAlpha());
        EXPECT_EQ(traceRoi, traceRoiRef);
    }
}

TEST(SequelSparseROI,Condense)
{
    SequelSparseROI traceRoiRef(SequelSensorROI::SequelAlpha());
    EXPECT_EQ("null",PacBio::IPC::RenderJSON( traceRoiRef.Condense()));

    traceRoiRef.AddRectangle(102, 64, 3, 96);
    EXPECT_EQ("[[102,64,3,96]]",PacBio::IPC::RenderJSON( traceRoiRef.Condense()));

    traceRoiRef.AddRectangle(0, 0, 1, 32);
    EXPECT_EQ("[[0,0,1,32],[102,64,3,96]]",PacBio::IPC::RenderJSON( traceRoiRef.Condense()));

    traceRoiRef.AddRectangle(0, 0, 1, 64);
    EXPECT_EQ("[[0,0,1,64],[102,64,3,96]]",PacBio::IPC::RenderJSON( traceRoiRef.Condense()));

    traceRoiRef.AddRectangle(0, 0, 2, 32);
    EXPECT_EQ("[[0,0,1,64],[1,0,1,32],[102,64,3,96]]",PacBio::IPC::RenderJSON( traceRoiRef.Condense()));

    traceRoiRef.AddRectangle(4, 0, 1, 64);
    EXPECT_EQ("[[0,0,1,64],[1,0,1,32],[4,0,1,64],[102,64,3,96]]",PacBio::IPC::RenderJSON( traceRoiRef.Condense()));

    traceRoiRef.AddRectangle(5, 32, 1, 32);
    EXPECT_EQ("[[0,0,1,64],[1,0,1,32],[4,0,1,64],[5,32,1,32],[102,64,3,96]]",PacBio::IPC::RenderJSON( traceRoiRef.Condense()));

    traceRoiRef.AddRectangle(0, 320, 1, 32);
    EXPECT_EQ("[[0,0,1,64],[0,320,1,32],[1,0,1,32],[4,0,1,64],[5,32,1,32],[102,64,3,96]]",PacBio::IPC::RenderJSON( traceRoiRef.Condense()));
}

TEST(SequelSparseROI,Condense2)
{
    SequelSparseROI traceRoiRef(SequelSensorROI::SequelAlpha());
    EXPECT_EQ("null", PacBio::IPC::RenderJSON(traceRoiRef.Condense()));

    traceRoiRef.AddRectangle(0, 0, 8, 2016);
    EXPECT_EQ("[[0,0,8,2016]]", PacBio::IPC::RenderJSON(traceRoiRef.Condense()));

    traceRoiRef.AddRectangle(0, 0, 8, 2048);
    EXPECT_EQ("[[0,0,8,2048]]", PacBio::IPC::RenderJSON(traceRoiRef.Condense()));

    traceRoiRef.AddRectangle(0, 0, 1144, 2048);
    EXPECT_EQ("[[0,0,1144,2048]]", PacBio::IPC::RenderJSON(traceRoiRef.Condense()));

    traceRoiRef.PostAddRectangle();
    EXPECT_EQ("[[0,0,1144,2048]]", PacBio::IPC::RenderJSON(traceRoiRef.GetJson()));
}

TEST(SequelRectangularROI,SpiderChipClass)
{
    auto sensorROI = SequelSensorROI::Spider();
    EXPECT_EQ(0, sensorROI.GetTileOffsetOfPixel(0, 0));
    EXPECT_EQ(1, sensorROI.GetTileOffsetOfPixel(0, 32));

    SequelRectangularROI roi0(1, 0, 2, 32, sensorROI);

    EXPECT_EQ(91, roi0.SensorROI().GetTileOffsetOfPixel(1, 1));
    EXPECT_EQ(91, roi0.SensorROI().GetTileOffsetOfPixel(1, 2));
    EXPECT_TRUE(roi0.ContainsPixel(RowPixels(1), ColPixels(1)));
    EXPECT_TRUE(roi0.ContainsPixel(RowPixels(1), ColPixels(2)));
    EXPECT_TRUE(roi0.ContainsPixel(RowPixels(1), ColPixels(3)));
    EXPECT_TRUE(roi0.ContainsPixel(RowPixels(2), ColPixels(1)));
    EXPECT_TRUE(roi0.ContainsPixel(RowPixels(2), ColPixels(2)));
    EXPECT_TRUE(roi0.ContainsPixel(RowPixels(2), ColPixels(3)));
    EXPECT_FALSE(roi0.ContainsPixel(RowPixels(1), ColPixels(33)));
    EXPECT_FALSE(roi0.ContainsPixel(RowPixels(2), ColPixels(33)));
    EXPECT_FALSE(roi0.ContainsPixel(RowPixels(0), ColPixels(1)));
    EXPECT_FALSE(roi0.ContainsPixel(RowPixels(3), ColPixels(1)));

#if 0
    // if the ROI class supports non-modulo 32 columnar settings, this might work
    SequelRectangularROI roi1(1,1,2,3,layout.GetSensorROI());
    EXPECT_EQ(0,layout.GetSensorROI().GetTileOffsetOfPixel(0,0));
    EXPECT_EQ(1,layout.GetSensorROI().GetTileOffsetOfPixel(0,32));

    EXPECT_EQ(0,roi1.SensorROI().GetTileOffsetOfPixel(1,1));
    EXPECT_EQ(0,roi1.SensorROI().GetTileOffsetOfPixel(1,2));
    EXPECT_TRUE(roi1.ContainsPixel(RowPixels(1), ColPixels(1)));
    EXPECT_TRUE(roi1.ContainsPixel(RowPixels(1), ColPixels(2)));
    EXPECT_TRUE(roi1.ContainsPixel(RowPixels(1), ColPixels(3)));
    EXPECT_TRUE(roi1.ContainsPixel(RowPixels(2), ColPixels(1)));
    EXPECT_TRUE(roi1.ContainsPixel(RowPixels(2), ColPixels(2)));
    EXPECT_TRUE(roi1.ContainsPixel(RowPixels(2), ColPixels(3)));
    EXPECT_FALSE(roi1.ContainsPixel(RowPixels(1), ColPixels(4)));
    EXPECT_FALSE(roi1.ContainsPixel(RowPixels(2), ColPixels(4)));
    EXPECT_FALSE(roi1.ContainsPixel(RowPixels(0), ColPixels(1)));
    EXPECT_FALSE(roi1.ContainsPixel(RowPixels(3), ColPixels(1)));
#endif
}
