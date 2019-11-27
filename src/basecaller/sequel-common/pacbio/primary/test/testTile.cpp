#include <stdexcept>
#include <random>

#include "gtest/gtest.h"

#include <pacbio/primary/Tile.h>

using namespace std;


using namespace PacBio::Primary;

TEST(Tile,Patterns)
{
    std::unique_ptr<Tile> tile(Tile::make());
    tile->SetPattern(42);
    EXPECT_NO_THROW(tile->CheckPattern(42));
    tile->data[3] = 1; // corrupt the tile, to force a bad pattern
    EXPECT_THROW(tile->CheckPattern(42), std::runtime_error);
}

TEST(Tile,Alignment)
{
    unique_ptr<Tile> pTile(Tile::make());
    EXPECT_EQ(0, ((uint64_t)pTile.get()) & 4095) << "4K aligned heap. Pointer returned was " << (void*)pTile.get() ;
}

TEST(Tile,ArrayAlignment)
{
    unique_ptr<Tile[]> aTiles( Tile::make(4));
    EXPECT_EQ(0, ((uint64_t)aTiles.get()) & 4095) << "4K aligned heap. Pointer returned was " << (void*)aTiles.get() ;
}

TEST(Tile,Header)
{
    unique_ptr<Tile> aTile( Tile::make());
    aTile->CreateHeaderTile(1,2,3);
    EXPECT_EQ(1,aTile->FirstFrameIndex());
    EXPECT_EQ(2,aTile->FirstFrameTimeStamp());
    EXPECT_EQ(3,aTile->FirstFrameConfig());
    EXPECT_EQ(512,aTile->LastFrameIndex());
    EXPECT_LT(2,aTile->LastFrameTimeStamp());
    EXPECT_EQ(3,aTile->LastFrameConfig());
}

TEST(Tile,ErrorFrames)
{
    unique_ptr<Tile> aTile(Tile::make());

    aTile->CreateHeaderTile(0,0,0);
    auto badFrames0 = aTile->ErroredFrames();
    EXPECT_EQ(0,badFrames0.size());

    aTile->data[1024] = 0x83;
    aTile->data[1024 + 128/8] = 1;
    aTile->data[1087] = 0x80;
    auto badFrames = aTile->ErroredFrames();
    EXPECT_EQ(5,badFrames.size());
    EXPECT_EQ(0,badFrames[0]);
    EXPECT_EQ(1,badFrames[1]);
    EXPECT_EQ(7,badFrames[2]);
    EXPECT_EQ(128,badFrames[3]);
    EXPECT_EQ(511,badFrames[4]);
}


TEST(Tile,Pattern3)
{
    EXPECT_EQ(0, Tile::GetPattern3Pixel(0, 0));
    EXPECT_EQ(10, Tile::GetPattern3Pixel(1, 0));
    EXPECT_EQ(2, Tile::GetPattern3Pixel(0, 2));
    EXPECT_EQ(33, Tile::GetPattern3Pixel(3, 3));

    unique_ptr<Tile> aTile(Tile::make());
    std::srand(0);// don't want a real random seed.
    for (int i=0;i<100;i++)
    {
        uint32_t pixelBase = std::rand() & 0x1FFFFFF & ~31;
        uint32_t frameBase = std::rand() & ~511;

        aTile->SetPattern3(pixelBase, frameBase);
        for(uint32_t ipixel=0;ipixel<Tile::NumPixels;ipixel++)
        {
            for (uint32_t iframe = 0; iframe < Tile::NumFrames ; iframe++)
            {
                ASSERT_EQ(Tile::GetPattern3Pixel(pixelBase+ipixel, frameBase+iframe), aTile->GetPixel(ipixel, iframe))
                                    << "(" << pixelBase << "," << frameBase << ") + (" << ipixel << "," << iframe << ") #" << i;
            }
        }
    }
}
