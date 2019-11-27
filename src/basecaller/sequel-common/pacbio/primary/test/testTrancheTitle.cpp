//
// Created by mlakata on 4/24/15.
//

#include <gtest/gtest.h>

#include <pacbio/primary/TrancheTitle.h>
#include <pacbio/primary/Tile.h>

using namespace PacBio::Primary;


TEST(TrancheTitle,Accessors)
{
    TrancheTitle title;

    title.AddOffset(0);
    title.FrameIndexStart(0);
    title.TimeStampStart( 1);
    title.ConfigWord(2);
    title.FrameCount(3);
    title.ZmwIndex(4);
    title.MicOffset(1);
    title.PixelLane(1000);
    title.SuperChunkIndex(2000);
    title.ZmwNumber(0x00200024);
    title.StopStatus(ITranche::StopStatusType::BAD_DATA_QUALITY);


    uint8_t buffer[1024];
    uint32_t l = title.Serialize(buffer, sizeof(buffer));
    EXPECT_EQ(16 + // TitleBase fixed size
    8 + // one offset
        2*sizeof(uint64_t) + 9*sizeof(uint32_t),// TrancheTitle specific ()
    l);

    TrancheTitle titleB;
    uint32_t m = titleB.Deserialize(buffer, l);
    EXPECT_EQ(m, l);
    EXPECT_EQ(1,titleB.GetNumOffsets());
    EXPECT_EQ(sizeof(PacBio::Primary::Tile), titleB.GetElementSize());
    EXPECT_EQ(0, titleB.FrameIndexStart());
    EXPECT_EQ(1, titleB.TimeStampStart());
    EXPECT_EQ(2, titleB.ConfigWord());
    EXPECT_EQ(3, titleB.FrameCount());
    EXPECT_EQ(4, titleB.ZmwIndex());
    EXPECT_EQ(1, titleB.MicOffset());
    EXPECT_EQ(1000, titleB.PixelLane());
    EXPECT_EQ(2000, titleB.SuperChunkIndex());
    EXPECT_EQ(0x00200024, title.ZmwNumber());
    EXPECT_EQ(ITranche::StopStatusType::BAD_DATA_QUALITY, titleB.StopStatus());
}

