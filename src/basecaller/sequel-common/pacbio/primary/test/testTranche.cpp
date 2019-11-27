#include <pacbio/primary/Tranche.h>
#include "gtest/gtest.h"
#include <pacbio/dev/gtest-extras.h>
#include <pacbio/ipc/Message.h>

using namespace std;

#include <stdexcept>

using namespace PacBio::Primary;

TEST(Tranche,Constructor)
{
    Tranche t;
#ifdef  SPIDER_TRANCHES
    t.Create2C2ATitle();
#endif
    TrancheTitle& title = t.Title();
    title.AddOffset(0);
    title.FrameIndexStart(12345678901234567890ULL);
    title.TimeStampStart(8765432109876543210ULL);
    title.ConfigWord(2);
    title.FrameCount(3);
    title.ZmwIndex(4);
    title.ZmwNumber(0x00200024);
    title.PixelLane(5);
    title.MicOffset(1);
    title.SuperChunkIndex(123);
    title.StopStatus(ITranche::StopStatusType::NORMAL);
    title.TimeStampDelta(defaultExposureUsec);

    t.StopStatus(title.StopStatus());

    EXPECT_EQ(nullptr, t.TraceData());
    EXPECT_EQ(0, t.order);
    EXPECT_EQ(2, t.ConfigWord());
    EXPECT_EQ(3, t.FrameCount());
    EXPECT_EQ(12345678901234567890ULL, t.FrameIndex());
    EXPECT_EQ(8765432109876543210ULL, t.TimeStampStart());
    EXPECT_EQ(4, t.ZmwIndex());
    EXPECT_EQ(5, t.ZmwLaneIndex());
    EXPECT_EQ(0x00200024, t.ZmwNumber());
    EXPECT_EQ(Tranche::MessageType::Unknown, t.Type());
    EXPECT_EQ(ITranche::StopStatusType::NORMAL, title.StopStatus());
    EXPECT_EQ(ITranche::StopStatusType::NORMAL, t.StopStatus());
    EXPECT_EQ(defaultExposureUsec, t.TimeStampDelta());

    ITranche& it = t;
    EXPECT_EQ(nullptr, it.TraceData());
    EXPECT_EQ(123, it.SuperChunkIndex());
    EXPECT_EQ(2, it.ConfigWord());
    EXPECT_EQ(3, it.FrameCount());
    EXPECT_EQ(12345678901234567890ULL, it.FrameIndex());
    EXPECT_EQ(8765432109876543210ULL, it.TimeStampStart());
    EXPECT_EQ(5, it.ZmwLaneIndex());
    EXPECT_EQ(0x00200024, it.ZmwNumber());
    EXPECT_EQ(ITranche::StopStatusType::NORMAL, title.StopStatus());
    EXPECT_EQ(ITranche::StopStatusType::NORMAL, it.StopStatus());
    EXPECT_EQ(defaultExposureUsec, it.TimeStampDelta());

    std::vector<Tile*> alltiles;
    std::unique_ptr<Tile> tile(Tile::make());
    tile->SetPattern2(0);
    alltiles.push_back(tile.get());
    t.AssignTileDataPointers(alltiles);
    std::vector<int16_t> lots(1000000,0);
    ITranche::Pointer ptr;
    it.CopyTraceData(lots.data(),ptr,5 /* frames */);
    EXPECT_EQ(0,lots[0]);
    EXPECT_EQ(1,lots[1]);
    EXPECT_EQ(10,lots[32]);
    EXPECT_EQ(11,lots[33]);
    EXPECT_EQ(20,lots[64]);
    EXPECT_EQ(21,lots[65]);
    EXPECT_EQ(30,lots[96]);
    EXPECT_EQ(31,lots[97]);
    EXPECT_EQ(40,lots[128]);
    EXPECT_EQ(41,lots[129]);
    EXPECT_EQ(0,lots[160]);
    EXPECT_EQ(5,ptr.frame);
    EXPECT_EQ(0,ptr.tile);
    EXPECT_EQ(13880288255492001792ULL,t.Checksum());

    EXPECT_THROW(it.CopyTraceData(lots.data(),ptr,50000 /* frames */),std::exception);

}

static uint32_t sentDeeds = 0;

static void thisSendDeed(const PacBio::IPC::Deed& /*deed*/)
{
    sentDeeds++;
//    std::cout << deed << std::endl;
}

TEST(Tranche,SharedTitles)
{
    sentDeeds = 0;
    std::shared_ptr<TrancheTitle> title(new TrancheTitle, [this](TrancheTitle* p) {
        PacBio::IPC::Deed deed("deed", *p);
        thisSendDeed(deed);
        delete p;
    });
    EXPECT_EQ(1,title.use_count());

    std::shared_ptr<TrancheTitle> a = title;
    EXPECT_EQ(2,title.use_count());

    {
        std::shared_ptr<TrancheTitle> b = title;
        EXPECT_EQ(3,title.use_count());
    }
    EXPECT_EQ(2,title.use_count());
    a.reset();
    EXPECT_EQ(1,title.use_count());
    EXPECT_EQ(0,sentDeeds);
    title.reset();
    EXPECT_EQ(0,title.use_count());
    EXPECT_EQ(1,sentDeeds);
}

TEST(Tranche,SharedTitles2)
{
#ifdef  SPIDER_TRANCHES
    sentDeeds = 0;
    std::shared_ptr<TrancheTitle> title(new TrancheTitle, [this](TrancheTitle* p) {
        PacBio::IPC::Deed deed("deed", *p);
        thisSendDeed(deed);
        delete p;
    });
    EXPECT_EQ(1, title.use_count());

    {
        Tranche x;
        x.SetSharedTitle(title, 0, Tranche::PixelFormat::Format2C2A);
        EXPECT_EQ(2, title.use_count());

        {
            Tranche y;
            y.SetSharedTitle(title, 0, Tranche::PixelFormat::Format2C2A);
            EXPECT_EQ(3, title.use_count());
        }
        EXPECT_EQ(2, title.use_count());
    }
    EXPECT_EQ(1,title.use_count());
    EXPECT_EQ(0,sentDeeds);
    title.reset();
    EXPECT_EQ(0,title.use_count());
    EXPECT_EQ(1,sentDeeds);
#endif
}

TEST(Tranche,SpiderCopyData)
{
#ifdef  SPIDER_TRANCHES
    std::shared_ptr<TrancheTitle> title(new TrancheTitle);
    title->AddOffset(0);
    title->FrameIndexStart(12345678901234567890ULL);
    title->TimeStampStart(8765432109876543210ULL);
    title->ConfigWord(2);
    title->FrameCount(3);
    title->ZmwIndex(4);
    title->ZmwNumber(0x00200024);
    title->PixelLane(5);
    title->MicOffset(1);
    title->SuperChunkIndex(123);
    title->StopStatus(ITranche::StopStatusType::NORMAL);
    title->TimeStampDelta(defaultExposureUsec);

    Tranche t1;
    Tranche t2;

    t1.SetSharedTitle(title,0,Tranche::PixelFormat::Format1C4A_RT);
    t2.SetSharedTitle(title,1,Tranche::PixelFormat::Format1C4A_RT);

    EXPECT_EQ(12345678901234567890ULL,t1.FrameIndex());
    EXPECT_EQ(12345678901234567890ULL,t2.FrameIndex());
    EXPECT_EQ(4 ,t1.ZmwIndex());
    EXPECT_EQ(20,t2.ZmwIndex());
    EXPECT_EQ(0x00200024 ,t1.ZmwNumber());
    EXPECT_EQ(0x00200034 ,t2.ZmwNumber());
    EXPECT_EQ(5, t1.PixelLaneIndex());
    EXPECT_EQ(5, t2.PixelLaneIndex());
    EXPECT_EQ(10,t1.ZmwLaneIndex());
    EXPECT_EQ(11,t2.ZmwLaneIndex());
    EXPECT_EQ(0, t1.Subset());
    EXPECT_EQ(1, t2.Subset());

    EXPECT_EQ(Tranche::PixelFormat::Format1C4A_RT, t1.Format());
    EXPECT_EQ(Tranche::PixelFormat::Format1C4A_RT, t2.Format());

    // check setter, getter round-trip
    uint32_t laneIndex = 1234;
    t1.ZmwLaneIndex(laneIndex);
    EXPECT_EQ(617,t1.PixelLaneIndex());
    EXPECT_EQ(laneIndex,t1.ZmwLaneIndex());

    uint32_t zmwNumber = 0x12345678;
    t1.ZmwNumber(zmwNumber);
    EXPECT_EQ(zmwNumber, t1.ZmwNumber());

    uint32_t zmwIndex = 567890;
    t1.ZmwIndex(zmwIndex);
    EXPECT_EQ(zmwIndex, t1.ZmwIndex());


    EXPECT_EQ(617,t2.PixelLaneIndex());
    t2.ZmwLaneIndex(laneIndex+1);
    EXPECT_EQ(laneIndex+1,t2.ZmwLaneIndex());

    t2.ZmwNumber(zmwNumber+16);
    EXPECT_EQ(zmwNumber+16, t2.ZmwNumber());

    t2.ZmwIndex(zmwIndex+16);
    EXPECT_EQ(zmwIndex+16, t2.ZmwIndex());

    ITranche& it1 = t1;
    std::vector<Tile*> alltiles;
    std::unique_ptr<Tile> tile0(Tile::make());
    std::unique_ptr<Tile> tile1(Tile::make());
    tile0->SetPattern2(0); // value = pixel + iframe * 10
    tile1->SetPattern2(5120); // value = pixel + iframe * 10
    alltiles.push_back(tile0.get());
    alltiles.push_back(tile1.get());
    t1.AssignTileDataPointers(alltiles);
    t2.AssignTileDataPointers(alltiles);

    EXPECT_EQ(0,tile0->GetPixel(0,0));
    EXPECT_EQ(1,tile0->GetPixel(1,0));
    EXPECT_EQ(2,tile0->GetPixel(2,0));
    EXPECT_EQ(14,tile0->GetPixel(14,0));
    EXPECT_EQ(15,tile0->GetPixel(15,0));
    EXPECT_EQ(16,tile0->GetPixel(16,0));
    EXPECT_EQ(10,tile0->GetPixel(0,1));

    std::vector<int16_t> lots(1000000,0);
    ITranche::Pointer ptr1;
    it1.CopyTraceData(lots.data(),ptr1,5 /* frames */);
    EXPECT_EQ(5,ptr1.frame);
    EXPECT_EQ(0,ptr1.tile);

    EXPECT_EQ(0,lots[0]);
    EXPECT_EQ(1,lots[1]);
    EXPECT_EQ(2,lots[2]);
    EXPECT_EQ(7,lots[7]);
    EXPECT_EQ(8,lots[8]);
    EXPECT_EQ(14,lots[14]);
    EXPECT_EQ(15,lots[15]);
    EXPECT_EQ(0,lots[16]);
    EXPECT_EQ(15,lots[31]);
    EXPECT_EQ(10+0,lots[1*32 + 0]); // frame 1
    EXPECT_EQ(10+1,lots[1*32 + 1]);
    EXPECT_EQ(20+0,lots[2*32 + 0]); // frame 2
    EXPECT_EQ(20+1,lots[2*32 + 1]);
    EXPECT_EQ(30+0,lots[3*32 + 0]); // frame 3
    EXPECT_EQ(30+1,lots[3*32 + 1]);
    EXPECT_EQ(40+0,lots[4*32 + 0]); // frame 4
    EXPECT_EQ(40+1,lots[4*32 + 1]);

//    EXPECT_EQ(13880288255492001792ULL,t1.Checksum());
    EXPECT_EQ(9403905804232368128ULL,t1.Checksum());

    ITranche& itUpper = t2;
    ITranche::Pointer ptr2;
    EXPECT_EQ(0,ptr2.frame);
    EXPECT_EQ(0,ptr2.tile);
    itUpper.CopyTraceData(lots.data(),ptr2,1024 /* frames */);
    EXPECT_EQ(0,ptr2.frame);
    EXPECT_EQ(2,ptr2.tile);

    EXPECT_EQ(16,lots[0]);
    EXPECT_EQ(17,lots[1]);
    EXPECT_EQ(31,lots[15]);
    EXPECT_EQ(16,lots[16]);
    EXPECT_EQ(31,lots[31]);
    EXPECT_EQ(10 + 16,lots[1*32 + 0]); // frame 1
    EXPECT_EQ(10 + 17,lots[1*32 + 1]);
    EXPECT_EQ(20 + 16,lots[2*32 + 0]); // frame 2
    EXPECT_EQ(20 + 17,lots[2*32 + 1]);
    EXPECT_EQ(30 + 16,lots[3*32 + 0]); // frame 3
    EXPECT_EQ(30 + 17,lots[3*32 + 1]);
    EXPECT_EQ(40 + 16,lots[4*32 + 0]); // frame 4
    EXPECT_EQ(40 + 17,lots[4*32 + 1]);

    EXPECT_EQ(5120 + 16,lots[512*32 + 0]); // frame 512
    EXPECT_EQ(5120 + 17,lots[512*32 + 1]);

    EXPECT_EQ(10230 + 16,lots[1023*32 + 0]); // frame 1023
    EXPECT_EQ(10230 + 17,lots[1023*32 + 1]);
#endif
}
