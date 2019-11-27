//
// Created by mlakata on 11/13/18.
//

#include <gtest/gtest.h>
#include <pacbio/primary/EventObject.h>
#include <pacbio/primary/LaserPowerChange.h>

using namespace PacBio::Primary;


TEST(LaserPowerChange,Basic)
{
    std::array<uint64_t,2> frameInterval;
    std::array<float,2> topPower;
    std::array<float,2> bottomPower;

    frameInterval[0] = 21;
    frameInterval[1] = 29;
    topPower[0] = 1.0;
    topPower[1] = 2.0;
    bottomPower[0] = 3.0;
    bottomPower[1] = -1.0;
    LaserPowerChange lpc(123456789.0, frameInterval, topPower, bottomPower);

    EXPECT_DOUBLE_EQ(123456789.0, lpc.TimeStamp());
    EXPECT_FLOAT_EQ(21, lpc.FrameInterval().Lower());
    EXPECT_FLOAT_EQ(29, lpc.FrameInterval().Upper());
    EXPECT_EQ(21, lpc.StartFrame());
    EXPECT_EQ(29, lpc.StopFrame());
    EXPECT_FLOAT_EQ(1.0, lpc.TopPower()[0]);
    EXPECT_FLOAT_EQ(2.0, lpc.TopPower()[1]);
    EXPECT_FLOAT_EQ(-1.0, lpc.BottomPower()[0]); // nb, the 3.0 is NOT expected!
    EXPECT_FLOAT_EQ(-1.0, lpc.BottomPower()[1]);

    lpc.TranslateFrameInterval(20);
    EXPECT_EQ(1, lpc.StartFrame());
    EXPECT_EQ(9, lpc.StopFrame());

    lpc.TranslateFrameInterval(100);
    EXPECT_EQ(0, lpc.StartFrame());
    EXPECT_EQ(0, lpc.StopFrame());
}

TEST(LaserPowerChange,EventObjectImport)
{
    EventObject eo;
    eo.eventType = EventObject::EventType::laserpower;
    eo.timeStamp = "2018-11-12T12:00:00Z";
    eo.startFrame = 123;
    eo.stopFrame  = 456;
    LaserPowerObject lpo1;
    lpo1.name = LaserPowerObject::LaserName::topLaser;
    lpo1.startPower_mW = 666.0;
    lpo1.stopPower_mW = 999.0;
    eo.lasers[0] = lpo1;
    // no bottom laser here on purpose
    eo.token = "abc-def";

    LaserPowerChange lpc(eo);

    EXPECT_EQ(123,lpc.StartFrame());
    EXPECT_EQ(456,lpc.StopFrame());
    EXPECT_DOUBLE_EQ(eo.timestamp_epoch(),lpc.TimeStamp());
    EXPECT_FLOAT_EQ(666.0, lpc.TopPower()[0]);
    EXPECT_FLOAT_EQ(999.0, lpc.TopPower()[1]);
    EXPECT_FLOAT_EQ(-1.0, lpc.BottomPower()[0]);
    EXPECT_FLOAT_EQ(-1.0, lpc.BottomPower()[1]);

    eo.lasers[0].name = LaserPowerObject::LaserName::unknown;
    EXPECT_THROW(LaserPowerChange lpc1(eo), std::exception);

}
