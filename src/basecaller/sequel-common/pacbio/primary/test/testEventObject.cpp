//
// Created by mlakata on 11/13/18.
//

#include <gtest/gtest.h>
#include <pacbio/primary/EventObject.h>

using namespace PacBio::Primary;

TEST(EventObject,A)
{
    EventObject eo;
    eo.timeStamp = "2018-11-12T12:00:00Z";
    EXPECT_FLOAT_EQ(1.5420239e+09, eo.timestamp_epoch());
    EXPECT_EQ("2018-11-12T12:00:00Z",eo.timeStamp());

    eo.timestamp_epoch = 0.0;
    EXPECT_EQ("1970-01-01T00:00:00.000Z", eo.timeStamp());
}

TEST(EventObject,B)
{
    EventObject eo;
    eo.Load(R"( { "timeStamp" : "2018-11-12T12:00:00Z" } )");
    EXPECT_FLOAT_EQ(1.5420239e+09, eo.timestamp_epoch());
    EXPECT_EQ("2018-11-12T12:00:00Z",eo.timeStamp());
}

TEST(EventObject,C)
{
    EventObject eo;
    eo.Load(R"( { "timeStamp" : "2018-11-12T12:00:00Z" , "timestamp_epoch":1.5420239e+09 } )");
    EXPECT_FLOAT_EQ(1.5420239e+09, eo.timestamp_epoch());
    EXPECT_EQ("2018-11-12T12:00:00Z",eo.timeStamp());
}

