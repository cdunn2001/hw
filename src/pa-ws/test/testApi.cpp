#include <pa-ws/api/SocketObject.h>

#include <pacbio/ipc/JSON.h>

#include <gtest/gtest.h>

using namespace PacBio::API;

TEST(API,ONE)
{

    const std::string json1 = R"JSON(
    {
        "movieMaxFrames" : 100,
        "movieMaxSeconds": 60.0,
        "movieNumber": 123,
        "bazUrl" : "http://localhost:23632/storage/m123456/mine.baz",
        "logUrl" : "http://localhost:23632/storage/m123456/log.txt",
        "logLevel" : "DEBUG",
        "chiplayout" : "Minesweeper",
        "crosstalkFilter" :
        [
            [ 0.1,  0.2, 0.3 ],
            [ -0.1, 0.8, -0.2],
            [ 0.4,  0.5, 0.6 ]
        ],
        "traceFileRoi":
        [ [ 0, 0, 13, 32 ] ]
    }
    )JSON";

    const Json::Value json = 
        PacBio::IPC::ParseJSON(json1);

    SocketBasecallerObject sbc(json);

    EXPECT_EQ(sbc.movieMaxFrames, 100);
    EXPECT_EQ(0.2, sbc.crosstalkFilter[0][1]);
    EXPECT_EQ(13, sbc.traceFileRoi[0][2]);
    EXPECT_EQ("discard:",sbc.traceFileUrl);
    EXPECT_EQ("http://localhost:23632/storage/m123456/log.txt",sbc.logUrl);


    // typo
    const std::string json2 = R"JSON(
    {
        "movieMaxFrmes" : 100,
    }
    )JSON";

    EXPECT_THROW( SocketBasecallerObject x( PacBio::IPC::ParseJSON(json2)), std::exception);

    // bad structure (passing 1D array instead of 2D array)
    const std::string json3 = R"JSON(
    {
         "traceFileRoi": [ 0, 0, 13, 32 ]
    }
    )JSON";

    EXPECT_THROW( SocketBasecallerObject x( PacBio::IPC::ParseJSON(json3)), std::exception);
}
