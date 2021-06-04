#include <pa-ws/api/SocketObject.h>

#include <pacbio/ipc/JSON.h>

#include <gtest/gtest.h>

using namespace PacBio::API;

TEST(API,ONE)
{

    const std::string json1 = R"JSON(
    {
        "movie_max_frames" : 100,
        "movie_max_time": 60.0,
        "movie_number": 123,
        "baz_url" : "http://localhost:23632/storage/m123456/mine.baz",
        "log_url" : "http://localhost:23632/storage/m123456/log.txt",
        "log_level" : "DEBUG",
        "chiplayout" : "Minesweeper",
        "crosstalk_filter" :
        [
            [ 0.1,  0.2, 0.3 ],
            [ -0.1, 0.8, -0.2],
            [ 0.4,  0.5, 0.6 ]
        ],
        "trace_file_roi":
        [ [ 0, 0, 13, 32 ] ]
    }
    )JSON";

    const Json::Value json = 
        PacBio::IPC::ParseJSON(json1);

    SocketBasecallerObject sbc(json);

    EXPECT_EQ(sbc.movie_max_frames, 100);
    EXPECT_EQ(0.2, sbc.crosstalk_filter[0][1]);
    EXPECT_EQ(13, sbc.trace_file_roi[0][2]);
    EXPECT_EQ("discard:",sbc.trace_file_url);
    EXPECT_EQ("http://localhost:23632/storage/m123456/log.txt",sbc.log_url);


    // typo
    const std::string json2 = R"JSON(
    {
        "movie_max_frmes" : 100,
    }
    )JSON";

    EXPECT_THROW( SocketBasecallerObject x( PacBio::IPC::ParseJSON(json2)), std::exception);

    // bad structure (passing 1D array instead of 2D array)
    const std::string json3 = R"JSON(
    {
         "trace_file_roi": [ 0, 0, 13, 32 ]
    }
    )JSON";

    EXPECT_THROW( SocketBasecallerObject x( PacBio::IPC::ParseJSON(json3)), std::exception);
}
