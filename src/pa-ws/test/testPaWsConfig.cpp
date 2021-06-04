#include <gtest/gtest.h>
#include <pa-ws/PaWsConfig.h>

using namespace PacBio::Primary::PaWs;
using namespace testing;
using namespace PacBio::Sensor;

TEST(PaWsConfig,Basic)
{
    Json::Value json;
    json["numSRAs"] = 2;
    PaWsConfig conf1(json);
    EXPECT_FALSE(conf1.logHttpGets);
    EXPECT_EQ(2,conf1.numSRAs);
}

TEST(PaWsConfig,Factory)
{
    Json::Value json;
    json["platform"] = "Sequel2Lvl2";
    PaWsConfig conf1(json);

    conf1.platform = Platform::Sequel2Lvl1;
    FactoryConfig(&conf1);
    EXPECT_DOUBLE_EQ(conf1.numSRAs,1);

    conf1.platform = Platform::Mongo;
    FactoryConfig(&conf1);
    EXPECT_DOUBLE_EQ(conf1.numSRAs,4);

    conf1.platform = Platform::Kestrel;
    FactoryConfig(&conf1);
    EXPECT_DOUBLE_EQ(conf1.numSRAs,4);
}
