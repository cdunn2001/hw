#include <gtest/gtest.h>
#include <pa-ws/PaWsConfig.h>

using namespace PacBio::Primary::PaWs;
using namespace testing;
using namespace PacBio::Sensor;

TEST(PaWsConfig,Basic)
{
    Json::Value json;
    json["socketIds"] = Json::Value(Json::arrayValue);
    //std::vector<std::string>{"1", "2"};
    PaWsConfig conf1(json);
    EXPECT_FALSE(conf1.logHttpGets);
    //EXPECT_EQ(2,conf1.lastSocket);
}

TEST(PaWsConfig,Factory)
{
    Json::Value json;
    json["platform"] = "Sequel2Lvl2";
    PaWsConfig conf1(json);

    conf1.platform = Platform::Sequel2Lvl1;
    FactoryConfig(&conf1);
    //EXPECT_DOUBLE_EQ(conf1.lastSocket,1);

    conf1.platform = Platform::Mongo;
    FactoryConfig(&conf1);
    //EXPECT_DOUBLE_EQ(conf1.lastSocket,4);

    conf1.platform = Platform::Kestrel;
    FactoryConfig(&conf1);
    //EXPECT_DOUBLE_EQ(conf1.lastSocket,4);
}
