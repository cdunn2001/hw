#include <gtest/gtest.h>
#include <pa-ws/PaWsConfig.h>

using namespace PacBio::Primary::PaWs;
using namespace testing;
using namespace PacBio::Sensor;

TEST(PaWsConfig,Basic)
{
    Json::Value json;
    json["socketIds"] = Json::Value(Json::arrayValue);
    PaWsConfig conf1(json);
    EXPECT_FALSE(conf1.logHttpGets);
}

TEST(PaWsConfig,Factory)
{
    {
        Json::Value json;
        json["platform"] = "Sequel2Lvl1";
        PaWsConfig conf1(json);
        EXPECT_EQ(conf1.platform, Platform::Sequel2Lvl1);
        FactoryConfig(&conf1);
        const std::vector<std::string> expected{"1"};
        EXPECT_EQ(conf1.socketIds, std::vector<std::string>{"1"});
    }
    {
        Json::Value json;
        json["platform"] = "Sequel2Lvl2";
        PaWsConfig conf1(json);
        EXPECT_EQ(conf1.platform, Platform::Sequel2Lvl2);
        FactoryConfig(&conf1);
        const std::vector<std::string> expected{"1"};
        EXPECT_EQ(conf1.socketIds, std::vector<std::string>{"1"});
    }
    {
        Json::Value json;
        json["platform"] = "Kestrel";
        PaWsConfig conf1(json);
        EXPECT_EQ(conf1.platform, Platform::Kestrel);
        FactoryConfig(&conf1);
        const std::vector<std::string> expected{"1", "2", "3", "4"};
        EXPECT_EQ(conf1.socketIds, expected);
    }
}
