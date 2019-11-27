//
// Created by mlakata on 6/8/16.
//

#include <gtest/gtest.h>

#include <pacbio/dev/gtest-extras.h>
#include <pacbio/dev/AutoTimer.h>
#include <pacbio/primary/Event.h>
#include <pacbio/ipc/JSON.h>
#include <pacbio/PBException.h>
#include <pacbio/logging/Logger.h>
#include <pacbio/utilities/ISO8601.h>

using namespace PacBio::Primary;


TEST(Event,UnitTestJson)
{
    // .. requirement:: Event object properties

    TEST_COUT << "Hello\n";

    Event event("primary.test.events.test","Testing");

    auto now = PacBio::Utilities::ISO8601::TimeString();

    std::string s = event.RenderJSON();

    Json::Value v;
    std::stringstream ss;
    ss << s;
    ss >> v;

    EXPECT_EQ("primary.test.events.test",v["id"].asString()) << s;
    EXPECT_EQ("Testing",v["name"].asString()) << s;

    double delta = PacBio::Utilities::ISO8601::EpochTime(now) -
                   PacBio::Utilities::ISO8601::EpochTime(v["whenChanged"].asString());
    EXPECT_LT(std::abs(delta),1) << s;
}
