#include <common/LaneArray.h>

#include <gtest/gtest.h>

namespace PacBio {
namespace Mongo {

TEST(TestLaneMask, Ops)
{
    LaneMask<laneSize> x{true};
    EXPECT_TRUE(all(x));
    x[0] = false;
    EXPECT_FALSE(all(x));
    EXPECT_TRUE(any(x));
    EXPECT_FALSE(none(x));

    LaneMask<laneSize> y{true};
    EXPECT_TRUE(all(x | y));
    EXPECT_FALSE(all(x & y));

    EXPECT_TRUE(none(!y));
}

TEST(TestLaneArray, Ops)
{
    LaneArray<short, laneSize> one{1};
    LaneArray<short, laneSize> zero{0};

    EXPECT_TRUE(all(one >= zero));
    EXPECT_TRUE(all((one * zero) == zero));
    EXPECT_TRUE(none(((one + one) * zero) > one));
    EXPECT_TRUE(any((one - one) == zero));
}

}}  // namespace PacBio::Mongo

