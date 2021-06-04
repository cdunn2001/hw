#include <gtest/gtest.h>

#include <common/MemInfo.h>

using namespace testing;
using namespace PacBio::Utilities;

TEST(MemInfo,A)
{
    NodeMemInfo nmi(0);
    EXPECT_GT(nmi.HugePages_Total(),0);
    EXPECT_GE(nmi.HugePages_Free(),0);
}
