//
// Created by mlakata on 1/5/21.
//

#include <gtest/gtest.h>

#include <pacbio/dev/gtest-extras.h>
#include <dataTypes/LabelsBatch.h>

TEST(ManagedAllocations,SourceMarker)
{
    auto x = SOURCE_MARKER();
    auto y = SOURCE_MARKER();
    EXPECT_FALSE(x == y);
    EXPECT_NE(x.AsString(),y.AsString());
    EXPECT_NE(x.AsHash(),y.AsHash());
    TEST_COUT << "x:"<<x.AsString()<< " " << (void*) x.AsHash() << std::endl;
    TEST_COUT << "y:"<<y.AsString()<< " " << (void*) y.AsHash() << std::endl;

    auto w = SOURCE_MARKER(); auto z = SOURCE_MARKER();
    EXPECT_EQ(w,z);
    EXPECT_EQ(w.AsString(), z.AsString());
    EXPECT_EQ(w.AsHash(), z.AsHash());
    TEST_COUT << "w:"<<w.AsString()<< " " << (void*) w.AsHash() << std::endl;
}
