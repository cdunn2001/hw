//
// Created by mlakata on 1/11/18.
//

#include <gtest/gtest.h>
#include <pacbio/primary/Sparse2DArray.h>
using namespace PacBio::Primary;

TEST(Sparse2DArray,Simple)
{
    Sparse2DArray<uint8_t> sa;
    const size_t numNonZmws = 6;
    int16_t xes[] = {1,2,3,5,2,3};
    int16_t yes[] = {1,1,1,1,2,3};
    uint8_t zes[] = {5,5,5,5,6,7};

    // array looks like
    // 5 5 5
    // 1 6 1
    // 1 1 7

    sa.CompressSparseArray(numNonZmws, xes, yes, zes,1);

    EXPECT_EQ(5,sa.Value(1,1));
    EXPECT_EQ(5,sa.Value(2,1));
    EXPECT_EQ(5,sa.Value(3,1));
    EXPECT_EQ(1,sa.Value(4,1));
    EXPECT_EQ(5,sa.Value(5,1));
    EXPECT_EQ(6,sa.Value(2,2));
    EXPECT_EQ(7,sa.Value(3,3));

    EXPECT_EQ(1,sa.Value(1,2));
    EXPECT_EQ(1,sa.Value(3,2));
    EXPECT_EQ(1,sa.Value(1,3));
    EXPECT_EQ(1,sa.Value(2,3));
}
