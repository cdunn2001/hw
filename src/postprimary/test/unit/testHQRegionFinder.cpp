//PrimaryAnalysis/basecaller-dha/Sequel/ppa/test/unit/HQRegionFinder_Test.cpp#2 - edit change 176394 (text)
#include <gtest/gtest.h>

#include <assert.h>
#include <iostream>
#include <cmath>
#include <tuple>

#include <postprimary/hqrf/BlockHQRegionFinder.h>
#include <postprimary/hqrf/HQRegionFinderParams.h>
#include <postprimary/hqrf/ClassificationTree.h>

#include <bazio/BlockActivityLabels.h>

using namespace PacBio::Primary::Postprimary;
using namespace PacBio::Primary::ActivityLabeler;


TEST(HQRegionFinder, TestEmptyZmw)
{
    BlockHQRegionFinder<HQRFMethod::TRAINED_CART_HMM> hqrf(80.0, 3.75, false);
    std::vector<HQRFState> decoding;
    float loglikelihood;
    size_t hqStart, hqEnd;

    // Empty ZMW; expect a null HQR
    std::vector<Activity> emptyZmw = { A0, A0, A0, A0, A0 };
    std::tie(hqStart, hqEnd) = hqrf.FindBlockHQRegion(emptyZmw, &decoding, &loglikelihood);

    std::vector<HQRFState> expectedDecoding = { POST_QUIET, POST_QUIET, POST_QUIET, POST_QUIET, POST_QUIET };
    EXPECT_EQ((size_t)0, hqStart);
    EXPECT_EQ((size_t)0, hqEnd);
    EXPECT_EQ(expectedDecoding, decoding);
    EXPECT_FLOAT_EQ(std::log(0.33f * std::pow((0.965f), 5)), loglikelihood);
}

TEST(HQRegionFinder, TestFullHQR)
{
    BlockHQRegionFinder<HQRFMethod::TRAINED_CART_HMM> hqrf(80.0, 3.75, false);
    std::vector<HQRFState> decoding;
    float loglikelihood;
    size_t hqStart, hqEnd;

    // End-to-end HQR
    std::vector<Activity> fullHQ = { A1, A1, A1, A1, A1 };
    std::tie(hqStart, hqEnd) = hqrf.FindBlockHQRegion(fullHQ, &decoding, &loglikelihood);

    std::vector<HQRFState> expectedDecoding = { HQ, HQ, HQ, HQ, HQ };
    EXPECT_EQ((size_t)0, hqStart);
    EXPECT_EQ((size_t)5, hqEnd);
    EXPECT_EQ(expectedDecoding, decoding);

    float a = 1.0/5;
    EXPECT_FLOAT_EQ(log(0.33 * pow(0.867, 5) * pow(1-a, 5)), loglikelihood);
}

TEST(HQRegionFinder, TestFullyA2)
{
    BlockHQRegionFinder<HQRFMethod::TRAINED_CART_HMM> hqrf(80.0, 3.75, false);
    std::vector<HQRFState> decoding;
    float loglikelihood;
    size_t hqStart, hqEnd;

    // End-to-end HQR
    std::vector<Activity> fullyA2 = { A2, A2, A2, A2, A2 };
    std::tie(hqStart, hqEnd) = hqrf.FindBlockHQRegion(fullyA2, &decoding, &loglikelihood);

    std::vector<HQRFState> expectedDecoding = { PRE, PRE, PRE, PRE, PRE };
    EXPECT_EQ(expectedDecoding, decoding);
    EXPECT_EQ((size_t)0, hqStart);
    EXPECT_EQ((size_t)0, hqEnd);

    float a = 1.0/5;
    EXPECT_FLOAT_EQ(log(0.33 * pow(0.939 , 5) * pow(1-a, 5)), loglikelihood);
}



TEST(HQRegionFinder, TestPattern1)
{
    // Pattern 1 is when we go 2 active -> 1 active -> 0 active
    BlockHQRegionFinder<HQRFMethod::TRAINED_CART_HMM> hqrf(80.0, 3.75, false);
    std::vector<HQRFState> decoding;
    float loglikelihood;
    size_t hqStart, hqEnd;

    //                                  0   1   2   3   4   5   6   7   8   9  10  11
    std::vector<Activity> pattern1 = { A2, A2, A1, A1, A0, A1, A1, A1, A0, A0, A0, A0 };
    std::tie(hqStart, hqEnd) = hqrf.FindBlockHQRegion(pattern1, &decoding, &loglikelihood);


    std::vector<HQRFState> expectedDecoding = { PRE, PRE,
                                                HQ, HQ, HQ, HQ, HQ, HQ,
                                                POST_QUIET, POST_QUIET, POST_QUIET, POST_QUIET };
    EXPECT_EQ(expectedDecoding, decoding);
    EXPECT_EQ((size_t)2, hqStart);
    EXPECT_EQ((size_t)8, hqEnd);
}


TEST(HQRegionFinder, TestPattern2)
{
    // Pattern 2 is when we go 2 active -> 1 active -> 2 active
    BlockHQRegionFinder<HQRFMethod::TRAINED_CART_HMM> hqrf(80.0, 3.75, false);
    std::vector<HQRFState> decoding;
    float loglikelihood;
    size_t hqStart, hqEnd;

    //                                  0   1   2   3   4   5   6   7   8   9  10  11
    std::vector<Activity> pattern2 = { A2, A2, A1, A1, A1, A0, A1, A1, A2, A2, A2, A2 };
    std::tie(hqStart, hqEnd) = hqrf.FindBlockHQRegion(pattern2, &decoding, &loglikelihood);


    std::vector<HQRFState> expectedDecoding = { PRE, PRE,
                                                HQ, HQ, HQ, HQ, HQ, HQ,
                                                POST_ACTIVE, POST_ACTIVE, POST_ACTIVE, POST_ACTIVE };
    EXPECT_EQ(expectedDecoding, decoding);
    EXPECT_EQ((size_t)2, hqStart);
    EXPECT_EQ((size_t)8, hqEnd);
}

TEST(HQRegionFinder, TestClassificationTree)
{

    size_t hqStart, hqEnd;
    // Empty ZMW; expect a null HQR
    std::vector<Activity> emptyZmw = { A0, A0, A0, A0, A0 };
    std::tie(hqStart, hqEnd) = ActivityLabeler::LabelRegions(emptyZmw, 2);
    EXPECT_EQ((size_t)0, hqStart);
    EXPECT_EQ((size_t)0, hqEnd);

    // All Single
    std::vector<Activity> fullHQ = { A1, A1, A1, A1, A1 };
    std::tie(hqStart, hqEnd) = ActivityLabeler::LabelRegions(fullHQ, 2);
    EXPECT_EQ((size_t)0, hqStart);
    EXPECT_EQ((size_t)5, hqEnd);

    // All multi
    std::vector<Activity> fullA2 = { A2, A2, A2, A2, A2 };
    std::tie(hqStart, hqEnd) = ActivityLabeler::LabelRegions(fullA2, 2);
    EXPECT_EQ((size_t)0, hqStart);
    EXPECT_EQ((size_t)0, hqEnd);

    // NOTE: The following results won't be perfect. This model is still under
    // development, and some compromises are being made for the sake of
    // "realtime" region finding
    std::vector<Activity> pattern1 = { A2, A2, A1, A1, A0, A1, A1, A1, A0, A0, A0, A0 };
    std::tie(hqStart, hqEnd) = ActivityLabeler::LabelRegions(pattern1, 2);
    EXPECT_EQ((size_t)5, hqStart);
    EXPECT_EQ((size_t)12, hqEnd);

    std::vector<Activity> pattern2 = { A2, A2, A1, A1, A1, A0, A1, A1, A2, A2, A2, A2 };
    std::tie(hqStart, hqEnd) = ActivityLabeler::LabelRegions(pattern2, 2);
    EXPECT_EQ((size_t)4, hqStart);
    EXPECT_EQ((size_t)8, hqEnd);

}
