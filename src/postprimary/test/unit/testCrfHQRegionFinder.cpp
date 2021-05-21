#include <gtest/gtest.h>

#include <assert.h>
#include <iostream>
#include <cmath>
#include <tuple>

#include <postprimary/hqrf/BlockHQRegionFinder.h>

#include <postprimary/hqrf/HQRegionFinder.h>
#include <postprimary/hqrf/SpiderCrfParams.h>
#include <postprimary/hqrf/SequelCrfParams.h>

#include "ReadSimulator.h"

using namespace PacBio::Primary;
using namespace PacBio::Primary::Postprimary;
using namespace PacBio::Primary::ActivityLabeler;

template <typename Model>
class CrfModelTest : public ::testing::Test {};

typedef ::testing::Types<SpiderCrfModel, SequelCrfModel> CrfModels;

TYPED_TEST_SUITE(CrfModelTest, CrfModels);

TYPED_TEST(CrfModelTest, TestSimRead)
{
    // Mechanical test. These simulated reads aren't quite right

    ReadConfig config{};
    config.PKMID_A = 120;
    config.PKMID_C = 220;
    config.PKMID_G = 320;
    config.PKMID_T = 420;

    config.PKMAX_A = 160;
    config.PKMAX_C = 260;
    config.PKMAX_G = 360;
    config.PKMAX_T = 460;
    config.dumpBasesInLastBlock = false;
    const auto& metrics = SimulateMetrics(config);

    BlockHQRegionFinder<HQRFMethod::SPIDER_CRF_HMM> hqrf(80.0f, 3.75f, false);
    auto labels = hqrf.LabelActivities(metrics);

    EXPECT_TRUE(std::all_of(labels.begin(), labels.end(), [](Activity label) {
                return (label == Activity::A1) || (label == Activity::A2);}));
}

TYPED_TEST(CrfModelTest, TestSimReadLateStartSingle)
{
    // Mechanical test. These simulated reads aren't quite right

    ReadConfig config{};
    config.dumpBasesInLastBlock = false;
    config.PKMID_A = 120;
    config.PKMID_C = 220;
    config.PKMID_G = 320;
    config.PKMID_T = 420;

    config.PKMAX_A = 160;
    config.PKMAX_C = 260;
    config.PKMAX_G = 360;
    config.PKMAX_T = 460;
    config.seqstart = 288000;
    //config.numBases *= (config.numFrames - 288000)/config.numFrames;
    const auto& metrics = SimulateMetrics(config);

    BlockHQRegionFinder<HQRFMethod::SPIDER_CRF_HMM> hqrf(80.0f, 3.75f, false);
    auto labels = hqrf.LabelActivities(metrics);

    EXPECT_TRUE(std::all_of(labels.begin(), labels.begin() + 71,
                [](Activity label) {return label == Activity::A0;}));
    EXPECT_TRUE(std::all_of(labels.begin() + 71, labels.end(),
                [](Activity label) {
                return (label == Activity::A1) || (label == Activity::A2);}));
}
