
#include <gtest/gtest.h>

#include <iostream>
#include <memory>
#include <vector>
#include <thread>
#include <chrono>
#include <random>
#include <sstream>
#include <string>
#include <assert.h>

//#define private public
#include <pacbio/primary/MetricBlock.h>
//#undef private

using namespace PacBio::Primary;
using Mb = SequelMetricBlock;

TEST(metrics, failRange)
{
    Mb m1;
    m1.BaselineSds({-40000,-40000})
        .Baselines({-40000,-40000})
        .PkmidA(-40000)
        .PkmidC(-40000)
        .PkmidG(-40000)
        .PkmidT(-40000);

    ASSERT_FLOAT_EQ(-40000, m1.BaselineSds()[0]);
    ASSERT_FLOAT_EQ(-40000, m1.BaselineSds()[1]);
    ASSERT_FLOAT_EQ(-40000, m1.Baselines()[0]);
    ASSERT_FLOAT_EQ(-40000, m1.Baselines()[1]);
    ASSERT_FLOAT_EQ(-40000, m1.PkmidA());
    ASSERT_FLOAT_EQ(-40000, m1.PkmidC());
    ASSERT_FLOAT_EQ(-40000, m1.PkmidG());
    ASSERT_FLOAT_EQ(-40000, m1.PkmidT());

    Mb m2;
    m2.BaselineSds({+40000,+40000})
        .Baselines({+40000,+40000})
        .PkmidA(+40000)
        .PkmidC(+40000)
        .PkmidG(+40000)
        .PkmidT(+40000);

    ASSERT_FLOAT_EQ(40000, m2.BaselineSds()[0]);
    ASSERT_FLOAT_EQ(40000, m2.BaselineSds()[1]);
    ASSERT_FLOAT_EQ(40000, m2.Baselines()[0]);
    ASSERT_FLOAT_EQ(40000, m2.Baselines()[1]);
    ASSERT_FLOAT_EQ(40000, m2.PkmidA());
    ASSERT_FLOAT_EQ(40000, m2.PkmidC());
    ASSERT_FLOAT_EQ(40000, m2.PkmidG());
    ASSERT_FLOAT_EQ(40000, m2.PkmidT());
}
TEST(metrics, raw)
{
    Mb m1;
    m1.NumBasesA(7)
        .NumBasesC(4)
        .NumBasesG(4)
        .NumBasesT(4)
        .BaselineSds({1.2,2})
        .NumPulses(3)
        .PulseWidth(4)
        .BaseWidth(5)
        .Baselines({8,9})
        .PkmidA(-10)
        .PkmidC(11)
        .PkmidG(-12)
        .PkmidT(13)
        .NumPkmidFramesA(14)
        .NumPkmidFramesC(15)
        .NumPkmidFramesG(16)
        .NumPkmidFramesT(17)
        .NumFrames(18);

    EXPECT_EQ(7,m1.NumBasesA());
    EXPECT_EQ(4,m1.NumBasesC());
    EXPECT_EQ(4,m1.NumBasesG());
    EXPECT_EQ(4,m1.NumBasesT());
    ASSERT_FLOAT_EQ(1.2,m1.BaselineSds()[0]);
    EXPECT_EQ(2,m1.BaselineSds()[1]);
    EXPECT_EQ(3,m1.NumPulses());
    EXPECT_EQ(4,m1.PulseWidth());
    EXPECT_EQ(5,m1.BaseWidth());
    EXPECT_EQ(8,m1.Baselines()[0]);
    EXPECT_EQ(9,m1.Baselines()[1]);
    EXPECT_EQ(-10,m1.PkmidA());
    EXPECT_EQ(11,m1.PkmidC());
    EXPECT_EQ(-12,m1.PkmidG());
    EXPECT_EQ(13,m1.PkmidT());
    EXPECT_EQ(14,m1.NumPkmidFramesA());
    EXPECT_EQ(15,m1.NumPkmidFramesC());
    EXPECT_EQ(16,m1.NumPkmidFramesG());
    EXPECT_EQ(17,m1.NumPkmidFramesT());
    EXPECT_EQ(18,m1.NumFrames());

    EXPECT_EQ(static_cast<size_t>(7),m1.numBasesA_);
    EXPECT_EQ(static_cast<size_t>(4),m1.numBasesC_);
    EXPECT_EQ(static_cast<size_t>(4),m1.numBasesG_);
    EXPECT_EQ(static_cast<size_t>(4),m1.numBasesT_);
    ASSERT_FLOAT_EQ(1.2,m1.baselineSds_[0]);
    ASSERT_FLOAT_EQ(2,m1.baselineSds_[1]);
    EXPECT_EQ(static_cast<size_t>(3), m1.numPulses_);
    EXPECT_EQ(static_cast<size_t>(4), m1.pulseWidth_);
    EXPECT_EQ( static_cast<size_t>(5), m1.baseWidth_);
    ASSERT_FLOAT_EQ(8,m1.baselines_[0]);
    ASSERT_FLOAT_EQ(9,m1.baselines_[1]);
    ASSERT_FLOAT_EQ(-10,m1.pkmidA_);
    ASSERT_FLOAT_EQ(11,m1.pkmidC_);
    ASSERT_FLOAT_EQ(-12,m1.pkmidG_);
    ASSERT_FLOAT_EQ(13,m1.pkmidT_);
    EXPECT_EQ(static_cast<size_t>(14), m1.numpkmidFramesA_);
    EXPECT_EQ(static_cast<size_t>(15), m1.numpkmidFramesC_);
    EXPECT_EQ(static_cast<size_t>(16), m1.numpkmidFramesG_);
    EXPECT_EQ(static_cast<size_t>(17), m1.numpkmidFramesT_);
    EXPECT_EQ(static_cast<size_t>(18), m1.numFrames_);

}

void TestMerge(Mb& m, size_t numParents)
{
    size_t sum = 10 * numParents;
    EXPECT_EQ(sum, m.NumBasesA());
    EXPECT_EQ(sum, m.NumBasesC());
    EXPECT_EQ(sum, m.NumBasesG());
    EXPECT_EQ(sum, m.NumBasesT());
    EXPECT_EQ(10,m.BaselineSds()[0]);
    EXPECT_EQ(10, m.BaselineSds()[1]);
    EXPECT_EQ(sum, m.NumPulses());
    EXPECT_EQ(sum, m.PulseWidth());
    EXPECT_EQ(sum, m.BaseWidth());
    EXPECT_FLOAT_EQ(10, m.Baselines()[0]);
    EXPECT_FLOAT_EQ(10, m.Baselines()[1]);

    EXPECT_EQ(10, m.PkmidA());
    EXPECT_EQ(10, m.PkmidC());
    EXPECT_EQ(10, m.PkmidG());
    EXPECT_EQ(10, m.PkmidT());
    EXPECT_EQ(sum, m.NumPkmidFramesA());
    EXPECT_EQ(sum, m.NumPkmidFramesC());
    EXPECT_EQ(sum, m.NumPkmidFramesG());
    EXPECT_EQ(sum, m.NumPkmidFramesT());
    EXPECT_EQ(sum, m.NumFrames());

    EXPECT_EQ(sum, m.numBasesA_);
    EXPECT_EQ(sum, m.numBasesC_);
    EXPECT_EQ(sum, m.numBasesG_);
    EXPECT_EQ(sum, m.numBasesT_);
    ASSERT_FLOAT_EQ(10, m.baselineSds_[0]);
    ASSERT_FLOAT_EQ(10, m.baselineSds_[1]);
    EXPECT_EQ(sum, m.numPulses_);
    EXPECT_EQ(static_cast<size_t>(sum), m.pulseWidth_);
    EXPECT_EQ(static_cast<size_t>(sum), m.baseWidth_);
    ASSERT_FLOAT_EQ(10, m.baselines_[0]);
    ASSERT_FLOAT_EQ(10, m.baselines_[1]);
    ASSERT_FLOAT_EQ(10, m.pkmidA_);
    ASSERT_FLOAT_EQ(10, m.pkmidC_);
    ASSERT_FLOAT_EQ(10, m.pkmidG_);
    ASSERT_FLOAT_EQ(10, m.pkmidT_);
    EXPECT_EQ(sum, m.numpkmidFramesA_);
    EXPECT_EQ(sum, m.numpkmidFramesC_);
    EXPECT_EQ(sum, m.numpkmidFramesG_);
    EXPECT_EQ(sum, m.numpkmidFramesT_);
    EXPECT_EQ(sum, m.numFrames_);

    float var = 3.0f;
    EXPECT_EQ(var, m.BpzvarA());
    EXPECT_EQ(var, m.BpzvarC());
    EXPECT_EQ(var, m.BpzvarG());
    EXPECT_EQ(var, m.BpzvarT());

    float zvar = 1.7f;
    EXPECT_EQ(zvar, m.PkzvarA());
    EXPECT_EQ(zvar, m.PkzvarC());
    EXPECT_EQ(zvar, m.PkzvarG());
    EXPECT_EQ(zvar, m.PkzvarT());
}

TEST(metrics, aggregate)
{
    Mb m1;
    m1.BaselineSds({10,10})
      .NumPulses(10)
      .PulseWidth(10)
      .BaseWidth(10)
      .Baselines({10,10})
      .NumBaselineFrames({1,1})
      .PkmidA(10)
      .PkmidC(10)
      .PkmidG(10)
      .PkmidT(10)
      .NumPkmidFramesA(10)
      .NumPkmidFramesC(10)
      .NumPkmidFramesG(10)
      .NumPkmidFramesT(10)
      .NumBasesA(10)
      .NumBasesC(10)
      .NumBasesG(10)
      .NumBasesT(10)
      .NumPkmidBasesA(2)
      .NumPkmidBasesC(2)
      .NumPkmidBasesG(2)
      .NumPkmidBasesT(2)
      .BpzvarA(3)
      .BpzvarC(3)
      .BpzvarG(3)
      .BpzvarT(3)
      .PkzvarA(1.7)
      .PkzvarC(1.7)
      .PkzvarG(1.7)
      .PkzvarT(1.7)
      .NumFrames(10);

    Mb m2(m1);

    std::vector<Mb> vec;
    vec.push_back(m1);
    vec.push_back(m2);

    //
    // Average two blocks
    
    // Average via vector
    Mb merge12(vec);
    TestMerge(merge12, 2);

    // Average via ptr and length
    Mb merge12ptr(&vec[0], vec.size());
    TestMerge(merge12ptr, 2);

    // Average via initializer list
    Mb merge12initList{m1, m2};
    TestMerge(merge12initList, 2);

    //
    // Average three blocks
    Mb m3(m1);
    vec.push_back(m3);

    // Average via vector
    Mb merge13(vec);
    TestMerge(merge13, 3);

    // Average via ptr and length
    Mb merge13ptr(&vec[0], vec.size());
    TestMerge(merge13ptr, 3);

    // Average via initializer list
    Mb merge13initList{m1, m2, m3};
    TestMerge(merge13initList, 3);
}
