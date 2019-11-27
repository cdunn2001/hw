//
// Created by mlakata on 10/23/15.
//

#include <cmath>

#include <pacbio/primary/AnalogMode.h>
#include <gtest/gtest.h>
#include <pacbio/dev/gtest-extras.h>

using namespace PacBio::Primary;

TEST(AnalogMode,1CConstructor)
{
    std::array<float, 1> green{{1.0}};
    AnalogMode am('A',green,1.0);

    AnalogMode ay('T',green,1.0);

    AnalogMode ar('G',green,1.0);
}


TEST(AnalogMode,2CConstructor)
{
    std::array<float, 2> green{{1.0,0.0}};
    std::array<float, 2> yellow{{0.5,0.5}};
    std::array<float, 2> red{{0.0,1.0}};
    AnalogMode am('A',green,1.0);
    EXPECT_FLOAT_EQ(0, am.SpectralAngle());

    AnalogMode ay('A',yellow,1.0);
    EXPECT_FLOAT_EQ(M_PI_2 * 0.5, ay.SpectralAngle());

    AnalogMode ar('A',red,1.0);
    EXPECT_FLOAT_EQ(M_PI_2, ar.SpectralAngle());

    EXPECT_FLOAT_EQ(0.0, ar.pw2SlowStepRatio);
    EXPECT_FLOAT_EQ(0.0, ar.ipd2SlowStepRatio);
}

TEST(AnalogConfigEx,Parser1)
{
    const std::string json = R"(
{
  "base": "C",
  "spectralAngle": 0.5,
  "pw2SlowStepRatio": 0.14,
  "ipd2SlowStepRatio": 0.15
}
    )";

    AnalogConfigEx config;
    config.Load(json);

    EXPECT_FLOAT_EQ(0.14, config.pw2SlowStepRatio);
    EXPECT_FLOAT_EQ(0.15, config.ipd2SlowStepRatio);
    ASSERT_EQ(2,config.spectrumValues().size());
    EXPECT_FLOAT_EQ(0.64670402, config.spectrumValues()[0]);
    EXPECT_FLOAT_EQ(0.35329601, config.spectrumValues()[1]);
}

TEST(AnalogConfigEx,Parser2)
{
    const std::string json = R"(
{
  "base": "C",
  "spectrumValues": [0.1,0.9],
  "pw2SlowStepRatio": 0.14,
  "ipd2SlowStepRatio": 0.15
}
    )";

    AnalogConfigEx config;
    config.Load(json);

    EXPECT_EQ("C",config.base());
    EXPECT_FLOAT_EQ(1.4601392, config.spectralAngle()) << config;
    ASSERT_EQ(2,config.spectrumValues().size()) << config;
    EXPECT_FLOAT_EQ(0.1, config.spectrumValues()[0]) << config;
    EXPECT_FLOAT_EQ(0.9, config.spectrumValues()[1]) << config;

    EXPECT_FLOAT_EQ(0.14, config.pw2SlowStepRatio) << config;
    EXPECT_FLOAT_EQ(0.15, config.ipd2SlowStepRatio) << config;
}
