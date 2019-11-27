
#include <gtest/gtest.h>

#include <algorithm>
#include <pacbio/primary/SmrtSensor.h>

using PacBio::Primary::SmrtSensor;

TEST (SmrtSensor, DefaultContruction2Color)
{
    static const auto nFilt = 2;

    SmrtSensor ss(nFilt);

    EXPECT_EQ(nFilt,ss.NumFilters());

    // Filter map is [0, 1] or [1, 0].
    const std::vector<uint16_t> fm0 {{0, 1}};
    const std::vector<uint16_t> fm1 {{1, 0}};
    EXPECT_TRUE (ss.FilterMap() == fm0 || ss.FilterMap() == fm1);
    EXPECT_EQ(2, ss.FilterMapArray<2>().size());

    ss.FilterMap(fm1);
    EXPECT_TRUE (ss.FilterMap() == fm1);

#if 0
    // Image PSF size is (2, 5, 5).
    static const unsigned int psfSize = 5;
    // There was an intention to refactor the SmrtSensor class to
    // support storage of the ImagePsfs and XtalkCorrection, but during the refactoring of the Acquisitio::Setup
    // class, supporting this unused feature was busywork. So SmrtSensor does not support ImagePsf() or XtalkCorrection()
    // any more. I'll leave this test code in case someone wants to revive it.
    const auto& imgPsf = ss.ImagePsf();
    ASSERT_EQ (nFilt, imgPsf.shape()[0]);
    ASSERT_EQ (psfSize, imgPsf.shape()[1]);
    ASSERT_EQ (psfSize, imgPsf.shape()[2]);

    // Image PSFs are normalized, and elements are not negative,
    // and central elements are > 0.5.
    for (unsigned int f = 0; f < imgPsf.shape()[0]; ++f)
    {
        EXPECT_GT (imgPsf[f][psfSize/2][psfSize/2], 0.5f);
        float sum = 0.0f;
        for (unsigned int r = 0; r < psfSize; ++r)
        {
            for (unsigned int c = 0; c < psfSize; ++c)
            {
                const auto x = imgPsf[f][r][c];
                EXPECT_GE (x, 0.0f);
                sum += x;
            }
        }
        EXPECT_NEAR (1.0f, sum, 0.0001f);
    }

    // XtalkCorrectionFilter size is (7, 7).
    static const unsigned int xtcfSize = 7;
    const auto& xtcFilter = ss.XtalkCorrection();
    ASSERT_EQ (xtcfSize, xtcFilter.shape()[0]);
    ASSERT_EQ (xtcfSize, xtcFilter.shape()[1]);

    // XtalkCorrectionFilter is normalized.
    float sum = 0.0f;
    for (unsigned int row = 0; row < xtcfSize; ++row)
    {
        for (unsigned int col = 0; col < xtcfSize; ++col)
        {
            sum += xtcFilter[row][col];
        }
    }
    EXPECT_NEAR (1.0f, sum, 0.0001f);
#endif

    // RefDwsSnr is positive.
    EXPECT_GT (ss.RefDwsSnr(), 0.0f);

    // RefSpectrum elements are non-negative and normalized.
    ASSERT_EQ (nFilt, ss.RefSpectrum().size());
    ASSERT_EQ (nFilt, ss.RefSpectrumArray<2>().size());
    float sum = 0.0f;
    for (auto x : ss.RefSpectrum())
    {
        EXPECT_GE (x, 0.0f);
        sum += x;
    }
    EXPECT_NEAR (sum, 1.0f, 0.0001f);
    const std::vector<float> spec {{0.5f, 0.5f}};
    ss.RefSpectrum(spec);
    EXPECT_EQ(spec,ss.RefSpectrum());

    EXPECT_EQ(1.0, ss.PhotoelectronSensitivity());
    ss.PhotoelectronSensitivity(1.5f);
    EXPECT_EQ(1.5, ss.PhotoelectronSensitivity());

    EXPECT_EQ(100.0, ss.FrameRate());
    ss.FrameRate(80.0);
    EXPECT_EQ(80.0, ss.FrameRate());

    auto shotNoise = ss.ShotNoiseCovar(0);
    EXPECT_EQ(3,shotNoise.size());

    ss.RefDwsSnr(50.0);
    EXPECT_EQ(50.0, ss.RefDwsSnr());
}

TEST (SmrtSensor, DefaultContruction1Color)
{
    static const auto nFilt = 1;

    SmrtSensor ss(nFilt);

    EXPECT_EQ(nFilt,ss.NumFilters());

    // Filter map is [0, 1] or [1, 0].
    const std::vector<uint16_t> fm0 {{0}};
    EXPECT_TRUE (ss.FilterMap() == fm0);
    EXPECT_EQ(1, ss.FilterMapArray<1>().size());
    EXPECT_NO_THROW(ss.FilterMap(fm0.cbegin(), fm0.cend()));

    // RefSpectrum elements are non-negative and normalized.
    ASSERT_EQ (nFilt, ss.RefSpectrum().size());
    ASSERT_EQ (nFilt, ss.RefSpectrumArray<1>().size());
    float sum = 0.0f;
    for (auto x : ss.RefSpectrum())
    {
        EXPECT_GE (x, 0.0f);
        sum += x;
    }
    EXPECT_NEAR (sum, 1.0f, 0.0001f);
    const std::vector<float> spec {{0.5f}};
    EXPECT_THROW(ss.RefSpectrum(spec),std::exception); // not normalized

    auto shotNoise = ss.ShotNoiseCovar(0);
    EXPECT_EQ(1,shotNoise.size());
}

TEST(SmrtSensor, Copies)
{
    static const auto nFilt = 2;

    const SmrtSensor ss(nFilt);
    const SmrtSensor ss2(ss);
    EXPECT_EQ(nFilt,ss2.NumFilters());

    SmrtSensor ss3(nFilt);
    ss3 = ss;
    EXPECT_EQ(nFilt,ss3.NumFilters());
}
