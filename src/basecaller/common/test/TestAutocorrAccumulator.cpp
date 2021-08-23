// Copyright (c) 2021, Pacific Biosciences of California, Inc.
//
// All rights reserved.
//
// THIS SOFTWARE CONSTITUTES AND EMBODIES PACIFIC BIOSCIENCES' CONFIDENTIAL
// AND PROPRIETARY INFORMATION.
//
// Disclosure, redistribution and use of this software is subject to the
// terms and conditions of the applicable written agreement(s) between you
// and Pacific Biosciences, where "you" refers to you or your company or
// organization, as applicable.  Any other disclosure, redistribution or
// use is prohibited.
//
// THIS SOFTWARE IS PROVIDED BY PACIFIC BIOSCIENCES AND ITS CONTRIBUTORS "AS
// IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
// THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL PACIFIC BIOSCIENCES OR ITS
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
// OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
// WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
// OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
// ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
//  Description:
//  Defines unit tests for template class AutocorrAccumulator.


#include "common/AutocorrAccumulator.h"

#include <vector>
#include <random>
#include <algorithm>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include <common/LaneArray.h>


using namespace PacBio::Mongo;

using FloatArray = LaneArray<float>;


namespace {

TEST(TestAutocorrAccumulator, SimpleLag4)
{
    auto n = 9;
    AutocorrAccumulator<FloatArray> aca(0.0f);
    auto m00 = aca.Count().data()[0][0];                 EXPECT_EQ(0, m00);
    auto m10 = aca.Mean().data()[0][0];                  EXPECT_TRUE(isnan(m10));
    auto m20 = aca.Variance().data()[0][0];              EXPECT_TRUE(isnan(m20));
    auto acorr0 = aca.Autocorrelation().data()[0][0];    EXPECT_TRUE(isnan(acorr0));

    auto x = 0.0f;
    aca.AddSample(x);
    auto m01 = aca.Count().data()[0][0];                 EXPECT_EQ(1, m01);
    auto m11 = aca.Mean().data()[0][0];                  EXPECT_EQ(x, m11);
    auto m21 = aca.Variance().data()[0][0];              EXPECT_TRUE(isnan(m21));
    auto acorr1 = aca.Autocorrelation().data()[0][0];    EXPECT_TRUE(isnan(acorr1));

    for (auto i = 1; i < n; ++i)
    {
        aca.AddSample(i);
    }

    auto m02 = aca.Count().data()[0][0];                 EXPECT_EQ(n, m02);
    auto m12 = aca.Mean().data()[0][0];                  EXPECT_EQ(4, m12);
    auto m22 = aca.Variance().data()[0][0];              EXPECT_EQ(7.5, m22);
    auto acorr2 = aca.Autocorrelation().data()[0][0];    EXPECT_EQ(4.0f/15, acorr2);
}

TEST(TestAutocorrAccumulator, AutocorrSine)
{
    const uint32_t n = 100;
    const float a = 0.1;
    AutocorrAccumulator<FloatArray> aca(0.0f);

    Eigen::Matrix<float, n, 16> x;
    for (uint32_t i = 0; i < n; ++i)
    {
        auto val = std::sin(a*i);
        x.row(i).array() = val;
        aca.AddSample(val);
    }

    // Compare against pre-calculated values
    auto m01 = aca.Count().data()[0][0];                 EXPECT_EQ(n, m01);
    auto m11 = aca.Mean().data()[0][0];                  EXPECT_FLOAT_EQ(0.1864739, m11);
    auto m21 = aca.Variance().data()[0][0];              EXPECT_FLOAT_EQ(0.4454547, m21);

    // Expected value for autocorrelation, taken from
    // https://en.wikipedia.org/wiki/Autocorrelation#Estimation
    // Python code:
    // a = np.sin(0.1 * np.arange(100))
    // l, ac = 4, a - np.mean(a)
    // autocorr_l = np.sum(ac[l:] * ac[:-l]) / (len(ac)-l) / np.var(a, ddof=1)
    // autocorr_l = 0.9281753966122696
    const float expectAutocorr = 0.928175397f;
    auto acorr0 = aca.Autocorrelation().data()[0][0];
    EXPECT_NEAR(expectAutocorr, acorr0, 1.0e-4f);

    // Compare against Eigen
    auto em1 = x.colwise().mean();
    auto em2 = (x.rowwise() - em1).array().square().colwise().sum() / (n-1);
    EXPECT_FLOAT_EQ(em1[0], m11);
    EXPECT_FLOAT_EQ(em2[0], m21);

    // all columns contain the same value, take #1
    auto aci = 1; auto l = AutocorrAccumState::lag;
    auto acr = x.col(aci).array() - em1[aci];
    auto acorr1 = (acr.head(n-l) * acr.tail(n-l)).sum() / (n-l) / em2[aci];

    EXPECT_NEAR(acorr1, acorr0, 1.0e-4f);

    assert(true);
}

struct TestAutocorrAccumulatorLinspace
    : public ::testing::TestWithParam<std::tuple<int, int>>
{
    const float a = 0.2;
    size_t w = laneSize;
    size_t n = 1080;
    std::vector<float> data;
    size_t threshold;

    TestAutocorrAccumulatorLinspace()
        : data (n)
    {
        auto nom = std::get<0>(GetParam());
        auto dnom = std::get<1>(GetParam());
        threshold = n * nom / dnom;

        for (decltype(n) i = 0; i < n; ++i)
        {
            data[i] = a*i;
        }
    }
};

TEST_P(TestAutocorrAccumulatorLinspace, MergeThreshold)
{
    AutocorrAccumulator<FloatArray> aca0, aca1, aca2;

    decltype(n) i = 0;
    for (; i < threshold; ++i)
    {
        aca0.AddSample(data[i]); // Add all the samples to aca0
        aca1.AddSample(data[i]); // And samples BEFORE the threshold
    }

    for (; i < n; ++i)
    {
        aca0.AddSample(data[i]); // Add all the samples to aca0
        aca2.AddSample(data[i]); // And samples AFTER the threshold
    }

    aca1.Merge(aca2);

    // frame_idx  relative tolerance      abs tolerance
    auto aci = 0;  auto rtol = 1e-5f; auto atol = 1e-5f;

    // Test full and merged accumulator
    auto cnt0 = aca0.Count().data()[0][aci];           auto cnt1 = aca1.Count().data()[0][aci];
    auto m10  = aca0.Mean().data()[0][aci];            auto m11  = aca1.Mean().data()[0][aci];
    auto m20  = aca0.Variance().data()[0][aci];        auto m21  = aca1.Variance().data()[0][aci];
    auto crr0 = aca0.Autocorrelation().data()[0][aci]; auto crr1 = aca1.Autocorrelation().data()[0][aci];

    EXPECT_EQ (cnt0, cnt1);
    EXPECT_NEAR (m10, m11,         atol);
    EXPECT_LE (fabs(m20-m21)/m20,    rtol);
    EXPECT_LE (fabs(crr0-crr1)/crr0, rtol);
    EXPECT_NEAR (crr0, crr1,       atol);
}

struct TestAutocorrAccumulatorRand
    : public ::testing::TestWithParam<std::tuple<int, int>>
{
    const float a = 0.2;
    size_t w = laneSize;
    size_t n = 1080;
    std::vector<float> data;
    size_t threshold;

    TestAutocorrAccumulatorRand()
        : data (n*w)
    {
        auto nom = std::get<0>(GetParam());
        auto dnom = std::get<1>(GetParam());
        threshold = n * nom / dnom;

        std::default_random_engine gnr;
        std::normal_distribution<float> dist(2.0, 2.0);
        std::generate(data.begin(), data.end(), std::bind(dist, gnr));
        // for (decltype(n) i = 0; i < w*n; ++i) data[i] = a*i;
    }
};

TEST_P(TestAutocorrAccumulatorRand, MergeThreshold2)
{
    std::vector<FloatArray> farr(n);
    for (decltype(n) i = 0; i < n; ++i)
    {
        farr[i] = FloatArray(MemoryRange<float, laneSize>(&data[i*w]));
    }

    decltype(n) i = 0;
    AutocorrAccumulator<FloatArray> aca0(FloatArray(0.0f)), aca1(FloatArray(0.0f)), aca2(FloatArray(0.0f));

    for (; i < threshold; ++i)
    {
        aca0.AddSample(farr[i]); // Add all the samples to aca0
        aca1.AddSample(farr[i]); // And samples BEFORE the threshold
    }

    for (; i < n; ++i)
    {
        aca0.AddSample(farr[i]); // Add all the samples to aca0
        aca2.AddSample(farr[i]); // And samples AFTER the threshold
    }

    aca1.Merge(aca2);

    // frame_idx  relative tolerance      abs tolerance
    auto aci = 3;  auto rtol = 5e-5f; auto atol = 1e-5f; auto l = AutocorrAccumState::lag;

    // Test full and merged accumulator
    auto cnt0 = aca0.Count().data()[0][aci];           auto cnt1 = aca1.Count().data()[0][aci];
    auto m10  = aca0.Mean().data()[0][aci];            auto m11  = aca1.Mean().data()[0][aci];
    auto m20  = aca0.Variance().data()[0][aci];        auto m21  = aca1.Variance().data()[0][aci];
    auto crr0 = aca0.Autocorrelation().data()[0][aci]; auto crr1 = aca1.Autocorrelation().data()[0][aci];

    EXPECT_EQ (cnt0, cnt1);
    EXPECT_NEAR (m10, m11,         atol);
    EXPECT_LE (fabs(m20-m21)/m20,    rtol);
    EXPECT_LE (fabs(crr0-crr1)/crr0, rtol);
    EXPECT_NEAR (crr0, crr1,       atol);

    // Test against eigen
    Eigen::Map<Eigen::MatrixXf> xr(data.data(), w, n);
    auto em1 = xr.rowwise().mean();
    auto em2 = (xr.colwise() - em1).array().square().rowwise().sum() / (n - 1);

    auto acr1 = xr.row(aci).array() - em1[aci];
    auto acorr1 = (acr1.head(n-l) * acr1.tail(n-l)).sum() / (n-l) / em2[aci];

    EXPECT_FLOAT_EQ (m10, em1[aci]);
    EXPECT_LE (fabs(m20-em2[aci])/em2[aci],  rtol);
    EXPECT_LE (fabs(crr0-acorr1)/acorr1,     rtol);
    EXPECT_NEAR (crr0, acorr1,             atol);
}

const auto& testParams = ::testing::Combine(
            ::testing::Values(2, 4, 7),
            ::testing::Values(8, 12, 20)
            );

INSTANTIATE_TEST_SUITE_P (TestAutoCorrMergeThreshold, TestAutocorrAccumulatorLinspace,
                         testParams);
INSTANTIATE_TEST_SUITE_P (TestAutoCorrMergeThreshold2, TestAutocorrAccumulatorRand,
                         testParams);

//#endif
}
