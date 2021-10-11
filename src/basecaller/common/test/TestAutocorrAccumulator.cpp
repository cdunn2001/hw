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

TEST(TestAutocorrAccumulator, SimpleOnePass)
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
    auto m12 = aca.Mean().data()[0][0];                  EXPECT_FLOAT_EQ(4, m12);
    auto m22 = aca.Variance().data()[0][0];              EXPECT_FLOAT_EQ(7.5, m22);
    auto acorr2 = aca.Autocorrelation().data()[0][0];    EXPECT_FLOAT_EQ(-4.0/15, acorr2);
}

TEST(TestAutocorrAccumulator, SquareOnePass)
{
    auto n = 14;
    AutocorrAccumulator<FloatArray> aca(0.0f);

    for (auto i = 0.0f; i < n; i += 1.0f)
    {
        aca.AddSample(i*i);
    }

    auto m02 = aca.Count().data()[0][0];                 EXPECT_EQ(n,      m02);
    auto m12 = aca.Mean().data()[0][0];                  EXPECT_FLOAT_EQ(58.5f,   m12);
    auto m22 = aca.Variance().data()[0][0];              EXPECT_FLOAT_EQ(3181.5f, m22);
    auto acorr2 = aca.Autocorrelation().data()[0][0];    EXPECT_FLOAT_EQ(0.20589344f, acorr2);
}

void AccCompare(const AutocorrAccumulator<FloatArray> &exp, const AutocorrAccumulator<FloatArray> &act,
                float rtol=1e-05, float atol=1e-08)
{
    auto z = 0; // zmw_idx

    // Expected data                                 // Actual data                                 
    auto cnt0 = exp.Count().data()[0][z];            auto cnt1 = act.Count().data()[0][z];          
    auto m10  = exp.Mean().data()[0][z];             auto m11  = act.Mean().data()[0][z];           
    auto m20  = exp.Variance().data()[0][z];         auto m21  = act.Variance().data()[0][z];       
    auto crr0 = exp.Autocorrelation().data()[0][z];  auto crr1 = act.Autocorrelation().data()[0][z];

    EXPECT_EQ (cnt0, cnt1);
    EXPECT_LE(abs(m10-m11),   atol + rtol*abs(m11));
    EXPECT_LE(abs(m20-m21),   atol + rtol*abs(m21));
    EXPECT_LE(abs(crr0-crr1), atol + rtol*abs(crr1));
}

TEST(TestAutocorrAccumulator, SimpleMerge)
{
    auto n = 30; // (lag + 2 + lag) * 3
    AutocorrAccumulator<FloatArray> aca, aca1, aca2, aca3;

    decltype(n) i = 0;
    for (; i < 1 * n / 3; ++i)
    {
        aca.AddSample(i);
        aca1.AddSample(i);
    }

    for (; i < 2 * n / 3; ++i)
    {
        aca.AddSample(i);
        aca2.AddSample(i);
    }

    for (; i < n; ++i)
    {
        aca.AddSample(i);
        aca3.AddSample(i);
    }

    // Copy for another merging order
    AutocorrAccumulator<FloatArray> acb1(aca1), acb2(aca2), acb3(aca3);

    aca1.Merge(aca2).Merge(aca3);
    acb1.Merge(acb2.Merge(acb3));

    AutocorrAccumulator<FloatArray> aca_empty;
    aca_empty.Merge(aca1);

    AccCompare(aca, aca1);       // Test full vs merged accumulator A
    AccCompare(aca, acb1);       // Test full vs merged accumulator B
    AccCompare(aca, aca_empty);  // Test full vs initially empty accumulator
}

TEST(TestAutocorrAccumulator, PartialMerge)
{
    auto n = 30; // (lag + 2 + lag) * 3
    AutocorrAccumulator<FloatArray> aca, aca1, aca2, aca3, aca4;

    decltype(n) i = 0;
    for (; i < 1; ++i)   // 1st is shorter than lag
    {
        aca.AddSample(i+1);
        aca1.AddSample(i+1);
    }

    for (; i < 8; ++i)   // 2nd is shorter than 2*lag
    {
        aca.AddSample(i+1);
        aca2.AddSample(i+1);
    }

    for (; i < 27; ++i)  // 3rd is typically long
    {
        aca.AddSample(i+1);
        aca3.AddSample(i+1);
    }

    for (; i < n; ++i)   // 4th is shorter than lag
    {
        aca.AddSample(i+1);
        aca4.AddSample(i+1);
    }

    // Copy to test another merging order
    AutocorrAccumulator<FloatArray> acb1(aca1), acb2(aca2), acb3(aca3), acb4(aca4);
    AutocorrAccumulator<FloatArray> acc1(aca1), acc2(aca2), acc3(aca3), acc4(aca4);

    aca1.Merge(aca2).Merge(aca3).Merge(aca4);
    acb1.Merge(acb2).Merge(acb3.Merge(acb4));
    acc1.Merge(acc2.Merge(acc3.Merge(acc4)));

    AccCompare(aca, aca1);     // Test full vs merged accumulator A
    AccCompare(aca, acb1);     // Test full vs merged accumulator B
    AccCompare(aca, acc1);     // Test full vs merged accumulator C
}

TEST(TestAutocorrAccumulator, SimpleSerializationMerge)
{
    auto n = 30; // (lag + 2 + lag) * 3 : Regular case
    AutocorrAccumulator<FloatArray> aca, aca1, aca2, aca3;

    decltype(n) i = 0;
    for (; i < 1 * n / 3; ++i)
    {
        aca.AddSample(i);
        aca1.AddSample(i);
    }

    for (; i < 2 * n / 3; ++i)
    {
        aca.AddSample(i);
        aca2.AddSample(i);
    }

    for (; i < n; ++i)
    {
        aca.AddSample(i);
        aca3.AddSample(i);
    }

    // Copy for another merging order
    AutocorrAccumulator<FloatArray> acb1(aca1.GetState()), acb2(aca2.GetState()), acb3(aca3.GetState());

    aca1.Merge(aca2).Merge(aca3);
    acb1.Merge(acb2.Merge(acb3));

    AccCompare(aca, aca1);     // Test full vs merged accumulator A
    AccCompare(aca, acb1);     // Test full vs merged accumulator B
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

    //            relative tolerance    abs tolerance
    auto z = 1;  auto rtol = 5e-5f; auto atol = 5e-5f; auto l = AutocorrAccumState::lag;

    const float expectAutocorr = 0.928175397f;
    auto acorr0 = aca.Autocorrelation().data()[0][z];
    EXPECT_NEAR(expectAutocorr, acorr0, 1.0e-4f);

    // Compare against Eigen
    auto em1 = x.colwise().mean();
    auto em2 = (x.rowwise() - em1).array().square().colwise().sum() / (n-1);
    EXPECT_FLOAT_EQ(em1[0], m11);
    EXPECT_FLOAT_EQ(em2[0], m21);

    auto acr = x.col(z).array() - em1[z];
    auto acorr1 = (acr.head(n-l) * acr.tail(n-l)).sum() / (n-l) / em2[z];

    EXPECT_LE(abs(acorr0-acorr1),   atol + rtol*abs(acorr1));
}

struct TestAutocorrAccumulatorLinspace
    : public ::testing::TestWithParam<std::tuple<int, int>>
{
    const float a = 0.6;
    size_t w = laneSize;
    size_t n = 1080;  // can be evenly divided to all denominators
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
            data[i] = 1*std::sin(a*i);
        }
    }
};

TEST_P(TestAutocorrAccumulatorLinspace, MergeThreshold)
{
    AutocorrAccumulator<FloatArray> aca0, aca1, aca2;

    decltype(n) i = 0;
    for (; i < threshold; ++i)
    {
        aca1.AddSample(data[i]); // And samples BEFORE the threshold
    }

    for (; i < n; ++i)
    {
        aca2.AddSample(data[i]); // And samples AFTER the threshold
    }
    aca0.AddSamples(data.begin(), data.end()); // Add all the samples to aca0

    aca1.Merge(aca2);

    AccCompare(aca0, aca1);     // Test full vs merged accumulator
}

struct TestAutocorrAccumulatorRand
    : public ::testing::TestWithParam<std::tuple<int, int>>
{
    size_t w = laneSize;
    size_t n = 1080;  // can be evenly divided to all denominators
    std::vector<float> data;
    size_t threshold;

    TestAutocorrAccumulatorRand()
        : data (n*w)
    {
        auto nom = std::get<0>(GetParam());
        auto dnom = std::get<1>(GetParam());
        threshold = n * nom / dnom;

        std::default_random_engine gnr;
        std::normal_distribution<float> dist(300.0, 20.0);
        std::generate(data.begin(), data.end(), std::bind(dist, gnr));
        // for (decltype(n) i = 0; i < w*n; ++i) data[i] = 0.2*i;
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
        aca1.AddSample(farr[i]); // And samples BEFORE the threshold
    }

    for (; i < n; ++i)
    {
        aca2.AddSample(farr[i]); // And samples AFTER the threshold
    }
    aca0.AddSamples(farr.begin(), farr.end()); // Add all the samples to aca0

    aca1.Merge(aca2);

    // frame_idx  relative tolerance      abs tolerance
    auto z = 3;  auto rtol = 5e-4f; auto atol = 5e-4f; auto l = AutocorrAccumState::lag;

    AccCompare(aca0, aca1, rtol, atol);     // Test full vs merged accumulator

    // Test full and merged accumulator
    auto cnt0 = aca0.Count().data()[0][z];
    auto m10  = aca0.Mean().data()[0][z];
    auto m20  = aca0.Variance().data()[0][z];
    auto crr0 = aca0.Autocorrelation().data()[0][z];

    // Test against eigen
    Eigen::Map<Eigen::MatrixXf> xr(data.data(), w, n);
    auto em1 = xr.rowwise().mean();
    auto em2 = (xr.colwise() - em1).array().square().rowwise().sum() / (n - 1);

    auto acr1 = xr.row(z).array() - em1[z];
    auto acorr1 = (acr1.head(n-l) * acr1.tail(n-l)).sum() / (n-l) / em2[z];

    EXPECT_EQ(cnt0, n);
    EXPECT_LE(abs(em1[z]-m10),   atol + rtol*abs(m10));
    EXPECT_LE(abs(em2[z]-m20),   atol + rtol*abs(m20));
    EXPECT_LE(abs(acorr1-crr0),  atol + rtol*abs(crr0));
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
