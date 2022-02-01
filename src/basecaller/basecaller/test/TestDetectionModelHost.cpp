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
//  Unit tests for class DetectionModelHost.

#include <basecaller/traceAnalysis/DetectionModelHost.h>

#include <iostream>

#include <gtest/gtest.h>

#include <common/LaneArray.h>
#include <dataTypes/LaneDetectionModel.h>

namespace {

template <typename T>
bool equalToHalfPrecision(const T& a, const T& b)
{
    constexpr float halfEps = 0.000978;
    return all(abs(a - b) < halfEps * max(a, b));
}

}   // anonymous namespace

namespace PacBio::Mongo::Data {

using DetModelHost = DetectionModelHost<LaneArray<float>>;

// Test conversion from DetectionModelHost to LaneDetectionModel and back.
TEST(TestDetectionModelHost, LaneDetectionModelConversion)
{
    const FrameIntervalType fi {0u, 42u};
    DetModelHost dmh0 {MockLaneDetectionModel<Cuda::PBHalf>(), fi};
    DetModelHost dmh1 {dmh0};
    EXPECT_TRUE(all(dmh0 == dmh1));
    LaneDetectionModel<Cuda::PBHalf> ldm;

    const auto roundTrip = [&]()
    {
        dmh0.ExportTo(&ldm);

        // Move assignment invalidates references to elements of DetectionModes.
        dmh1 = DetModelHost{ldm, fi};
    };

    // Check roundtrip after modifying each property of dmh0.
    {
        auto& blm0 = dmh0.BaselineMode();
        const auto& blm1 = dmh1.BaselineMode();

        blm0.Weight(blm0.Weight() + 0.02f);
        EXPECT_FALSE(equalToHalfPrecision(blm0.Weight(), blm1.Weight()));
        roundTrip();
        EXPECT_TRUE(equalToHalfPrecision(blm0.Weight(), blm1.Weight()));

        blm0.SignalMean(blm0.SignalMean() + 1.0f);
        EXPECT_FALSE(equalToHalfPrecision(blm0.SignalMean(), blm1.SignalMean()));
        roundTrip();
        EXPECT_TRUE(equalToHalfPrecision(blm0.SignalMean(), blm1.SignalMean()));

        blm0.SignalCovar(blm0.SignalCovar() + 2.0f);
        EXPECT_FALSE(equalToHalfPrecision(blm0.SignalCovar(), blm1.SignalCovar()));
        roundTrip();
        EXPECT_TRUE(equalToHalfPrecision(blm0.SignalCovar(), blm1.SignalCovar()));
    }

    auto& dm0 = dmh0.DetectionModes();
    const auto& dm1 = dmh1.DetectionModes();
    ASSERT_EQ(dm0.size(), dm1.size());
    for (unsigned int i = 0; i < dm0.size(); ++i)
    {
        dm0[i].Weight(dm0[i].Weight() + 0.01f);
        EXPECT_FALSE(equalToHalfPrecision(dm0[i].Weight(), dm1[i].Weight()));
        roundTrip();
        EXPECT_TRUE(equalToHalfPrecision(dm0[i].Weight(), dm1[i].Weight()));

        dm0[i].SignalMean(dm0[i].SignalMean() + 1.0f);
        EXPECT_FALSE(equalToHalfPrecision(dm0[i].SignalMean(), dm1[i].SignalMean()));
        roundTrip();
        EXPECT_TRUE(equalToHalfPrecision(dm0[i].SignalMean(), dm1[i].SignalMean()));

        dm0[i].SignalCovar(dm0[i].SignalCovar() + 2.0f);
        EXPECT_FALSE(equalToHalfPrecision(dm0[i].SignalCovar(), dm1[i].SignalCovar()));
        roundTrip();
        EXPECT_TRUE(equalToHalfPrecision(dm0[i].SignalCovar(), dm1[i].SignalCovar()));
    }

    dmh0.Confidence(dmh0.Confidence() + 0.2f);
    EXPECT_FALSE(equalToHalfPrecision(dmh0.Confidence(), dmh1.Confidence()));
    roundTrip();
    EXPECT_TRUE(equalToHalfPrecision(dmh0.Confidence(), dmh1.Confidence()));
}

TEST(TestDetectionModelHost, EvolveConfidence)
{
    DetModelHost dm {MockLaneDetectionModel<Cuda::PBHalf>(), {0u, 100u}};
    dm.Confidence(1.0f);
    dm.EvolveConfidence({100u, 200u}, 100.0f);
    EXPECT_TRUE(all(dm.Confidence() == 0.5f));

    dm.EvolveConfidence({200u, 300u}, 200.f);
    const float expect = 0.5f * std::sqrt(0.5f);
    EXPECT_TRUE(all(abs(dm.Confidence() - expect) < 1.0e-6f * expect));
}

}   // namespace PacBio::Mongo::Data
