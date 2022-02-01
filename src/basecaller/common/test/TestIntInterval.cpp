
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
//  Unit tests for IntInterval

#include <common/IntInterval.h>

#include <cmath>
#include <limits>

#include <gtest/gtest.h>

namespace PacBio::Mongo {

struct TestIntInterval : public ::testing::Test
{
    using IntervalT = IntInterval<unsigned int>;
    static_assert(std::is_trivially_copy_constructible_v<IntervalT>);

    const IntervalT ii0;
    const IntervalT ii1 {1u, 2u};
    const IntervalT ii2 {2u, 5u};
    const IntervalT ii3 {3u, 8u};
    const IntervalT ii4 {4u, 6u};
};

TEST_F(TestIntInterval, Basic)
{
    EXPECT_TRUE(ii0.Empty());
    EXPECT_EQ(0u, ii0.Size());

    EXPECT_FALSE(ii1.Empty());
    EXPECT_EQ(1u, ii1.Lower());
    EXPECT_EQ(2u, ii1.Upper());
    EXPECT_EQ(1u, ii1.Size());

    EXPECT_FALSE(ii2.Empty());
    EXPECT_EQ(2u, ii2.Lower());
    EXPECT_EQ(5u, ii2.Upper());
}

TEST_F(TestIntInterval, Compare)
{
    EXPECT_EQ(ii0, ii0);
    EXPECT_EQ(ii1, ii1);
    EXPECT_EQ(IntervalT(2u, 5u), ii2);

    EXPECT_NE(ii0, ii1);
    EXPECT_NE(ii1, ii0);
    EXPECT_NE(ii1, ii2);
    EXPECT_NE(ii2, ii1);
    EXPECT_NE(ii1, ii3);
    EXPECT_NE(ii3, ii1);
    EXPECT_NE(ii3, ii4);
    EXPECT_NE(ii4, ii3);

    IntervalT ii;
    ii = ii1;
    EXPECT_EQ(ii1, ii);
    ii = ii0;
    EXPECT_TRUE(ii.Empty());
}

TEST_F(TestIntInterval, Center)
{
    EXPECT_TRUE(std::isnan(ii0.Center()));
    EXPECT_EQ(1.0f, ii1.Center());
    EXPECT_EQ(3.0f, ii2.Center());
    EXPECT_EQ(5.0f, ii3.Center());
    EXPECT_EQ(4.5f, ii4.Center());
}

TEST_F(TestIntInterval, Disjoint)
{
    // Empty interval is disjoint from any other interval (by definition).
    EXPECT_TRUE(Disjoint(ii0, ii0));
    EXPECT_TRUE(Disjoint(ii0, ii1));
    EXPECT_TRUE(Disjoint(ii0, ii2));
    EXPECT_TRUE(Disjoint(ii3, ii0));

    // "Disjoint" is defined as whether the pair have an empty intersection.
    EXPECT_FALSE(Disjoint(ii1, ii1));
    EXPECT_TRUE(Disjoint(ii1, ii2));
    EXPECT_TRUE(Disjoint(ii2, ii1));
    EXPECT_TRUE(Disjoint(ii1, ii3));
    EXPECT_TRUE(Disjoint(ii3, ii1));
    EXPECT_FALSE(Disjoint(ii2, ii3));
    EXPECT_FALSE(Disjoint(ii3, ii2));
    EXPECT_FALSE(Disjoint(ii3, ii4));
    EXPECT_FALSE(Disjoint(ii4, ii3));
}

TEST_F(TestIntInterval, IsUnionConnected)
{
    // If either operand is empty, result is by definition true.
    EXPECT_TRUE(IsUnionConnected(ii0, ii0));
    EXPECT_TRUE(IsUnionConnected(ii0, ii1));
    EXPECT_TRUE(IsUnionConnected(ii2, ii0));
    EXPECT_TRUE(IsUnionConnected(ii0, ii3));
    EXPECT_TRUE(IsUnionConnected(ii0, ii4));

    EXPECT_TRUE(IsUnionConnected(ii1, ii2));
    EXPECT_TRUE(IsUnionConnected(ii2, ii1));

    EXPECT_FALSE(IsUnionConnected(ii1, ii3));
    EXPECT_FALSE(IsUnionConnected(ii3, ii1));

    EXPECT_TRUE(IsUnionConnected(ii2, ii3));
    EXPECT_TRUE(IsUnionConnected(ii3, ii2));

    EXPECT_TRUE(IsUnionConnected(ii3, ii4));
    EXPECT_TRUE(IsUnionConnected(ii4, ii3));
}

TEST_F(TestIntInterval, Intersection)
{
    EXPECT_TRUE(Intersection(ii0, ii0).Empty());
    EXPECT_TRUE(Intersection(ii0, ii1).Empty());
    EXPECT_TRUE(Intersection(ii1, ii0).Empty());

    EXPECT_TRUE(Intersection(ii1, ii2).Empty());
    EXPECT_TRUE(Intersection(ii2, ii1).Empty());
    EXPECT_TRUE(Intersection(ii1, ii3).Empty());
    EXPECT_TRUE(Intersection(ii3, ii1).Empty());

    const IntervalT ii2i3 {3u, 5u};
    EXPECT_EQ(ii2i3, Intersection(ii2, ii3));
    EXPECT_EQ(ii2i3, Intersection(ii3, ii2));

    EXPECT_EQ(ii1, Intersection(ii1, ii1));
    EXPECT_EQ(ii2, Intersection(ii2, ii2));
    EXPECT_EQ(ii4, Intersection(ii3, ii4));
    EXPECT_EQ(ii4, Intersection(ii4, ii3));
}

TEST_F(TestIntInterval, Hull)
{
    EXPECT_TRUE(Hull(ii0, ii0).Empty());
    EXPECT_EQ(ii1, Hull(ii0, ii1));
    EXPECT_EQ(ii2, Hull(ii2, ii0));

    EXPECT_EQ(ii2, Hull(ii2, ii2));
    EXPECT_EQ(ii3, Hull(ii3, ii3));

    EXPECT_EQ(IntervalT(1u, 5u), Hull(ii1, ii2));
    EXPECT_EQ(IntervalT(1u, 5u), Hull(ii2, ii1));
    EXPECT_EQ(IntervalT(1u, 8u), Hull(ii1, ii3));
    EXPECT_EQ(IntervalT(1u, 8u), Hull(ii3, ii1));
    EXPECT_EQ(IntervalT(2u, 8u), Hull(ii2, ii3));
    EXPECT_EQ(IntervalT(2u, 8u), Hull(ii3, ii2));

    EXPECT_EQ(ii3, Hull(ii3, ii4));
    EXPECT_EQ(ii3, Hull(ii4, ii3));
}

TEST_F(TestIntInterval, Add)
{
    // Check addition on empty interval.
    const IntervalT::ElementType a {42u};
    IntervalT i0 {ii0};
    i0 += a;
    EXPECT_TRUE(i0.Empty());
    EXPECT_TRUE((ii0 + a).Empty());
    i0.AddWithSaturation(a);
    EXPECT_TRUE(i0.Empty());


    // Check += and AddWithSaturation w/o actual saturation.
    for (const auto& iix : {ii1, ii2, ii3})
    {
        IntervalT i {iix};
        const auto s = i.Size();
        i += a;
        EXPECT_EQ(s, i.Size());
        EXPECT_EQ(iix.Lower() + a, i.Lower());
        EXPECT_EQ(iix.Upper() + a, i.Upper());

        IntervalT j {iix};
        j.AddWithSaturation(a);     // Expect no saturation here.
        EXPECT_EQ(i, j);
    }

    // Check integer + interval + integer.
    for (const auto& iix : {ii1, ii2, ii3})
    {
        const IntervalT i {a + iix + a};
        EXPECT_EQ(iix.Size(), i.Size());
        EXPECT_EQ(iix.Lower() + 2u*a, i.Lower());
        EXPECT_EQ(iix.Upper() + 2u*a, i.Upper());
    }

    // Check AddWithSaturation w/ actual saturation.
    constexpr auto mx = std::numeric_limits<IntervalT::ElementType>::max();
    {
        IntervalT i {ii2};
        auto b = mx - ii2.Upper();
        // Expect no saturation.
        i.AddWithSaturation(b);
        EXPECT_EQ(ii2.Size(), i.Size());
        EXPECT_EQ(ii2.Lower() + b, i.Lower());
        EXPECT_EQ(ii2.Upper() + b, i.Upper());

        i = ii2;
        b = mx - 3;
        ASSERT_LT(mx - b, ii2.Upper());
        ASSERT_GT(mx - b, ii2.Lower());
        // Expect saturation of Upper only.
        i.AddWithSaturation(b);
        EXPECT_LT(i.Size(), ii2.Size());
        EXPECT_EQ(ii2.Lower() + b, i.Lower());
        EXPECT_EQ(mx, i.Upper());

        i = ii2;
        b = mx - ii2.Lower();
        // Expect saturation of Lower and Upper.
        i.AddWithSaturation(b);
        EXPECT_TRUE(i.Empty());
    }
}

TEST_F(TestIntInterval, Subtract)
{
    const IntervalT::ElementType a {1u};
    IntervalT i0 {ii0};
    i0 -= a;
    EXPECT_TRUE(i0.Empty());
    i0.SubtractWithSaturation(a);
    EXPECT_TRUE(i0.Empty());

    for (const auto& iix : {ii1, ii2, ii3})
    {
        IntervalT i {iix};
        const auto s = i.Size();
        i -= a;
        EXPECT_EQ(s, i.Size());
        EXPECT_EQ(iix.Lower() - a, i.Lower());
        EXPECT_EQ(iix.Upper() - a, i.Upper());

        IntervalT j {iix};
        j.SubtractWithSaturation(a);     // Expect no saturation here.
        EXPECT_EQ(i, j);
    }

    {
        IntervalT i {ii2};
        auto b = ii2.Lower();
        // Expect no saturation.
        i.SubtractWithSaturation(b);
        EXPECT_EQ(ii2.Size(), i.Size());
        EXPECT_EQ(ii2.Lower() - b, i.Lower());
        EXPECT_EQ(ii2.Upper() - b, i.Upper());

        i = ii2;
        b = 3;
        ASSERT_LT(b, ii2.Upper());
        ASSERT_GT(b, ii2.Lower());
        // Expect saturation of Lower only.
        i.SubtractWithSaturation(b);
        EXPECT_LT(i.Size(), ii2.Size());
        EXPECT_EQ(0, i.Lower());
        EXPECT_EQ(ii2.Upper() - b, i.Upper());

        i = ii2;
        b = ii2.Upper();
        // Expect saturation of Lower and Upper.
        i.SubtractWithSaturation(b);
        EXPECT_TRUE(i.Empty());
    }
}

}   // namespace PacBio::Mongo
