
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
#include <vector>

#include <gtest/gtest.h>

namespace PacBio::Mongo {

template <typename IntT>
struct TestIntInterval : public ::testing::Test
{
    using IntervalT = IntInterval<IntT>;
    static_assert(std::is_trivially_copy_constructible_v<IntervalT>);

    // Can specialize this for signed IntType so that some negative values are
    // used in the tests.
    static const IntT shift;

    // In several places, we assume that these bounds are far from the limits of
    // the representational.
    const IntervalT ii0;
    const IntervalT ii1 {1 + shift, 2 + shift};
    const IntervalT ii2 {2 + shift, 5 + shift};
    const IntervalT ii3 {3 + shift, 8 + shift};
    const IntervalT ii4 {4 + shift, 6 + shift};
};

template <typename IntT>
const IntT TestIntInterval<IntT>::shift = 0;
template<> const int TestIntInterval<int>::shift = -3;

using IntegerTypes = ::testing::Types<unsigned int, int>;
TYPED_TEST_SUITE(TestIntInterval, IntegerTypes);

TYPED_TEST(TestIntInterval, Basic)
{
    // Making local copies to avoid the tedious need for `this->`.
    const auto ii0 = this->ii0;
    const auto ii1 = this->ii1;
    const auto ii2 = this->ii2;
    const auto shift = TestFixture::shift;
    using IntType = typename TestFixture::IntervalT::ElementType;
    using SizeType = typename TestFixture::IntervalT::SizeType;

    EXPECT_TRUE(ii0.Empty());
    EXPECT_EQ(SizeType{0}, ii0.Size());

    EXPECT_FALSE(ii1.Empty());
    EXPECT_EQ(IntType{1} + shift, ii1.Lower());
    EXPECT_EQ(IntType{2} + shift, ii1.Upper());
    EXPECT_EQ(SizeType{1}, ii1.Size());

    EXPECT_FALSE(ii2.Empty());
    EXPECT_EQ(IntType{2} + shift, ii2.Lower());
    EXPECT_EQ(IntType{5} + shift, ii2.Upper());
}

TYPED_TEST(TestIntInterval, Compare)
{
    // Making local copies to avoid the tedious need for `this->`.
    const auto ii0 = this->ii0;
    const auto ii1 = this->ii1;
    const auto ii2 = this->ii2;
    const auto ii3 = this->ii3;
    const auto ii4 = this->ii4;
    const auto shift = TestFixture::shift;
    using IntervalT = typename TestFixture::IntervalT;

    EXPECT_EQ(ii0, ii0);
    EXPECT_EQ(ii1, ii1);
    EXPECT_EQ(IntervalT(2 + shift, 5 + shift), ii2);

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

TYPED_TEST(TestIntInterval, Clear)
{
    auto i = this->ii0;
    i.Clear();
    EXPECT_TRUE(i.Empty());

    i = this->ii1;
    EXPECT_TRUE(i.Clear().Empty());

    i = this->ii3;
    EXPECT_TRUE(i.Clear().Empty());
}

TYPED_TEST(TestIntInterval, Center)
{
    const auto shift = TestFixture::shift;

    EXPECT_TRUE(std::isnan(this->ii0.Center()));
    EXPECT_EQ(1.0f + shift, this->ii1.Center());
    EXPECT_EQ(3.0f + shift, this->ii2.Center());
    EXPECT_EQ(5.0f + shift, this->ii3.Center());
    EXPECT_EQ(4.5f + shift, this->ii4.Center());
}

TYPED_TEST(TestIntInterval, Disjoint)
{
    // Making local copies to avoid the tedious need for `this->`.
    const auto ii0 = this->ii0;
    const auto ii1 = this->ii1;
    const auto ii2 = this->ii2;
    const auto ii3 = this->ii3;
    const auto ii4 = this->ii4;

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

TYPED_TEST(TestIntInterval, IsUnionConnected)
{
    // Making local copies to avoid the tedious need for `this->`.
    const auto ii0 = this->ii0;
    const auto ii1 = this->ii1;
    const auto ii2 = this->ii2;
    const auto ii3 = this->ii3;
    const auto ii4 = this->ii4;

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

TYPED_TEST(TestIntInterval, AreOrderedAdjacent)
{
    // Making local copies to avoid the tedious need for `this->`.
    const auto ii0 = this->ii0;
    const auto ii1 = this->ii1;
    const auto ii2 = this->ii2;
    const auto ii3 = this->ii3;
    const auto ii4 = this->ii4;

    // Empty interval is ordered and adjacent to any other interval, including
    // itself.
    EXPECT_TRUE(AreOrderedAdjacent(ii0, ii0));
    EXPECT_TRUE(AreOrderedAdjacent(ii0, ii1));
    EXPECT_TRUE(AreOrderedAdjacent(ii1, ii0));
    EXPECT_TRUE(AreOrderedAdjacent(ii0, ii3));
    EXPECT_TRUE(AreOrderedAdjacent(ii3, ii0));

    // Any non-empty interval is not OA with itself.
    EXPECT_FALSE(AreOrderedAdjacent(ii1, ii1));
    EXPECT_FALSE(AreOrderedAdjacent(ii2, ii2));
    EXPECT_FALSE(AreOrderedAdjacent(ii4, ii4));

    // In general AOA is not symmetric.
    EXPECT_TRUE(AreOrderedAdjacent(ii1, ii2));
    EXPECT_FALSE(AreOrderedAdjacent(ii2, ii1));

    // Disjoint but not adjacent example.
    EXPECT_FALSE(AreOrderedAdjacent(ii1, ii3));
    EXPECT_FALSE(AreOrderedAdjacent(ii3, ii1));

    // Overlapping example.
    EXPECT_FALSE(AreOrderedAdjacent(ii2, ii3));
    EXPECT_FALSE(AreOrderedAdjacent(ii3, ii2));

    // Non-trivially nested example.
    EXPECT_FALSE(AreOrderedAdjacent(ii3, ii4));
    EXPECT_FALSE(AreOrderedAdjacent(ii4, ii3));
}

TYPED_TEST(TestIntInterval, Intersection)
{
    // Making local copies to avoid the tedious need for `this->`.
    const auto ii0 = this->ii0;
    const auto ii1 = this->ii1;
    const auto ii2 = this->ii2;
    const auto ii3 = this->ii3;
    const auto ii4 = this->ii4;
    const auto shift = TestFixture::shift;
    using IntervalT = typename TestFixture::IntervalT;

    EXPECT_TRUE(Intersection(ii0, ii0).Empty());
    EXPECT_TRUE(Intersection(ii0, ii1).Empty());
    EXPECT_TRUE(Intersection(ii1, ii0).Empty());

    EXPECT_TRUE(Intersection(ii1, ii2).Empty());
    EXPECT_TRUE(Intersection(ii2, ii1).Empty());
    EXPECT_TRUE(Intersection(ii1, ii3).Empty());
    EXPECT_TRUE(Intersection(ii3, ii1).Empty());

    const IntervalT ii2i3 {3 + shift, 5 + shift};
    EXPECT_EQ(ii2i3, Intersection(ii2, ii3));
    EXPECT_EQ(ii2i3, Intersection(ii3, ii2));

    EXPECT_EQ(ii1, Intersection(ii1, ii1));
    EXPECT_EQ(ii2, Intersection(ii2, ii2));
    EXPECT_EQ(ii4, Intersection(ii3, ii4));
    EXPECT_EQ(ii4, Intersection(ii4, ii3));
}

TYPED_TEST(TestIntInterval, Hull)
{
    // Making local copies to avoid the tedious need for `this->`.
    const auto ii0 = this->ii0;
    const auto ii1 = this->ii1;
    const auto ii2 = this->ii2;
    const auto ii3 = this->ii3;
    const auto ii4 = this->ii4;
    const auto shift = TestFixture::shift;
    using IntervalT = typename TestFixture::IntervalT;

    EXPECT_TRUE(Hull(ii0, ii0).Empty());
    EXPECT_EQ(ii1, Hull(ii0, ii1));
    EXPECT_EQ(ii2, Hull(ii2, ii0));

    EXPECT_EQ(ii2, Hull(ii2, ii2));
    EXPECT_EQ(ii3, Hull(ii3, ii3));

    EXPECT_EQ(IntervalT(1 + shift, 5 + shift), Hull(ii1, ii2));
    EXPECT_EQ(IntervalT(1 + shift, 5 + shift), Hull(ii2, ii1));
    EXPECT_EQ(IntervalT(1 + shift, 8 + shift), Hull(ii1, ii3));
    EXPECT_EQ(IntervalT(1 + shift, 8 + shift), Hull(ii3, ii1));
    EXPECT_EQ(IntervalT(2 + shift, 8 + shift), Hull(ii2, ii3));
    EXPECT_EQ(IntervalT(2 + shift, 8 + shift), Hull(ii3, ii2));

    EXPECT_EQ(ii3, Hull(ii3, ii4));
    EXPECT_EQ(ii3, Hull(ii4, ii3));
}

TYPED_TEST(TestIntInterval, Add)
{
    // Making local copies to avoid the tedious need for `this->`.
    const auto ii0 = this->ii0;
    const auto ii1 = this->ii1;
    const auto ii2 = this->ii2;
    const auto ii3 = this->ii3;
    const auto ii4 = this->ii4;
    using IntervalT = typename TestFixture::IntervalT;
    using IntType = typename IntervalT::ElementType;

    std::vector<IntType> arguments {42};
    if constexpr (std::is_signed_v<IntType>) arguments.push_back(-47);

    for (const IntType a : arguments)
    {
        // Check addition on empty interval.
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
    }

    // Check AddWithSaturation w/ actual saturation.
    constexpr auto mx = std::numeric_limits<IntType>::max();
    {
        IntervalT i {ii2};
        auto b = mx - ii2.Upper();
        // Expect no saturation.
        i.AddWithSaturation(b);
        EXPECT_EQ(ii2.Size(), i.Size());
        EXPECT_EQ(ii2.Lower() + b, i.Lower());
        EXPECT_EQ(ii2.Upper() + b, i.Upper());

        i = ii2;
        b = (i.Lower() + i.Upper())/IntType(2);
        ASSERT_LT(b, ii2.Upper());
        ASSERT_GT(b, ii2.Lower());
        ASSERT_GE(b, IntType(0))
            << "Test error: next statement would cause integer overflow.";
        b = mx - b;
        // Expect saturation of Upper only.
        i.AddWithSaturation(b);
        EXPECT_LT(i.Size(), ii2.Size());
        EXPECT_EQ(ii2.Lower() + b, i.Lower());
        EXPECT_EQ(mx, i.Upper());

        i = ii4;
        b = i.Lower();
        ASSERT_GE(b, IntType(0))
            << "Test error: next statement would cause integer overflow.";
        b = mx - b;
        // Expect saturation of Lower and Upper.
        i.AddWithSaturation(b);
        EXPECT_TRUE(i.Empty());
    }
}

TYPED_TEST(TestIntInterval, Subtract)
{
    // Making local copies to avoid the tedious need for `this->`.
    const auto ii0 = this->ii0;
    const auto ii1 = this->ii1;
    const auto ii2 = this->ii2;
    const auto ii3 = this->ii3;
    const auto shift = TestFixture::shift;
    using IntervalT = typename TestFixture::IntervalT;
    using IntType = typename IntervalT::ElementType;

    constexpr auto mn = std::numeric_limits<IntType>::min();

    std::vector<IntType> arguments {1};
    if constexpr (std::is_signed_v<IntType>) arguments.push_back(-5);

    for (const IntType a : arguments)
    {
        IntervalT i0 {ii0};
        i0 -= a;
        EXPECT_TRUE(i0.Empty());
        i0.SubtractWithSaturation(a);
        EXPECT_TRUE(i0.Empty());

        // Here only use test intervals for which Lower() - a >= mn.
        for (const auto& iix : {ii1, ii2, ii3})
        {
            if (a < 0) ASSERT_GE(iix.Lower() - a, mn);
            else ASSERT_GE(iix.Lower(), mn + a);
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
    }

    // Note that for signed integers, typically min = -max - 1, and -min overflows.
    // Split mn into two parts.
    constexpr auto mnh1 = mn/2;
    constexpr auto mnh2 = mn - mnh1;
    ASSERT_EQ(mn, mnh1 + mnh2) << "Test error.";
    {
        IntervalT i {ii2};
        auto b = ii2.Lower();
        // Expect no saturation.
        i.SubtractWithSaturation(b - mnh1);
        i.SubtractWithSaturation(-mnh2);
        EXPECT_EQ(ii2.Size(), i.Size());
        EXPECT_EQ(mn, i.Lower());
        EXPECT_EQ(mn + ii2.Size(), i.Upper());

        i = ii2;
        b = IntType(3) + shift;
        ASSERT_LT(b, ii2.Upper());
        ASSERT_GT(b, ii2.Lower());
        // Expect saturation of Lower only.
        i.SubtractWithSaturation(b - mnh1);
        i.SubtractWithSaturation(-mnh2);
        EXPECT_LT(i.Size(), ii2.Size());
        EXPECT_EQ(mn, i.Lower());
        EXPECT_EQ(mn + ii2.Upper() - b, i.Upper());

        i = ii2;
        b = ii2.Upper();
        // Expect saturation of Lower and Upper.
        i.SubtractWithSaturation(b - mnh1);
        i.SubtractWithSaturation(-mnh2);
        EXPECT_TRUE(i.Empty());
    }
}

// Tests relationships involving operations and predicates.
TYPED_TEST(TestIntInterval, OperationalRelationships)
{
    // Making local copies to avoid the tedious need for `this->`.
    const auto ii0 = this->ii0;
    const auto ii1 = this->ii1;
    const auto ii2 = this->ii2;
    const auto ii3 = this->ii3;
    const auto ii4 = this->ii4;

    const auto intervalList = {ii0, ii1, ii2, ii3, ii4};
    for (const auto i1 : intervalList)
    {
        for (const auto i2 : intervalList)
        {
            if (Disjoint(i1, i2))
            {
                EXPECT_TRUE(Intersection(i1, i2).Empty());
                EXPECT_GE(Hull(i1, i2).Size(), i1.Size() + i2.Size());
            }
            else {
                EXPECT_LT(Hull(i1, i2).Size(), i1.Size() + i2.Size());
                EXPECT_TRUE(IsUnionConnected(i1, i2));
            }

            if (AreOrderedAdjacent(i1, i2))
            {
                EXPECT_TRUE(Disjoint(i1, i2));
                EXPECT_TRUE(IsUnionConnected(i1, i2));
                EXPECT_EQ(Hull(i1, i2).Size(), i1.Size() + i2.Size());
            }
        }
    }
}

}   // namespace PacBio::Mongo
