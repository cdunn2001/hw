
#include "IntInterval.h"

#include <limits>
#include <boost/numeric/conversion/cast.hpp>
#include <gtest/gtest.h>

using boost::numeric_cast;

// TODO: Restore this when we add support for signed integers in IntInterval.
//struct TestIntIntervalWithInt
//{
//    using IntervalType = PacBio::Primary::IntInterval<int>;

//    static constexpr IntervalType empty {};

//    static constexpr IntervalType a {-3, -1};
//    static constexpr IntervalType b {-1, 1};
//    static constexpr IntervalType c {1, 3};
//    static constexpr IntervalType d {3, 5};

//    static constexpr IntervalType ab {-3, 1};
//    static constexpr IntervalType bc {-1, 3};
//    static constexpr IntervalType cd {1, 5};

//    static constexpr IntervalType abc {-3, 3};
//    static constexpr IntervalType bcd {-1, 5};

//    static constexpr IntervalType abcd {-3, 5};
//};


struct TestIntIntervalWithUInt64
{
    using IntervalType = PacBio::Primary::IntInterval<uint64_t>;

    static constexpr IntervalType empty {};

    static constexpr IntervalType a {0, 2};
    static constexpr IntervalType b {2, 4};
    static constexpr IntervalType c {4, 6};
    static constexpr IntervalType d {6, 8};

    static constexpr IntervalType ab {0, 4};
    static constexpr IntervalType bc {2, 6};
    static constexpr IntervalType cd {4, 8};

    static constexpr IntervalType abc {0, 6};
    static constexpr IntervalType bcd {2, 8};

    static constexpr IntervalType abcd {0, 8};
};


template <typename T>
struct TestIntInterval : public ::testing::Test
{
    using ElementType = typename T::IntervalType::ElementType;
};


using TestTypes = ::testing::Types<TestIntIntervalWithUInt64>;
TYPED_TEST_CASE(TestIntInterval, TestTypes);


TYPED_TEST(TestIntInterval, Basic)
{
    using IntervalType = typename TypeParam::IntervalType;
    constexpr auto empty = TypeParam::empty;
    constexpr auto a = TypeParam::a;
    constexpr auto b = TypeParam::b;

    EXPECT_TRUE(empty.Empty());

    IntervalType x (a);
    EXPECT_EQ(a, x);

    x = b;
    EXPECT_EQ(b, x);

    EXPECT_EQ(b.Lower() + b.Size(), b.Upper());
}

TYPED_TEST(TestIntInterval, OpEqualEqual)
{
    using IntervalType = typename TypeParam::IntervalType;
    constexpr auto empty = TypeParam::empty;
    constexpr auto a = TypeParam::a;
    constexpr auto b = TypeParam::b;
    constexpr auto c = TypeParam::c;
    constexpr auto bc = TypeParam::bc;

    const IntervalType e1 {42, 42};

    EXPECT_TRUE(empty == empty);
    EXPECT_TRUE(empty == e1);
    EXPECT_FALSE(empty == a);
    EXPECT_FALSE(a == e1);

    EXPECT_TRUE(b == b);
    EXPECT_FALSE(b == c);
    EXPECT_FALSE(b == bc);
    EXPECT_FALSE(c == bc);
}

TYPED_TEST(TestIntInterval, OpPlusEqual)
{
    using IntervalType = typename TypeParam::IntervalType;
    using ElementType = typename IntervalType::ElementType;

    constexpr auto a = TypeParam::a;
    constexpr auto cd = TypeParam::cd;

    IntervalType ii {};
    ii += 42;
    EXPECT_TRUE(ii.Empty());

    ii = IntervalType{0, numeric_cast<ElementType>(a.Size())};
    ii += a.Lower();
    EXPECT_EQ(a, ii);

    ii = IntervalType{0, numeric_cast<ElementType>(cd.Size())};
    ii += cd.Lower();
    EXPECT_EQ(cd, ii);
}

TYPED_TEST(TestIntInterval, OpMinusEqual)
{
    using IntervalType = typename TypeParam::IntervalType;
    constexpr auto c = TypeParam::c;
    constexpr auto ab = TypeParam::ab;

    IntervalType ii {};
    ii -= 42;
    EXPECT_TRUE(ii.Empty());

    ii = c;
    ii -= c.Lower();
    EXPECT_EQ(IntervalType(0, c.Size()), ii);

    ii = ab;
    ii -= ab.Lower();
    EXPECT_EQ(IntervalType(0, ab.Size()), ii);
}

TYPED_TEST(TestIntInterval, AddWithSaturation)
{
    using IntervalType = typename TypeParam::IntervalType;
    using ElementType = typename IntervalType::ElementType;
    constexpr auto a = TypeParam::a;
    constexpr auto b = TypeParam::b;

    constexpr auto highest = std::numeric_limits<ElementType>::max();
    IntervalType ii {highest - 3, highest};
    ii.AddWithSaturation(1);
    EXPECT_EQ(highest, ii.Upper());
    EXPECT_EQ(3-1, ii.Size());

    ii = a;
    ii.AddWithSaturation(a.Lower());
    EXPECT_EQ(2*a.Lower(), ii.Lower());
    EXPECT_EQ(a.Size(), ii.Size());

    ii = b;
    ii.AddWithSaturation(0);
    EXPECT_EQ(b, ii);
}

TYPED_TEST(TestIntInterval, SubtractWithSaturation)
{
    using IntervalType = typename TypeParam::IntervalType;
    using ElementType = typename IntervalType::ElementType;
    constexpr auto a = TypeParam::a;
    constexpr auto b = TypeParam::b;

    constexpr auto lowest = std::numeric_limits<ElementType>::min();
    IntervalType ii {lowest, lowest + 5};
    ii.SubtractWithSaturation(2);
    EXPECT_EQ(lowest, ii.Lower());
    EXPECT_EQ(5-2, ii.Size());
    ii.SubtractWithSaturation(10);
    EXPECT_TRUE(ii.Empty());

    ii = a;
    ii.SubtractWithSaturation(a.Lower());
    EXPECT_EQ(0, ii.Lower());
    EXPECT_EQ(a.Size(), ii.Upper());

    ii = b;
    ii.SubtractWithSaturation(0);
    EXPECT_EQ(b, ii);
}

TYPED_TEST(TestIntInterval, Disjoint)
{
    constexpr auto empty = TypeParam::empty;
    constexpr auto a = TypeParam::a;
    constexpr auto b = TypeParam::b;
    constexpr auto c = TypeParam::c;
    constexpr auto ab = TypeParam::ab;
    constexpr auto bc = TypeParam::bc;
    constexpr auto cd = TypeParam::cd;
    constexpr auto abc = TypeParam::abc;

    // Empty set is disjoint from any other set, including itself.
    EXPECT_TRUE(Disjoint(empty, empty));
    EXPECT_TRUE(Disjoint(empty, a));
    EXPECT_TRUE(Disjoint(empty, bc));
    EXPECT_TRUE(Disjoint(abc, empty));

    // No non-empty set is disjoint from itself.
    EXPECT_FALSE(Disjoint(a, a));
    EXPECT_FALSE(Disjoint(bc, bc));
    EXPECT_FALSE(Disjoint(cd, cd));

    // Non-trivial cases.
    EXPECT_FALSE(Disjoint(ab, bc));
    EXPECT_FALSE(Disjoint(bc, ab));
    EXPECT_TRUE(Disjoint(a, b));
    EXPECT_TRUE(Disjoint(c, b));
    EXPECT_FALSE(Disjoint(a, ab));
    EXPECT_FALSE(Disjoint(ab, a));
    EXPECT_FALSE(Disjoint(b, abc));
    EXPECT_FALSE(Disjoint(abc, b));
}

TYPED_TEST(TestIntInterval, IsUnionConnected)
{
    constexpr auto empty = TypeParam::empty;
    constexpr auto a = TypeParam::a;
    constexpr auto b = TypeParam::b;
    constexpr auto c = TypeParam::c;
    constexpr auto d = TypeParam::d;
    constexpr auto ab = TypeParam::ab;
    constexpr auto bc = TypeParam::bc;
    constexpr auto abc = TypeParam::abc;

    EXPECT_TRUE(IsUnionConnected(empty, empty));
    EXPECT_TRUE(IsUnionConnected(empty, a));
    EXPECT_TRUE(IsUnionConnected(a, empty));

    EXPECT_TRUE(IsUnionConnected(a, b));
    EXPECT_TRUE(IsUnionConnected(b, a));
    EXPECT_TRUE(IsUnionConnected(b, abc));
    EXPECT_TRUE(IsUnionConnected(abc, b));
    EXPECT_TRUE(IsUnionConnected(bc, ab));

    EXPECT_FALSE(IsUnionConnected(a, c));
    EXPECT_FALSE(IsUnionConnected(c, a));
    EXPECT_FALSE(IsUnionConnected(b, d));
    EXPECT_FALSE(IsUnionConnected(d, b));
}

TYPED_TEST(TestIntInterval, Intersection)
{
    constexpr auto empty = TypeParam::empty;
    constexpr auto a = TypeParam::a;
    constexpr auto b = TypeParam::b;
    constexpr auto c = TypeParam::c;
    constexpr auto d = TypeParam::d;
    constexpr auto bc = TypeParam::bc;
    constexpr auto cd = TypeParam::cd;
    constexpr auto abc = TypeParam::abc;

    EXPECT_TRUE((Intersection(empty, a).Empty()));
    EXPECT_TRUE((Intersection(a, empty).Empty()));

    EXPECT_EQ(a, Intersection(a, abc));
    EXPECT_EQ(a, Intersection(abc, a));
    EXPECT_EQ(c, Intersection(bc, cd));
    EXPECT_EQ(c, Intersection(cd, bc));

    EXPECT_EQ(b, Intersection(b, abc));
    EXPECT_EQ(b, Intersection(abc, b));

    EXPECT_TRUE(Intersection(a, b).Empty());
    EXPECT_TRUE(Intersection(b, a).Empty());
    EXPECT_TRUE(Intersection(c, d).Empty());
    EXPECT_TRUE(Intersection(d, b).Empty());
}

TYPED_TEST(TestIntInterval, Hull)
{
    constexpr auto empty = TypeParam::empty;
    constexpr auto a = TypeParam::a;
    constexpr auto b = TypeParam::b;
    constexpr auto c = TypeParam::c;
    constexpr auto ab = TypeParam::ab;
    constexpr auto bc = TypeParam::bc;
    constexpr auto abc = TypeParam::abc;

    EXPECT_TRUE(Hull(empty, empty).Empty());
    EXPECT_EQ(a, Hull(empty, a));
    EXPECT_EQ(a, Hull(a, empty));

    EXPECT_EQ(a, Hull(a, a));
    EXPECT_EQ(bc, Hull(b, bc));
    EXPECT_EQ(bc, Hull(bc, b));
    EXPECT_EQ(ab, Hull(a, b));
    EXPECT_EQ(ab, Hull(b, a));
    EXPECT_EQ(abc, Hull(abc, b));
    EXPECT_EQ(abc, Hull(abc, b));
    EXPECT_EQ(abc, Hull(a, c));
    EXPECT_EQ(abc, Hull(c, a));
}
