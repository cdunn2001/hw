// Copyright (c) 2019, Pacific Biosciences of California, Inc.
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
//  Defines unit tests for classes LaneArray, LaneMask, LaneArrayRef, and
//  ConstLaneArrayRef.

#include <common/LaneArray.h>

#include <algorithm>
#include <vector>

#include <gtest/gtest.h>

namespace PacBio {
namespace Mongo {

TEST(TestLaneMask, Ops)
{
    LaneMask<laneSize> x{true};
    EXPECT_TRUE(all(x));
    x[0] = false;
    EXPECT_FALSE(all(x));
    EXPECT_TRUE(any(x));
    EXPECT_FALSE(none(x));

    LaneMask<laneSize> y{true};
    EXPECT_TRUE(all(x | y));
    EXPECT_FALSE(all(x & y));

    EXPECT_TRUE(none(!y));
}


struct TestLaneArrayBase
{
    static constexpr size_t n = 10u;
    using ElementType = int;
    using VecType = std::vector<ElementType>;

    const VecType vPrime {2,  3,  5,  7, 11, 13, 17, 19, 23, 29};
    const VecType vFibon {1,  2,  3,  5,  8, 13, 21, 34, 55, 89};
};


struct TestConstLaneArrayRef : public ::testing::Test, TestLaneArrayBase
{
    using ConstLarType = ConstLaneArrayRef<ElementType, n>;

    void SetUp()
    {
        ASSERT_LE(ConstLarType::Size(), vPrime.size());
        ASSERT_LE(ConstLarType::Size(), vFibon.size());
    }
};


TEST_F(TestConstLaneArrayRef, CopyAndElementAccess)
{
    VecType va (vPrime);
    ConstLarType lara (va.data());
    ConstLarType larb (lara);

    EXPECT_EQ(va.data(), lara.Data());
    EXPECT_EQ(va.data(), larb.Data());

    for (size_t i = 0; i < n; ++i)
    {
        ASSERT_EQ(va[i], lara[i]);
        ASSERT_EQ(va[i], larb[i]);
    }

    ASSERT_EQ(vFibon.size(), va.size());
    std::copy(vFibon.cbegin(), vFibon.cend(), va.begin());

    for (size_t i = 0; i < n; ++i)
    {
        ASSERT_EQ(vFibon[i], va[i]);
        ASSERT_EQ(vFibon[i], lara[i]);
        ASSERT_EQ(vFibon[i], lara[i]);
    }
}


TEST_F(TestConstLaneArrayRef, Iterators)
{
    ConstLarType larPrime (vPrime.data());
    EXPECT_EQ(larPrime.end(), larPrime.begin() + larPrime.Size());

    const auto j = n/2;
    auto vi = vPrime.cbegin();
    auto lari = larPrime.begin();
    EXPECT_EQ(vi[j], lari[j]);
    while (vi != vPrime.cend())
    {
        ASSERT_NE(larPrime.end(), lari);
        ASSERT_EQ(*vi++, *lari++);
    }
}

TEST_F(TestConstLaneArrayRef, Comparisons)
{
    ConstLarType larPrime (vPrime.data());
    ConstLarType larFibon (vFibon.data());

    auto result = (larPrime == larFibon);
    for (unsigned int i = 0; i < n; ++i)
    {
        ASSERT_EQ(vPrime[i] == vFibon[i], result[i]);
    }

    result = (larPrime != larFibon);
    for (unsigned int i = 0; i < n; ++i)
    {
        ASSERT_EQ(vPrime[i] != vFibon[i], result[i]);
    }

    result = (larPrime < larFibon);
    for (unsigned int i = 0; i < n; ++i)
    {
        ASSERT_EQ(vPrime[i] < vFibon[i], result[i]);
    }

    result = (larPrime <= larFibon);
    for (unsigned int i = 0; i < n; ++i)
    {
        ASSERT_EQ(vPrime[i] <= vFibon[i], result[i]);
    }

    result = (larPrime > larFibon);
    for (unsigned int i = 0; i < n; ++i)
    {
        ASSERT_EQ(vPrime[i] > vFibon[i], result[i]);
    }

    result = (larPrime >= larFibon);
    for (unsigned int i = 0; i < n; ++i)
    {
        ASSERT_EQ(vPrime[i] >= vFibon[i], result[i]);
    }
}


struct TestLaneArrayRef : public ::testing::Test, TestLaneArrayBase
{
    using LarType = LaneArrayRef<ElementType, n>;
    using ConstLarType = ConstLaneArrayRef<ElementType, n>;

    void SetUp()
    {
        ASSERT_LE(LarType::Size(), vPrime.size());
        ASSERT_LE(LarType::Size(), vFibon.size());
    }
};


TEST_F(TestLaneArrayRef, CopyAndElementAccess)
{
    VecType v (vPrime);
    LarType lar1 (v.data());
    LarType lar2 (lar1);

    EXPECT_EQ(v.data(), lar1.Data());
    EXPECT_EQ(v.data(), lar2.Data());

    for (size_t i = 0; i < n; ++i)
    {
        ASSERT_EQ(v[i], lar1[i]);
        ASSERT_EQ(v[i], lar2[i]);
    }

    ASSERT_EQ(vFibon.size(), lar2.Size());
    for (unsigned int i = 0; i < n; ++i)
    {
        lar2[i] = vFibon[i];
        ASSERT_EQ(vFibon[i], v[i]);
        ASSERT_EQ(vFibon[i], lar1[i]);
        ASSERT_EQ(vFibon[i], lar1[i]);
    }
}


TEST_F(TestLaneArrayRef, Assignment)
{
    VecType v (n, 0);
    LarType lar (v.data());

    // Assignment from ConstLaneArrayRef.
    const ConstLarType clarPrime (vPrime.data());
    lar = clarPrime;
    for (unsigned int i = 0; i < n; ++i)
    {
        ASSERT_EQ(vPrime[i], lar[i]);
        ASSERT_EQ(vPrime[i], v[i]);
    }

    // Assignment from LaneArrayRef.
    VecType vf (vFibon);
    const LarType larFibon (vf.data());
    lar = larFibon;
    for (unsigned int i = 0; i < n; ++i)
    {
        ASSERT_EQ(vFibon[i], lar[i]);
        ASSERT_EQ(vFibon[i], v[i]);
    }

    // Assign fixed value.
    lar = 3;
    for (unsigned int i = 0; i < n; ++i)
    {
        ASSERT_EQ(3, v[i]);
    }
}

TEST_F(TestLaneArrayRef, Iterators)
{
    VecType v (vPrime);
    LarType lar (v.data());
    std::sort(lar.begin(), lar.end(), std::greater<int>());
    ASSERT_EQ(vPrime.size(), lar.Size());
    const auto mm = std::mismatch(vPrime.crbegin(), vPrime.crend(), lar.cbegin());
    EXPECT_EQ(mm.first, vPrime.crend());
    EXPECT_EQ(mm.second, lar.cend());
    for (unsigned int i = 0; i < n; ++i)
    {
        ASSERT_EQ(vPrime[n-1-i], v[i]);
    }
}


TEST_F(TestLaneArrayRef, CompoundAssignmentScalar)
{
    const ElementType a = 7;

    {   // +=
        VecType v (vFibon);
        LarType lar (v.data());
        lar += a;
        for (unsigned int i = 0; i < n; ++i)
        {
            ASSERT_EQ(vFibon[i] + a, v[i]);
        }
    }

    {   // -=
        VecType v (vFibon);
        LarType lar (v.data());
        lar -= a;
        for (unsigned int i = 0; i < n; ++i)
        {
            ASSERT_EQ(vFibon[i] - a, v[i]);
        }
    }

    {   // *=
        VecType v (vFibon);
        LarType lar (v.data());
        lar *= a;
        for (unsigned int i = 0; i < n; ++i)
        {
            ASSERT_EQ(vFibon[i] * a, v[i]);
        }
    }

    {   // /=
        VecType v (vFibon);
        LarType lar (v.data());
        lar /= a;
        for (unsigned int i = 0; i < n; ++i)
        {
            ASSERT_EQ(vFibon[i] / a, v[i]);
        }
    }
}


TEST_F(TestLaneArrayRef, CompoundAssignmentArray)
{
    VecType a (vPrime);
    for (auto& x : a) x /= 2;
    ConstLarType lara (a.data());

    {   // +=
        VecType v (vFibon);
        LarType lar (v.data());
        lar += lara;
        for (unsigned int i = 0; i < n; ++i)
        {
            ASSERT_EQ(vFibon[i] + a[i], v[i]);
        }
    }

    {   // -=
        VecType v (vFibon);
        LarType lar (v.data());
        lar -= lara;
        for (unsigned int i = 0; i < n; ++i)
        {
            ASSERT_EQ(vFibon[i] - a[i], v[i]);
        }
    }

    {   // *=
        VecType v (vFibon);
        LarType lar (v.data());
        lar *= lara;
        for (unsigned int i = 0; i < n; ++i)
        {
            ASSERT_EQ(vFibon[i] * a[i], v[i]);
        }
    }

    {   // /=
        VecType v (vFibon);
        LarType lar (v.data());
        lar /= lara;
        for (unsigned int i = 0; i < n; ++i)
        {
            ASSERT_EQ(vFibon[i] / a[i], v[i]);
        }
    }
}


struct TestLaneArray : public ::testing::Test, TestLaneArrayBase
{
    using LaneArrayType = LaneArray<ElementType, n>;
    using ConstLarType = ConstLaneArrayRef<ElementType, n>;


    void SetUp()
    {
        ASSERT_LE(LaneArrayType::Size(), vPrime.size());
        ASSERT_LE(LaneArrayType::Size(), vFibon.size());
    }
};


TEST_F(TestLaneArray, Ops)
{
    LaneArray<short, laneSize> one{1};
    LaneArray<short, laneSize> zero{0};

    EXPECT_TRUE(all(one >= zero));
    EXPECT_TRUE(all((one * zero) == zero));
    EXPECT_TRUE(none(((one + one) * zero) > one));
    EXPECT_TRUE(any((one - one) == zero));
}


TEST_F(TestLaneArray, ConstructAndElementAccess)
{
    // Constructor that initializes with fixed value.
    LaneArrayType five (5);
    for (unsigned int i = 0; i < n; ++i)
    {
        ASSERT_EQ(5, five[i]);
    }

    // Initialization with sequence range represented by iterator pair.
    const LaneArrayType laPrime (vPrime.begin(), vPrime.end());
    for (unsigned int i = 0; i < n; ++i)
    {
        ASSERT_EQ(vPrime[i], laPrime[i]);
    }

    // Initialize with ConstLaneArrayRef.
    ConstLarType larFibon (vFibon.data());
    {
        LaneArrayType la (larFibon);
        for (unsigned int i = 0; i < n; ++i)
        {
            ASSERT_EQ(vFibon[i], la[i]);
        }
    }

    // Copy construction.
    {
        LaneArrayType la (laPrime);
        for (unsigned int i = 0; i < n; ++i)
        {
            ASSERT_EQ(vPrime[i], la[i]);
        }
    }

    // Default construction and copy assignment.
    LaneArrayType la;
    {
        la = laPrime;
        for (unsigned int i = 0; i < n; ++i)
        {
            ASSERT_EQ(vPrime[i], la[i]);
        }
    }

    // Assignment from ConstLaneArrayRef.
    la = larFibon;
    for (unsigned int i = 0; i < n; ++i)
    {
        ASSERT_EQ(vFibon[i], la[i]);
    }

    // Assign fixed value.
    la = 7;
    for (unsigned int i = 0; i < n; ++i)
    {
        ASSERT_EQ(7, la[i]);
    }
}


TEST_F(TestLaneArray, Iterators)
{
    LaneArrayType la (vPrime.begin(), vPrime.end());
    const LaneArrayType cla (vFibon.begin(), vFibon.end());
    std::copy(cla.begin(), cla.end(), la.begin());
    for (unsigned int i = 0; i < n; ++i)
    {
        ASSERT_EQ(vFibon[i], la[i]);
    }
}

TEST_F(TestLaneArray, ComparisonOps)
{
    const LaneArrayType lap (vPrime.begin(), vPrime.end());
    const LaneArrayType laf (vFibon.begin(), vFibon.end());

    // ==
    auto result = lap == laf;
    for (unsigned int i = 0; i < n; ++i)
    {
        ASSERT_EQ(vPrime[i] == vFibon[i], result[i]);
    }

    // !=
    result = lap != laf;
    for (unsigned int i = 0; i < n; ++i)
    {
        ASSERT_EQ(vPrime[i] != vFibon[i], result[i]);
    }

    // <
    result = lap < laf;
    for (unsigned int i = 0; i < n; ++i)
    {
        ASSERT_EQ(vPrime[i] < vFibon[i], result[i]);
    }

    // <=
    result = lap <= laf;
    for (unsigned int i = 0; i < n; ++i)
    {
        ASSERT_EQ(vPrime[i] <= vFibon[i], result[i]);
    }

    // >
    result = lap > laf;
    for (unsigned int i = 0; i < n; ++i)
    {
        ASSERT_EQ(vPrime[i] > vFibon[i], result[i]);
    }

    // >=
    result = lap >= laf;
    for (unsigned int i = 0; i < n; ++i)
    {
        ASSERT_EQ(vPrime[i] >= vFibon[i], result[i]);
    }
}


TEST_F(TestLaneArray, NamedBinaryOps)
{
    const LaneArrayType lap (vPrime.begin(), vPrime.end());
    const LaneArrayType laf (vFibon.begin(), vFibon.end());

    // max
    auto result = max(lap, laf);
    for (unsigned int i = 0; i < n; ++i)
    {
        ASSERT_EQ(std::max(vPrime[i], vFibon[i]), result[i]);
    }

    // min
    result = min(lap, laf);
    for (unsigned int i = 0; i < n; ++i)
    {
        ASSERT_EQ(std::min(vPrime[i], vFibon[i]), result[i]);
    }
}


TEST_F(TestLaneArray, BinaryArithmeticOps)
{
    const LaneArrayType lap (vPrime.begin(), vPrime.end());
    const LaneArrayType laf (vFibon.begin(), vFibon.end());
    const ElementType a = 2;
    const ElementType b = 3;

    // +
    auto result = a + lap + laf + b;
    for (unsigned int i = 0; i < n; ++i)
    {
        ASSERT_EQ(a + vPrime[i] + vFibon[i] + b, result[i]);
    }

    // -
    result = a - lap - laf - b;
    for (unsigned int i = 0; i < n; ++i)
    {
        ASSERT_EQ(a - vPrime[i] - vFibon[i] - b, result[i]);
    }

    // *
    result = a * lap * laf * b;
    for (unsigned int i = 0; i < n; ++i)
    {
        ASSERT_EQ(a * vPrime[i] * vFibon[i] * b, result[i]);
    }

    // /
    result = 10000 / laf / lap / b;
    for (unsigned int i = 0; i < n; ++i)
    {
        ASSERT_EQ(10000 / vFibon[i] / vPrime[i] / b, result[i]);
    }
}

}}  // namespace PacBio::Mongo
