// Copyright (c) 2015, Pacific Biosciences of California, Inc.
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
//  Unit tests for CircularArray

#include <common/CircularArray.h>

#include <algorithm>
#include <iterator>
#include <vector>

#include <gtest/gtest.h>

#include <pacbio/PBException.h>

namespace PacBio::Mongo {

TEST(TestCircularArray, PushPopFront)
{
    constexpr unsigned int n = 4;
    std::array<int, n+1u> a {2, -3, 5, -7, 11};

    CircularArray<int, n> ca;
    ASSERT_EQ(n, ca.capacity());
    ASSERT_TRUE(ca.empty());

    unsigned int i = 0;
    while (i < n)
    {
        ca.push_front(a.at(i++));
        EXPECT_EQ(i, ca.size());
        for (unsigned int j = 0; j < ca.size(); ++j)
        {
            EXPECT_EQ(a.at(i - j - 1), ca[j]);
        }
    }

    ASSERT_TRUE(ca.full());
    ca.push_front(a.at(i++));
    EXPECT_EQ(n, ca.size());
    for (unsigned int j = 0; j < ca.size(); ++j)
    {
        EXPECT_EQ(a.at(i - j - 1), ca[j]);
    }

    EXPECT_EQ(n, ca.capacity());
    EXPECT_EQ(n, ca.size());

    while (!ca.empty())
    {
        const auto cas = ca.size();
        EXPECT_EQ(a.at(cas), ca.front());
        for (unsigned int j = 0; j < cas; ++j)
        {
            EXPECT_EQ(a.at(cas - j), ca[j]);
        }
        ca.pop_front();
    }
}

TEST(TestCircularArray, PushPopBack)
{
    constexpr unsigned int n = 4;
    std::array<int, n+1u> a {2, -3, 5, -7, 11};

    CircularArray<int, n> ca;
    ASSERT_EQ(n, ca.capacity());
    ASSERT_TRUE(ca.empty());

    unsigned int i = 0;
    while (i < n)
    {
        ca.push_back(a.at(i++));
        EXPECT_EQ(i, ca.size());
        EXPECT_EQ(a.at(i-1), ca.back());
        for (unsigned int j = 0; j < ca.size(); ++j) {
            EXPECT_EQ(a.at(j), ca[j]);
        }
    }

    ASSERT_TRUE(ca.full());
    ca.push_back(a.at(i++));
    EXPECT_EQ(n, ca.size());
    for (size_t j = 0; j < ca.size(); ++j)
    {
        EXPECT_EQ(a.at(j + 1), ca[j]);
    }

    EXPECT_EQ(n, ca.capacity());
    EXPECT_EQ(n, ca.size());

    while (!ca.empty())
    {
        EXPECT_EQ(a.at(ca.size()), ca.back());
        for (unsigned int j = 0; j < ca.size(); ++j)
        {
            EXPECT_EQ(a.at(j + 1), ca[j]);
        }
        ca.pop_back();
    }

}

TEST(TestCircularArray, Fill)
{
    constexpr unsigned int n = 50;
    ASSERT_EQ(0u, n % 2u);

    CircularArray<int, n> ca;
    ASSERT_TRUE(ca.empty());
    ASSERT_EQ(0u, ca.size());
    ASSERT_EQ(n, ca.capacity());

    std::fill_n(std::back_inserter(ca), ca.capacity(), 42);
    EXPECT_EQ(n, ca.capacity());
    EXPECT_EQ(n, ca.size());
    EXPECT_TRUE(ca.full());
    for (unsigned int i = 0; i < ca.size(); ++i) EXPECT_EQ(42, ca[i]);

    std::fill_n(std::front_inserter(ca), ca.capacity()/2u, 13);
    unsigned int i = 0;
    while (i < ca.size()/2u) EXPECT_EQ(13, ca[i++]);
    while (i < ca.size()) EXPECT_EQ(42, ca[i++]);

    EXPECT_FALSE(ca.empty());
    ca.clear();
    EXPECT_TRUE(ca.empty());
}

TEST(TestCircularArray, HalfFull)
{
    constexpr unsigned int n = 60;
    ASSERT_EQ(0u, n % 2u);

    CircularArray<int, n> ca;
    ASSERT_TRUE(ca.empty());
    ASSERT_EQ(0u, ca.size());
    ASSERT_EQ(n, ca.capacity());

    for (unsigned int i = 0; i < n / 2u; ++i) ca.push_back(i);

    EXPECT_EQ(n, ca.capacity());
    EXPECT_EQ(n / 2u, ca.size());
    EXPECT_FALSE(ca.full());
    EXPECT_FALSE(ca.empty());
    for (unsigned int i = 0; i < ca.size(); ++i) EXPECT_EQ(i, ca[i]);

    EXPECT_FALSE(ca.empty());
    ca.clear();
    EXPECT_TRUE(ca.empty());
}

}   // namespace PacBio::Mongo
