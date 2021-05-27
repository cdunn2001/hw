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

#include <gtest/gtest.h>

#include <common/cuda/utility/CudaTuple.h>

using namespace PacBio::Cuda::Utility;

struct Foo {};

TEST(CudaTuple, Types)
{
    CudaTuple<Foo, int, float, double, Foo> tuple;
    (void)tuple;

    EXPECT_TRUE((std::is_same<Foo&, decltype(tuple.Get<0>())>::value));
    EXPECT_TRUE((std::is_same<int&, decltype(tuple.Get<1>())>::value));
    EXPECT_TRUE((std::is_same<float&, decltype(tuple.Get<2>())>::value));
    EXPECT_TRUE((std::is_same<double&, decltype(tuple.Get<3>())>::value));
    EXPECT_TRUE((std::is_same<Foo&, decltype(tuple.Get<4>())>::value));

    // Extracting Foo by type is impossible, because we don't know
    // if we want the first or the last one
    EXPECT_TRUE((std::is_same<int&, decltype(tuple.Get<int>())>::value));
    EXPECT_TRUE((std::is_same<float&, decltype(tuple.Get<float>())>::value));
    EXPECT_TRUE((std::is_same<double&, decltype(tuple.Get<double>())>::value));

    // Make sure const access works as well.
    const auto& constTuple = tuple;
    (void)constTuple;
    EXPECT_TRUE((std::is_same<const Foo&, decltype(constTuple.Get<0>())>::value));
    EXPECT_TRUE((std::is_same<const int&, decltype(constTuple.Get<1>())>::value));
    EXPECT_TRUE((std::is_same<const float&, decltype(constTuple.Get<2>())>::value));
    EXPECT_TRUE((std::is_same<const double&, decltype(constTuple.Get<3>())>::value));
    EXPECT_TRUE((std::is_same<const Foo&, decltype(constTuple.Get<4>())>::value));

    EXPECT_TRUE((std::is_same<const int&, decltype(constTuple.Get<int>())>::value));
    EXPECT_TRUE((std::is_same<const float&, decltype(constTuple.Get<float>())>::value));
    EXPECT_TRUE((std::is_same<const double&, decltype(constTuple.Get<double>())>::value));

}

TEST(CudaTuple, Update)
{
    CudaTuple<float, int> tuple{};

    EXPECT_FLOAT_EQ(tuple.Get<0>(), 0.);
    EXPECT_FLOAT_EQ(tuple.Get<float>(), 0.);
    EXPECT_EQ(tuple.Get<1>(), 0);
    EXPECT_EQ(tuple.Get<int>(), 0);

    tuple.Get<0>() += 1.1;
    EXPECT_FLOAT_EQ(tuple.Get<0>(), 1.1);
    EXPECT_FLOAT_EQ(tuple.Get<float>(), 1.1);
    EXPECT_EQ(tuple.Get<1>(), 0);
    EXPECT_EQ(tuple.Get<int>(), 0);

    tuple.Get<float>() = 11.;
    EXPECT_FLOAT_EQ(tuple.Get<0>(), 11);
    EXPECT_FLOAT_EQ(tuple.Get<float>(), 11);
    EXPECT_EQ(tuple.Get<1>(), 0);
    EXPECT_EQ(tuple.Get<int>(), 0);

    tuple.Get<1>() += 3;
    EXPECT_FLOAT_EQ(tuple.Get<0>(), 11);
    EXPECT_FLOAT_EQ(tuple.Get<float>(), 11);
    EXPECT_EQ(tuple.Get<1>(), 3);
    EXPECT_EQ(tuple.Get<int>(), 3);

    tuple.Get<int>() = 13;
    EXPECT_FLOAT_EQ(tuple.Get<0>(), 11);
    EXPECT_FLOAT_EQ(tuple.Get<float>(), 11);
    EXPECT_EQ(tuple.Get<1>(), 13);
    EXPECT_EQ(tuple.Get<int>(), 13);
}

TEST(CudaTuple, ConstructAssign)
{
    CudaTuple<int, unsigned int, float, double> tuple(0,1,2,3);
    EXPECT_EQ(tuple.Get<0>(), 0);
    EXPECT_EQ(tuple.Get<1>(), 1);
    EXPECT_EQ(tuple.Get<2>(), 2.f);
    EXPECT_EQ(tuple.Get<3>(), 3.);

    CudaTuple<int, unsigned int, float, double> tuple2(tuple);
    EXPECT_EQ(tuple2.Get<0>(), 0);
    EXPECT_EQ(tuple2.Get<1>(), 1);
    EXPECT_EQ(tuple2.Get<2>(), 2.f);
    EXPECT_EQ(tuple2.Get<3>(), 3.);

    CudaTuple<int, unsigned int, float, double> tuple3{};
    EXPECT_EQ(tuple3.Get<0>(), 0);
    EXPECT_EQ(tuple3.Get<1>(), 0);
    EXPECT_EQ(tuple3.Get<2>(), 0.f);
    EXPECT_EQ(tuple3.Get<3>(), 0.);

    tuple3 = tuple;
    EXPECT_EQ(tuple3.Get<0>(), 0);
    EXPECT_EQ(tuple3.Get<1>(), 1);
    EXPECT_EQ(tuple3.Get<2>(), 2.f);
    EXPECT_EQ(tuple3.Get<3>(), 3.);
}
