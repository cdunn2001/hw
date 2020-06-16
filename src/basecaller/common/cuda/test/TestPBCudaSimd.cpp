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

#include <thread>

#include <gtest/gtest.h>

#include <common/cuda/PBCudaSimd.h>

// Note: This file still needs tests for PBShort2 and PBHalf2.

using namespace PacBio::Cuda;

TEST(PBCudaSimd, Float2_Construct)
{
    PBFloat2 f1(1.3f);
    EXPECT_EQ(f1.X(), 1.3f);
    EXPECT_EQ(f1.Y(), 1.3f);

    PBFloat2 f2(4.0f, 5.0f);
    EXPECT_EQ(f2.X(), 4.0f);
    EXPECT_EQ(f2.Y(), 5.0f);
}

TEST(PBCudaSimd, Float2_BasicArithmetic)
{
    auto sum = PBFloat2(1, 2) + PBFloat2(3, 4);
    EXPECT_EQ(sum.X(), 4);
    EXPECT_EQ(sum.Y(), 6);

    sum += PBFloat2(5, 6);
    EXPECT_EQ(sum.X(), 9);
    EXPECT_EQ(sum.Y(), 12);

    auto sub = PBFloat2(1, 2) - PBFloat2(3, 4);
    EXPECT_EQ(sub.X(), -2);
    EXPECT_EQ(sub.Y(), -2);

    sub -= PBFloat2(5, 6);
    EXPECT_EQ(sub.X(), -7);
    EXPECT_EQ(sub.Y(), -8);

    auto mul = PBFloat2(1, 2) * PBFloat2(3, 4);
    EXPECT_EQ(mul.X(), 3);
    EXPECT_EQ(mul.Y(), 8);

    mul *= PBFloat2(5, 6);
    EXPECT_EQ(mul.X(), 15);
    EXPECT_EQ(mul.Y(), 48);

    auto div = PBFloat2(1, 2) / PBFloat2(3, 4);
    EXPECT_NEAR(div.X(), .333f, 1e-3);
    EXPECT_NEAR(div.Y(), .5f,   1e-3);

    div /= PBFloat2(5, 6);
    EXPECT_NEAR(div.X(), 0.0666f, 1e-4);
    EXPECT_NEAR(div.Y(), 0.08333, 1e-4);
}
