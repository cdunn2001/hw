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

#include <bazio/encoding/FieldTransforms.h>

#include <gtest/gtest.h>

using namespace PacBio::BazIO;

template <size_t val_>
struct Int {
    static constexpr auto val = val_;
};

TEST(BazTransform, NoTransform)
{
    EXPECT_EQ(Identity::Apply(12u), 12u);
    EXPECT_EQ(Identity::Apply(-12), -12);
    EXPECT_EQ(Identity::Apply(1ul<<54), 1ul<<54);

    EXPECT_EQ(Identity::Revert(12u), 12u);
    EXPECT_EQ(Identity::Revert(-12), -12);
    EXPECT_EQ(Identity::Revert(1ul<<54), 1ul<<54);
}

TEST(BazTransform, FixedPoint)
{
    EXPECT_EQ(FixedPointU32::Apply(3.125412, FixedPointScale{10}), 31);
    EXPECT_EQ(FixedPointU32::Apply(3.125412, FixedPointScale{100}), 313);
    EXPECT_EQ(FixedPointU32::Apply(3.125412, FixedPointScale{1000}), 3125);
    EXPECT_EQ(FixedPointU32::Apply(3.125412, FixedPointScale{30}), 94);

    EXPECT_FLOAT_EQ(FixedPointU32::Revert(31, FixedPointScale{10}), 3.1);
    EXPECT_FLOAT_EQ(FixedPointU32::Revert(313, FixedPointScale{100}), 3.13);
    EXPECT_FLOAT_EQ(FixedPointU32::Revert(3125, FixedPointScale{1000}), 3.125);
    EXPECT_FLOAT_EQ(FixedPointU32::Revert(94, FixedPointScale{30}), 3.13333333);
}

TEST(BazTransform, Codec)
{
    auto VerifyToCode = [](NumBits mainBits, NumBits groupBits)
    {
        int numRounds = 1 << groupBits;
        size_t errors = 0;
        size_t successes = 0;
        auto entriesPerRound = 1 << mainBits;

        int stride = 1;
        uint32_t frame = 0;
        uint32_t code = 0;
        for (int i = 0; i < numRounds; ++i)
        {
            for (int j = 0; j < entriesPerRound; ++j)
            {
                for (int k = 0; k < stride; ++k, frame++)
                {
                    bool good = ::Codec::Apply(frame, mainBits) == code;
                    if (errors < 10)
                        EXPECT_TRUE(good) << code;
                    good ? successes++ : errors++;

                    if (k == (stride-1)/2) code++;
                }
            }
            stride *= 2;
        }
        return successes;
    };

    auto VerifyToFrame = [](NumBits mainBits, NumBits groupBits)
    {
        int numRounds = 1 << groupBits;
        size_t errors = 0;
        size_t successes = 0;
        auto entriesPerRound = 1 << mainBits;

        int stride = 1;
        uint32_t frame = 0;
        uint32_t code = 0;
        for (int i = 0; i < numRounds; ++i)
        {
            for (int j = 0; j < entriesPerRound; ++j, code++, frame+=stride)
            {
                bool good = ::Codec::Apply(frame, mainBits) == code;
                if (errors < 10)
                    EXPECT_TRUE(good) << code;
                good ? successes++ : errors++;
            }
            stride *= 2;
        }

        return successes;
    };

    // 2^6 = 64
    // groups = 2^2 = 4
    // Expected frame range = 64+128+256+512 = 960
    EXPECT_EQ(VerifyToCode(NumBits{6}, NumBits{2}), 960);
    // The number of reverse values is simpler, it's just
    // 2 ** (6+2) = 256
    EXPECT_EQ(VerifyToFrame(NumBits{6}, NumBits{2}), 256);

    // 2^6 = 64
    // groups = 2^1 = 2
    // Expected frame range = 64+128 = 192
    EXPECT_EQ(VerifyToCode(NumBits{6}, NumBits{1}), 192);
    // The number of reverse values is simpler, it's just
    // 2 ** (6+1) = 128
    EXPECT_EQ(VerifyToFrame(NumBits{6}, NumBits{1}), 128);

    // 2^5 = 32
    // groups = 2^2 = 4
    // Expected frame range = 32+64+128+256 = 480
    EXPECT_EQ(VerifyToCode(NumBits{5}, NumBits{2}), 480);
    // The number of reverse values is simpler, it's just
    // 2 ** (5+2) = 128
    EXPECT_EQ(VerifyToFrame(NumBits{5}, NumBits{2}), 128);
}
