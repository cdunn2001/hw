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
    EXPECT_EQ(NoOp::Apply(12u, StoreSigned{false}), 12u);
    EXPECT_EQ(NoOp::Apply(-12, StoreSigned{true}), static_cast<uint64_t>(-12));
    EXPECT_EQ(NoOp::Apply(1ul<<54, StoreSigned{false}), 1ul<<54);

    EXPECT_EQ(NoOp::Revert<uint32_t>(12u, StoreSigned{false}), 12u);
    EXPECT_EQ(NoOp::Revert<int32_t>(static_cast<uint64_t>(-12), StoreSigned{true}), -12);
    EXPECT_EQ(NoOp::Revert<uint64_t>(1ul<<54, StoreSigned{false}), 1ul<<54);
}

TEST(BazTransform, FixedPoint)
{
    EXPECT_EQ(FixedPoint::Apply(3.125412, StoreSigned{false}, FixedPointScale{10}), 31);
    EXPECT_EQ(FixedPoint::Apply(3.125412, StoreSigned{false}, FixedPointScale{100}), 313);
    EXPECT_EQ(FixedPoint::Apply(3.125412, StoreSigned{false}, FixedPointScale{1000}), 3125);
    EXPECT_EQ(FixedPoint::Apply(3.125412, StoreSigned{false}, FixedPointScale{30}), 94);

    EXPECT_EQ(FixedPoint::Apply(-3.125412, StoreSigned{true}, FixedPointScale{10}), static_cast<uint64_t>(-31));
    EXPECT_EQ(FixedPoint::Apply(-3.125412, StoreSigned{true}, FixedPointScale{100}), static_cast<uint64_t>(-313));
    EXPECT_EQ(FixedPoint::Apply(-3.125412, StoreSigned{true}, FixedPointScale{1000}), static_cast<uint64_t>(-3125));
    EXPECT_EQ(FixedPoint::Apply(-3.125412, StoreSigned{true}, FixedPointScale{30}), static_cast<uint64_t>(-94));

    EXPECT_FLOAT_EQ(FixedPoint::Revert<float>(31, StoreSigned{false}, FixedPointScale{10}), 3.1);
    EXPECT_FLOAT_EQ(FixedPoint::Revert<float>(313, StoreSigned{false}, FixedPointScale{100}), 3.13);
    EXPECT_FLOAT_EQ(FixedPoint::Revert<float>(3125, StoreSigned{false}, FixedPointScale{1000}), 3.125);
    EXPECT_FLOAT_EQ(FixedPoint::Revert<float>(94, StoreSigned{false}, FixedPointScale{30}), 3.13333333);

    EXPECT_FLOAT_EQ(FixedPoint::Revert<float>(static_cast<uint64_t>(-31), StoreSigned{true}, FixedPointScale{10}), -3.1);
    EXPECT_FLOAT_EQ(FixedPoint::Revert<float>(static_cast<uint64_t>(-313), StoreSigned{true}, FixedPointScale{100}), -3.13);
    EXPECT_FLOAT_EQ(FixedPoint::Revert<float>(static_cast<uint64_t>(-3125), StoreSigned{true}, FixedPointScale{1000}), -3.125);
    EXPECT_FLOAT_EQ(FixedPoint::Revert<float>(static_cast<uint64_t>(-94), StoreSigned{true}, FixedPointScale{30}), -3.13333333);
}

TEST(BazTransform, FloatFixedCodec)
{
    auto roundTrip = [](float input, StoreSigned storeSigned, FixedPointScale scale, NumBytes numBytes)
    {
        auto transformed = FloatFixedCodec::Apply(input, storeSigned, scale, numBytes);
        return FloatFixedCodec::Revert<float>(transformed, storeSigned, scale);
    };

    // First make sure we can handle the same vanilla cases as the FixedPoint transform
    EXPECT_FLOAT_EQ(roundTrip(3.125412, StoreSigned{false}, FixedPointScale{10}, NumBytes{2}), 3.1);
    EXPECT_FLOAT_EQ(roundTrip(3.125412, StoreSigned{false}, FixedPointScale{100}, NumBytes{2}), 3.13);
    EXPECT_FLOAT_EQ(roundTrip(3.125412, StoreSigned{false}, FixedPointScale{1000}, NumBytes{2}), 3.125);
    EXPECT_FLOAT_EQ(roundTrip(3.125412, StoreSigned{false}, FixedPointScale{30}, NumBytes{2}), 3.13333333);

    EXPECT_FLOAT_EQ(roundTrip(-3.125412, StoreSigned{true}, FixedPointScale{10}, NumBytes{2}), -3.1);
    EXPECT_FLOAT_EQ(roundTrip(-3.125412, StoreSigned{true}, FixedPointScale{100}, NumBytes{2}), -3.13);
    EXPECT_FLOAT_EQ(roundTrip(-3.125412, StoreSigned{true}, FixedPointScale{1000}, NumBytes{2}), -3.125);
    EXPECT_FLOAT_EQ(roundTrip(-3.125412, StoreSigned{true}, FixedPointScale{30}, NumBytes{2}), -3.13333333);

    const auto nan = std::numeric_limits<float>::quiet_NaN();
    const auto inf = std::numeric_limits<float>::infinity();
    const auto ninf = -std::numeric_limits<float>::infinity();

    // Now try a bunch of settings with literal nan input
    EXPECT_TRUE(std::isnan(roundTrip(nan, StoreSigned{false}, FixedPointScale{10},  NumBytes{1})));
    EXPECT_TRUE(std::isnan(roundTrip(nan, StoreSigned{false}, FixedPointScale{100}, NumBytes{1})));
    EXPECT_TRUE(std::isnan(roundTrip(nan, StoreSigned{true},  FixedPointScale{10},  NumBytes{1})));
    EXPECT_TRUE(std::isnan(roundTrip(nan, StoreSigned{true},  FixedPointScale{100}, NumBytes{1})));
    EXPECT_TRUE(std::isnan(roundTrip(nan, StoreSigned{false}, FixedPointScale{10},  NumBytes{4})));
    EXPECT_TRUE(std::isnan(roundTrip(nan, StoreSigned{false}, FixedPointScale{100}, NumBytes{4})));
    EXPECT_TRUE(std::isnan(roundTrip(nan, StoreSigned{true},  FixedPointScale{10},  NumBytes{4})));
    EXPECT_TRUE(std::isnan(roundTrip(nan, StoreSigned{true},  FixedPointScale{100}, NumBytes{8})));
    EXPECT_TRUE(std::isnan(roundTrip(nan, StoreSigned{false}, FixedPointScale{10},  NumBytes{8})));
    EXPECT_TRUE(std::isnan(roundTrip(nan, StoreSigned{false}, FixedPointScale{100}, NumBytes{8})));
    EXPECT_TRUE(std::isnan(roundTrip(nan, StoreSigned{true},  FixedPointScale{10},  NumBytes{8})));
    EXPECT_TRUE(std::isnan(roundTrip(nan, StoreSigned{true},  FixedPointScale{100}, NumBytes{8})));

    // Now try a bunch of settings with literal infinity input
    EXPECT_EQ(roundTrip(inf, StoreSigned{false}, FixedPointScale{10},  NumBytes{1}), inf);
    EXPECT_EQ(roundTrip(inf, StoreSigned{false}, FixedPointScale{100}, NumBytes{1}), inf);
    EXPECT_EQ(roundTrip(inf, StoreSigned{true},  FixedPointScale{10},  NumBytes{1}), inf);
    EXPECT_EQ(roundTrip(inf, StoreSigned{true},  FixedPointScale{100}, NumBytes{1}), inf);
    EXPECT_EQ(roundTrip(inf, StoreSigned{false}, FixedPointScale{10},  NumBytes{4}), inf);
    EXPECT_EQ(roundTrip(inf, StoreSigned{false}, FixedPointScale{100}, NumBytes{4}), inf);
    EXPECT_EQ(roundTrip(inf, StoreSigned{true},  FixedPointScale{10},  NumBytes{4}), inf);
    EXPECT_EQ(roundTrip(inf, StoreSigned{true},  FixedPointScale{100}, NumBytes{8}), inf);
    EXPECT_EQ(roundTrip(inf, StoreSigned{false}, FixedPointScale{10},  NumBytes{8}), inf);
    EXPECT_EQ(roundTrip(inf, StoreSigned{false}, FixedPointScale{100}, NumBytes{8}), inf);
    EXPECT_EQ(roundTrip(inf, StoreSigned{true},  FixedPointScale{10},  NumBytes{8}), inf);
    EXPECT_EQ(roundTrip(inf, StoreSigned{true},  FixedPointScale{100}, NumBytes{8}), inf);

    // Now try a bunch of settings with literal negative infinity input
    EXPECT_EQ(roundTrip(ninf, StoreSigned{false}, FixedPointScale{10},  NumBytes{1}), ninf);
    EXPECT_EQ(roundTrip(ninf, StoreSigned{false}, FixedPointScale{100}, NumBytes{1}), ninf);
    EXPECT_EQ(roundTrip(ninf, StoreSigned{true},  FixedPointScale{10},  NumBytes{1}), ninf);
    EXPECT_EQ(roundTrip(ninf, StoreSigned{true},  FixedPointScale{100}, NumBytes{1}), ninf);
    EXPECT_EQ(roundTrip(ninf, StoreSigned{false}, FixedPointScale{10},  NumBytes{4}), ninf);
    EXPECT_EQ(roundTrip(ninf, StoreSigned{false}, FixedPointScale{100}, NumBytes{4}), ninf);
    EXPECT_EQ(roundTrip(ninf, StoreSigned{true},  FixedPointScale{10},  NumBytes{4}), ninf);
    EXPECT_EQ(roundTrip(ninf, StoreSigned{true},  FixedPointScale{100}, NumBytes{8}), ninf);
    EXPECT_EQ(roundTrip(ninf, StoreSigned{false}, FixedPointScale{10},  NumBytes{8}), ninf);
    EXPECT_EQ(roundTrip(ninf, StoreSigned{false}, FixedPointScale{100}, NumBytes{8}), ninf);
    EXPECT_EQ(roundTrip(ninf, StoreSigned{true},  FixedPointScale{10},  NumBytes{8}), ninf);
    EXPECT_EQ(roundTrip(ninf, StoreSigned{true},  FixedPointScale{100}, NumBytes{8}), ninf);

    // A couple more checks, this time adding a finite input, but with a scale that makes
    // our rescaled value infinite.  Technically we could round trip a larger dynamic range
    // if we did some careful "long form" multiplication, but that would be both more
    // complex and slower, so this implementation just opts to go infinite instead.
    auto val = std::numeric_limits<float>::max() - 1000;
    EXPECT_EQ(roundTrip(val, StoreSigned{false}, FixedPointScale{2},  NumBytes{8}), inf);
    EXPECT_FLOAT_EQ(roundTrip(val, StoreSigned{false}, FixedPointScale{1},  NumBytes{8}), val);
    EXPECT_EQ(roundTrip(val, StoreSigned{true}, FixedPointScale{2},  NumBytes{8}), inf);
    EXPECT_FLOAT_EQ(roundTrip(val, StoreSigned{true}, FixedPointScale{1},  NumBytes{8}), val);

    val = std::numeric_limits<float>::lowest() + 1000;
    EXPECT_EQ(roundTrip(val, StoreSigned{false}, FixedPointScale{2},  NumBytes{8}), ninf);
    EXPECT_FLOAT_EQ(roundTrip(val, StoreSigned{false}, FixedPointScale{1},  NumBytes{8}), ninf);
    EXPECT_EQ(roundTrip(val, StoreSigned{true}, FixedPointScale{2},  NumBytes{8}), ninf);
    EXPECT_FLOAT_EQ(roundTrip(val, StoreSigned{true}, FixedPointScale{1},  NumBytes{8}), val);

    // This is borderline testing the implementation, but the implementation reserves a few
    // bit patterns near zero to represent special values.  Here we use those special values
    // as an edge case test, to make sure we don't clobber real values.
    EXPECT_EQ(roundTrip(FloatFixedCodec::nan,  StoreSigned{false}, FixedPointScale{1},  NumBytes{1}), FloatFixedCodec::nan);
    EXPECT_EQ(roundTrip(FloatFixedCodec::inf,  StoreSigned{false}, FixedPointScale{1},  NumBytes{1}), FloatFixedCodec::inf);
    EXPECT_EQ(roundTrip(FloatFixedCodec::ninf, StoreSigned{false}, FixedPointScale{1},  NumBytes{1}), FloatFixedCodec::ninf);
    EXPECT_EQ(roundTrip(FloatFixedCodec::nan,  StoreSigned{true},  FixedPointScale{1},  NumBytes{1}), FloatFixedCodec::nan);
    EXPECT_EQ(roundTrip(FloatFixedCodec::inf,  StoreSigned{true},  FixedPointScale{1},  NumBytes{1}), FloatFixedCodec::inf);
    EXPECT_EQ(roundTrip(FloatFixedCodec::ninf, StoreSigned{true},  FixedPointScale{1},  NumBytes{1}), FloatFixedCodec::ninf);

    // Now that we've handled literal float infinities, make sure that the mechanism for
    // setting a maximum number of bytes is functional.  The round trip value should appear
    // infinite if we don't allow enough storage bytes in the transformed value.
    EXPECT_EQ(roundTrip(256, StoreSigned{false}, FixedPointScale{1},  NumBytes{1}), inf);
    EXPECT_EQ(roundTrip(255, StoreSigned{false}, FixedPointScale{1},  NumBytes{1}), inf);
    EXPECT_EQ(roundTrip(254, StoreSigned{false}, FixedPointScale{1},  NumBytes{1}), inf);
    EXPECT_EQ(roundTrip(253, StoreSigned{false}, FixedPointScale{1},  NumBytes{1}), inf);
    EXPECT_EQ(roundTrip(252, StoreSigned{false}, FixedPointScale{1},  NumBytes{1}), 252);

    EXPECT_EQ(roundTrip(252, StoreSigned{true}, FixedPointScale{1},  NumBytes{1}), inf);
    EXPECT_EQ(roundTrip(127, StoreSigned{true}, FixedPointScale{1},  NumBytes{1}), inf);
    EXPECT_EQ(roundTrip(126, StoreSigned{true}, FixedPointScale{1},  NumBytes{1}), inf);
    EXPECT_EQ(roundTrip(125, StoreSigned{true}, FixedPointScale{1},  NumBytes{1}), inf);
    EXPECT_EQ(roundTrip(124, StoreSigned{true}, FixedPointScale{1},  NumBytes{1}), 124);

    EXPECT_EQ(roundTrip(-127, StoreSigned{true}, FixedPointScale{1},  NumBytes{1}), ninf);
    EXPECT_EQ(roundTrip(-124, StoreSigned{true}, FixedPointScale{1},  NumBytes{1}), -124);

    // A couple more random checks...
    EXPECT_EQ(roundTrip(7623.531, StoreSigned{false}, FixedPointScale{10}, NumBytes{2}), inf);
    EXPECT_NEAR(roundTrip(7623.531, StoreSigned{false}, FixedPointScale{10}, NumBytes{3}), 7623.5, 1e-2);

    EXPECT_EQ(roundTrip(3623.531, StoreSigned{true}, FixedPointScale{10}, NumBytes{2}), inf);
    EXPECT_NEAR(roundTrip(3623.531, StoreSigned{true}, FixedPointScale{10}, NumBytes{3}), 3623.5, 1e-2);

    EXPECT_EQ(roundTrip(-3623.531, StoreSigned{true}, FixedPointScale{10}, NumBytes{2}), ninf);
    EXPECT_NEAR(roundTrip(-3623.531, StoreSigned{true}, FixedPointScale{10}, NumBytes{3}), -3623.5, 1e-2);
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
                    bool good = ::LossySequelCodec::Apply(frame, StoreSigned{false}, mainBits) == code;
                    if (errors < 10)
                    {
                        EXPECT_TRUE(good) << code;
                    }
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
                bool good = ::LossySequelCodec::Apply(frame, StoreSigned{false}, mainBits) == code;
                if (errors < 10)
                {
                    EXPECT_TRUE(good) << code;
                }
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

TEST(BazTransform, DeltaCompression)
{
    DeltaCompression d;

    // Simple increasing sequencing
    EXPECT_EQ(d.Apply(10, StoreSigned{false}), 10);
    EXPECT_EQ(d.Apply(20, StoreSigned{false}), 10);
    EXPECT_EQ(d.Apply(30, StoreSigned{false}), 10);
    EXPECT_EQ(d.Apply(50, StoreSigned{false}), 20);
    EXPECT_EQ(d.Apply(55, StoreSigned{false}), 5);

    // Undo previous sequence.  We're stateful, so we
    // need to reset our state
    d = DeltaCompression{};
    EXPECT_EQ(d.Revert<int>(10, StoreSigned{false}), 10);
    EXPECT_EQ(d.Revert<int>(10, StoreSigned{false}), 20);
    EXPECT_EQ(d.Revert<int>(10, StoreSigned{false}), 30);
    EXPECT_EQ(d.Revert<int>(20, StoreSigned{false}), 50);
    EXPECT_EQ(d.Revert<int>(5,  StoreSigned{false}), 55);

    // Now handle non-monotonic sequences.
    d = DeltaCompression{};
    EXPECT_EQ(d.Apply(10, StoreSigned{true}), static_cast<uint64_t>(10));
    EXPECT_EQ(d.Apply(20, StoreSigned{true}), static_cast<uint64_t>(10));
    EXPECT_EQ(d.Apply(10, StoreSigned{true}), static_cast<uint64_t>(-10));
    EXPECT_EQ(d.Apply(15, StoreSigned{true}), static_cast<uint64_t>(5));
    EXPECT_EQ(d.Apply(5,  StoreSigned{true}), static_cast<uint64_t>(-10));

    // And again undoing the previous sequence
    d = DeltaCompression{};
    EXPECT_EQ(d.Revert<int>(static_cast<uint64_t>(10),  StoreSigned{true}), 10);
    EXPECT_EQ(d.Revert<int>(static_cast<uint64_t>(10),  StoreSigned{true}), 20);
    EXPECT_EQ(d.Revert<int>(static_cast<uint64_t>(-10), StoreSigned{true}), 10);
    EXPECT_EQ(d.Revert<int>(static_cast<uint64_t>(5),   StoreSigned{true}), 15);
    EXPECT_EQ(d.Revert<int>(static_cast<uint64_t>(-10), StoreSigned{true}), 5);
}

TEST(BazTransform, MultiTransform)
{
    using Stage1 = Transform<DeltaCompression>;
    using Stage2 = Transform<LossySequelCodec, NumBits_t<6>>;
    using Multi = MultiTransform<Stage1, Stage2>;

    Multi m;
    EXPECT_EQ(m.Apply(10, StoreSigned{false}), 10);
    EXPECT_EQ(m.Apply(20, StoreSigned{false}), 10);
    EXPECT_EQ(m.Apply(40, StoreSigned{false}), 20);
    EXPECT_EQ(m.Apply(80, StoreSigned{false}), 40);
    EXPECT_EQ(m.Apply(144, StoreSigned{false}), 64);
    // We jump by 67, we should now have entered the lossy portion
    // of the codec 67 gets rounded to 68, which then gets transformed
    // to 64 + 4/2 = 66.
    EXPECT_EQ(m.Apply(211, StoreSigned{false}), 66);

    m = Multi{};
    EXPECT_EQ(m.Revert<int>(10, StoreSigned{false}), 10);
    EXPECT_EQ(m.Revert<int>(10, StoreSigned{false}), 20);
    EXPECT_EQ(m.Revert<int>(20, StoreSigned{false}), 40);
    EXPECT_EQ(m.Revert<int>(40, StoreSigned{false}), 80);
    EXPECT_EQ(m.Revert<int>(64, StoreSigned{false}), 144);
    // 66 in the codec is really 64 + 2*2 = 68.  144+68=212
    EXPECT_EQ(m.Revert<int>(66, StoreSigned{false}), 212);

}
