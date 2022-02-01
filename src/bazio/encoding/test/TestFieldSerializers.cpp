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

#include <bazio/encoding/FieldSerializers.h>

#include <gtest/gtest.h>

using namespace PacBio::BazIO;

template <size_t val_>
struct Int {
    static constexpr auto val = val_;
};

TEST(BazFieldSerializers, Truncate)
{
    auto roundTrip = [](auto val, auto bits, StoreSigned storeSigned)
    {
        uint8_t data[8];
        auto * ptr = data;
        // running into an unfortunate ICE if I don't do this on it's own line
        static constexpr auto bval = bits.val;
        using S = Serialize<TruncateOverflow, NumBits_t<bval>>;
        auto truncated = S::ToBinary(val, ptr, storeSigned);
        truncated &= (1ul << bits.val)-1;
        auto const * ptr2 = data;
        return S::FromBinary(truncated, ptr2, storeSigned);
    };

    EXPECT_EQ(roundTrip(0u, Int<1>{}, StoreSigned{false}), 0u);
    EXPECT_EQ(roundTrip(1u, Int<1>{}, StoreSigned{false}), 1u);
    EXPECT_EQ(roundTrip(2u, Int<1>{}, StoreSigned{false}), 0u);
    EXPECT_EQ(roundTrip(3u, Int<1>{}, StoreSigned{false}), 1u);
    EXPECT_EQ(roundTrip(3u, Int<3>{}, StoreSigned{false}), 3u);
    EXPECT_EQ(roundTrip(5u, Int<3>{}, StoreSigned{false}), 5u);
    EXPECT_EQ(roundTrip(7u, Int<3>{}, StoreSigned{false}), 7u);
    EXPECT_EQ(roundTrip(9u, Int<3>{}, StoreSigned{false}), 1u);
    EXPECT_EQ(roundTrip(1ul << 33, Int<32>{}, StoreSigned{false}), 0);
    EXPECT_EQ(roundTrip(1ul << 33, Int<34>{}, StoreSigned{false}), 1ul << 33);

    EXPECT_EQ(roundTrip(0, Int<2>{}, StoreSigned{true}), 0);
    EXPECT_EQ(roundTrip(1, Int<2>{}, StoreSigned{true}), 1);
    EXPECT_EQ(roundTrip(2, Int<2>{}, StoreSigned{true}), -2);
    EXPECT_EQ(roundTrip(3, Int<2>{}, StoreSigned{true}), -1);
    EXPECT_EQ(roundTrip(3, Int<4>{}, StoreSigned{true}), 3);
    EXPECT_EQ(roundTrip(5, Int<4>{}, StoreSigned{true}), 5);
    EXPECT_EQ(roundTrip(7, Int<4>{}, StoreSigned{true}), 7);
    EXPECT_EQ(roundTrip(9, Int<4>{}, StoreSigned{true}), -7);
    EXPECT_EQ(roundTrip(1l << 33, Int<33>{}, StoreSigned{true}), 0);
    EXPECT_EQ(roundTrip(1l << 33, Int<35>{}, StoreSigned{true}), 1l << 33);

    EXPECT_EQ(roundTrip(-1, Int<2>{}, StoreSigned{true}), -1);
    EXPECT_EQ(roundTrip(-2, Int<2>{}, StoreSigned{true}), -2);
    EXPECT_EQ(roundTrip(-3, Int<2>{}, StoreSigned{true}), 1);
    EXPECT_EQ(roundTrip(-3, Int<4>{}, StoreSigned{true}), -3);
    EXPECT_EQ(roundTrip(-5, Int<4>{}, StoreSigned{true}), -5);
    EXPECT_EQ(roundTrip(-7, Int<4>{}, StoreSigned{true}), -7);
    EXPECT_EQ(roundTrip(-9, Int<4>{}, StoreSigned{true}), 7);
    EXPECT_EQ(roundTrip(-1*(1l << 33), Int<33>{}, StoreSigned{true}), 0);
    EXPECT_EQ(roundTrip(-1*(1l << 33), Int<35>{}, StoreSigned{true}), -1*(1l << 33));
}

TEST(BazFieldSerializers, SimpleOverflow)
{
    auto roundTrip = [](auto val, auto mainBits, auto overflowBytes, StoreSigned storeSigned)
    {
        uint8_t data[8]{};
        auto ptr = data;
        // running into an unfortunate ICE if I don't do this on it's own line
        constexpr auto mval = mainBits.val;
        constexpr auto oval = overflowBytes.val;
        using S = Serialize<SimpleOverflow, NumBits_t<mval>, NumBytes_t<oval>>;
        auto truncated = S::ToBinary(val, ptr, storeSigned);
        truncated &= ((1ul << mainBits.val) - 1);
        auto const* ptr2 = data;
        return S::FromBinary(truncated, ptr2, storeSigned);
    };

    // Same sequence as the previous test, but now we shouldn't lose data
    EXPECT_EQ(roundTrip(0u, Int<1>{}, Int<1>{}, StoreSigned{false}), 0u);
    EXPECT_EQ(roundTrip(1u, Int<1>{}, Int<1>{}, StoreSigned{false}), 1u);
    EXPECT_EQ(roundTrip(2u, Int<1>{}, Int<1>{}, StoreSigned{false}), 2u);
    EXPECT_EQ(roundTrip(3u, Int<1>{}, Int<1>{}, StoreSigned{false}), 3u);
    EXPECT_EQ(roundTrip(3u, Int<3>{}, Int<1>{}, StoreSigned{false}), 3u);
    EXPECT_EQ(roundTrip(5u, Int<3>{}, Int<1>{}, StoreSigned{false}), 5u);
    EXPECT_EQ(roundTrip(7u, Int<3>{}, Int<1>{}, StoreSigned{false}), 7u);
    EXPECT_EQ(roundTrip(9u, Int<3>{}, Int<1>{}, StoreSigned{false}), 9u);
    EXPECT_EQ(roundTrip(1ul << 33, Int<32>{}, Int<8>{}, StoreSigned{false}), 1ul << 33);
    EXPECT_EQ(roundTrip(1ul << 33, Int<34>{}, Int<8>{}, StoreSigned{false}), 1ul << 33);

    EXPECT_EQ(roundTrip(0, Int<2>{}, Int<1>{}, StoreSigned{true}), 0);
    EXPECT_EQ(roundTrip(1, Int<2>{}, Int<1>{}, StoreSigned{true}), 1);
    EXPECT_EQ(roundTrip(2, Int<2>{}, Int<1>{}, StoreSigned{true}), 2);
    EXPECT_EQ(roundTrip(3, Int<2>{}, Int<1>{}, StoreSigned{true}), 3);
    EXPECT_EQ(roundTrip(3, Int<4>{}, Int<1>{}, StoreSigned{true}), 3);
    EXPECT_EQ(roundTrip(5, Int<4>{}, Int<1>{}, StoreSigned{true}), 5);
    EXPECT_EQ(roundTrip(7, Int<4>{}, Int<1>{}, StoreSigned{true}), 7);
    EXPECT_EQ(roundTrip(9, Int<4>{}, Int<1>{}, StoreSigned{true}), 9);
    EXPECT_EQ(roundTrip(1l << 33, Int<33>{}, Int<8>{}, StoreSigned{true}), 1l << 33);
    EXPECT_EQ(roundTrip(1l << 33, Int<35>{}, Int<8>{}, StoreSigned{true}), 1l << 33);

    EXPECT_EQ(roundTrip(-1, Int<2>{}, Int<1>{}, StoreSigned{true}), -1);
    EXPECT_EQ(roundTrip(-1, Int<3>{}, Int<1>{}, StoreSigned{true}), -1);
    EXPECT_EQ(roundTrip(-2, Int<2>{}, Int<1>{}, StoreSigned{true}), -2);
    EXPECT_EQ(roundTrip(-3, Int<2>{}, Int<1>{}, StoreSigned{true}), -3);
    EXPECT_EQ(roundTrip(-3, Int<4>{}, Int<1>{}, StoreSigned{true}), -3);
    EXPECT_EQ(roundTrip(-5, Int<4>{}, Int<1>{}, StoreSigned{true}), -5);
    EXPECT_EQ(roundTrip(-7, Int<4>{}, Int<1>{}, StoreSigned{true}), -7);
    EXPECT_EQ(roundTrip(-9, Int<4>{}, Int<1>{}, StoreSigned{true}), -9);
    EXPECT_EQ(roundTrip(-1*(1l << 33), Int<33>{}, Int<8>{}, StoreSigned{true}), -1*(1l << 33));
    EXPECT_EQ(roundTrip(-1*(1l << 33), Int<35>{}, Int<8>{}, StoreSigned{true}), -1*(1l << 33));

    // Not really a feature, but still check that we truncate if we don't
    // have enough overflow bits
    EXPECT_EQ(roundTrip(-1*(1l << 33), Int<30>{}, Int<4>{}, StoreSigned{true}), 0);

    // Now check overflow memory usage
    // Extra parens to avoid macro issues with commas in templatelist
    EXPECT_EQ((Serialize<SimpleOverflow, NumBits_t<4>, NumBytes_t<1>>::OverflowBytes(1<<2, StoreSigned{true})), 0);
    EXPECT_EQ((Serialize<SimpleOverflow, NumBits_t<4>, NumBytes_t<1>>::OverflowBytes(1<<5, StoreSigned{true})), 1);
    EXPECT_EQ((Serialize<SimpleOverflow, NumBits_t<4>, NumBytes_t<4>>::OverflowBytes(1<<5, StoreSigned{true})), 4);
    EXPECT_EQ((Serialize<SimpleOverflow, NumBits_t<7>, NumBytes_t<4>>::OverflowBytes(1<<5, StoreSigned{true})), 0);
    EXPECT_EQ((Serialize<SimpleOverflow, NumBits_t<7>, NumBytes_t<4>>::OverflowBytes(1<<9, StoreSigned{true})), 4);
}

TEST(BazFieldSerializers, CompactOverflow)
{
    auto roundTrip = [](auto val, auto mainBits, StoreSigned storeSigned)
    {
        uint8_t data[8];
        auto ptr = data;
        // running into an unfortunate ICE if I don't do this on it's own line
        constexpr auto mval = mainBits.val;
        using S = Serialize<CompactOverflow, NumBits_t<mval>>;
        auto truncated = S::ToBinary(val, ptr, storeSigned);
        auto const* ptr2 = data;
        return S::FromBinary(truncated, ptr2, storeSigned);
    };

    // Same sequence as the truncation test, but now we shouldn't lose data,
    // though we do need to add one more bit to account for our overflow bit
    EXPECT_EQ(roundTrip(0u, Int<2>{}, StoreSigned{false}), 0u);
    EXPECT_EQ(roundTrip(1u, Int<2>{}, StoreSigned{false}), 1u);
    EXPECT_EQ(roundTrip(2u, Int<2>{}, StoreSigned{false}), 2u);
    EXPECT_EQ(roundTrip(3u, Int<2>{}, StoreSigned{false}), 3u);
    EXPECT_EQ(roundTrip(3u, Int<4>{}, StoreSigned{false}), 3u);
    EXPECT_EQ(roundTrip(5u, Int<4>{}, StoreSigned{false}), 5u);
    EXPECT_EQ(roundTrip(7u, Int<4>{}, StoreSigned{false}), 7u);
    EXPECT_EQ(roundTrip(9u, Int<4>{}, StoreSigned{false}), 9u);
    EXPECT_EQ(roundTrip(1ul << 33, Int<33>{}, StoreSigned{false}), 1ul << 33);
    EXPECT_EQ(roundTrip(1ul << 33, Int<35>{}, StoreSigned{false}), 1ul << 33);

    EXPECT_EQ(roundTrip(0, Int<3>{}, StoreSigned{true}), 0);
    EXPECT_EQ(roundTrip(1, Int<3>{}, StoreSigned{true}), 1);
    EXPECT_EQ(roundTrip(2, Int<3>{}, StoreSigned{true}), 2);
    EXPECT_EQ(roundTrip(3, Int<3>{}, StoreSigned{true}), 3);
    EXPECT_EQ(roundTrip(3, Int<5>{}, StoreSigned{true}), 3);
    EXPECT_EQ(roundTrip(5, Int<5>{}, StoreSigned{true}), 5);
    EXPECT_EQ(roundTrip(7, Int<5>{}, StoreSigned{true}), 7);
    EXPECT_EQ(roundTrip(9, Int<5>{}, StoreSigned{true}), 9);
    EXPECT_EQ(roundTrip(1l << 33, Int<34>{}, StoreSigned{true}), 1l << 33);
    EXPECT_EQ(roundTrip(1l << 33, Int<36>{}, StoreSigned{true}), 1l << 33);

    EXPECT_EQ(roundTrip(-1, Int<3>{}, StoreSigned{true}), -1);
    EXPECT_EQ(roundTrip(-2, Int<3>{}, StoreSigned{true}), -2);
    EXPECT_EQ(roundTrip(-3, Int<3>{}, StoreSigned{true}), -3);
    EXPECT_EQ(roundTrip(-3, Int<5>{}, StoreSigned{true}), -3);
    EXPECT_EQ(roundTrip(-5, Int<5>{}, StoreSigned{true}), -5);
    EXPECT_EQ(roundTrip(-7, Int<5>{}, StoreSigned{true}), -7);
    EXPECT_EQ(roundTrip(-9, Int<5>{}, StoreSigned{true}), -9);
    EXPECT_EQ(roundTrip(-1*(1l << 33), Int<34>{}, StoreSigned{true}), -1*(1l << 33));
    EXPECT_EQ(roundTrip(-1*(1l << 33), Int<36>{}, StoreSigned{true}), -1*(1l << 33));

    // Now check overflow memory usage
    EXPECT_EQ((Serialize<CompactOverflow, NumBits_t<4>>::OverflowBytes(1u<<2, StoreSigned{false})), 0);
    EXPECT_EQ((Serialize<CompactOverflow, NumBits_t<4>>::OverflowBytes(1<<2, StoreSigned{true})), 1);
    EXPECT_EQ((Serialize<CompactOverflow, NumBits_t<4>>::OverflowBytes(1u<<5, StoreSigned{false})), 1);
    EXPECT_EQ((Serialize<CompactOverflow, NumBits_t<7>>::OverflowBytes(1u<<5, StoreSigned{false})), 0);
    EXPECT_EQ((Serialize<CompactOverflow, NumBits_t<7>>::OverflowBytes(1<<9, StoreSigned{true})), 1);
}
