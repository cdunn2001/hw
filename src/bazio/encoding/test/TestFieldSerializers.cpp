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
    auto roundTrip = [](auto val, auto bits)
    {
        uint8_t data[8];
        auto ptr = data;
        auto truncated = Serialize<TruncateOverflow, NumBits_t<bits.val>>::ToBinary(val, ptr);
        truncated &= (1ul << bits.val)-1;
        constexpr auto isSigned = std::is_signed<decltype(val)>::value;
        ptr = data;
        return Serialize<TruncateOverflow, NumBits_t<bits.val>>::template FromBinary<isSigned>(truncated, ptr);
    };

    EXPECT_EQ(roundTrip(0u, Int<1>{}), 0u);
    EXPECT_EQ(roundTrip(1u, Int<1>{}), 1u);
    EXPECT_EQ(roundTrip(2u, Int<1>{}), 0u);
    EXPECT_EQ(roundTrip(3u, Int<1>{}), 1u);
    EXPECT_EQ(roundTrip(3u, Int<3>{}), 3u);
    EXPECT_EQ(roundTrip(5u, Int<3>{}), 5u);
    EXPECT_EQ(roundTrip(7u, Int<3>{}), 7u);
    EXPECT_EQ(roundTrip(9u, Int<3>{}), 1u);
    EXPECT_EQ(roundTrip(1ul << 33, Int<32>{}), 0);
    EXPECT_EQ(roundTrip(1ul << 33, Int<34>{}), 1ul << 33);

    EXPECT_EQ(roundTrip(0, Int<2>{}), 0);
    EXPECT_EQ(roundTrip(1, Int<2>{}), 1);
    EXPECT_EQ(roundTrip(2, Int<2>{}), -2);
    EXPECT_EQ(roundTrip(3, Int<2>{}), -1);
    EXPECT_EQ(roundTrip(3, Int<4>{}), 3);
    EXPECT_EQ(roundTrip(5, Int<4>{}), 5);
    EXPECT_EQ(roundTrip(7, Int<4>{}), 7);
    EXPECT_EQ(roundTrip(9, Int<4>{}), -7);
    EXPECT_EQ(roundTrip(1l << 33, Int<33>{}), 0);
    EXPECT_EQ(roundTrip(1l << 33, Int<35>{}), 1l << 33);

    EXPECT_EQ(roundTrip(-1, Int<2>{}), -1);
    EXPECT_EQ(roundTrip(-2, Int<2>{}), -2);
    EXPECT_EQ(roundTrip(-3, Int<2>{}), 1);
    EXPECT_EQ(roundTrip(-3, Int<4>{}), -3);
    EXPECT_EQ(roundTrip(-5, Int<4>{}), -5);
    EXPECT_EQ(roundTrip(-7, Int<4>{}), -7);
    EXPECT_EQ(roundTrip(-9, Int<4>{}), 7);
    EXPECT_EQ(roundTrip(-1*(1l << 33), Int<33>{}), 0);
    EXPECT_EQ(roundTrip(-1*(1l << 33), Int<35>{}), -1*(1l << 33));
}

TEST(BazFieldSerializers, SimpleOverflow)
{
    auto roundTrip = [](auto val, auto mainBits, auto overflowBytes)
    {
        uint8_t data[8]{};
        auto ptr = data;
        auto truncated = Serialize<SimpleOverflow, NumBits_t<mainBits.val>, NumBytes_t<overflowBytes.val>>::ToBinary(val, ptr);
        truncated &= ((1ul << mainBits.val) - 1);
        constexpr auto isSigned = std::is_signed<decltype(val)>::value;
        ptr = data;
        return Serialize<SimpleOverflow, NumBits_t<mainBits.val>, NumBytes_t<overflowBytes.val>>::template FromBinary<isSigned>(truncated, ptr);
    };

    // Same sequence as the previous test, but now we shouldn't lose data
    EXPECT_EQ(roundTrip(0u, Int<1>{}, Int<1>{}), 0u);
    EXPECT_EQ(roundTrip(1u, Int<1>{}, Int<1>{}), 1u);
    EXPECT_EQ(roundTrip(2u, Int<1>{}, Int<1>{}), 2u);
    EXPECT_EQ(roundTrip(3u, Int<1>{}, Int<1>{}), 3u);
    EXPECT_EQ(roundTrip(3u, Int<3>{}, Int<1>{}), 3u);
    EXPECT_EQ(roundTrip(5u, Int<3>{}, Int<1>{}), 5u);
    EXPECT_EQ(roundTrip(7u, Int<3>{}, Int<1>{}), 7u);
    EXPECT_EQ(roundTrip(9u, Int<3>{}, Int<1>{}), 9u);
    EXPECT_EQ(roundTrip(1ul << 33, Int<32>{}, Int<8>{}), 1ul << 33);
    EXPECT_EQ(roundTrip(1ul << 33, Int<34>{}, Int<8>{}), 1ul << 33);

    EXPECT_EQ(roundTrip(0, Int<2>{}, Int<1>{}), 0);
    EXPECT_EQ(roundTrip(1, Int<2>{}, Int<1>{}), 1);
    EXPECT_EQ(roundTrip(2, Int<2>{}, Int<1>{}), 2);
    EXPECT_EQ(roundTrip(3, Int<2>{}, Int<1>{}), 3);
    EXPECT_EQ(roundTrip(3, Int<4>{}, Int<1>{}), 3);
    EXPECT_EQ(roundTrip(5, Int<4>{}, Int<1>{}), 5);
    EXPECT_EQ(roundTrip(7, Int<4>{}, Int<1>{}), 7);
    EXPECT_EQ(roundTrip(9, Int<4>{}, Int<1>{}), 9);
    EXPECT_EQ(roundTrip(1l << 33, Int<33>{}, Int<8>{}), 1l << 33);
    EXPECT_EQ(roundTrip(1l << 33, Int<35>{}, Int<8>{}), 1l << 33);

    EXPECT_EQ(roundTrip(-1, Int<2>{}, Int<1>{}), -1);
    EXPECT_EQ(roundTrip(-1, Int<3>{}, Int<1>{}), -1);
    EXPECT_EQ(roundTrip(-2, Int<2>{}, Int<1>{}), -2);
    EXPECT_EQ(roundTrip(-3, Int<2>{}, Int<1>{}), -3);
    EXPECT_EQ(roundTrip(-3, Int<4>{}, Int<1>{}), -3);
    EXPECT_EQ(roundTrip(-5, Int<4>{}, Int<1>{}), -5);
    EXPECT_EQ(roundTrip(-7, Int<4>{}, Int<1>{}), -7);
    EXPECT_EQ(roundTrip(-9, Int<4>{}, Int<1>{}), -9);
    EXPECT_EQ(roundTrip(-1*(1l << 33), Int<33>{}, Int<8>{}), -1*(1l << 33));
    EXPECT_EQ(roundTrip(-1*(1l << 33), Int<35>{}, Int<8>{}), -1*(1l << 33));

    // Not really a feature, but still check that we truncate if we don't
    // have enough overflow bits
    EXPECT_EQ(roundTrip(-1*(1l << 33), Int<30>{}, Int<4>{}), 0);

    // Now check overflow memory usage
    // Extra parens to avoid macro issues with commas in templatelist
    EXPECT_EQ((Serialize<SimpleOverflow, NumBits_t<4>, NumBytes_t<1>>::OverflowBytes(1<<2)), 0);
    EXPECT_EQ((Serialize<SimpleOverflow, NumBits_t<4>, NumBytes_t<1>>::OverflowBytes(1<<5)), 1);
    EXPECT_EQ((Serialize<SimpleOverflow, NumBits_t<4>, NumBytes_t<4>>::OverflowBytes(1<<5)), 4);
    EXPECT_EQ((Serialize<SimpleOverflow, NumBits_t<7>, NumBytes_t<4>>::OverflowBytes(1<<5)), 0);
    EXPECT_EQ((Serialize<SimpleOverflow, NumBits_t<7>, NumBytes_t<4>>::OverflowBytes(1<<9)), 4);
}

TEST(BazFieldSerializers, CompactOverflow)
{
    auto roundTrip = [](auto val, auto mainBits)
    {
        uint8_t data[8];
        auto ptr = data;
        auto truncated = Serialize<CompactOverflow, NumBits_t<mainBits.val>>::ToBinary(val, ptr);
        constexpr auto isSigned = std::is_signed<decltype(val)>::value;
        ptr = data;
        return Serialize<CompactOverflow, NumBits_t<mainBits.val>>::template FromBinary<isSigned>(truncated, ptr);
    };

    // Same sequence as the truncation test, but now we shouldn't lose data,
    // though we do need to add one more bit to account for our overflow bit
    EXPECT_EQ(roundTrip(0u, Int<2>{}), 0u);
    EXPECT_EQ(roundTrip(1u, Int<2>{}), 1u);
    EXPECT_EQ(roundTrip(2u, Int<2>{}), 2u);
    EXPECT_EQ(roundTrip(3u, Int<2>{}), 3u);
    EXPECT_EQ(roundTrip(3u, Int<4>{}), 3u);
    EXPECT_EQ(roundTrip(5u, Int<4>{}), 5u);
    EXPECT_EQ(roundTrip(7u, Int<4>{}), 7u);
    EXPECT_EQ(roundTrip(9u, Int<4>{}), 9u);
    EXPECT_EQ(roundTrip(1ul << 33, Int<33>{}), 1ul << 33);
    EXPECT_EQ(roundTrip(1ul << 33, Int<35>{}), 1ul << 33);

    EXPECT_EQ(roundTrip(0, Int<3>{}), 0);
    EXPECT_EQ(roundTrip(1, Int<3>{}), 1);
    EXPECT_EQ(roundTrip(2, Int<3>{}), 2);
    EXPECT_EQ(roundTrip(3, Int<3>{}), 3);
    EXPECT_EQ(roundTrip(3, Int<5>{}), 3);
    EXPECT_EQ(roundTrip(5, Int<5>{}), 5);
    EXPECT_EQ(roundTrip(7, Int<5>{}), 7);
    EXPECT_EQ(roundTrip(9, Int<5>{}), 9);
    EXPECT_EQ(roundTrip(1l << 33, Int<34>{}), 1l << 33);
    EXPECT_EQ(roundTrip(1l << 33, Int<36>{}), 1l << 33);

    EXPECT_EQ(roundTrip(-1, Int<3>{}), -1);
    EXPECT_EQ(roundTrip(-2, Int<3>{}), -2);
    EXPECT_EQ(roundTrip(-3, Int<3>{}), -3);
    EXPECT_EQ(roundTrip(-3, Int<5>{}), -3);
    EXPECT_EQ(roundTrip(-5, Int<5>{}), -5);
    EXPECT_EQ(roundTrip(-7, Int<5>{}), -7);
    EXPECT_EQ(roundTrip(-9, Int<5>{}), -9);
    EXPECT_EQ(roundTrip(-1*(1l << 33), Int<34>{}), -1*(1l << 33));
    EXPECT_EQ(roundTrip(-1*(1l << 33), Int<36>{}), -1*(1l << 33));

    // Now check overflow memory usage
    EXPECT_EQ((Serialize<CompactOverflow, NumBits_t<4>>::OverflowBytes(1u<<2)), 0);
    EXPECT_EQ((Serialize<CompactOverflow, NumBits_t<4>>::OverflowBytes(1<<2)), 1);
    EXPECT_EQ((Serialize<CompactOverflow, NumBits_t<4>>::OverflowBytes(1u<<5)), 1);
    EXPECT_EQ((Serialize<CompactOverflow, NumBits_t<7>>::OverflowBytes(1u<<5)), 0);
    EXPECT_EQ((Serialize<CompactOverflow, NumBits_t<7>>::OverflowBytes(1<<9)), 1);
}
