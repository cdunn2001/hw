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

#include <bazio/BazEventData.h>
#include <bazio/DataParsing.h>
#include <bazio/encoding/ObjectToBaz.h>
#include <bazio/encoding/test/TestingPulse.h>

using namespace PacBio::BazIO;
using namespace PacBio::Primary;

namespace {

constexpr std::array<char, 4> charMap {'A', 'C', 'G', 'T'};

}

TEST(BazRoundTrip, KestrelLosslessCompact)
{
    using Serializer = ObjectToBaz<
        Field<PacketFieldName::Label,
              StoreSigned_t<false>,
              Transform<NoOp>,
              Serialize<TruncateOverflow, NumBits_t<2>>>,
        Field<PacketFieldName::PulseWidth,
              StoreSigned_t<false>,
              Transform<NoOp>,
              Serialize<CompactOverflow, NumBits_t<7>>>,
        Field<PacketFieldName::StartFrame,
              StoreSigned_t<false>,
              Transform<DeltaCompression>,
              Serialize<CompactOverflow, NumBits_t<7>>>>;

    std::array<Pulse, 8> pulsesIn {};

    pulsesIn[0].Label(Pulse::NucleotideLabel::A);
    pulsesIn[1].Label(Pulse::NucleotideLabel::C);
    pulsesIn[2].Label(Pulse::NucleotideLabel::G);
    pulsesIn[3].Label(Pulse::NucleotideLabel::T);
    pulsesIn[4].Label(Pulse::NucleotideLabel::A);
    pulsesIn[5].Label(Pulse::NucleotideLabel::C);
    pulsesIn[6].Label(Pulse::NucleotideLabel::G);
    pulsesIn[7].Label(Pulse::NucleotideLabel::T);

    pulsesIn[0].Width(4);
    pulsesIn[1].Width(120);
    pulsesIn[2].Width(55);
    pulsesIn[3].Width(32);
    pulsesIn[4].Width(182);
    pulsesIn[5].Width(5);
    pulsesIn[6].Width(1);
    pulsesIn[7].Width(300);

    uint32_t start = 0;
    pulsesIn[0].Start(start += 14);
    pulsesIn[1].Start(start += pulsesIn[0].Width() + 130);
    pulsesIn[2].Start(start += pulsesIn[1].Width() + 5);
    pulsesIn[3].Start(start += pulsesIn[2].Width() + 18);
    pulsesIn[4].Start(start += pulsesIn[3].Width() + 22);
    pulsesIn[5].Start(start += pulsesIn[4].Width() + 500);
    pulsesIn[6].Start(start += pulsesIn[5].Width() + 64);
    pulsesIn[7].Start(start += pulsesIn[6].Width() + 33);

    Serializer serializer;
    size_t len = 0;
    for (const auto& p : pulsesIn) len += serializer.BytesRequired(p);

    EXPECT_EQ(len, 24);
    auto data = std::make_unique<uint8_t[]>(len);

    serializer = Serializer{};
    uint8_t* ptr = data.get();
    for (const auto& p : pulsesIn) ptr = serializer.Serialize(p, ptr);
    ASSERT_EQ(ptr, data.get() + len);

    BazEventData events(ParsePackets(Serializer::Params(), data.get(), len, pulsesIn.size()));
    ASSERT_EQ(events.NumEvents(), pulsesIn.size());

    for (size_t i = 0; i < pulsesIn.size(); ++i)
    {
        EXPECT_EQ(pulsesIn[i].Width(), events.PulseWidths(i));
        EXPECT_EQ(charMap[static_cast<int>(pulsesIn[i].Label())], events.Readouts()[i]);
        EXPECT_EQ(pulsesIn[i].Start(), events.StartFrames()[i]);
    }
}

TEST(BazRoundTrip, KestrelLosslessSimple)
{

    using Serializer = ObjectToBaz<
        Field<PacketFieldName::Label,
              StoreSigned_t<false>,
              Transform<NoOp>,
              Serialize<TruncateOverflow, NumBits_t<2>>>,
        Field<PacketFieldName::PulseWidth,
              StoreSigned_t<false>,
              Transform<NoOp>,
              Serialize<SimpleOverflow, NumBits_t<7>, NumBytes_t<4>>>,
        Field<PacketFieldName::StartFrame,
              StoreSigned_t<false>,
              Transform<DeltaCompression>,
              Serialize<SimpleOverflow, NumBits_t<7>, NumBytes_t<4>>>>;

    std::array<Pulse, 8> pulsesIn {};

    pulsesIn[0].Label(Pulse::NucleotideLabel::A);
    pulsesIn[1].Label(Pulse::NucleotideLabel::C);
    pulsesIn[2].Label(Pulse::NucleotideLabel::G);
    pulsesIn[3].Label(Pulse::NucleotideLabel::T);
    pulsesIn[4].Label(Pulse::NucleotideLabel::A);
    pulsesIn[5].Label(Pulse::NucleotideLabel::C);
    pulsesIn[6].Label(Pulse::NucleotideLabel::G);
    pulsesIn[7].Label(Pulse::NucleotideLabel::T);

    pulsesIn[0].Width(4);
    pulsesIn[1].Width(120);
    pulsesIn[2].Width(55);
    pulsesIn[3].Width(32);
    pulsesIn[4].Width(182);
    pulsesIn[5].Width(5);
    pulsesIn[6].Width(1);
    pulsesIn[7].Width(300);

    uint32_t start = 0;
    pulsesIn[0].Start(start += 14);
    pulsesIn[1].Start(start += pulsesIn[0].Width() + 130);
    pulsesIn[2].Start(start += pulsesIn[1].Width() + 5);
    pulsesIn[3].Start(start += pulsesIn[2].Width() + 18);
    pulsesIn[4].Start(start += pulsesIn[3].Width() + 22);
    pulsesIn[5].Start(start += pulsesIn[4].Width() + 500);
    pulsesIn[6].Start(start += pulsesIn[5].Width() + 64);
    pulsesIn[7].Start(start += pulsesIn[6].Width() + 33);

    Serializer serializer;
    size_t len = 0;
    for (const auto& p : pulsesIn) len += serializer.BytesRequired(p);

    EXPECT_EQ(len, 32);
    auto data = std::make_unique<uint8_t[]>(len);

    serializer = Serializer{};
    uint8_t* ptr = data.get();
    for (const auto& p : pulsesIn) ptr = serializer.Serialize(p, ptr);
    ASSERT_EQ(ptr, data.get() + len);

    BazEventData events(ParsePackets(Serializer::Params(), data.get(), len, pulsesIn.size()));
    ASSERT_EQ(events.NumEvents(), pulsesIn.size());

    for (size_t i = 0; i < pulsesIn.size(); ++i)
    {
        EXPECT_EQ(pulsesIn[i].Width(), events.PulseWidths(i));
        EXPECT_EQ(charMap[static_cast<int>(pulsesIn[i].Label())], events.Readouts()[i]);
        EXPECT_EQ(pulsesIn[i].Start(), events.StartFrames()[i]);
    }
}
