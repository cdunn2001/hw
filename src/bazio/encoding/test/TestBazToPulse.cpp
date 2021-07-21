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

#include <bazio/encoding/BazToPulse.h>
#include <dataTypes/Pulse.h>

using namespace PacBio::BazIO;

TEST(BazToPulse, KestrelLosslessCompact)
{
    std::vector<FieldParams> info;
    info.push_back({PacketFieldName::Label,
                    StoreSigned{false},
                    {NoOpTransformParams{}},
                    TruncateParams{ NumBits{2} }
        });
    info.push_back({PacketFieldName::Pw,
                    StoreSigned{false},
                    {NoOpTransformParams{}},
                    CompactOverflowParams{ NumBits{7} }
        });
    info.push_back({PacketFieldName::StartFrame,
                    StoreSigned{false},
                    {DeltaCompressionParams{}},
                    CompactOverflowParams{ NumBits{7} }
        });

    std::array<PacBio::Mongo::Data::Pulse, 8> pulsesIn{};
    std::array<PacBio::Mongo::Data::Pulse, 8> pulsesOut{};

    pulsesIn[0].Label(PacBio::Mongo::Data::Pulse::NucleotideLabel::A);
    pulsesIn[1].Label(PacBio::Mongo::Data::Pulse::NucleotideLabel::C);
    pulsesIn[2].Label(PacBio::Mongo::Data::Pulse::NucleotideLabel::G);
    pulsesIn[3].Label(PacBio::Mongo::Data::Pulse::NucleotideLabel::T);
    pulsesIn[4].Label(PacBio::Mongo::Data::Pulse::NucleotideLabel::A);
    pulsesIn[5].Label(PacBio::Mongo::Data::Pulse::NucleotideLabel::C);
    pulsesIn[6].Label(PacBio::Mongo::Data::Pulse::NucleotideLabel::G);
    pulsesIn[7].Label(PacBio::Mongo::Data::Pulse::NucleotideLabel::T);

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

    BazToPulse parser(info);
    size_t len = 0;
    for (const auto& p : pulsesIn) len += parser.BytesRequired(p);

    EXPECT_EQ(len, 24);
    auto data = std::make_unique<uint8_t[]>(len);

    parser.Reset();
    uint8_t* ptr = data.get();
    for (const auto& p : pulsesIn) ptr = parser.Serialize(p, ptr);
    ASSERT_EQ(ptr, data.get()+len);

    parser.Reset();
    auto const* ptr2 = data.get();
    for (auto& p : pulsesOut) ptr2 = parser.Deserialize(p, ptr2);
    ASSERT_EQ(ptr, data.get()+len);

    for (size_t i = 0; i < pulsesIn.size(); ++i)
    {
        EXPECT_TRUE(pulsesIn[i] == pulsesOut[i]) << i;
    }
}

TEST(BazToPulse, KestrelLosslessSimple)
{
    std::vector<FieldParams> info;
    info.push_back({PacketFieldName::Label,
                    StoreSigned{false},
                    {NoOpTransformParams{}},
                    TruncateParams{ NumBits{2} }
        });
    info.push_back({PacketFieldName::Pw,
                    StoreSigned{false},
                    {NoOpTransformParams{}},
                    SimpleOverflowParams{ NumBits{7}, NumBytes{4} }
        });
    info.push_back({PacketFieldName::StartFrame,
                    StoreSigned{false},
                    {DeltaCompressionParams{}},
                    SimpleOverflowParams{ NumBits{7}, NumBytes{4} }
        });

    std::array<PacBio::Mongo::Data::Pulse, 8> pulsesIn{};
    std::array<PacBio::Mongo::Data::Pulse, 8> pulsesOut{};

    pulsesIn[0].Label(PacBio::Mongo::Data::Pulse::NucleotideLabel::A);
    pulsesIn[1].Label(PacBio::Mongo::Data::Pulse::NucleotideLabel::C);
    pulsesIn[2].Label(PacBio::Mongo::Data::Pulse::NucleotideLabel::G);
    pulsesIn[3].Label(PacBio::Mongo::Data::Pulse::NucleotideLabel::T);
    pulsesIn[4].Label(PacBio::Mongo::Data::Pulse::NucleotideLabel::A);
    pulsesIn[5].Label(PacBio::Mongo::Data::Pulse::NucleotideLabel::C);
    pulsesIn[6].Label(PacBio::Mongo::Data::Pulse::NucleotideLabel::G);
    pulsesIn[7].Label(PacBio::Mongo::Data::Pulse::NucleotideLabel::T);

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

    BazToPulse parser(info);
    size_t len = 0;
    for (const auto& p : pulsesIn) len += parser.BytesRequired(p);

    EXPECT_EQ(len, 32);
    auto data = std::make_unique<uint8_t[]>(len);

    parser.Reset();
    uint8_t* ptr = data.get();
    for (const auto& p : pulsesIn) ptr = parser.Serialize(p, ptr);
    ASSERT_EQ(ptr, data.get()+len);

    parser.Reset();
    auto const* ptr2 = data.get();
    for (auto& p : pulsesOut) ptr2 = parser.Deserialize(p, ptr2);
    ASSERT_EQ(ptr, data.get()+len);

    for (size_t i = 0; i < pulsesIn.size(); ++i)
    {
        EXPECT_TRUE(pulsesIn[i] == pulsesOut[i]) << i;
    }
}
