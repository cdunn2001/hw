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

#include <common/cuda/memory/AllocationViews.cuh>
#include <common/cuda/memory/UnifiedCudaArray.h>
#include <common/cuda/streams/LaunchManager.cuh>
#include <common/cuda/utility/CudaArray.h>

#include <bazio/encoding/PulseGroups.h>
#include <bazio/encoding/PulseToBaz.h>

#include <dataTypes/Pulse.h>

#include <iostream>

using namespace PacBio::BazIO;
using namespace PacBio::Cuda::Memory;
using namespace PacBio::Cuda::Utility;

// This is annoying.  This function is invoked with HostView arguments
// on the Host, and DeviceView arguments on the device.  There does not
// exist a code path where we can invoke this with HostView arguemnts
// on the device, but the cuda compiler does not appear to realize that.
// This will generate several warnings about potentially invoking host
// functions on the device, so here we explicitly squash that warning.
#pragma nv_exec_check_disable
template <typename PulseToBaz_t, size_t expectedLen,
          typename PulsesIn,
          typename PulsesOut,
          typename Overrun>
__host__ __device__ void TestPulseToBaz(PulsesIn pulsesIn,
                                        PulsesOut pulsesOut,
                                        Overrun overrun)
{
    PulseToBaz_t t{};
    size_t len = 0;
    for (size_t i = 0; i < pulsesIn.Size(); ++i)
    {
        len += t.BytesRequired(pulsesIn[i]);
    }

    if (len != expectedLen)
    {
        overrun[0] = true;
        return;
    }

    CudaArray<uint8_t, expectedLen> raw{};

    t.Reset();
    uint8_t* ptr = raw.data();

    for (size_t i = 0; i < pulsesIn.Size(); ++i)
    {
        ptr = t.Serialize(pulsesIn[i], ptr);
    }
    if (ptr != raw.data() + len)
    {
        overrun[0] = true;
        return;
    }

    t.Reset();
    auto const* ptr2 = raw.data();
    for (size_t i = 0; i < pulsesOut.Size(); ++i)
    {
        ptr2 = t.Deserialize(pulsesOut[i], ptr2);
    }
    if (ptr2 != raw.data() + len)
    {
        overrun[0] = true;
        return;
    }
    overrun[0] = false;
}

template <typename PulseToBaz_t, size_t expectedLen>
__global__ void RunGpuTest(DeviceView<const PacBio::Mongo::Data::Pulse> pulsesIn,
                           DeviceView<PacBio::Mongo::Data::Pulse> pulsesOut,
                           DeviceView<bool> overrun)
{
    // Neither the serializers nor the test itself is multi-threaded
    assert(blockDim.x * blockDim.y * blockDim.z == 1);
    assert(gridDim.x * gridDim.y * gridDim.z == 1);

    TestPulseToBaz<PulseToBaz_t, expectedLen>(pulsesIn, pulsesOut, overrun);
}

// Originally written as a mistake when I misunderstood which configurations
// we were considering.  Keeping it, because it exercises the truncate path
// that we still do want for things like the pulse label, and because it can
// still be used to get a feel for maximual throughput, where there are no
// checks/tweaks on the data that we might be tempted to optomize.
TEST(PulseToBaz, KestrelLossyTruncate)
{
    using Test = PulseToBaz<Field<PacketFieldName::Label,
                                  StoreSigned_t<false>,
                                  Transform<NoOp>,
                                  Serialize<TruncateOverflow, NumBits_t<2>>
                                  >,
                            Field<PacketFieldName::Pw,
                                  StoreSigned_t<false>,
                                  Transform<NoOp>,
                                  Serialize<TruncateOverflow, NumBits_t<7>>
                                  >,
                            Field<PacketFieldName::StartFrame,
                                  StoreSigned_t<false>,
                                  Transform<DeltaCompression>,
                                  Serialize<TruncateOverflow,
                                            NumBits_t<7>>
                                  >
                            >;

    UnifiedCudaArray<PacBio::Mongo::Data::Pulse>  pulsesIn{8, SyncDirection::Symmetric, SOURCE_MARKER()};
    UnifiedCudaArray<PacBio::Mongo::Data::Pulse> pulsesOut{8, SyncDirection::Symmetric, SOURCE_MARKER()};
    UnifiedCudaArray<bool> Overrun{1, SyncDirection::Symmetric, SOURCE_MARKER()};

    {
        auto pulsesInV = pulsesIn.GetHostView();
        pulsesInV[0].Label(PacBio::Mongo::Data::Pulse::NucleotideLabel::A);
        pulsesInV[1].Label(PacBio::Mongo::Data::Pulse::NucleotideLabel::C);
        pulsesInV[2].Label(PacBio::Mongo::Data::Pulse::NucleotideLabel::G);
        pulsesInV[3].Label(PacBio::Mongo::Data::Pulse::NucleotideLabel::T);
        pulsesInV[4].Label(PacBio::Mongo::Data::Pulse::NucleotideLabel::A);
        pulsesInV[5].Label(PacBio::Mongo::Data::Pulse::NucleotideLabel::C);
        pulsesInV[6].Label(PacBio::Mongo::Data::Pulse::NucleotideLabel::G);
        pulsesInV[7].Label(PacBio::Mongo::Data::Pulse::NucleotideLabel::T);

        pulsesInV[0].Width(4);
        pulsesInV[1].Width(120);
        pulsesInV[2].Width(55);
        pulsesInV[3].Width(32);
        pulsesInV[4].Width(182);
        pulsesInV[5].Width(5);
        pulsesInV[6].Width(1);
        pulsesInV[7].Width(300);

        uint32_t start = 0;
        pulsesInV[0].Start(start += 14);
        pulsesInV[1].Start(start += pulsesInV[0].Width() + 130);
        pulsesInV[2].Start(start += pulsesInV[1].Width() + 5);
        pulsesInV[3].Start(start += pulsesInV[2].Width() + 18);
        pulsesInV[4].Start(start += pulsesInV[3].Width() + 22);
        pulsesInV[5].Start(start += pulsesInV[4].Width() + 500);
        pulsesInV[6].Start(start += pulsesInV[5].Width() + 64);
        pulsesInV[7].Start(start += pulsesInV[6].Width() + 33);
    }

    auto Validate = [&]()
    {
        auto startFrameTrunc = 0;
        auto startFrameTruth = 0;
        auto pulsesInV = pulsesIn.GetHostView();
        auto pulsesOutV = pulsesOut.GetHostView();
        for (size_t i = 0; i < pulsesIn.Size(); ++i)
        {
            EXPECT_EQ(pulsesInV[i].Label(), pulsesOutV[i].Label()) << i;
            EXPECT_EQ(pulsesInV[i].Width()%128, pulsesOutV[i].Width()) << i;
            auto deltaTruth = pulsesInV[i].Start() - startFrameTruth;
            startFrameTruth = pulsesInV[i].Start();
            auto deltaTrunc = pulsesOutV[i].Start() - startFrameTrunc;
            startFrameTrunc = pulsesOutV[i].Start();
            EXPECT_EQ(deltaTruth%128, deltaTrunc) << i;
        }
    };

    static constexpr size_t expectedLen = 16;

    PacBio::Cuda::PBLauncher(RunGpuTest<Test, expectedLen>, 1, 1)(pulsesIn, pulsesOut, Overrun);
    EXPECT_FALSE(Overrun.GetHostView()[0]);
    Validate();

    TestPulseToBaz<Test, expectedLen>(pulsesIn.GetHostView(), pulsesOut.GetHostView(), Overrun.GetHostView());
    EXPECT_FALSE(Overrun.GetHostView()[0]);
    Validate();
}

// Checks the "simple" overflow scheme, where if we overflow we just automatically write
// a 4 byte overflow value
TEST(PulseToBaz, KestrelLosslessSimple)
{
    using Test = PulseToBaz<Field<PacketFieldName::Label,
                                  StoreSigned_t<false>,
                                  Transform<NoOp>,
                                  Serialize<TruncateOverflow,  NumBits_t<2>>
                                  >,
                            Field<PacketFieldName::Pw,
                                  StoreSigned_t<false>,
                                  Transform<NoOp>,
                                  Serialize<SimpleOverflow, NumBits_t<7>, NumBytes_t<4>>
                                  >,
                            Field<PacketFieldName::StartFrame,
                                  StoreSigned_t<false>,
                                  Transform<DeltaCompression>,
                                  Serialize<SimpleOverflow, NumBits_t<7>, NumBytes_t<4>>
                                  >
                            >;

    UnifiedCudaArray<PacBio::Mongo::Data::Pulse>  pulsesIn{8, SyncDirection::Symmetric, SOURCE_MARKER()};
    UnifiedCudaArray<PacBio::Mongo::Data::Pulse> pulsesOut{8, SyncDirection::Symmetric, SOURCE_MARKER()};
    UnifiedCudaArray<bool> Overrun{1, SyncDirection::Symmetric, SOURCE_MARKER()};

    {
        auto pulsesInV = pulsesIn.GetHostView();
        pulsesInV[0].Label(PacBio::Mongo::Data::Pulse::NucleotideLabel::A);
        pulsesInV[1].Label(PacBio::Mongo::Data::Pulse::NucleotideLabel::C);
        pulsesInV[2].Label(PacBio::Mongo::Data::Pulse::NucleotideLabel::G);
        pulsesInV[3].Label(PacBio::Mongo::Data::Pulse::NucleotideLabel::T);
        pulsesInV[4].Label(PacBio::Mongo::Data::Pulse::NucleotideLabel::A);
        pulsesInV[5].Label(PacBio::Mongo::Data::Pulse::NucleotideLabel::C);
        pulsesInV[6].Label(PacBio::Mongo::Data::Pulse::NucleotideLabel::G);
        pulsesInV[7].Label(PacBio::Mongo::Data::Pulse::NucleotideLabel::T);

        pulsesInV[0].Width(4);
        pulsesInV[1].Width(120);
        pulsesInV[2].Width(55);
        pulsesInV[3].Width(32);
        pulsesInV[4].Width(182);
        pulsesInV[5].Width(5);
        pulsesInV[6].Width(1);
        pulsesInV[7].Width(300);

        uint32_t start = 0;
        pulsesInV[0].Start(start += 14);
        pulsesInV[1].Start(start += pulsesInV[0].Width() + 130);
        pulsesInV[2].Start(start += pulsesInV[1].Width() + 5);
        pulsesInV[3].Start(start += pulsesInV[2].Width() + 18);
        pulsesInV[4].Start(start += pulsesInV[3].Width() + 22);
        pulsesInV[5].Start(start += pulsesInV[4].Width() + 500);
        pulsesInV[6].Start(start += pulsesInV[5].Width() + 64);
        pulsesInV[7].Start(start += pulsesInV[6].Width() + 33);
    }

    auto Validate = [&]()
    {
        auto pulsesInV = pulsesIn.GetHostView();
        auto pulsesOutV = pulsesOut.GetHostView();
        for (size_t i = 0; i < pulsesIn.Size(); ++i)
        {
            EXPECT_EQ(pulsesInV[i].Label(), pulsesOutV[i].Label()) << i;
            EXPECT_EQ(pulsesInV[i].Width(), pulsesOutV[i].Width()) << i;
            EXPECT_EQ(pulsesInV[i].Start(), pulsesOutV[i].Start()) << i;
        }
    };

    static constexpr size_t expectedLen = 32;

    PacBio::Cuda::PBLauncher(RunGpuTest<Test, expectedLen>, 1, 1)(pulsesIn, pulsesOut, Overrun);
    EXPECT_FALSE(Overrun.GetHostView()[0]);
    Validate();

    TestPulseToBaz<Test, expectedLen>(pulsesIn.GetHostView(), pulsesOut.GetHostView(), Overrun.GetHostView());
    EXPECT_FALSE(Overrun.GetHostView()[0]);
    Validate();
}

// Checks the "compact" scheme, where we only write as many bytes as required to represent the value,
// at the cost of one byte per bit to encode if there are subsequent bytes to read/write
TEST(PulseToBaz, KestrelLosslessCompact)
{
    using Test = PulseToBaz<Field<PacketFieldName::Label,
                                  StoreSigned_t<false>,
                                  Transform<NoOp>,
                                  Serialize<TruncateOverflow, NumBits_t<2>>
                                  >,
                            Field<PacketFieldName::Pw,
                                  StoreSigned_t<false>,
                                  Transform<NoOp>,
                                  Serialize<CompactOverflow, NumBits_t<7>>
                                  >,
                            Field<PacketFieldName::StartFrame,
                                  StoreSigned_t<false>,
                                  Transform<DeltaCompression>,
                                  Serialize<CompactOverflow, NumBits_t<7>>
                                  >
                            >;

    UnifiedCudaArray<PacBio::Mongo::Data::Pulse>  pulsesIn{8, SyncDirection::Symmetric, SOURCE_MARKER()};
    UnifiedCudaArray<PacBio::Mongo::Data::Pulse> pulsesOut{8, SyncDirection::Symmetric, SOURCE_MARKER()};
    UnifiedCudaArray<bool> Overrun{1, SyncDirection::Symmetric, SOURCE_MARKER()};

    {
        auto pulsesInV = pulsesIn.GetHostView();
        pulsesInV[0].Label(PacBio::Mongo::Data::Pulse::NucleotideLabel::A);
        pulsesInV[1].Label(PacBio::Mongo::Data::Pulse::NucleotideLabel::C);
        pulsesInV[2].Label(PacBio::Mongo::Data::Pulse::NucleotideLabel::G);
        pulsesInV[3].Label(PacBio::Mongo::Data::Pulse::NucleotideLabel::T);
        pulsesInV[4].Label(PacBio::Mongo::Data::Pulse::NucleotideLabel::A);
        pulsesInV[5].Label(PacBio::Mongo::Data::Pulse::NucleotideLabel::C);
        pulsesInV[6].Label(PacBio::Mongo::Data::Pulse::NucleotideLabel::G);
        pulsesInV[7].Label(PacBio::Mongo::Data::Pulse::NucleotideLabel::T);

        pulsesInV[0].Width(4);
        pulsesInV[1].Width(120);
        pulsesInV[2].Width(55);
        pulsesInV[3].Width(32);
        pulsesInV[4].Width(182);
        pulsesInV[5].Width(5);
        pulsesInV[6].Width(1);
        pulsesInV[7].Width(300);

        uint32_t start = 0;
        pulsesInV[0].Start(start += 14);
        pulsesInV[1].Start(start += pulsesInV[0].Width() + 130);
        pulsesInV[2].Start(start += pulsesInV[1].Width() + 5);
        pulsesInV[3].Start(start += pulsesInV[2].Width() + 18);
        pulsesInV[4].Start(start += pulsesInV[3].Width() + 22);
        pulsesInV[5].Start(start += pulsesInV[4].Width() + 500);
        pulsesInV[6].Start(start += pulsesInV[5].Width() + 64);
        pulsesInV[7].Start(start += pulsesInV[6].Width() + 33);
    }

    auto Validate = [&]()
    {
        auto pulsesInV = pulsesIn.GetHostView();
        auto pulsesOutV = pulsesOut.GetHostView();
        for (size_t i = 0; i < pulsesIn.Size(); ++i)
        {
            EXPECT_EQ(pulsesInV[i].Label(), pulsesOutV[i].Label()) << i;
            EXPECT_EQ(pulsesInV[i].Width(), pulsesOutV[i].Width()) << i;
            EXPECT_EQ(pulsesInV[i].Start(), pulsesOutV[i].Start()) << i;
        }
    };

    static constexpr size_t expectedLen = 24;

    PacBio::Cuda::PBLauncher(RunGpuTest<Test, expectedLen>, 1, 1)(pulsesIn, pulsesOut, Overrun);
    EXPECT_FALSE(Overrun.GetHostView()[0]);
    Validate();

    TestPulseToBaz<Test, expectedLen>(pulsesIn.GetHostView(), pulsesOut.GetHostView(), Overrun.GetHostView());
    EXPECT_FALSE(Overrun.GetHostView()[0]);
    Validate();
}

TEST(PulseToBaz, Params)
{
    auto pp_params = ProductionPulses::Params();
    EXPECT_EQ(1, pp_params.size());
    auto gp = pp_params.front();
    EXPECT_EQ(PacketFieldName::Base, gp.members[0].name);
    EXPECT_EQ(PacketFieldName::Pw, gp.members[1].name);
    EXPECT_EQ(PacketFieldName::StartFrame, gp.members[2].name);
    EXPECT_EQ(2, gp.numBits[0]);
    EXPECT_EQ(7, gp.numBits[1]);
    EXPECT_EQ(7, gp.numBits[2]);
    EXPECT_EQ(16, gp.totalBits);
    EXPECT_TRUE(gp.members[0].transform.front().var.Visit(
                                                          [](const NoOpTransformParams& v) { return true; },
                                                          [](const CodecParams& v) { return false; },
                                                          [](const FixedPointParams& v) { return false; },
                                                          [](const DeltaCompressionParams& v) { return false; }));
    EXPECT_TRUE(gp.members[0].serialize.var.Visit(
                                                  [](const TruncateParams& v) { return v.numBits == 2; },
                                                  [](const SimpleOverflowParams& v) { return false; },
                                                  [](const CompactOverflowParams& v) { return false; }));
    EXPECT_TRUE(gp.members[1].serialize.var.Visit(
                                                  [](const TruncateParams& v) { return false; },
                                                  [](const SimpleOverflowParams& v) { return false; },
                                                  [](const CompactOverflowParams& v) { return v.numBits == 7; }));
    EXPECT_TRUE(gp.members[2].transform.front().var.Visit([](const NoOpTransformParams& v) { return false; },
                                                          [](const CodecParams& v) { return false; },
                                                          [](const FixedPointParams& v) { return false; },
                                                          [](const DeltaCompressionParams& v) { return true; }));
    auto json = gp.Serialize();
    EXPECT_EQ(16, json["totalBits"].asUInt());
    EXPECT_EQ(2, json["numBits"][0].asUInt());
    EXPECT_EQ(7, json["numBits"][1].asUInt());
    EXPECT_EQ(7, json["numBits"][2].asUInt());
    EXPECT_EQ(2, json["members"][0]["serialize"]["var"]["TruncateParams"]["numBits"].asUInt());
    EXPECT_TRUE(json["members"][0]["transform"][0]["var"]["NoOpTransformParams"].isNull());

    auto ip_params = InternalPulses::Params();
    EXPECT_EQ(5, ip_params.size());
    gp = ip_params.front();
    EXPECT_EQ(2, gp.numBits[0]);
    EXPECT_EQ(7, gp.numBits[1]);
    EXPECT_EQ(7, gp.numBits[2]);
    EXPECT_EQ(16, gp.totalBits);
    auto json2 = ip_params[0].Serialize();
    EXPECT_EQ(json["totalBits"].asUInt(), json2["totalBits"].asUInt());
    EXPECT_EQ("Base", json2["members"][0]["name"].asString());

    gp = ip_params[1];
    EXPECT_EQ(1, gp.members.size());
    EXPECT_EQ(PacketFieldName::Pkmax, gp.members[0].name);
    EXPECT_EQ(1, gp.numBits.size());
    EXPECT_EQ(gp.numBits.front(), gp.totalBits);
    EXPECT_TRUE(gp.members[0].storeSigned);
    EXPECT_TRUE(gp.members[0].transform[0].var.Visit(
            [](const NoOpTransformParams& v) { return false; },
            [](const CodecParams& v) { return false; },
            [](const FixedPointParams& v) { return v.scale == 10; },
            [](const DeltaCompressionParams& v) { return false; }
            ));
    EXPECT_TRUE(gp.members[0].serialize.var.Visit(
            [](const TruncateParams& v) { return false; },
            [](const SimpleOverflowParams& v) { return v.numBits == 8 && v.overflowBytes == 2; },
            [](const CompactOverflowParams& v) { return false; }
            ));
    gp = ip_params[2];
    EXPECT_EQ(1, gp.members.size());
    EXPECT_EQ(PacketFieldName::Pkmid, gp.members[0].name);
    EXPECT_EQ(1, gp.numBits.size());
    EXPECT_EQ(gp.numBits.front(), gp.totalBits);
    EXPECT_TRUE(gp.members[0].transform[0].var.Visit(
            [](const NoOpTransformParams& v) { return false; },
            [](const CodecParams& v) { return false; },
            [](const FixedPointParams& v) { return v.scale == 10; },
            [](const DeltaCompressionParams& v) { return false; }
    ));
    EXPECT_TRUE(gp.members[0].serialize.var.Visit(
            [](const TruncateParams& v) { return false; },
            [](const SimpleOverflowParams& v) { return v.numBits == 8 && v.overflowBytes == 2; },
            [](const CompactOverflowParams& v) { return false; }
    ));
}
