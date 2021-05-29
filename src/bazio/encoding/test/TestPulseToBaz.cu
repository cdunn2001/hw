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
    ptr = raw.data();
    for (size_t i = 0; i < pulsesOut.Size(); ++i)
    {
        ptr = t.Deserialize(pulsesOut[i], ptr);
    }
    if (ptr != raw.data() + len)
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
    using Test = PulseToBaz<Field<PacketFieldName::Base,
                                  Transform<Identity>,
                                  Serialize<TruncateOverflow, NumBits_t<2>>
                                  >,
                            Field<PacketFieldName::Ipd,
                                  Transform<Identity>,
                                  Serialize<TruncateOverflow,
                                            NumBits_t<7>>
                                  >,
                            Field<PacketFieldName::Pw,
                                  Transform<Identity>,
                                  Serialize<TruncateOverflow, NumBits_t<7>>
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
        pulsesInV[1].Start(start += 130);
        pulsesInV[2].Start(start += 5);
        pulsesInV[3].Start(start += 18);
        pulsesInV[4].Start(start += 22);
        pulsesInV[5].Start(start += 500);
        pulsesInV[6].Start(start += 64);
        pulsesInV[7].Start(start += 33);
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
            auto ipdTruth = pulsesInV[i].Start() - startFrameTruth;
            startFrameTruth += ipdTruth;
            auto ipdTrunc = pulsesOutV[i].Start() - startFrameTrunc;
            startFrameTrunc += ipdTrunc;
            EXPECT_EQ(ipdTruth%128, ipdTrunc) << i;
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
    using Test = PulseToBaz<Field<PacketFieldName::Base,
                                  Transform<Identity>,
                                  Serialize<TruncateOverflow,  NumBits_t<2>>
                                  >,
                            Field<PacketFieldName::Ipd,
                                  Transform<Identity>,
                                  Serialize<SimpleOverflow, NumBits_t<7>, NumBytes_t<4>>
                                  >,
                            Field<PacketFieldName::Pw,
                                  Transform<Identity>,
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
        pulsesInV[1].Start(start += 130);
        pulsesInV[2].Start(start += 5);
        pulsesInV[3].Start(start += 18);
        pulsesInV[4].Start(start += 22);
        pulsesInV[5].Start(start += 500);
        pulsesInV[6].Start(start += 64);
        pulsesInV[7].Start(start += 33);
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
    using Test = PulseToBaz<Field<PacketFieldName::Base,
                                  Transform<Identity>,
                                  Serialize<TruncateOverflow, NumBits_t<2>>
                                  >,
                            Field<PacketFieldName::Ipd,
                                  Transform<Identity>,
                                  Serialize<CompactOverflow, NumBits_t<7>>
                                  >,
                            Field<PacketFieldName::Pw,
                                  Transform<Identity>,
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
        pulsesInV[1].Start(start += 130);
        pulsesInV[2].Start(start += 5);
        pulsesInV[3].Start(start += 18);
        pulsesInV[4].Start(start += 22);
        pulsesInV[5].Start(start += 500);
        pulsesInV[6].Start(start += 64);
        pulsesInV[7].Start(start += 33);
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

    static constexpr size_t expectedLen = 22;

    PacBio::Cuda::PBLauncher(RunGpuTest<Test, expectedLen>, 1, 1)(pulsesIn, pulsesOut, Overrun);
    EXPECT_FALSE(Overrun.GetHostView()[0]);
    Validate();

    TestPulseToBaz<Test, expectedLen>(pulsesIn.GetHostView(), pulsesOut.GetHostView(), Overrun.GetHostView());
    EXPECT_FALSE(Overrun.GetHostView()[0]);
    Validate();
}
