#include <gtest/gtest.h>

#include <iostream>
#include <memory>
#include <vector>
#include <thread>
#include <chrono>
#include <random>
#include <sstream>
#include <string>
#include <assert.h>

#include <pacbio/dev/gtest-extras.h>
#include <pacbio/dev/TemporaryDirectory.h>
#include <pacbio/smrtdata/Basecall.h>
#include <pacbio/smrtdata/Pulse.h>

#include <pacbio/primary/BazCore.h>
#include <pacbio/primary/BazReader.h>
#include <pacbio/primary/BazWriter.h>
#include <pacbio/primary/PrimaryToBaz.h>
#include <pacbio/primary/SmartMemory.h>
#include <pacbio/primary/Timing.h>

#include "ArminsFakeMovie.h"

using namespace PacBio::Primary;
using namespace PacBio::Logging;

static void write(const std::string& filename, int numChunks, int numZmws)
{
//    TEST_COUT << "Write numChunks:" << numChunks << " numZmws:" << numZmws << std::endl;

    std::vector<int> currentFrames(numZmws, 0);
    FileHeaderBuilder fhb = ArminsFakeMovie();
    fhb.MovieLengthFrames(16384 * numChunks);

    // LUT
    for (int i = 0; i < numZmws; ++i)
        fhb.AddZmwNumber(100000 + i);

    BazWriter<SequelMetricBlock> writer(filename, fhb, PacBio::Primary::BazIOConfig{}, 100000);
    for (int c = 0; c < numChunks; ++c)
    {
        auto now = std::chrono::high_resolution_clock::now();
        for (int j = 0; j < numZmws; ++j)
        {
            // Simulate base calls
            auto numEvents = 5 * 160;
            Basecall* basecall = new Basecall[numEvents];
            for (int i = 0; i < numEvents; ++i)
            {
                int ipd = 8;
                basecall[i].Base(NucleotideLabel::C);
                basecall[i].DeletionTag(NucleotideLabel::G);
                basecall[i].SubstitutionTag(NucleotideLabel::T);
                basecall[i].DeletionQV(4);
                basecall[i].SubstitutionQV(5);
                basecall[i].InsertionQV(6);
                basecall[i].GetPulse().MergeQV(7);
                basecall[i].GetPulse().Start(currentFrames[j] + ipd);
                basecall[i].GetPulse().Width(1);
                currentFrames[j] += ipd + basecall[i].GetPulse().Width();
            }
            auto numHFmetrics = 16;
            std::vector<SequelMetricBlock> metrics;
            metrics.resize(numHFmetrics);
            for (int i = 0; i < numHFmetrics; ++i)
            {
                metrics[i].NumBasesA(10)
                          .NumBasesC(10)
                          .NumBasesG(15)
                          .NumBasesT(15)
                          .NumPulses(5 * 160 / 16)
                          .PulseWidth(2)
                          .BaseWidth(3)
                          .Baselines({{7,6}})
                          .PkmidA(10000)
                          .PkmidC(9)
                          .PkmidG(1.0)
                          .PkmidT(11)
                          .BaselineSds({{1.3, 111.2}})
                          .NumBaselineFrames({{1,1}})
                          .NumPkmidFramesA(14)
                          .NumPkmidFramesC(15)
                          .NumPkmidFramesG(16)
                          .NumPkmidFramesT(17)
                          .NumFrames(2 * 5 * 160 / 16)
                          .NumHalfSandwiches(20)
                          .NumPulseLabelStutters(21)
                          .PixelChecksum(-22)
                          .PkmaxA((i+1)%4)
                          .PkmaxC((i+1)%4)
                          .PkmaxG((i+1)%4)
                          .PkmaxT((i+1)%4)
                          .PulseDetectionScore(2.5);
            }
            
            const bool r = writer.AddZmwSlice(basecall, numEvents, std::move(metrics), j);
            EXPECT_TRUE(r);
        }
        Timing::PrintTime(now, "Simulation");
        EXPECT_TRUE(writer.Flush());
    }
    writer.WaitForTermination();

    CheckFileSizes(writer);
}

TEST(endToEnd, Test)
{
    LogSeverityContext _(LogLevel::ERROR);
    const std::string FILENAME1 = "end2end_test1.baz";
    static PacBio::Dev::TemporaryDirectory tmpdir;
    std::string FILENAME = tmpdir.DirName() + "/" + FILENAME1;

    write(FILENAME, 2, 10);
    BazReader reader(FILENAME);
    EXPECT_TRUE(reader.HasNext());

    while (reader.HasNext())
    {
        auto zmwData = reader.NextSlice();
        EXPECT_EQ(static_cast<uint32_t>(10), zmwData.size());

        for (const auto& data : zmwData)
        {
            const auto& packets = ParsePackets(reader.Fileheader(), data);
            EXPECT_TRUE(packets.HasPacketField(PacketFieldName::READOUT));
            EXPECT_FALSE(packets.HasPacketField(PacketFieldName::DEL_TAG));
            EXPECT_FALSE(packets.HasPacketField(PacketFieldName::SUB_TAG));
            EXPECT_FALSE(packets.HasPacketField(PacketFieldName::DEL_QV));
            EXPECT_FALSE(packets.HasPacketField(PacketFieldName::SUB_QV));
            EXPECT_FALSE(packets.HasPacketField(PacketFieldName::INS_QV));
            EXPECT_FALSE(packets.HasPacketField(PacketFieldName::MRG_QV));
            EXPECT_TRUE(packets.HasPacketField(PacketFieldName::IPD_V1));
            EXPECT_FALSE(packets.HasPacketField(PacketFieldName::IPD_LL));
            EXPECT_FALSE(packets.HasPacketField(PacketFieldName::PW_LL));
            EXPECT_TRUE(packets.HasPacketField(PacketFieldName::PW_V1));
            EXPECT_FALSE(packets.HasPacketField(PacketFieldName::IPD16_LL));
            EXPECT_FALSE(packets.HasPacketField(PacketFieldName::PW16_LL));

            EXPECT_EQ(static_cast<size_t>(5 * 160 * 2),
                      packets.PacketField(PacketFieldName::READOUT).size());

            for (const auto& x : packets.PacketField(PacketFieldName::READOUT))
                EXPECT_EQ(static_cast<uint32_t>('C'), x);
            for (const auto& x : packets.PacketField(PacketFieldName::IPD_V1))
                EXPECT_EQ(static_cast<uint32_t>(8), x);
            for (const auto& x : packets.PacketField(PacketFieldName::PW_V1))
                EXPECT_EQ(static_cast<uint32_t>(1), x);

            const auto& metrics = ParseMetrics(reader.Fileheader(), data, false);
            EXPECT_EQ(metrics.NumPulsesAll().size(), 8);

            for (const auto& x : metrics.NumBasesAll().data())
                EXPECT_EQ(x, static_cast<uint32_t>(200));

            for (const auto& x : metrics.PkMid().data().A)
                ASSERT_FLOAT_EQ(10000, x);
            for (const auto& x : metrics.PkMid().data().C)
                ASSERT_FLOAT_EQ(9, x);
            for (const auto& x : metrics.PkMid().data().G)
                ASSERT_FLOAT_EQ(1.0, x);
            for (const auto& x : metrics.PkMid().data().T)
                ASSERT_FLOAT_EQ(11, x);

            for (const auto& x : metrics.NumHalfSandwiches().data())
                EXPECT_EQ(static_cast<uint32_t>(80), x);
            for (const auto& x : metrics.NumPulseLabelStutters().data())
                EXPECT_EQ(static_cast<uint32_t>(84), x);
            for (const auto& x : metrics.NumPulsesAll().data())
                EXPECT_EQ(static_cast<uint32_t>(200), x);
            for (const auto& x : metrics.PkMax().data().A)
                ASSERT_FLOAT_EQ(3, x);
            for (const auto& x : metrics.PkMax().data().C)
                ASSERT_FLOAT_EQ(3, x);
            for (const auto& x : metrics.PkMax().data().G)
                ASSERT_FLOAT_EQ(3, x);
            for (const auto& x : metrics.PkMax().data().T)
                ASSERT_FLOAT_EQ(3, x);
            for (const auto& x : metrics.PulseDetectionScore().data())
                ASSERT_FLOAT_EQ(2.5, x);

            for (const auto& x : metrics.BaselineSD().data().red)
                ASSERT_FLOAT_EQ(111.1875, x);
            for (const auto& x : metrics.BaselineSD().data().green)
                ASSERT_FLOAT_EQ(1.2998047, x);
            for (const auto& x : metrics.BaselineMean().data().green)
                ASSERT_FLOAT_EQ(7, x);
            for (const auto& x : metrics.BaselineMean().data().red)
                ASSERT_FLOAT_EQ(6, x);

            for (const auto& x : metrics.PixelChecksum().data())
                EXPECT_EQ(static_cast<int16_t>(4 * -22), x);
        }
    }
}


TEST(endToEnd, FastFileSizeChecks)
{
    LogSeverityContext _(LogLevel::WARN);
    const std::string FILENAME1 = "end2end_FastFileSizeChecks.baz";
    static PacBio::Dev::TemporaryDirectory tmpdir;
    std::string FILENAME = tmpdir.DirName() + "/" + FILENAME1;


    // file size checks
    write(FILENAME, 2, 100);
    write(FILENAME, 10, 10);
    write(FILENAME, 2, 1000);
}
// these are slow, so they are disabled.
TEST(endToEnd, DISABLED_SlowFileSizeChecks)
{
    const std::string FILENAME1 = "end2end_SlowFileSizeChecks.baz";
    static PacBio::Dev::TemporaryDirectory tmpdir;
    std::string FILENAME = tmpdir.DirName() + "/" + FILENAME1;

    // file size checks
    write(FILENAME, 2,10000);
    write(FILENAME, 10,100000);
    write(FILENAME, 100,10000);
}
