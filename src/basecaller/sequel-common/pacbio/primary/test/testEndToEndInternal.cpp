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


#include <pacbio/dev/TemporaryDirectory.h>
#include <pacbio/smrtdata/Basecall.h>
#include <pacbio/smrtdata/Pulse.h>
#include <pacbio/logging/Logger.h>

#include <pacbio/primary/BazWriter.h>
#include <pacbio/primary/BazReader.h>
#include <pacbio/primary/BazCore.h>
#include <pacbio/primary/SmartMemory.h>
#include <pacbio/primary/Timing.h>
#include <pacbio/primary/PrimaryToBaz.h>

#include "ArminsFakeMovie.h"

using namespace PacBio::Primary;
using namespace PacBio::Logging;

static void write(const std::string& filename)
{
    int numZmws = 10;
    int numChunks = 10;
    std::vector<int> currentFrames(numZmws, 0);
    FileHeaderBuilder fhb = ArminsFakeMovieHighVerbosity( );
    fhb.MovieLengthFrames(16384 * numChunks);

    // LUT
    for (int i = 0; i < numZmws; ++i)
        fhb.AddZmwNumber(100000 + i);

    BazWriter<SequelMetricBlock> writer(filename, fhb, PacBio::Primary::BazIOConfig{}, 1000000);
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
                int ipd = !(i % 20) ? 65280 : 12;
                basecall[i].Base(NucleotideLabel::C);
                basecall[i].DeletionTag(NucleotideLabel::G);
                basecall[i].SubstitutionTag(NucleotideLabel::T);
                basecall[i].DeletionQV(4);
                basecall[i].SubstitutionQV(5);
                basecall[i].InsertionQV(6);
                basecall[i].GetPulse().MergeQV(7);
                basecall[i].GetPulse().AltLabelQV(8);
                basecall[i].GetPulse().LabelQV(9);
                basecall[i].GetPulse().AltLabel(NucleotideLabel::G);
                basecall[i].GetPulse().Label(NucleotideLabel::T);
                basecall[i].GetPulse().MeanSignal(!(i % 20) ? 6528 : 14);
                basecall[i].GetPulse().MidSignal(!(i % 20) ? 6528 : 15);
                basecall[i].GetPulse().Start(currentFrames[j] + ipd);
                basecall[i].GetPulse().Width(!(i % 20) ? 65280 : 13);
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
                          .NumPulses(50)
                          .PulseWidth(2)
                          .BaseWidth(3)
                          .Baselines({{7,6}})
                          .NumBaselineFrames({{1,1}})
                          .PkmidA(8)
                          .PkmidC(9)
                          .PkmidG(10)
                          .PkmidT(11)
                          .BaselineSds({{13,12}})
                          .NumPkmidFramesA(14)
                          .NumPkmidFramesC(15)
                          .NumPkmidFramesG(16)
                          .NumPkmidFramesT(17)
                          .NumFrames(2 * 5 * 160 / 16)
                          .PkmaxA((i+1)%4)
                          .PkmaxC((i+1)%4)
                          .PkmaxG((i+1)%4)
                          .PkmaxT((i+1)%4)
                          .PulseDetectionScore(2.5);
            }
            
            writer.AddZmwSlice(basecall, numEvents, std::move(metrics), j);
        }
        Timing::PrintTime(now, "Simulation");
        writer.Flush();
    }
    writer.WaitForTermination();

    CheckFileSizes(writer);
}

TEST(endToEndInternal, NoData)
{
    LogSeverityContext _(LogLevel::WARN);
    const std::string FILENAME1 = "end2endinternal_test.baz";
    static PacBio::Dev::TemporaryDirectory tmpdir;
    std::string FILENAME = tmpdir.DirName() + "/" + FILENAME1;

    write(FILENAME);
    BazReader reader(FILENAME);
    EXPECT_TRUE(reader.HasNext());

    while (reader.HasNext())
    {
        auto zmwData = reader.NextSlice();
        EXPECT_EQ(zmwData.size(), static_cast<size_t>(10));

        for (const auto& data : zmwData)
        {
            const auto& packets = ParsePackets(reader.Fileheader(), data);
            EXPECT_TRUE(packets.HasPacketField(PacketFieldName::READOUT));
            EXPECT_TRUE(packets.HasPacketField(PacketFieldName::DEL_TAG));
            EXPECT_TRUE(packets.HasPacketField(PacketFieldName::SUB_TAG));
            EXPECT_TRUE(packets.HasPacketField(PacketFieldName::DEL_QV));
            EXPECT_TRUE(packets.HasPacketField(PacketFieldName::SUB_QV));
            EXPECT_TRUE(packets.HasPacketField(PacketFieldName::INS_QV));
            EXPECT_TRUE(packets.HasPacketField(PacketFieldName::MRG_QV));
            EXPECT_TRUE(packets.HasPacketField(PacketFieldName::ALT_QV));
            EXPECT_TRUE(packets.HasPacketField(PacketFieldName::LAB_QV));
            EXPECT_TRUE(packets.HasPacketField(PacketFieldName::IPD_LL));
            EXPECT_FALSE(packets.HasPacketField(PacketFieldName::IPD_V1));
            EXPECT_FALSE(packets.HasPacketField(PacketFieldName::IPD16_LL));
            EXPECT_TRUE(packets.HasPacketField(PacketFieldName::PW_LL));
            EXPECT_FALSE(packets.HasPacketField(PacketFieldName::PW_V1));
            EXPECT_FALSE(packets.HasPacketField(PacketFieldName::PW16_LL));
            EXPECT_TRUE(packets.HasPacketField(PacketFieldName::PKMEAN_LL));
            EXPECT_TRUE(packets.HasPacketField(PacketFieldName::PKMID_LL));
            EXPECT_TRUE(packets.HasPacketField(PacketFieldName::LABEL));
            EXPECT_TRUE(packets.HasPacketField(PacketFieldName::ALT_LABEL));

            for (const auto& x : packets.PacketField(PacketFieldName::READOUT))
                EXPECT_EQ(x, static_cast<uint32_t>('C'));
            for (const auto& x : packets.PacketField(PacketFieldName::DEL_TAG))
                EXPECT_EQ(x, static_cast<uint32_t>('G'));
            for (const auto& x : packets.PacketField(PacketFieldName::SUB_TAG))
                EXPECT_EQ(x, static_cast<uint32_t>('T'));
            for (const auto& x : packets.PacketField(PacketFieldName::DEL_QV))
                EXPECT_EQ(x, static_cast<uint32_t>(4+33));
            for (const auto& x : packets.PacketField(PacketFieldName::SUB_QV))
                EXPECT_EQ(x, static_cast<uint32_t>(5+33));
            for (const auto& x : packets.PacketField(PacketFieldName::INS_QV))
                EXPECT_EQ(x, static_cast<uint32_t>(6+33));
            for (const auto& x : packets.PacketField(PacketFieldName::MRG_QV))
                EXPECT_EQ(x, static_cast<uint32_t>(7+33));
            for (const auto& x : packets.PacketField(PacketFieldName::ALT_QV))
                EXPECT_EQ(x, static_cast<uint32_t>(8+33));
            for (const auto& x : packets.PacketField(PacketFieldName::LAB_QV))
                EXPECT_EQ(x, static_cast<uint32_t>(9+33));
            for (const auto& x : packets.PacketField(PacketFieldName::LABEL))
                EXPECT_EQ(x, static_cast<uint32_t>('T'));
            for (const auto& x : packets.PacketField(PacketFieldName::ALT_LABEL))
                EXPECT_EQ(x, static_cast<uint32_t>('G'));

            for (const auto& x : packets.PacketField(PacketFieldName::IPD_LL))
            {
                if (x > 255)
                    EXPECT_EQ(x, static_cast<uint32_t>(65280));
                else
                    EXPECT_EQ(x, static_cast<uint32_t>(12));
            }
            for (const auto& x : packets.PacketField(PacketFieldName::PW_LL))
            {
                if (x > 255)
                    EXPECT_EQ(x, static_cast<uint32_t>(65280));
                else
                    EXPECT_EQ(x, static_cast<uint32_t>(13));
            }
            for (const auto& x : packets.PacketField(PacketFieldName::PKMEAN_LL))
            {
                if (x > 255)
                    EXPECT_EQ(x, static_cast<uint32_t>(65280));
                else
                    EXPECT_EQ(x, static_cast<uint32_t>(140));
            }
            for (const auto& x : packets.PacketField(PacketFieldName::PKMID_LL))
            {
                if (x > 255)
                    EXPECT_EQ(x, static_cast<uint32_t>(65280));
                else
                    EXPECT_EQ(x, static_cast<uint32_t>(150));
            }

            const auto& metrics = ParseMetrics(reader.Fileheader(), data, true);
            EXPECT_EQ(40, metrics.NumPulsesAll().size());

            for (const auto& x : metrics.NumPulsesAll().data())
                EXPECT_EQ(x, static_cast<uint32_t>(200));

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

            for (const auto& x : metrics.PkMid().data().A)
                EXPECT_EQ(x, static_cast<uint32_t>(8));
            for (const auto& x : metrics.PkMid().data().C)
                EXPECT_EQ(x, static_cast<uint32_t>(9));
            for (const auto& x : metrics.PkMid().data().G)
                EXPECT_EQ(x, static_cast<uint32_t>(10));
            for (const auto& x : metrics.PkMid().data().T)
                EXPECT_EQ(x, static_cast<uint32_t>(11));
            for (const auto& x : metrics.BaselineSD().data().red)
                EXPECT_EQ(x, static_cast<uint32_t>(12));
            for (const auto& x : metrics.BaselineSD().data().green)
                EXPECT_EQ(x, static_cast<uint32_t>(13));
            for (const auto& x : metrics.BaselineMean().data().green)
                EXPECT_EQ(x, static_cast<uint32_t>(7));
            for (const auto& x : metrics.BaselineMean().data().red)
                EXPECT_EQ(x, static_cast<uint32_t>(6));

            for (const auto& x : metrics.PulseWidth().data())
                EXPECT_EQ(x, static_cast<uint32_t>(2 * 4));
            for (const auto& x : metrics.BaseWidth().data())
                EXPECT_EQ(x, static_cast<uint32_t>(3 * 4));
            for (const auto& x : metrics.NumBasesAll().data())
                EXPECT_EQ(x, static_cast<uint32_t>(200));
        }
    }
}
