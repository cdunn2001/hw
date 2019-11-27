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

#include <pacbio/primary/BazCore.h>
#include <pacbio/primary/BazReader.h>
#include <pacbio/primary/BazWriter.h>
#include <pacbio/primary/PrimaryToBaz.h>
#include <pacbio/primary/SmartMemory.h>
#include <pacbio/primary/Timing.h>

#include "ArminsFakeMovie.h"

using namespace PacBio::Primary;
using namespace PacBio::Logging;

static const size_t numBases = 2 * 5 * 163;

static void write(const std::string& filename)
{
    std::vector<int> currentFrames(10, 0);
    FileHeaderBuilder fhb = ArminsFakeNoMetricsMovie();
    fhb.MovieLengthFrames(2 * 16384);

    // LUT
    for (int i = 0; i < 10; ++i)
        fhb.AddZmwNumber(100000 + i);
    
    BazWriter<SequelMetricBlock> writer(filename, fhb, PacBio::Primary::BazIOConfig{}, 100000);
    for (int c = 0; c < 2; ++c)
    {
        auto now = std::chrono::high_resolution_clock::now();
        for (int j = 0; j < 10; ++j)
        {
            // Simulate base calls
            auto numEvents = 5 * 163;
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
            writer.AddZmwSlice(basecall, numEvents, std::vector<SequelMetricBlock>(), j);
        }
        Timing::PrintTime(now, "Simulation");
        writer.Flush();
    }
    writer.WaitForTermination();
    CheckFileSizes(writer);

}
TEST(end2endProductionNoMetrics, Test123)
{
    LogSeverityContext _(LogLevel::WARN);
    const std::string FILENAME1 = "end2endNoMetrics_test.baz";
    static PacBio::Dev::TemporaryDirectory tmpdir;
    std::string FILENAME = tmpdir.DirName() + "/" + FILENAME1;

    write(FILENAME);
    BazReader reader(FILENAME);
    EXPECT_TRUE(reader.HasNext());

    while (reader.HasNext())
    {
        auto zmwData = reader.NextSlice();
        EXPECT_EQ(zmwData.size(), static_cast<uint32_t>(10));

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

            EXPECT_EQ(packets.PacketField(PacketFieldName::READOUT).size(), numBases);
            EXPECT_EQ(packets.PacketField(PacketFieldName::IPD_V1).size(), numBases);
            EXPECT_EQ(packets.PacketField(PacketFieldName::PW_V1).size(), numBases);

            for (const auto& x : packets.PacketField(PacketFieldName::READOUT))
                EXPECT_EQ(x, static_cast<uint32_t>('C'));
            for (const auto& x : packets.PacketField(PacketFieldName::IPD_V1))
                EXPECT_EQ(x, static_cast<uint32_t>(8));
            for (const auto& x : packets.PacketField(PacketFieldName::PW_V1))
                EXPECT_EQ(x, static_cast<uint32_t>(1));

            const auto& metrics = ParseMetrics(reader.Fileheader(), data, false);
            EXPECT_EQ(0, metrics.NumPulsesAll().size());
        }
    }
}

