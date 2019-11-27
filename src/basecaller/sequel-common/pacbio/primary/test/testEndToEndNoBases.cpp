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

#include <pacbio/primary/BazCore.h>
#include <pacbio/primary/BazReader.h>
#include <pacbio/primary/BazWriter.h>
#include <pacbio/primary/PrimaryToBaz.h>
#include <pacbio/primary/SmartMemory.h>
#include <pacbio/primary/Timing.h>

#include "ArminsFakeMovie.h"

using namespace PacBio::Primary;
using namespace PacBio::Logging;


static void write(const std::string& filename)
{
    std::vector<int> currentFrames(10, 0);
    FileHeaderBuilder fhb = ArminsFakeMovie();
    fhb.MovieLengthFrames(16384 *2);

    // LUT
    for (int i = 0; i < 10; ++i)
        fhb.AddZmwNumber(100000 + i);
    
    BazWriter<SequelMetricBlock> writer(filename, fhb, PacBio::Primary::BazIOConfig{}, 100000);
    for (int c = 0; c < 2; ++c)
    {
        // auto now = std::chrono::high_resolution_clock::now();
        for (int j = 0; j < 10; ++j)
        {
            // Simulate base calls
            auto numEvents = 0;
            Basecall* basecall = nullptr;
            auto numHFmetrics = 16;
            std::vector<SequelMetricBlock> metrics;
            metrics.resize(numHFmetrics);
            for (int i = 0; i < numHFmetrics; ++i)
            {
                metrics[i].NumBasesA(0)
                          .NumBasesC(0)
                          .NumBasesG(0)
                          .NumBasesT(0)
                          .NumPulses(0)
                          .PulseWidth(2)
                          .BaseWidth(3)
                          .Baselines({{7,6}})
                          .PkmidA(8)
                          .PkmidC(9)
                          .PkmidG(10)
                          .PkmidT(11)
                          .BaselineSds({{13,12}})
                          .NumPkmidFramesA(14)
                          .NumPkmidFramesC(15)
                          .NumPkmidFramesG(16)
                          .NumPkmidFramesT(17)
                          .NumBaselineFrames({{1,1}})
                          .NumFrames(18);
            }
            
            writer.AddZmwSlice(basecall, numEvents, std::move(metrics), j);
        }
        // Timing::PrintTime(now, "Simulation");
        writer.Flush();
    }
    writer.WaitForTermination();
    CheckFileSizes(writer);

}
TEST(endToEndNoBases, Test)
{
    LogSeverityContext _(LogLevel::WARN);

    const std::string FILENAME1 = "end2endNoBases_test.baz";
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
            EXPECT_FALSE(packets.HasPacketField(PacketFieldName::READOUT));
            EXPECT_FALSE(packets.HasPacketField(PacketFieldName::DEL_TAG));
            EXPECT_FALSE(packets.HasPacketField(PacketFieldName::SUB_TAG));
            EXPECT_FALSE(packets.HasPacketField(PacketFieldName::DEL_QV));
            EXPECT_FALSE(packets.HasPacketField(PacketFieldName::SUB_QV));
            EXPECT_FALSE(packets.HasPacketField(PacketFieldName::INS_QV));
            EXPECT_FALSE(packets.HasPacketField(PacketFieldName::MRG_QV));
            EXPECT_FALSE(packets.HasPacketField(PacketFieldName::IPD_V1));
            EXPECT_FALSE(packets.HasPacketField(PacketFieldName::IPD_LL));
            EXPECT_FALSE(packets.HasPacketField(PacketFieldName::PW_LL));
            EXPECT_FALSE(packets.HasPacketField(PacketFieldName::PW_V1));
            EXPECT_FALSE(packets.HasPacketField(PacketFieldName::IPD16_LL));
            EXPECT_FALSE(packets.HasPacketField(PacketFieldName::PW16_LL));

            const auto& metrics = ParseMetrics(reader.Fileheader(), data, false);
            EXPECT_EQ(8, metrics.NumPulsesAll().size());

            for (const auto& x : metrics.NumBasesAll().data())
                EXPECT_EQ(x, static_cast<uint32_t>(0));

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
            for (const auto& x : metrics.NumPulsesAll().data())
                EXPECT_EQ(x, static_cast<uint32_t>(0));
        }
    }
}
