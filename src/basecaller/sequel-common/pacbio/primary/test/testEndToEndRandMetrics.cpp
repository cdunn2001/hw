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

static void write(const std::string& filename)
{
    std::vector<int> currentFrames(10, 0);
    FileHeaderBuilder fhb = ArminsFakeMovie();
    fhb.MovieLengthFrames(16384 *1);
    
    // LUT
    for (int i = 0; i < 10; ++i)
        fhb.AddZmwNumber(100000 + i);

    BazWriter<SequelMetricBlock> writer(filename, fhb, PacBio::Primary::BazIOConfig{}, 100000);
    for (int c = 0; c < 1; ++c)
    {
        auto now = std::chrono::high_resolution_clock::now();
        for (int j = 0; j < 10; ++j)
        {
            // Simulate base calls
            auto numEvents = 110;
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
            auto numHFmetrics = 11;
            std::vector<SequelMetricBlock> metrics;
            metrics.resize(numHFmetrics);
            for (int i = 0; i < numHFmetrics; ++i)
            {
                metrics[i].NumBasesA(2)
                          .NumBasesC(2)
                          .NumBasesG(3)
                          .NumBasesT(3)
                          .NumPulses(10)
                          .PulseWidth(2)
                          .BaseWidth(3)
                          .Baselines({{6,7}})
                          .PkmidA(8)
                          .PkmidC(9)
                          .PkmidG(10)
                          .PkmidT(11)
                          .BaselineSds({{12,13}})
                          .NumPkmidFramesA(14)
                          .NumPkmidFramesC(15)
                          .NumPkmidFramesG(16)
                          .NumPkmidFramesT(17)
                          .NumFrames(18)
                          .NumHalfSandwiches(10)
                          .NumPulseLabelStutters(10);
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
TEST(endToEndRandMetrics, Test)
{
    LogSeverityContext _(LogLevel::WARN);
    const std::string FILENAME1 = "end2endMetrics_test.baz";
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
            const auto& metrics = ParseMetrics(reader.Fileheader(), data, false);
            EXPECT_EQ(3, metrics.NumPulsesAll().size());
        }
    }
}
