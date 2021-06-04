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

#include <bazio/BazWriter.h>
#include <bazio/BazReader.h>
#include <bazio/BazCore.h>
#include <bazio/SmartMemory.h>
#include <bazio/Timing.h>
#include <bazio/PrimaryToBaz.h>

#include <postprimary/hqrf/HQRegionFinder.h>

#include "ArminsFakeMovie.h"

using namespace PacBio::Primary;
using namespace PacBio::Primary::Postprimary;

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
                metrics[i].NumBasesA(20)
                          .NumBasesC(30)
                          .NumBasesG(20)
                          .NumBasesT(30)
                          .NumPkmidBasesA(20)
                          .NumPkmidBasesC(30)
                          .NumPkmidBasesG(20)
                          .NumPkmidBasesT(30)
                          .NumPulses(100)
                          .PulseWidth(2)
                          .BaseWidth(3)
                          .Baselines({{6,7}})
                          .PkmidA(190)
                          .PkmidC(410)
                          .PkmidG(210)
                          .PkmidT(390)
                          .PkzvarA(8)
                          .PkzvarC(9)
                          .PkzvarG(10)
                          .PkzvarT(11)
                          .BpzvarA(.08)
                          .BpzvarC(0.09)
                          .BpzvarG(0.10)
                          .BpzvarT(0.11)
                          .PkmaxA(200)
                          .PkmaxC(420)
                          .PkmaxG(220)
                          .PkmaxT(400)
                          .BaselineSds({{25,26}})
                          .NumBaselineFrames({{90,90}})
                          .NumPkmidFramesA(140)
                          .NumPkmidFramesC(150)
                          .NumPkmidFramesG(160)
                          .NumPkmidFramesT(170)
                          .NumFrames(4096)
                          .NumHalfSandwiches(10)
                          .NumSandwiches(10)
                          .PulseDetectionScore(-7)
                          .TraceAutocorr(.8)
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

TEST(HQRegionFinderFeatures, HQMetrics)
{
    const std::string FILENAME1 = "HQRFFeatures_test.baz";
    static PacBio::Dev::TemporaryDirectory tmpdir;
    std::string FILENAME = tmpdir.DirName() + "/" + FILENAME1;

    write(FILENAME);
    BazReader reader(FILENAME);

    // Check Experiment Metadata, which is needed for some features:
    const std::vector<float> relampExp{1, 0.946, 0.529, 0.553};
    EXPECT_EQ(relampExp, reader.Fileheader().RelativeAmplitudes());
    EXPECT_EQ("CTAG", reader.Fileheader().BaseMap());

    while (reader.HasNext())
    {
        auto zmwData = reader.NextSlice();

        for (const auto& data: zmwData)
        {
            const auto& metrics = ParseMetrics(reader.Fileheader(), data, false);
            EXPECT_EQ(3, metrics.NumPulsesAll().size());

            EXPECT_EQ(3, metrics.PkMid().size());
            EXPECT_EQ(190, metrics.PkMid().data().A[0]);

            const std::vector<float>& exp{190, 190, 190};
            const auto& obs = metrics.PkMid().data().A;
            EXPECT_EQ(static_cast<size_t>(3), obs.size());
            EXPECT_TRUE(std::equal(exp.begin(), exp.end(), obs.begin()));

            auto TestUIntMetric = [](const std::vector<uint32_t>& obs, std::vector<uint32_t> exp)
            {
                if (exp.size() != obs.size())
                    PBLOG_INFO << exp.size() << " " << obs.size();
                EXPECT_EQ(exp.size(), obs.size());
                if (!std::equal(exp.begin(), exp.end(),
                                obs.begin()))
                {
                    for (auto val : obs) std::cout << val << ", ";
                    std::cout << std::endl;
                    for (auto val : exp) std::cout << val << ", ";
                    std::cout << std::endl;
                }
                EXPECT_TRUE(std::equal(exp.begin(), exp.end(),
                                       obs.begin()));
            };

            auto TestFloatMetric = [](const std::vector<float>& obs, std::vector<float> exp, float delta=0.01)
            {
                const auto almostEqual = [delta](float a, float b)->bool {return std::abs(a - b) < delta;};
                if (exp.size() != obs.size())
                    PBLOG_INFO << exp.size() << " " << obs.size();
                EXPECT_EQ(exp.size(), obs.size());
                if (!std::equal(exp.begin(), exp.end(),
                            obs.begin(), almostEqual))
                {
                    for (auto val : obs) std::cout << val << ", ";
                    std::cout << std::endl;
                    for (auto val : exp) std::cout << val << ", ";
                    std::cout << std::endl;
                }
                EXPECT_TRUE(std::equal(exp.begin(), exp.end(),
                            obs.begin(), almostEqual));
            };

            TestUIntMetric(metrics.NumPulsesAll().data(), {400, 400, 300});
            TestFloatMetric(metrics.PulseWidth().data(), {8, 8, 6});
            TestUIntMetric(metrics.NumHalfSandwiches().data(), {40, 40, 30});
            TestUIntMetric(metrics.NumSandwiches().data(), {40, 40, 30});
            TestUIntMetric(metrics.NumPulseLabelStutters().data(), {40, 40, 30});
            TestUIntMetric(metrics.NumFrames().data(), {16384, 16384, 12288});
            TestFloatMetric(metrics.PkMid().data().A, {190, 190, 190});
            TestFloatMetric(metrics.PkMid().data().C, {410, 410, 410});
            TestFloatMetric(metrics.PkMid().data().G, {210, 210, 210});
            TestFloatMetric(metrics.PkMid().data().T, {390, 390, 390});
            TestFloatMetric(metrics.PkMax().data().A, {200, 200, 200});
            TestFloatMetric(metrics.PkMax().data().C, {420, 420, 420});
            TestFloatMetric(metrics.PkMax().data().G, {220, 220, 220});
            TestFloatMetric(metrics.PkMax().data().T, {400, 400, 400});
            TestFloatMetric(metrics.pkzvar().data().A, {8, 8, 8}),
            TestFloatMetric(metrics.pkzvar().data().C, {9, 9, 9}),
            TestFloatMetric(metrics.pkzvar().data().G, {10, 10, 10}),
            TestFloatMetric(metrics.pkzvar().data().T, {11, 11, 11}),
            TestFloatMetric(metrics.bpzvar().data().A, {0.08, 0.08, 0.08}),
            TestFloatMetric(metrics.bpzvar().data().C, {0.09, 0.09, 0.09}),
            TestFloatMetric(metrics.bpzvar().data().G, {0.10, 0.10, 0.10}),
            TestFloatMetric(metrics.bpzvar().data().T, {0.11, 0.11, 0.11}),
            TestFloatMetric(metrics.BaselineSD().data().green, {25, 25, 25});
            TestFloatMetric(metrics.BaselineSD().data().red, {25, 25, 25});
            TestFloatMetric(metrics.PulseDetectionScore().data(), {-7, -7, -7});
            TestFloatMetric(metrics.TraceAutocorr().data(), {.8, .8, .8});
            // (mdsmith) I've confirmed these manually:
            TestFloatMetric(metrics.ChannelMinSNR().data().red, {8.34, 8.34, 8.34});
            TestFloatMetric(metrics.ChannelMinSNR().data().green, {8.34, 8.34, 8.34});
            TestFloatMetric(metrics.BlockLowSNR().data(), {8.34, 8.34, 8.34});
        }
    }
}
