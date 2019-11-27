//
// Created by jnguyen on 9/30/15.
//

#include <pacbio/primary/BasecallingMetrics.h>
#include <pacbio/primary/ZmwResultBuffer.h>
#include "gtest/gtest.h"

using namespace std;
using namespace PacBio::Primary;
using namespace PacBio::SmrtData;

TEST(TraceAnalysisMetrics, Memcpy)
{
    std::array<DetectionMode<float,2>,5> detModes =
        {{
            DetectionMode<float,2>({{ 200, 140 }}, {{ 32, 34, 34 }}),
            DetectionMode<float,2>({{ 140, 234 }}, {{ 20, 24, 56 }}),
            DetectionMode<float,2>({{ 300, 101 }}, {{ 20, 34, 23 }}),
            DetectionMode<float,2>({{ 140, 234 }}, {{ 20, 24, 56 }}),
            DetectionMode<float,2>({{ 140, 234 }}, {{ 20, 24, 56 }})
         }};

    TraceAnalysisMetrics<float,2> tm;
    tm.DetectionModes(detModes);
    tm.StartFrame(0);
    tm.NumFrames(1024);

    uint8_t* data = new uint8_t[sizeof(TraceAnalysisMetrics<float,2>)];
    std::memcpy(data, &tm, sizeof(TraceAnalysisMetrics<float,2>));
    const auto* tPtr = reinterpret_cast<const TraceAnalysisMetrics<float,2>*>(data);
    EXPECT_EQ(tPtr->StartFrame(), 0);
    EXPECT_EQ(tPtr->NumFrames(), 1024);
    auto blm = tPtr->Baseline().SignalMean();
    EXPECT_EQ(blm[0], 200);
    EXPECT_EQ(blm[1], 140);
    auto covr = tPtr->Baseline().SignalCovar();
    EXPECT_EQ(covr[1], 34);

    delete data;

    TraceAnalysisMetrics<float,1> tm2;
    tm2.FrameBaselineDWS() = {{200}};
    tm2.FrameBaselineVarianceDWS() = {{20}};

    EXPECT_EQ(200, tm2.FrameBaselineDWS()[0]);
    EXPECT_EQ(20, tm2.FrameBaselineVarianceDWS()[0]);
}

TEST(BasecallingMetrics, SNR)
{
    std::array<DetectionMode<float,2>,5> detModes =
        {{
             DetectionMode<float,2>({{ 200, 140 }}, {{ 16, 25, 36 }}),
             DetectionMode<float,2>({{ 140, 234 }}, {{ 36, 24, 56 }}),
             DetectionMode<float,2>({{ 300, 101 }}, {{ 36, 34, 23 }}),
             DetectionMode<float,2>({{ 140, 234 }}, {{ 36, 24, 56 }}),
             DetectionMode<float,2>({{ 140, 234 }}, {{ 36, 24, 56 }})
         }};

    BasecallingMetrics<float,2> metric;
    auto& tm = metric.TraceMetrics();
    tm.DetectionModes(detModes);
    tm.StartFrame(0).NumFrames(1024);
    tm.FrameBaselineVarianceDWS() = {{25, 16}};

    Pulse p1;
    p1.Start(0).Width(3);
    p1.MidSignal(100);
    p1.SignalM2(12000);
    p1.Label(NucleotideLabel::A);
    Basecall b1(p1);
    b1.Base(NucleotideLabel::A);

    PulseLookback<2> pl;
    uint32_t swT = 0;
    metric.AddBasecall(b1, pl, swT);

    EXPECT_NEAR(100, metric.PkmidSignal()[0], 1e-4);
    EXPECT_NEAR(100, metric.PkmidMean()[0], 1e-4);
    EXPECT_NEAR(25,  metric.FrameSnr()[0], 1e-4);
    // interpulse M2 is accumulated
    EXPECT_NEAR(10000, metric.Bpzvar()[0], 1e-4);
    // total M2 is accumulated
    EXPECT_NEAR(12000, metric.Pkzvar()[0], 1e-4);

    Pulse p2;
    p2.Start(0).Width(5);
    p2.MidSignal(100);
    p2.SignalM2(42000);
    p2.Label(NucleotideLabel::A);
    Basecall b2(p2);
    b2.Base(NucleotideLabel::A);

    metric.AddBasecall(b2, pl, swT);

    EXPECT_NEAR(400, metric.PkmidSignal()[0], 1e-4);
    EXPECT_NEAR(100, metric.PkmidMean()[0], 1e-4);
    EXPECT_NEAR(25, metric.FrameSnr()[0], 1e-4);
    EXPECT_NEAR(40000, metric.Bpzvar()[0], 1e-4);
    EXPECT_NEAR(54000, metric.Pkzvar()[0], 1e-4);

    Pulse p3;
    p3.Start(0).Width(10);
    p3.MidSignal(100);
    p3.SignalM2(90000);
    p3.Label(NucleotideLabel::T);
    Basecall b3(p3);
    b3.Base(NucleotideLabel::T);

    metric.AddBasecall(b3, pl, swT);

    EXPECT_NEAR(80000, metric.Bpzvar()[3], 1e-4);
    EXPECT_NEAR(90000, metric.Pkzvar()[3], 1e-4);

    Pulse p4;
    p4.Start(0).Width(3);
    p4.MidSignal(50);
    p4.SignalM2(5000);
    p4.Label(NucleotideLabel::T);
    Basecall b4(p4);
    b4.Base(NucleotideLabel::T);

    metric.AddBasecall(b4, pl, swT);

    EXPECT_NEAR(800 + 50, metric.PkmidSignal()[3], 1e-4);
    EXPECT_NEAR(850/static_cast<float>(9), metric.PkmidMean()[3], 1e-4);
    EXPECT_NEAR((850/static_cast<float>(9))/5, metric.FrameSnr()[3], 1e-4);
    EXPECT_NEAR(82500, metric.Bpzvar()[3], 1e-2);
    EXPECT_NEAR(95000, metric.Pkzvar()[3], 1e-2);

    Pulse p5;
    p5.Start(0).Width(10);
    p5.MidSignal(60);
    p5.SignalM2(90000);
    p5.Label(NucleotideLabel::T);
    Basecall b5(p5);
    b5.Base(NucleotideLabel::T);

    metric.AddBasecall(b5, pl, swT);

    EXPECT_NEAR(111300, metric.Bpzvar()[3], 1e-2);
    EXPECT_NEAR(185000, metric.Pkzvar()[3], 1e-2);

    metric.FinalizeVariance();

    EXPECT_NEAR(0.06941, metric.Bpzvar()[3], 1e-2);
    EXPECT_NEAR(419.3540, metric.Pkzvar()[3], 1e-2);
}

TEST(BasecallingMetrics, PulseAndBaseWidth)
{
    BasecallingMetrics<float,2> metric;
    auto& tm = metric.TraceMetrics();
    tm.StartFrame(0).NumFrames(1024);

    Pulse p1;
    p1.Start(0).Width(3);
    p1.MidSignal(100);
    p1.Label(NucleotideLabel::A);
    Basecall b1(p1);
    b1.Base(NucleotideLabel::A);

    PulseLookback<2> pl;
    uint32_t swT = 0;
    metric.AddBasecall(b1, pl, swT);

    // Add pulse and base.
    EXPECT_EQ(3, metric.PulseWidth());
    EXPECT_EQ(3, metric.BaseWidth());

    Pulse p2;
    p2.Start(0).Width(6);
    p2.MidSignal(100);
    p2.Label(NucleotideLabel::A);
    Basecall b2(p2);
    b2.Base(NucleotideLabel::NONE);

    // Add pulse but no base.
    metric.AddBasecall(b2, pl, swT);

    EXPECT_EQ(4.5, metric.PulseWidth());
    EXPECT_EQ(3, metric.BaseWidth());
}


TEST(BasecallingMetrics, Memcpy)
{
    std::array<DetectionMode<float,2>,5> detModes =
    {{
        DetectionMode<float,2>({{ 200, 140 }}, {{ 32, 34, 34 }}),
        DetectionMode<float,2>({{ 140, 234 }}, {{ 20, 24, 56 }}),
        DetectionMode<float,2>({{ 300, 101 }}, {{ 20, 34, 23 }}),
        DetectionMode<float,2>({{ 140, 234 }}, {{ 20, 24, 56 }}),
        DetectionMode<float,2>({{ 140, 234 }}, {{ 20, 24, 56 }})
    }};

    BasecallingMetrics<float,2> metric;
    auto& tm = metric.TraceMetrics();
    tm.DetectionModes(detModes);
    tm.StartFrame(0).NumFrames(1024);
    tm.FrameBaselineDWS() = {{140, 200}};
    tm.FrameBaselineVarianceDWS() = {{34*34,32*32}};

    metric.NumPulseFrames(20);
    metric.NumPulses(20);
    metric.NumBaseFrames(20);
    metric.NumBases(20);
    metric.PkmidSignal() = {{ 200.0f, 150.0f, 300.0f, 210.0f }};
    metric.Bpzvar() = {{ 30.0f, 32.0f, 24.0f, 45.0f }};
    metric.PkmidNumFrames() = {{ 4, 4, 4, 4 }};
    metric.NumPkmidBasesByAnalog() = {{ 5, 5, 5, 5 }};

    uint8_t* data = new uint8_t[sizeof(BasecallingMetrics<float,2>)*16];
    for (int i = 0; i < 16; i++)
        std::memcpy(data + (i*sizeof(BasecallingMetrics<float,2>)), &metric, sizeof(BasecallingMetrics<float,2>));

    const auto* mPtr = reinterpret_cast<const BasecallingMetrics<float,2>*>(data);

    for (int i = 0; i < 16; i++)
    {
        EXPECT_EQ(mPtr[i].NumPulseFrames(), 20);
        EXPECT_EQ(mPtr[i].NumPulses(), 20);
        EXPECT_EQ(mPtr[i].NumBaseFrames(), 20);
        EXPECT_EQ(mPtr[i].NumBases(), 20);
        EXPECT_EQ(mPtr[i].PkmidNumFrames()[0], 4);
        EXPECT_EQ(mPtr[i].NumPkmidBasesByAnalog()[3], 5);
        EXPECT_EQ(mPtr[i].PkmidSignal()[1], 150.0f);
        EXPECT_EQ(mPtr[i].TraceMetrics().FrameBaselineDWS()[0], 140);
        EXPECT_EQ(mPtr[i].TraceMetrics().FrameBaselineSigmaDWS()[1], 32);

    }

    delete data;

    BasecallingMetrics<float,2> m1;
    BasecallingMetrics<float,2>& m1_ref(m1);
    m1_ref = metric;

    EXPECT_EQ(m1_ref.NumBases(), metric.NumBases());
    EXPECT_EQ(m1_ref.NumPulses(), metric.NumPulses());
    EXPECT_EQ(m1_ref.NumBaseFrames(), metric.NumBaseFrames());
    EXPECT_EQ(m1_ref.NumPulseFrames(), metric.NumPulseFrames());
    EXPECT_EQ(m1_ref.PkmidSignal()[1], metric.PkmidSignal()[1]);
    EXPECT_EQ(m1_ref.TraceMetrics().FrameBaselineDWS()[0], metric.TraceMetrics().FrameBaselineDWS()[0]);
    EXPECT_EQ(m1_ref.NumPkmidBasesByAnalog()[3], metric.NumPkmidBasesByAnalog()[3]);
}
