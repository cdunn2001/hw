#include "gtest/gtest.h"

#include <pacbio/smrtdata/Basecall.h>
#include <pacbio/smrtdata/NucleotideLabel.h>
#include <pacbio/smrtdata/Pulse.h>
#include <pacbio/ipc/PoolMalloc.h>
#include <pacbio/primary/ZmwResultBuffer.h>

using namespace PacBio::Primary;
using namespace PacBio::SmrtData;

TEST(ReadBuffer, Mem)
{
    PrimaryConfig pc(0.0);
    pc.chipClass = ChipClass::Sequel;
    pc.platform = Platform::Sequel1PAC1;
    ReadBuffer::Configure(pc);
    size_t rbBytes = ReadBuffer::SizeOf();
    ASSERT_EQ(36864, rbBytes);
    ASSERT_EQ(0, rbBytes % 4096);
    PacBio::IPC::PoolMalloc<ReadBuffer>      readBufferPool(10, rbBytes);
    readBufferPool.Create();

    ASSERT_LT(0, ReadBuffer::MaxNumSamplesPerBuffer());


    for (size_t i = 0; i < 10; i++)
    {
        ReadBuffer* r = ReadBuffer::Initialize::FromMemoryLocation(readBufferPool.GetPointerByIndex(i));
        r->Reset(i);
    }

    for (size_t i = 0; i < 10; i++)
    {
        ReadBuffer* r = readBufferPool.GetPointerByIndex(i);
        ASSERT_NE(nullptr, r->Metrics<BasecallingMetricsT::Sequel>());
        ASSERT_NE(nullptr, r->Samples());
        ASSERT_EQ(0, r->NumSamples());
        ASSERT_EQ(i, r->ZmwIndex());
        ASSERT_EQ(2960, r->Size());
    }

    ReadBuffer* r = readBufferPool.GetPointerByIndex(0);

    Pulse p {};
    p.Start(1234).Width(12);
    p.Label(NucleotideLabel::T).LabelQV(42);
    p.AltLabel(NucleotideLabel::G).AltLabelQV(3);
    p.MeanSignal(42.0).MidSignal(47.47).MaxSignal(50.5);
    p.MergeQV(5);
    Basecall bIn {p};
    bIn.Base(NucleotideLabel::A);
    bIn.InsertionQV(8).SubstitutionQV(20).DeletionQV(120);
    bIn.SubstitutionTag(NucleotideLabel::C).DeletionTag(NucleotideLabel::T);

    const Basecall* bOut = r->Samples();

    ASSERT_EQ(1, r->BackInsertFeatures(&bIn, 1));
    ASSERT_EQ(1, r->NumSamples());
    ASSERT_EQ(NucleotideLabel::A, bOut[0].Base());

    r->BackInsertFeatures(&bIn, 1);
    ASSERT_EQ(2, r->NumSamples());
    ASSERT_EQ(8, bOut[1].InsertionQV());

    r->BackInsertFeatures(&bIn, 1);
    ASSERT_EQ(3, r->NumSamples());
    ASSERT_EQ(120, bOut[2].DeletionQV());

    Basecall bInArr[4];
    NucleotideLabel nLabels[4] = { NucleotideLabel::A, NucleotideLabel::C, NucleotideLabel::G, NucleotideLabel::T };
    for (size_t i = 0; i < 4; i++)
    {
        bInArr[i].Base(nLabels[i]);
        bInArr[i].InsertionQV(i*1).SubstitutionQV(i*2).DeletionQV(i*3);
    }
    ASSERT_EQ(4, r->BackInsertFeatures(bInArr, 4));
    for (size_t i = 0; i < 4; i++)
    {
        ASSERT_EQ(nLabels[i], bOut[3+i].Base());
        ASSERT_EQ(i*1, bOut[3+i].InsertionQV());
        ASSERT_EQ(i*2, bOut[3+i].SubstitutionQV());
        ASSERT_EQ(i*3, bOut[3+i].DeletionQV());
    }
    ASSERT_EQ(7, r->NumSamples());

    ReadBuffer* r2 = ReadBuffer::Initialize::FromMemoryLocation(readBufferPool.GetPointerByIndex(1));
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
    metric.NumPulseFrames(20);
    metric.NumPulses(20);
    metric.NumBaseFrames(20);
    metric.NumBases(20);
    metric.PkmidSignal() = {{ 200.0f, 150.0f, 300.0f, 210.0f }};
    metric.Bpzvar() = {{ 30.0f, 32.0f, 24.0f, 45.0f }};
    metric.PkmidNumFrames() = {{ 4, 4, 4, 4 }};
    metric.NumPkmidBasesByAnalog() = {{ 5, 5, 5, 5 }};

    ASSERT_EQ(1, r2->BackInsertMetrics(&metric, 1));
    ASSERT_EQ(1, r2->NumMetrics());

    const auto* metrics = r2->Metrics<BasecallingMetrics<float,2>>();
    ASSERT_EQ(20, metrics[0].NumPulseFrames());

    BasecallingMetrics<float,2> metricArr[3];
    for (size_t i = 0; i < 3; i++)
    {
        metricArr[i].NumPulses(i*30);
        metricArr[i].NumBases(i*20);
        metricArr[i].PkmidSignal() = {{ i*100.0f, i*200.0f, i*300.0f, i*400.0f }};
        ASSERT_EQ(1, r2->BackInsertMetrics(metricArr+i, 1));
    }

    ASSERT_EQ(4, r2->NumMetrics());
    for (size_t i = 0; i < 3; i++)
    {
        ASSERT_EQ(i*30, metrics[1+i].NumPulses());
        ASSERT_EQ(i*20, metrics[1+i].NumBases());
        ASSERT_EQ(i*100.0f, metrics[1+i].PkmidSignal()[0]);
        ASSERT_EQ(i*200.0f, metrics[1+i].PkmidSignal()[1]);
        ASSERT_EQ(i*300.0f, metrics[1+i].PkmidSignal()[2]);
        ASSERT_EQ(i*400.0f, metrics[1+i].PkmidSignal()[3]);
    }
}

