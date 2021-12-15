#include <gtest/gtest.h>

#include <iostream>
#include <memory>
#include <vector>
#include <thread>
#include <chrono>
#include <random>
#include <string>
#include <assert.h>
#include <ctime>
#include <chrono>
#include <sstream>

#include <postprimary/stats/ZmwMetrics.h>

#include "ReadSimulator.h"

using namespace PacBio::Primary;
using namespace PacBio::Primary::Postprimary;


TEST(zmwMetrics, SignalMetrics)
{
    const auto& bazMetrics = SimulateMetrics();
    const auto& signalMetrics = SignalMetrics(bazMetrics.GetFullRegion(), bazMetrics);

    EXPECT_EQ(20, signalMetrics.Baseline().green);
    EXPECT_EQ(20, signalMetrics.Baseline().red);

    EXPECT_EQ(40, signalMetrics.BaselineSD().green);
    EXPECT_EQ(40, signalMetrics.BaselineSD().red);

    EXPECT_EQ(10, signalMetrics.PkMid().A);
    EXPECT_EQ(20, signalMetrics.PkMid().C);
    EXPECT_EQ(30, signalMetrics.PkMid().G);
    EXPECT_EQ(40, signalMetrics.PkMid().T);

    EXPECT_NEAR(0.25, signalMetrics.Snr().A, 0.01);
    EXPECT_NEAR(0.5, signalMetrics.Snr().C, 0.01);
    EXPECT_NEAR(0.75, signalMetrics.Snr().G, 0.01);
    EXPECT_NEAR(1, signalMetrics.Snr().T, 0.01);

    // These two are slightly weird, because the definition of MinSnr does a weighted
    // average scaled by expected relative amplitude, but the simulated snr does not
    // produce data consistent with the expected relative amplitudes.  So the A/C
    // happens to come out about right, but the G/T channel ends up somewhere in
    // the middle of the two values.
    EXPECT_NEAR(0.45, signalMetrics.MinSnr().red, 0.01);
    EXPECT_NEAR(0.45, signalMetrics.MinSnr().green, 0.01);

}

TEST(zmwMetrics, ExcludedPulseMetrics)
{
    std::vector<InsertState> states{
            InsertState::BASE,
            InsertState::BASE,
            InsertState::EX_SHORT_PULSE,
            InsertState::BASE,
            InsertState::BURST_PULSE,
            InsertState::EX_SHORT_PULSE,
            InsertState::BURST_PULSE,
            InsertState::BASE,
            InsertState::EX_SHORT_PULSE,
            InsertState::BASE,
            InsertState::PAUSE_PULSE,
            InsertState::PAUSE_PULSE,
            InsertState::BASE,
            InsertState::BASE,
    };

    const auto& excludedPulseMetrics = ExcludedPulseMetrics(states);

    EXPECT_EQ(excludedPulseMetrics.InsertCounts().base, 3);
    EXPECT_EQ(excludedPulseMetrics.InsertCounts().exShortPulse, 3);
    EXPECT_EQ(excludedPulseMetrics.InsertCounts().burstPulse, 1);
    EXPECT_EQ(excludedPulseMetrics.InsertCounts().pausePulse, 1);

    EXPECT_EQ(excludedPulseMetrics.InsertLengths().base, 7);
    EXPECT_EQ(excludedPulseMetrics.InsertLengths().exShortPulse, 3);
    EXPECT_EQ(excludedPulseMetrics.InsertLengths().burstPulse, 2);
    EXPECT_EQ(excludedPulseMetrics.InsertLengths().pausePulse, 2);
}

TEST(zmwMetrics, PulseMetrics)
{
    ReadConfig config;
    config.numBases = 131072;
    config.numFrames = 262144;
    config.excludePulses = true;
    config.seqend = 262144;
    config.hqend = 100000;
    
    const auto& bazMetrics = SimulateMetrics(config);
    const auto& fh = config.GenerateHeader();

    const auto& hqRegion = config.GenerateHQRegion();
    const auto& pulseMetrics = PulseMetrics(fh.FrameRateHz(), hqRegion, bazMetrics);

    EXPECT_EQ(2048 / (fh.MFMetricFrames()/fh.FrameRateHz()), pulseMetrics.Rate());
    EXPECT_NEAR(400.0f / 2048.0f / fh.FrameRateHz(), pulseMetrics.Width(), 0.1);
    EXPECT_EQ(98304, pulseMetrics.TotalCount());
}

TEST(zmwMetrics, BaseMetrics)
{
    ReadConfig config;
    config.excludePulses = true;
    config.numBases = 131072;
    config.numFrames = 262144;
    config.seqend = 262144;
    config.hqend = 100000;
    
    const auto& bazMetrics = SimulateMetrics(config);
    const auto& events = SimulateEventData(config);
    const auto& fh = config.GenerateHeader();
    const auto& hqRegion = config.GenerateHQRegion();

    const auto& baseMetrics = BaseMetrics(fh.FrameRateHz(), hqRegion, bazMetrics, events);

    double expbr = 2048 / (fh.MFMetricFrames()/fh.FrameRateHz());
    EXPECT_EQ(expbr, baseMetrics.Rate());

    double expbw = 400.0f / 2048.0f / fh.FrameRateHz();
    EXPECT_NEAR(expbw, baseMetrics.Width(), 0.1);

    double expipd = (5 / log(2))/ fh.FrameRateHz();
    EXPECT_FLOAT_EQ(expipd, baseMetrics.Ipd());

    double explbr = (47000 * (expipd + expbw)) - expipd;
    EXPECT_NEAR(47000 / explbr, baseMetrics.LocalRate(), 0.02);

    EXPECT_EQ(0, baseMetrics.Pausiness());
    
    // HQ region only extends for 100000 bases, and turning on pulse exclusion
    // knocks out 10% of the bases, so 100000/(0.9 * 131072) = .8477
    EXPECT_NEAR(0.8477f, baseMetrics.HQRatio(), 0.01);

    // HQ region is defined above as 100000 long, but
    // turning on pulse exclusion knocks out every 4th
    // base, and since the simulator assigns them round
    // robbin, that means they all come from A and T.
    EXPECT_EQ(22222, baseMetrics.Counts().A);
    EXPECT_EQ(27778, baseMetrics.Counts().C);
    EXPECT_EQ(27778, baseMetrics.Counts().G);
    EXPECT_EQ(22222, baseMetrics.Counts().T);
}
TEST(zmwMetrics, ProductivityReadMetrics)
{
    ReadConfig config;
    size_t numMBlocks = 10;
    config.numBases = 512 * numMBlocks;
    config.numFrames = 4096 * numMBlocks;
    config.seqstart = 0;
    config.seqend = config.numFrames;
    config.hqstart = 0;
    config.hqend = config.numBases / 2;

    const auto& fh = config.GenerateHeader();
    const auto& metrics = SimulateMetrics(config);
    const auto& events = SimulateEventData(config);
    const auto& hqRegion = config.GenerateHQRegion();

    ProductivityMetrics prodClass(0.25, minEmptyTime, emptyOutlierTime);
    auto prod = prodClass.ComputeProductivityInfo(hqRegion, metrics, true);

    // Should be multi-load.
    EXPECT_EQ(ProductivityClass::PRODUCTIVE, prod.productivity);
    EXPECT_EQ(ReadTypeClass::PARTIALHQREAD1, prod.readType);

    ReadMetrics readMetrics(fh.MovieTimeInHrs(), fh.ZmwUnitFeatures(events.ZmwIndex()),
                            hqRegion, events, prod);

    EXPECT_EQ(0, readMetrics.UnitFeatures());
    EXPECT_EQ(4194368, readMetrics.HoleNumber());
    EXPECT_EQ(2560, readMetrics.ReadLength());
    EXPECT_EQ(5120, readMetrics.PolyLength());
    EXPECT_EQ(true, readMetrics.Internal());
    EXPECT_EQ(true, readMetrics.IsRead());
}

TEST(zmwMetrics, SubreadMetrics)
{
    // Test all interior inserts.
    RegionLabel hqRegion(0, 3100, 0, RegionLabelType::HQREGION);
    std::vector<RegionLabel> adapters;
    adapters.emplace_back(0, 100, 0, RegionLabelType::ADAPTER);
    adapters.emplace_back(1000, 1100, 0, RegionLabelType::ADAPTER);
    adapters.emplace_back(2000, 2100, 0, RegionLabelType::ADAPTER);
    adapters.emplace_back(3000, 3100, 0, RegionLabelType::ADAPTER);
    auto i1 = SubreadMetrics(hqRegion, adapters);
    EXPECT_EQ(900, i1.MeanLength());
    EXPECT_EQ(900, i1.MaxSubreadLength());
    EXPECT_EQ(900, i1.MedianLength());
    EXPECT_EQ(900, i1.Umy());

    // Test with 1 partial pass.
    adapters.clear();
    hqRegion = RegionLabel(0, 10000, 0, RegionLabelType::HQREGION);
    adapters.emplace_back(0, 110, 0, RegionLabelType::ADAPTER);
    adapters.emplace_back(1000, 1110, 0, RegionLabelType::ADAPTER);
    adapters.emplace_back(2000, 2110, 0, RegionLabelType::ADAPTER);
    adapters.emplace_back(3000, 3110, 0, RegionLabelType::ADAPTER);
    auto i2 = SubreadMetrics(hqRegion, adapters);
    EXPECT_EQ(890, i2.MeanLength());
    EXPECT_EQ(6890, i2.MaxSubreadLength());
    EXPECT_EQ(890, i2.MedianLength());
    EXPECT_EQ(890, i2.Umy());

    // Test with 1 adapter.
    hqRegion = RegionLabel(0, 10100, 0, RegionLabelType::HQREGION);
    adapters.clear();
    adapters.emplace_back(3000, 3100, 0, RegionLabelType::ADAPTER);
    auto i3 = SubreadMetrics(hqRegion, adapters);
    EXPECT_EQ(0, i3.MeanLength());
    EXPECT_EQ(7000, i3.MaxSubreadLength());
    EXPECT_EQ(0, i3.MedianLength());
    EXPECT_EQ(7000, i3.Umy());

    // Test with no adapters.
    hqRegion = RegionLabel(0, 10100, 0, RegionLabelType::HQREGION);
    adapters.clear();
    auto i4 = SubreadMetrics(hqRegion, adapters);
    EXPECT_EQ(0, i4.MeanLength());
    EXPECT_EQ(10100, i4.MaxSubreadLength());
    EXPECT_EQ(0, i4.MedianLength());
    EXPECT_EQ(10100, i4.Umy());
}
