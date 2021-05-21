#include <assert.h>
#include <chrono>
#include <ctime>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include <thread>

#include <gtest/gtest.h>

#include <postprimary/stats/ZmwMetrics.h>

#include "ReadSimulator.h"

using namespace PacBio::Primary;
using namespace PacBio::Primary::Postprimary;


//TODO:
// H5: All of VsT
// Expand pretty much everything beyond the current base case

// H5: NumBases, ReadLength, InsertReadLength
TEST(zmwMetrics, ReadLength)
{
    // Using the default in the call, regen the default to get some numbers:
    ReadConfig config;
    config.excludePulses = false;
    auto hqlen = config.hqend - config.hqstart;

    // No adapter regions specified:
    SubreadMetrics insertStats = SubreadMetrics(GenerateEmpyHQ(),
                                                std::vector<RegionLabel>{});
    // By default: no adapters, therefore no insert.
    // InsertReadLength:
    EXPECT_EQ(0, insertStats.MaxSubreadLength());
    // MedianInsertLength:
    EXPECT_EQ(0, insertStats.MedianLength());

    const auto& events = SimulateEventData(config);
    const auto& metrics = SimulateMetrics(config);
    const auto& hqRegion = config.GenerateHQRegion();

    auto res = fillstats(events, metrics, hqRegion, config);
    PacBio::Primary::ZmwStats zmw = std::get<0>(res);
    std::unique_ptr<FileHeader> fh = std::move(std::get<1>(res));

    EXPECT_EQ(hqlen, zmw.ReadLength);

    // The official way
    // NumBases:
    EXPECT_EQ(config.numBases, zmw.NumBases);
    EXPECT_EQ(hqlen, zmw.ReadLength);
    // NumPulses
    EXPECT_EQ(config.numBases, zmw.NumPulses);

    // InsertReadLength:
    EXPECT_EQ(hqlen, zmw.InsertReadLength);
}

// H5: HQPkmid HQPkmidRatio
TEST(zmwMetrics, HQPkmid)
{
    const auto& events = SimulateEventData();
    const auto& metrics = SimulateMetrics();
    const auto& hqRegion = ReadConfig{}.GenerateHQRegion();

    // gcc seems to have an internal compiler error when this lambda isn't wrapping the
    // actual function call
    auto res = [&](){ return fillstats(events, metrics, hqRegion, ReadConfig{}); }();
    PacBio::Primary::ZmwStats zmw = std::get<0>(res);
    std::unique_ptr<FileHeader> fh = std::move(std::get<1>(res));

    std::array<double, 4> exppk = {{10, 20, 30, 40}};

    for (size_t i = 0; i < exppk.size(); i++)
    {
        EXPECT_NEAR(zmw.HQPkmid[i], exppk[i], 0.01);
    }
}

// H5: HQRegionEnd HQRegionEndTime HQRegionStart HQRegionStartTime
TEST(zmwMetrics, HQRegionStartEnd)
{
    ReadConfig config;
    config.numBases = 131072;
    config.numFrames = 262144;
    config.seqend = 262144;
    config.hqend = config.numBases;

    const auto& events = SimulateEventData(config);
    const auto& metrics = SimulateMetrics(config);
    const auto& hqRegion = config.GenerateHQRegion();

    auto res = fillstats(events, metrics, hqRegion, config);
    PacBio::Primary::ZmwStats zmw = std::get<0>(res);
    std::unique_ptr<FileHeader> fh = std::move(std::get<1>(res));

    EXPECT_EQ(0, zmw.HQRegionStart);
    EXPECT_EQ(131072, zmw.HQRegionEnd);
    EXPECT_EQ(0, zmw.HQRegionStartTime);
    EXPECT_EQ(3276, zmw.HQRegionEndTime);
}

// H5: HQRegionSnrMean, SnrMean
TEST(zmwMetrics, HQRegionSnrMean)
{
    const auto& events = SimulateEventData();
    const auto& metrics = SimulateMetrics();
    const auto& hqRegion = ReadConfig{}.GenerateHQRegion();

    // gcc seems to have an internal compiler error when this lambda isn't wrapping the
    // actual function call
    auto res = [&](){ return fillstats(events, metrics, hqRegion, ReadConfig{}); }();
    PacBio::Primary::ZmwStats zmw = std::get<0>(res);
    std::unique_ptr<FileHeader> fh = std::move(std::get<1>(res));

    auto snrs = zmw.HQRegionSnrMean;
    std::array<double,4> expsnrs = {{1/3.0f, 2/3.0f, 3/4.0f, 1}};
    for (int i = 0; i < 4; i++ )
    {
        EXPECT_NEAR(expsnrs[i], snrs[i], 0.01);
    }
}

// H5: Productivity
TEST(zmwMetrics, productivityReadType)
{
    uint32_t minEmptyTime0{60};

    const auto& metrics = SimulateMetrics();
    const auto& hqRegion = ReadConfig{}.GenerateHQRegion();

    // SNR cut (first arg) doesn't matter for this test
    ProductivityMetrics prodClass(4, minEmptyTime0, emptyOutlierTime);
    auto prod = prodClass.ComputeProductivityInfo(hqRegion, metrics, true);

    EXPECT_EQ(ProductivityClass::OTHER, prod.productivity);

    EXPECT_EQ(ReadTypeClass::FULLHQREAD0, prod.readType);
}
