#include <gtest/gtest.h>

#include <assert.h>
#include <ctime>
#include <chrono>
#include <iostream>
#include <memory>
#include <vector>
#include <thread>
#include <chrono>
#include <random>
#include <sstream>
#include <string>

#include "ReadSimulator.h"

using namespace PacBio::Primary;
using namespace PacBio::Primary::Postprimary;

namespace
{

ReadTypeClass ReadType(const RegionLabel& hqRegion,
                       const BlockLevelMetrics& metrics,
                       uint32_t minEmptyTime0,
                       uint32_t emptyOutlierTime0)
{
    ProductivityMetrics prodClass(4, minEmptyTime0, emptyOutlierTime0);
    auto prod = prodClass.ComputeProductivityInfo(hqRegion, metrics, true);
    return prod.readType;
}
ReadTypeClass ReadType(const ReadConfig& config,
                       uint32_t minEmptyTime0,
                       uint32_t emptyOutlierTime0)
{
    const auto& hqRegion = config.GenerateHQRegion();
    const auto& metrics = SimulateMetrics(config);
    return ReadType(hqRegion, metrics, minEmptyTime0, emptyOutlierTime0);
}

}

TEST(zmwMetrics, IsSequencingInRegion)
{
    auto makeBaseRate = [](const std::vector<float>& A,
            const std::vector<float>& C,
            const std::vector<float>& G,
            const std::vector<float>& T)
    {
        AnalogMetricData<std::vector<float>> data;
        data.A = A;
        data.C = C;
        data.G = G;
        data.T = T;
        return AnalogMetric<float>(std::move(data), MetricFrequency::MEDIUM, 80.0f, 4096);
    };
    // each channel has one high pulse rate in the same block
    std::vector<float> A {1, 10, 40, 0};
    auto prba = makeBaseRate(A, A, A, A);
    bool res = ProductivityMetrics::IsSequencingInRegion(prba.GetRegion(0, 4));
    EXPECT_EQ(res, true);

    // no channel has a high pulse rate
    std::vector<float> B {1, 10, 0, 0};
    prba = makeBaseRate(B, B, B, B);
    res = ProductivityMetrics::IsSequencingInRegion(prba.GetRegion(0, 4));
    EXPECT_EQ(res, false);

    // one channel has a high pulse rate
    prba = makeBaseRate(A, B, B, B);
    res = ProductivityMetrics::IsSequencingInRegion(prba.GetRegion(0, 4));
    EXPECT_EQ(res, false);

    // one channel has a low pulse rate
    prba = makeBaseRate(A, A, A, B);
    res = ProductivityMetrics::IsSequencingInRegion(prba.GetRegion(0, 4));
    EXPECT_EQ(res, false);

    // each channel has one high pulse rate, but in different blocks
    std::vector<float> C {1, 10, 0, 40};
    prba = makeBaseRate(A, A, A, C);
    res = ProductivityMetrics::IsSequencingInRegion(prba.GetRegion(0, 4));
    EXPECT_EQ(res, false);
}

TEST(zmwMetrics, SimReadIndeterminateNoHQ)
{
    ReadConfig config;
    config.hqstart = 0;
    config.hqend = 0;

    auto type = ReadType(config, minEmptyTime, emptyOutlierTime);
    EXPECT_EQ(type, ReadTypeClass::INDETERMINATE);
}

TEST(zmwMetrics, SimReadIndeterminateHQremoved)
{
    ReadConfig rc;
    rc.excludePulses = false;
    const auto& hqRegion = GenerateEmpyHQ();
    const auto& metrics = SimulateMetrics(rc);
    
    auto type = ReadType(hqRegion, metrics, minEmptyTime, emptyOutlierTime);
    EXPECT_EQ(type, ReadTypeClass::INDETERMINATE);
}

TEST(zmwMetrics, SimReadIndeterminateShortHQ)
{
    // Need to modify and pass around the read config object.
    ReadConfig config;
    config.hqstart = 100;
    config.hqend = 111;

    auto type = ReadType(config, minEmptyTime, emptyOutlierTime);
    EXPECT_EQ(type, ReadTypeClass::INDETERMINATE);
}

TEST(zmwMetrics, SimReadFullHQRead0NoPost)
{
    // Need to modify and pass around the read config object.
    ReadConfig config;
    config.numBases = 1000;
    config.hqstart = 0;
    config.hqend = 1000;
    config.seqstart = 0;
    config.seqend = 1000;

    const auto& hqRegion = config.GenerateHQRegion();
    const auto& metrics = SimulateMetrics(config);
    const auto& metricRegion = metrics.GetMetricRegion(hqRegion);
    const auto& baseRates = metrics.BasesVsTime().GetRegionAfter(metricRegion);

    EXPECT_FALSE(baseRates.empty());
    EXPECT_FALSE(ProductivityMetrics::IsSequencingInRegion(baseRates));

    auto type = ReadType(hqRegion, metrics, minEmptyTime, 0);
    EXPECT_EQ(type, ReadTypeClass::FULLHQREAD0);
}

TEST(zmwMetrics, SimReadFullHQRead0LastBlock)
{
    // Need to modify and pass around the read config object.
    ReadConfig config;
    config.hqstart = 0;
    config.hqend = config.numBases;
    config.seqstart = 0;
    config.seqend = config.numFrames;

    const auto& hqRegion = config.GenerateHQRegion();
    const auto& metrics = SimulateMetrics(config);
    const auto& metricRegion = metrics.GetMetricRegion(hqRegion);
    const auto& baseRates = metrics.BasesVsTime().GetRegionAfter(metricRegion);

    // Make sure we are HQregion to the last block:
    EXPECT_TRUE(baseRates.empty());
    EXPECT_FALSE(ProductivityMetrics::IsSequencingInRegion(baseRates));

    auto type = ReadType(hqRegion, metrics, minEmptyTime, emptyOutlierTime);
    EXPECT_EQ(type, ReadTypeClass::FULLHQREAD0);
}

TEST(zmwMetrics, SimReadPartialHQRead1SeqPost)
{
    // Need to modify and pass around the read config object.
    ReadConfig config;
    config.hqstart = 0;
    config.hqend = config.numBases/2;
    config.seqstart = 0;
    config.seqend = config.numFrames;
    
    const auto& hqRegion = config.GenerateHQRegion();
    const auto& metrics = SimulateMetrics(config);
    const auto& metricRegion = metrics.GetMetricRegion(hqRegion);
    const auto& baseRates = metrics.BasesVsTime();

    const auto& before = baseRates.GetRegionBefore(metricRegion);
    EXPECT_TRUE(before.empty());
    EXPECT_FALSE(ProductivityMetrics::IsSequencingInRegion(before));

    const auto& after = baseRates.GetRegionAfter(metricRegion);
    EXPECT_FALSE(after.empty());
    EXPECT_TRUE(ProductivityMetrics::IsSequencingInRegion(after));

    EXPECT_TRUE(ProductivityMetrics::IsSequencingInRegion(baseRates.GetRegion(metricRegion)));

    auto type = ReadType(hqRegion, metrics, minEmptyTime, emptyOutlierTime);
    EXPECT_EQ(type, ReadTypeClass::PARTIALHQREAD1);
}

TEST(zmwMetrics, SimReadPartialHQRead0SeqBefore)
{
    // Need to modify and pass around the read config object.
    ReadConfig config;
    config.numBases = 131072;
    config.numFrames = 262144;
    config.hqstart = config.numBases/2;
    config.hqend = config.numBases;
    config.seqstart = 0;
    config.seqend = config.numFrames;

    const auto& metrics = SimulateMetrics(config);
    const auto& hqRegion = config.GenerateHQRegion();
    const auto& metricRegion = metrics.GetMetricRegion(hqRegion);
    const auto& baseRates = metrics.BasesVsTime();

    const auto& after = baseRates.GetRegionAfter(metricRegion);
    EXPECT_TRUE(after.empty());
    EXPECT_FALSE(ProductivityMetrics::IsSequencingInRegion(after));

    const auto& before = baseRates.GetRegionBefore(metricRegion);
    EXPECT_EQ(before.size(), 32);
    EXPECT_TRUE(ProductivityMetrics::IsSequencingInRegion(before));

    EXPECT_TRUE(ProductivityMetrics::IsSequencingInRegion(baseRates.GetRegion(metricRegion)));

    auto type = ReadType(hqRegion, metrics, minEmptyTime, emptyOutlierTime);
    EXPECT_EQ(type, ReadTypeClass::PARTIALHQREAD0);
}

TEST(zmwMetrics, SimReadFullHQRead1LateStartEarlyStopNoOtherSeq)
{
    // Need to modify and pass around the read config object.
    ReadConfig config;
    config.numBases = 131072;
    config.numFrames = 262144;
    config.hqstart = 0;
    config.hqend = config.numBases;
    config.seqstart = config.numFrames/4;
    config.seqend = config.seqstart + 10000;

    const auto& hqRegion = config.GenerateHQRegion();
    const auto& metrics = SimulateMetrics(config);
    const auto& metricRegion = metrics.GetMetricRegion(hqRegion);
    const auto& baseRates = metrics.BasesVsTime();

    const auto& after = baseRates.GetRegionAfter(metricRegion);
    EXPECT_EQ(after.size(), 45);
    EXPECT_FALSE(ProductivityMetrics::IsSequencingInRegion(after));

    const auto& before = baseRates.GetRegionBefore(metricRegion);
    EXPECT_EQ(before.size(), 16);
    EXPECT_FALSE(ProductivityMetrics::IsSequencingInRegion(before));

    EXPECT_TRUE(ProductivityMetrics::IsSequencingInRegion(baseRates.GetRegion(metricRegion)));

    auto type = ReadType(hqRegion, metrics, 0, 0);
    EXPECT_EQ(type, ReadTypeClass::FULLHQREAD1);

    type = ReadType(hqRegion, metrics, minEmptyTime, 0);
    EXPECT_EQ(type, ReadTypeClass::FULLHQREAD1);
}

TEST(zmwMetrics, SimReadFullHQRead1LateStartFullStopNoOtherSeq)
{
    // Need to modify and pass around the read config object.
    ReadConfig config;
    config.numBases = 131072;
    config.numFrames = 262144;
    config.hqstart = 0;
    config.hqend = config.numBases;
    config.seqstart = config.numFrames/4;
    config.seqend = config.numFrames;

    const auto& hqRegion = config.GenerateHQRegion();
    const auto& metrics = SimulateMetrics(config);
    const auto& metricRegion = metrics.GetMetricRegion(hqRegion);
    const auto& baseRates = metrics.BasesVsTime();

    const auto& after = baseRates.GetRegionAfter(metricRegion);
    EXPECT_EQ(after.size(), 0);
    EXPECT_FALSE(ProductivityMetrics::IsSequencingInRegion(after));

    const auto& before = baseRates.GetRegionBefore(metricRegion);
    EXPECT_EQ(before.size(), 16);
    EXPECT_FALSE(ProductivityMetrics::IsSequencingInRegion(before));

    EXPECT_TRUE(ProductivityMetrics::IsSequencingInRegion(baseRates.GetRegion(metricRegion)));

    auto type = ReadType(hqRegion, metrics, 0, emptyOutlierTime);
    EXPECT_EQ(type, ReadTypeClass::FULLHQREAD1);

    type = ReadType(hqRegion, metrics, minEmptyTime, emptyOutlierTime);
    EXPECT_EQ(type, ReadTypeClass::FULLHQREAD1);
}

TEST(zmwMetrics, SimReadFullHQRead1LateStartFullStopNoOtherSeq_minEmptyTimeTest)
{
    // Need to modify and pass around the read config object.
    ReadConfig config;
    config.numBases = 131072;
    config.numFrames = 262144;
    config.hqstart = 0;
    config.hqend = config.numBases;
    // sequencing starts at block 60, ~60 minutes:
    config.seqstart = 4096 * 60;
    config.seqend = config.numFrames;

    const auto& hqRegion = config.GenerateHQRegion();
    const auto& metrics = SimulateMetrics(config);
    const auto& metricRegion = metrics.GetMetricRegion(hqRegion);
    const auto& baseRates = metrics.BasesVsTime();

    const auto& after = baseRates.GetRegionAfter(metricRegion);
    EXPECT_EQ(after.size(), 0);
    EXPECT_FALSE(ProductivityMetrics::IsSequencingInRegion(after));

    const auto& before = baseRates.GetRegionBefore(metricRegion);
    EXPECT_EQ(before.size(), 60);
    EXPECT_FALSE(ProductivityMetrics::IsSequencingInRegion(before));

    EXPECT_TRUE(ProductivityMetrics::IsSequencingInRegion(baseRates.GetRegion(metricRegion)));

    auto type = ReadType(hqRegion, metrics, 0, 0);
    EXPECT_EQ(type, ReadTypeClass::FULLHQREAD1);

    type = ReadType(hqRegion, metrics, 26, emptyOutlierTime);
    EXPECT_EQ(type, ReadTypeClass::EMPTY);
    type = ReadType(hqRegion, metrics, 51, emptyOutlierTime);
    EXPECT_EQ(type, ReadTypeClass::EMPTY);
    // so ~51 minutes translates to 60 blocks, for ~51 seconds per block
    type = ReadType(hqRegion, metrics, 52, 0);
    EXPECT_EQ(type, ReadTypeClass::FULLHQREAD1);
    type = ReadType(hqRegion, metrics, 3500, 0);
    EXPECT_EQ(type, ReadTypeClass::FULLHQREAD1);
    // we can go way over without issue:
    type = ReadType(hqRegion, metrics, 100000, 0);
    EXPECT_EQ(type, ReadTypeClass::FULLHQREAD1);
}

TEST(zmwMetrics, SimReadPartialHQRead2LateStartSeqAfter)
{
    // Not sure this is really possible...
    // Need to modify and pass around the read config object.
    ReadConfig config;
    config.numBases = 131072;
    config.numFrames = 262144;
    config.hqstart = 0;
    config.hqend = config.numBases/2;
    config.seqstart = config.numFrames/4;
    config.seqend = config.numFrames;

    const auto& hqRegion = config.GenerateHQRegion();
    const auto& metrics = SimulateMetrics(config);
    const auto& metricRegion = metrics.GetMetricRegion(hqRegion);
    const auto& baseRates = metrics.BasesVsTime();

    const auto& after = baseRates.GetRegionAfter(metricRegion);
    EXPECT_EQ(after.size(), 24);
    EXPECT_TRUE(ProductivityMetrics::IsSequencingInRegion(after));

    const auto& before = baseRates.GetRegionBefore(metricRegion);
    EXPECT_EQ(before.size(), 16);
    EXPECT_FALSE(ProductivityMetrics::IsSequencingInRegion(before));

    EXPECT_TRUE(ProductivityMetrics::IsSequencingInRegion(baseRates.GetRegion(metricRegion)));

    auto type = ReadType(hqRegion, metrics, 0, emptyOutlierTime);
    EXPECT_EQ(type, ReadTypeClass::PARTIALHQREAD2);

    type = ReadType(hqRegion, metrics, minEmptyTime, emptyOutlierTime);
    EXPECT_EQ(type, ReadTypeClass::PARTIALHQREAD2);
}

TEST(zmwMetrics, SimReadEmpty)
{
    // Simulate a read with no bases
    ReadConfig config;
    config.numBases = 0;
    config.hqstart = 0;
    config.hqend = 0;

    auto type = ReadType(config, minEmptyTime, emptyOutlierTime);
    EXPECT_EQ(type, ReadTypeClass::EMPTY);
}
