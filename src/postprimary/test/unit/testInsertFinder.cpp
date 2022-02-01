#include <gtest/gtest.h>

#include <assert.h>
#include <iostream>
#include <cmath>
#include <tuple>

#include <postprimary/insertfinder/InsertFinder.h>

#include "ReadSimulator.h"

using namespace PacBio;
using namespace PacBio::Primary;
using namespace PacBio::Primary::Postprimary;


namespace {

auto EmptyRawPackets()
{
    return std::map<BazIO::PacketFieldName, std::vector<uint32_t>>{};
}

}

TEST(InsertFinder, RecomputeIPD)
{
    BurstInsertFinder insertF(50);

    // Case 1: Burst occurs in middle of read. IPDs all set to 1
    constexpr size_t numPulses = 500;
    auto rawPackets = EmptyRawPackets();
    const int ipd = 1;
    size_t frame = ipd;
    for (unsigned int i = 0; i < numPulses; i++)
    {
        rawPackets[BazIO::PacketFieldName::StartFrame].push_back(frame);
        rawPackets[BazIO::PacketFieldName::Label].push_back(1);
        rawPackets[BazIO::PacketFieldName::IsBase].push_back(1);
        if (200 <= i && i < 300)
            rawPackets[BazIO::PacketFieldName::PulseWidth].push_back(1);
        else
            rawPackets[BazIO::PacketFieldName::PulseWidth].push_back(15);
        frame += ipd + rawPackets[BazIO::PacketFieldName::PulseWidth].back();
    }

    auto packets = BazIO::BazEventData(rawPackets, {});
    auto states1 = insertF.ClassifyInserts(packets);

    EXPECT_TRUE(std::all_of(states1.begin(), states1.begin()+200,
                            [](InsertState q) { return q == InsertState::BASE; }));

    EXPECT_TRUE(std::all_of(states1.begin()+200, states1.begin()+300,
                            [](InsertState q) { return q == InsertState::BURST_PULSE; }));

    EXPECT_TRUE(std::all_of(states1.begin()+300, states1.end(),
                            [](InsertState q) { return q == InsertState::BASE; }));

    auto processed = EventData(0, std::move(packets), std::move(states1));
    EXPECT_EQ(201, processed.Ipds()[300]);

    // Case 2: Same setup as 1) but sprinkle in EX_SHORT_PULSES.
    rawPackets = EmptyRawPackets();
    frame = ipd;
    for (unsigned int i = 0; i < numPulses; i++)
    {
        rawPackets[BazIO::PacketFieldName::StartFrame].push_back(frame);
        rawPackets[BazIO::PacketFieldName::Label].push_back(1);
        rawPackets[BazIO::PacketFieldName::IsBase].push_back(1);
        if (100 <= i && i < 400)
        {
            rawPackets[BazIO::PacketFieldName::PulseWidth].push_back(1);
            if (i % 2 == 0)
            {
                rawPackets[BazIO::PacketFieldName::IsBase][i] = 0;
            }
        }
        else
            rawPackets[BazIO::PacketFieldName::PulseWidth].push_back(15);
        frame += ipd + rawPackets[BazIO::PacketFieldName::PulseWidth].back();
    }

    packets = BazIO::BazEventData(rawPackets, {});
    auto states2 = insertF.ClassifyInserts(std::move(packets));

    EXPECT_TRUE(std::all_of(states2.begin(), states2.begin()+100,
                            [](InsertState q) { return q == InsertState::BASE; }));

    EXPECT_TRUE(std::all_of(states2.begin()+100, states2.begin()+400,
                            [](InsertState q) { return q == InsertState::BURST_PULSE || q == InsertState::EX_SHORT_PULSE; }));

    EXPECT_TRUE(std::all_of(states2.begin()+400, states2.end(),
                            [](InsertState q) { return q == InsertState::BASE; }));


    processed = EventData(0, std::move(packets), std::move(states2));
    EXPECT_EQ(601, processed.Ipds()[400]);
}

TEST(InsertFinder, TestInterspersedBase)
{
    BurstInsertFinder insertF(50);

    constexpr size_t numPulses = 200;
    auto rawPackets = EmptyRawPackets();
    const int ipd = 5;
    size_t frame = ipd;
    for (unsigned int i = 0; i < numPulses; i++)
    {
        rawPackets[BazIO::PacketFieldName::StartFrame].push_back(frame);
        rawPackets[BazIO::PacketFieldName::PulseWidth].push_back(1);
        auto isBase = (i % 2 == 0) ? true : false;
        rawPackets[BazIO::PacketFieldName::IsBase].push_back(isBase);
        rawPackets[BazIO::PacketFieldName::Label].push_back(1);
        frame += ipd + rawPackets[BazIO::PacketFieldName::PulseWidth].back();
    }

    auto packets = BazIO::BazEventData(rawPackets, {});
    const auto& states = insertF.ClassifyInserts(packets);

    std::vector<InsertState> burstPulses;
    std::vector<InsertState> exShortPulses;
    for (size_t i = 0; i < states.size(); i++)
    {
        if (i % 2 == 0) burstPulses.push_back(states[i]);
        else exShortPulses.push_back(states[i]);
    }

    EXPECT_TRUE(std::all_of(burstPulses.begin(), burstPulses.end(),
                            [](InsertState q) { return q == InsertState::BURST_PULSE; }));

    EXPECT_TRUE(std::all_of(exShortPulses.begin(), exShortPulses.end(),
                            [](InsertState q) { return q == InsertState::EX_SHORT_PULSE; }));
}

TEST(InsertFinder, InterspersedShortPulse)
{
    BurstInsertFinder insertF(1000);

    constexpr size_t numPulses = 300;
    auto rawPackets = EmptyRawPackets();
    const int ipd = 5;
    const int pw = 1;
    size_t frame = ipd;
    for (unsigned int i = 0; i < numPulses; i++)
    {
        rawPackets[BazIO::PacketFieldName::StartFrame].push_back(frame);
        rawPackets[BazIO::PacketFieldName::PulseWidth].push_back(1);
        auto isBase = (i % 3 == 0) ? true : false;
        rawPackets[BazIO::PacketFieldName::IsBase].push_back(isBase);
        rawPackets[BazIO::PacketFieldName::Label].push_back(pw);
        frame += ipd + pw;
    }

    auto packets = BazIO::BazEventData(rawPackets, {});
    const auto& states = insertF.ClassifyInserts(packets);
    EXPECT_TRUE(std::all_of(states.begin(), states.begin()+298, [](InsertState q) { return q != InsertState::BURST_PULSE; }));
}

TEST(InsertFinder, TestShortPulses)
{
    BurstInsertFinder insertF(50);

    constexpr size_t numPulses = 200;
    auto rawPackets = EmptyRawPackets();
    const int ipd = 5;
    const int pw = 1;
    size_t frame = ipd;
    for (unsigned int i = 0; i < numPulses; i++)
    {
        rawPackets[BazIO::PacketFieldName::StartFrame].push_back(frame);
        rawPackets[BazIO::PacketFieldName::PulseWidth].push_back(1);
        rawPackets[BazIO::PacketFieldName::IsBase].push_back(false);
        rawPackets[BazIO::PacketFieldName::Label].push_back(1);
        frame += ipd + pw;
    }

    constexpr auto numBases = 100;
    for (unsigned int i = 0; i < numBases; i++)
    {
        rawPackets[BazIO::PacketFieldName::IsBase][i+50] = true;
    }

    auto packets = BazIO::BazEventData(rawPackets, {});
    const auto& states = insertF.ClassifyInserts(packets);
    EXPECT_TRUE(std::all_of(states.begin(), states.begin()+50, [](InsertState q) { return q == InsertState::EX_SHORT_PULSE; }));
    EXPECT_TRUE(std::all_of(states.begin()+50, states.begin()+50+numBases, [](InsertState q) { return q == InsertState::BURST_PULSE; }));
    EXPECT_TRUE(std::all_of(states.begin()+50+numBases, states.end(), [](InsertState q) { return q == InsertState::EX_SHORT_PULSE; }));
}

TEST(InsertFinder, ShortBurst)
{
    BurstInsertFinder insertF(1000);

    constexpr size_t numPulses = 100;
    auto rawPackets = EmptyRawPackets();
    const int ipd = 5;
    const int pw = 1;
    size_t frame = ipd;
    for (unsigned int i = 0; i < numPulses; i++)
    {
        rawPackets[BazIO::PacketFieldName::StartFrame].push_back(frame);
        rawPackets[BazIO::PacketFieldName::PulseWidth].push_back(1);
        rawPackets[BazIO::PacketFieldName::IsBase].push_back(true);
        rawPackets[BazIO::PacketFieldName::Label].push_back(1);
        frame += ipd + pw;
    }

    auto packets = BazIO::BazEventData(rawPackets, {});
    const auto& states = insertF.ClassifyInserts(packets);
    EXPECT_TRUE(std::all_of(states.begin(), states.end(), [](InsertState q) { return q == InsertState::BASE; }));
}

TEST(InsertFinder, LongBurst)
{
    BurstInsertFinder insertF(50);

    // Add long pulses that should be called as bases.
    auto rawPackets = EmptyRawPackets();
    const int ipd = 5;
    size_t frame = ipd;
    for (unsigned int i = 0; i < 250; i++)
    {
        rawPackets[BazIO::PacketFieldName::StartFrame].push_back(frame);
        rawPackets[BazIO::PacketFieldName::PulseWidth].push_back(12);
        rawPackets[BazIO::PacketFieldName::IsBase].push_back(true);
        rawPackets[BazIO::PacketFieldName::Label].push_back(1);
        frame += ipd + rawPackets[BazIO::PacketFieldName::PulseWidth].back();
    }

    // Add short pulses that should be identified as burst.
    for (unsigned int i = 0; i < 500; i++)
    {
        rawPackets[BazIO::PacketFieldName::StartFrame].push_back(frame);
        rawPackets[BazIO::PacketFieldName::PulseWidth].push_back(1);
        rawPackets[BazIO::PacketFieldName::IsBase].push_back(true);
        rawPackets[BazIO::PacketFieldName::Label].push_back(1);
        frame += ipd + rawPackets[BazIO::PacketFieldName::PulseWidth].back();
    }

    for (unsigned int i = 0; i < 250; i++)
    {
        rawPackets[BazIO::PacketFieldName::StartFrame].push_back(frame);
        rawPackets[BazIO::PacketFieldName::PulseWidth].push_back(12);
        rawPackets[BazIO::PacketFieldName::IsBase].push_back(true);
        rawPackets[BazIO::PacketFieldName::Label].push_back(1);
        frame += ipd + rawPackets[BazIO::PacketFieldName::PulseWidth].back();
    }

    // This should get overturned as the burst is below the minimum length of 50.
    for (unsigned int i = 0; i < 40; i++)
    {
        rawPackets[BazIO::PacketFieldName::StartFrame].push_back(frame);
        rawPackets[BazIO::PacketFieldName::PulseWidth].push_back(1);
        rawPackets[BazIO::PacketFieldName::IsBase].push_back(true);
        rawPackets[BazIO::PacketFieldName::Label].push_back(1);
        frame += ipd + rawPackets[BazIO::PacketFieldName::PulseWidth].back();
    }

    for (unsigned int i = 0; i < 160; i++)
    {
        rawPackets[BazIO::PacketFieldName::StartFrame].push_back(frame);
        rawPackets[BazIO::PacketFieldName::PulseWidth].push_back(12);
        rawPackets[BazIO::PacketFieldName::IsBase].push_back(true);
        rawPackets[BazIO::PacketFieldName::Label].push_back(1);
        frame += ipd + rawPackets[BazIO::PacketFieldName::PulseWidth].back();
    }

    auto packets = BazIO::BazEventData(rawPackets, {});
    const auto& states = insertF.ClassifyInserts(packets);

    EXPECT_TRUE(std::all_of(states.begin(), states.begin()+250, [](InsertState q) { return q == InsertState::BASE; }));
    EXPECT_TRUE(std::all_of(states.begin()+250, states.begin()+250+500, [](InsertState q) { return q == InsertState::BURST_PULSE; }));
    EXPECT_TRUE(std::all_of(states.begin()+250+500, states.end(), [](InsertState q) { return q == InsertState::BASE; }));
}

