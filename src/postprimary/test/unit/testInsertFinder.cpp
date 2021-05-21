#include <gtest/gtest.h>

#include <assert.h>
#include <iostream>
#include <cmath>
#include <tuple>

#include <postprimary/insertfinder/InsertFinder.h>

#include "ReadSimulator.h"

using namespace PacBio::Primary;
using namespace PacBio::Primary::Postprimary;


namespace {

std::vector<std::vector<uint32_t>> EmptyRawPackets()
{
    static const size_t size = PacketFieldName::allValues().size();
    return std::vector<std::vector<uint32_t>>(size);
}

}

TEST(InsertFinder, RecomputeIPD)
{
    BurstInsertFinder insertF(50);

    // Case 1: Burst occurs in middle of read. IPDs all set to 1
    constexpr size_t numPulses = 500;
    auto rawPackets = EmptyRawPackets();
    for (unsigned int i = 0; i < numPulses; i++)
    {
        rawPackets[static_cast<uint8_t>(PacketFieldName::IPD_LL)].push_back(1);
        rawPackets[static_cast<uint8_t>(PacketFieldName::READOUT)].push_back(1);
        rawPackets[static_cast<uint8_t>(PacketFieldName::IS_BASE)].push_back(1);
        if (200 <= i && i < 300)
            rawPackets[static_cast<uint8_t>(PacketFieldName::PW_LL)].push_back(1);
        else
            rawPackets[static_cast<uint8_t>(PacketFieldName::PW_LL)].push_back(15);
    }

    auto packets = BazEventData(std::move(rawPackets));
    const auto& states1 = insertF.ClassifyInserts(packets);

    EXPECT_TRUE(std::all_of(states1.begin(), states1.begin()+200,
                            [](InsertState q) { return q == InsertState::BASE; }));

    EXPECT_TRUE(std::all_of(states1.begin()+200, states1.begin()+300,
                            [](InsertState q) { return q == InsertState::BURST_PULSE; }));

    EXPECT_TRUE(std::all_of(states1.begin()+300, states1.end(),
                            [](InsertState q) { return q == InsertState::BASE; }));

    auto processed = EventDataParent(std::move(packets), states1);
    EXPECT_EQ(201, processed.Ipds()[300]);

    // Case 2: Same setup as 1) but sprinkle in EX_SHORT_PULSES.
    rawPackets = EmptyRawPackets();
    for (unsigned int i = 0; i < numPulses; i++)
    {
        rawPackets[static_cast<uint8_t>(PacketFieldName::IPD_LL)].push_back(1);
        rawPackets[static_cast<uint8_t>(PacketFieldName::READOUT)].push_back(1);
        rawPackets[static_cast<uint8_t>(PacketFieldName::IS_BASE)].push_back(1);
        if (100 <= i && i < 400)
        {
            rawPackets[static_cast<uint8_t>(PacketFieldName::PW_LL)].push_back(1);
            if (i % 2 == 0)
            {
                rawPackets[static_cast<uint8_t>(PacketFieldName::IS_BASE)][i] = 0;
            }
        }
        else
            rawPackets[static_cast<uint8_t>(PacketFieldName::PW_LL)].push_back(15);
    }

    packets = BazEventData(std::move(rawPackets));
    const auto& states2 = insertF.ClassifyInserts(std::move(packets));

    EXPECT_TRUE(std::all_of(states2.begin(), states2.begin()+100,
                            [](InsertState q) { return q == InsertState::BASE; }));

    EXPECT_TRUE(std::all_of(states2.begin()+100, states2.begin()+400,
                            [](InsertState q) { return q == InsertState::BURST_PULSE || q == InsertState::EX_SHORT_PULSE; }));

    EXPECT_TRUE(std::all_of(states2.begin()+400, states2.end(),
                            [](InsertState q) { return q == InsertState::BASE; }));


    processed = EventDataParent(std::move(packets), states2);
    EXPECT_EQ(301, processed.Ipds()[400]);
}

TEST(InsertFinder, TestInterspersedBase)
{
    BurstInsertFinder insertF(50);

    constexpr size_t numPulses = 200;
    auto rawPackets = EmptyRawPackets();
    for (unsigned int i = 0; i < numPulses; i++)
    {
        rawPackets[static_cast<uint8_t>(PacketFieldName::IPD_LL)].push_back(5);
        rawPackets[static_cast<uint8_t>(PacketFieldName::PW_LL)].push_back(1);
        auto isBase = (i % 2 == 0) ? true : false;
        rawPackets[static_cast<uint8_t>(PacketFieldName::IS_BASE)].push_back(isBase);
    }

    auto packets = BazEventData(std::move(rawPackets));
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
    for (unsigned int i = 0; i < numPulses; i++)
    {
        rawPackets[static_cast<uint8_t>(PacketFieldName::IPD_LL)].push_back(5);
        rawPackets[static_cast<uint8_t>(PacketFieldName::PW_LL)].push_back(1);
        auto isBase = (i % 3 == 0) ? true : false;
        rawPackets[static_cast<uint8_t>(PacketFieldName::IS_BASE)].push_back(isBase);
    }

    auto packets = BazEventData(std::move(rawPackets));
    const auto& states = insertF.ClassifyInserts(packets);
    EXPECT_TRUE(std::all_of(states.begin(), states.begin()+298, [](InsertState q) { return q != InsertState::BURST_PULSE; }));
}

TEST(InsertFinder, TestShortPulses)
{
    BurstInsertFinder insertF(50);

    constexpr size_t numPulses = 200;
    auto rawPackets = EmptyRawPackets();
    for (unsigned int i = 0; i < numPulses; i++)
    {
        rawPackets[static_cast<uint8_t>(PacketFieldName::IPD_LL)].push_back(5);
        rawPackets[static_cast<uint8_t>(PacketFieldName::PW_LL)].push_back(1);
        rawPackets[static_cast<uint8_t>(PacketFieldName::IS_BASE)].push_back(false);
    }

    constexpr auto numBases = 100;
    for (unsigned int i = 0; i < numBases; i++)
    {
        rawPackets[static_cast<uint8_t>(PacketFieldName::IS_BASE)][i+50] = true;
    }

    auto packets = BazEventData(std::move(rawPackets));
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
    for (unsigned int i = 0; i < numPulses; i++)
    {
        rawPackets[static_cast<uint8_t>(PacketFieldName::IPD_LL)].push_back(5);
        rawPackets[static_cast<uint8_t>(PacketFieldName::PW_LL)].push_back(1);
        rawPackets[static_cast<uint8_t>(PacketFieldName::IS_BASE)].push_back(true);
    }

    auto packets = BazEventData(std::move(rawPackets));
    const auto& states = insertF.ClassifyInserts(packets);
    EXPECT_TRUE(std::all_of(states.begin(), states.end(), [](InsertState q) { return q == InsertState::BASE; }));
}

TEST(InsertFinder, LongBurst)
{
    BurstInsertFinder insertF(50);

    // Add long pulses that should be called as bases.
    auto rawPackets = EmptyRawPackets();
    for (unsigned int i = 0; i < 250; i++)
    {
        rawPackets[static_cast<uint8_t>(PacketFieldName::IPD_LL)].push_back(5);
        rawPackets[static_cast<uint8_t>(PacketFieldName::PW_LL)].push_back(12);
        rawPackets[static_cast<uint8_t>(PacketFieldName::IS_BASE)].push_back(true);
    }

    // Add short pulses that should be identified as burst.
    for (unsigned int i = 0; i < 500; i++)
    {
        rawPackets[static_cast<uint8_t>(PacketFieldName::IPD_LL)].push_back(5);
        rawPackets[static_cast<uint8_t>(PacketFieldName::PW_LL)].push_back(1);
        rawPackets[static_cast<uint8_t>(PacketFieldName::IS_BASE)].push_back(true);
    }

    for (unsigned int i = 0; i < 250; i++)
    {
        rawPackets[static_cast<uint8_t>(PacketFieldName::IPD_LL)].push_back(5);
        rawPackets[static_cast<uint8_t>(PacketFieldName::PW_LL)].push_back(12);
        rawPackets[static_cast<uint8_t>(PacketFieldName::IS_BASE)].push_back(true);
    }

    // This should get overturned as the burst is below the minimum length of 50.
    for (unsigned int i = 0; i < 40; i++)
    {
        rawPackets[static_cast<uint8_t>(PacketFieldName::IPD_LL)].push_back(5);
        rawPackets[static_cast<uint8_t>(PacketFieldName::PW_LL)].push_back(1);
        rawPackets[static_cast<uint8_t>(PacketFieldName::IS_BASE)].push_back(true);
    }

    for (unsigned int i = 0; i < 160; i++)
    {
        rawPackets[static_cast<uint8_t>(PacketFieldName::IPD_LL)].push_back(5);
        rawPackets[static_cast<uint8_t>(PacketFieldName::PW_LL)].push_back(12);
        rawPackets[static_cast<uint8_t>(PacketFieldName::IS_BASE)].push_back(true);
    }

    auto packets = BazEventData(std::move(rawPackets));
    const auto& states = insertF.ClassifyInserts(packets);

    EXPECT_TRUE(std::all_of(states.begin(), states.begin()+250, [](InsertState q) { return q == InsertState::BASE; }));
    EXPECT_TRUE(std::all_of(states.begin()+250, states.begin()+250+500, [](InsertState q) { return q == InsertState::BURST_PULSE; }));
    EXPECT_TRUE(std::all_of(states.begin()+250+500, states.end(), [](InsertState q) { return q == InsertState::BASE; }));
}

