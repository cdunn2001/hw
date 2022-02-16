// Copyright (c) 2020, Pacific Biosciences of California, Inc.
//
// All rights reserved.
//
// THIS SOFTWARE CONSTITUTES AND EMBODIES PACIFIC BIOSCIENCES' CONFIDENTIAL
// AND PROPRIETARY INFORMATION.
//
// Disclosure, redistribution and use of this software is subject to the
// terms and conditions of the applicable written agreement(s) between you
// and Pacific Biosciences, where "you" refers to you or your company or
// organization, as applicable.  Any other disclosure, redistribution or
// use is prohibited.
//
// THIS SOFTWARE IS PROVIDED BY PACIFIC BIOSCIENCES AND ITS CONTRIBUTORS "AS
// IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
// THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL PACIFIC BIOSCIENCES OR ITS
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
// OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
// WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
// OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
// ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <gtest/gtest.h>

#include <pacbio/datasource/ZmwFeatures.h>
#include <pacbio/logging/Logger.h>

#include <common/MongoConstants.h>

#include <appModules/SimulatedDataSource.h>
#include <pacbio/datasource/MallocAllocator.h>

using namespace PacBio::Application;
using namespace PacBio::DataSource;
using namespace PacBio::Memory;
using namespace PacBio::Mongo;
using namespace PacBio::Sensor;

// Validation routine to check the output of an individual SensorPacket
//
// `ExpectedFunct will have a function operator that takes a zmw and frame and
// returns an expected pixel value.
//
// ValidatePacket will return true iff all data in the packet matches expectations.
// If there are any mismatches this function will directly trigger gtest failures
// for the first small number of differences, after which it will stop trying
// so as to not flood the output with failure data.  It is expected that the true/false
// output of this function will also be tossed into a gtest assert, so that higher
// level information about which batch is failing also gets output.
template <typename T, typename ExpectedFunc>
bool ValidatePacket(const ExpectedFunc& expected, const SensorPacket& packet)
{
    const auto& layout = packet.Layout();
    auto blockZmw = packet.StartZmw();
    size_t errorCount = 0;
    constexpr size_t maxErrors = 10;
    for (size_t i = 0; i < layout.NumBlocks(); ++i)
    {
        auto data = reinterpret_cast<const T*>(packet.BlockData(i).Data());
        auto chunkFrame = packet.StartFrame();
        for (size_t frameIdx = 0; frameIdx < layout.NumFrames(); ++frameIdx)
        {
            for (size_t zmwIdx = 0; zmwIdx < layout.BlockWidth(); ++zmwIdx)
            {
                auto e = expected(blockZmw + zmwIdx, chunkFrame + frameIdx);
                EXPECT_EQ(e, *data) << "zmw, frame: " << blockZmw + zmwIdx << ", " << chunkFrame + frameIdx;
                if (e != *data) errorCount++;
                if (errorCount == maxErrors)
                {
                    ADD_FAILURE() << "Reached max errors per packet";
                    return false;
                }
                ++data;
            }
        }
        blockZmw += layout.BlockWidth();
    }
    return errorCount == 0;
}

// Validation routine to check the output of a full SimulatedDataSource.  Uses the
// above validation routine to check individual packets, and also adds some bits
// of extra information to the output in the event of a failure.
//
// Returns the number of packets that failed validation.
template <typename ExpectedFunc>
size_t ValidateData(const ExpectedFunc& expected, SimulatedDataSource& source)
{
    SensorPacketsChunk chunk;
    size_t numValidPackets = 0;
    size_t expectedNextStartFrame = 0;
    while (source.IsRunning())
    {
        source.ContinueProcessing();
        if(source.PopChunk(chunk, std::chrono::milliseconds{10}))
        {
            const bool validLayout = (chunk.IsValid() && expectedNextStartFrame == chunk.StartFrame());
            expectedNextStartFrame = chunk.StopFrame();
            EXPECT_TRUE(validLayout) << "Failed layout validation, skipping data validation";
            if (!validLayout) continue;

            for (const auto& packet : chunk)
            {
                auto packetValid = packet.Layout().Encoding() == PacketLayout::INT16
                    ? ValidatePacket<int16_t>(expected, packet)
                    : ValidatePacket<uint8_t>(expected, packet);
                EXPECT_TRUE(packetValid) << "Packet failed, StartZmw/StartFrame: " << packet.StartZmw() << "/" << packet.StartFrame();
                if (packetValid) numValidPackets++;
            }
        }
    }
    return numValidPackets;
}

struct TestingParams
{
    size_t framesPerBlock;
    size_t lanesPerPool;
    size_t totalFrames;
    size_t numZmw;
    size_t numSignals;
    size_t signalFrames;

    // Need this so gtest prints something sensible on test failure
    friend std::ostream& operator<<(std::ostream& os, const TestingParams& param)
    {
        os << std::endl << "framesPerBlock: " << param.framesPerBlock;
        os << std::endl << "lanesPerPool: " << param.lanesPerPool;
        os << std::endl << "numZmw: " << param.numZmw;
        os << std::endl << "totalFrames: " << param.totalFrames;
        os << std::endl << "numSignals: " << param.numSignals;
        os << std::endl << "signalFrames: " << param.signalFrames;
        return os;
    }

};

class SimDataSource : public testing::TestWithParam<TestingParams>
{
};

TEST_P(SimDataSource, Constant)
{
    const auto params = GetParam();

    PacBio::Logging::LogSeverityContext severity(PacBio::Logging::LogLevel::WARN);

    PacketLayout layout(PacketLayout::BLOCK_LAYOUT_DENSE, PacketLayout::INT16,
                        {params.lanesPerPool, params.framesPerBlock, laneSize});
    SimulatedDataSource::SimConfig simConf(params.numSignals, params.signalFrames);

    DataSourceBase::Configuration cfg(layout, std::make_unique<MallocAllocator>());
    cfg.numFrames = params.totalFrames;
    SimulatedDataSource source(params.numZmw,
                               simConf,
                               std::move(cfg),
                               std::make_unique<ConstantGenerator>());

    auto Expected = [&](size_t zmw, size_t) -> int16_t {
        auto expectedSignals = (params.numSignals + laneSize - 1) / laneSize * laneSize;
        return zmw % expectedSignals;
    };
    const auto expectedChunks = params.totalFrames / params.framesPerBlock;
    const auto expectedPools = params.numZmw / (params.lanesPerPool * laneSize);
    EXPECT_EQ(ValidateData(Expected, source), expectedChunks * expectedPools);
    EXPECT_EQ("SimulatedDataSource:ConstantGenerator", source.InstrumentName());
    EXPECT_EQ(Platform::DONT_CARE, source.Platform());
}

TEST_P(SimDataSource, LongSawtooth)
{
    const auto params = GetParam();

    PacBio::Logging::LogSeverityContext severity(PacBio::Logging::LogLevel::WARN);

    PacketLayout layout(PacketLayout::BLOCK_LAYOUT_DENSE, PacketLayout::INT16,
                        {params.lanesPerPool, params.framesPerBlock, laneSize});
    SimulatedDataSource::SimConfig simConf(params.numSignals, params.signalFrames);

    DataSourceBase::Configuration cfg(layout, std::make_unique<MallocAllocator>());
    cfg.numFrames = params.totalFrames;

    SawtoothGenerator::Config sawConfig;
    sawConfig.periodFrames = 6000;
    sawConfig.minAmp = 0;
    sawConfig.maxAmp = 3000;
    sawConfig.startFrameStagger = 1;
    SimulatedDataSource source(params.numZmw,
                               simConf,
                               std::move(cfg),
                               std::make_unique<SawtoothGenerator>(sawConfig));

    auto Expected = [&](size_t zmw, size_t frame) -> int16_t {
        auto expectedSignals = (params.numSignals + laneSize - 1) / laneSize * laneSize;
        auto expectedSimFrames = (params.signalFrames + params.framesPerBlock - 1) / params.framesPerBlock * params.framesPerBlock;

        auto wrappedZmw = zmw % expectedSignals;
        auto wrappedFrame = frame % expectedSimFrames;

        auto range = sawConfig.maxAmp - sawConfig.minAmp + 1;
        auto slope = range / static_cast<double>(sawConfig.periodFrames);
        auto ret = sawConfig.minAmp + static_cast<int16_t>((wrappedFrame + wrappedZmw * sawConfig.startFrameStagger) * slope) % range;
        return static_cast<int16_t>(ret);
    };

    const auto expectedChunks = params.totalFrames / params.framesPerBlock;
    const auto expectedPools = params.numZmw / (params.lanesPerPool * laneSize);
    EXPECT_EQ(ValidateData(Expected, source), expectedChunks * expectedPools);
    EXPECT_EQ("SimulatedDataSource:SawtoothGenerator", source.InstrumentName());
    EXPECT_EQ(Platform::DONT_CARE, source.Platform());
}

// Make sure that we can generate 8 bit data, but also that any
// out of bounds values that are generated are clamped to the
// 0-255 range
TEST_P(SimDataSource, 8BitWithSaturation)
{
    const auto params = GetParam();

    PacBio::Logging::LogSeverityContext severity(PacBio::Logging::LogLevel::WARN);

    PacketLayout layout(PacketLayout::BLOCK_LAYOUT_DENSE, PacketLayout::UINT8,
                        {params.lanesPerPool, params.framesPerBlock, laneSize});
    SimulatedDataSource::SimConfig simConf(params.numSignals, params.signalFrames);

    DataSourceBase::Configuration cfg(layout, std::make_unique<MallocAllocator>());
    cfg.numFrames = params.totalFrames;

    // Picking parameters that are mostly in bounds, but require some amount
    // of clamping to be 0-255
    SawtoothGenerator::Config sawConfig;
    sawConfig.periodFrames = 270;
    sawConfig.minAmp = -9;
    sawConfig.maxAmp = 260;
    sawConfig.startFrameStagger = 1;
    SimulatedDataSource source(params.numZmw,
                               simConf,
                               std::move(cfg),
                               std::make_unique<SawtoothGenerator>(sawConfig));

    auto Expected = [&](size_t zmw, size_t frame) -> uint8_t {
        auto expectedSignals = (params.numSignals + laneSize - 1) / laneSize * laneSize;
        auto expectedSimFrames = (params.signalFrames + params.framesPerBlock - 1) / params.framesPerBlock * params.framesPerBlock;

        auto wrappedZmw = zmw % expectedSignals;
        auto wrappedFrame = frame % expectedSimFrames;

        auto range = sawConfig.maxAmp - sawConfig.minAmp + 1;
        auto slope = range / static_cast<double>(sawConfig.periodFrames);
        auto ret = sawConfig.minAmp + static_cast<int16_t>((wrappedFrame + wrappedZmw * sawConfig.startFrameStagger) * slope) % range;
        return static_cast<uint8_t>(std::clamp(ret, 0, 255));
    };

    const auto expectedChunks = params.totalFrames / params.framesPerBlock;
    const auto expectedPools = params.numZmw / (params.lanesPerPool * laneSize);
    EXPECT_EQ(ValidateData(Expected, source), expectedChunks * expectedPools);
    EXPECT_EQ("SimulatedDataSource:SawtoothGenerator", source.InstrumentName());
    EXPECT_EQ(Platform::DONT_CARE, source.Platform());
}

TEST_P(SimDataSource, ShortSawtooth)
{
    const auto params = GetParam();

    PacBio::Logging::LogSeverityContext severity(PacBio::Logging::LogLevel::WARN);

    PacketLayout layout(PacketLayout::BLOCK_LAYOUT_DENSE, PacketLayout::INT16,
                        {params.lanesPerPool, params.framesPerBlock, laneSize});
    SimulatedDataSource::SimConfig simConf(params.numSignals, params.signalFrames);

    DataSourceBase::Configuration cfg(layout, std::make_unique<MallocAllocator>());
    cfg.numFrames = params.totalFrames;

    SawtoothGenerator::Config sawConfig;
    sawConfig.periodFrames = 100;
    sawConfig.minAmp = 0;
    sawConfig.maxAmp = 200;
    sawConfig.startFrameStagger = 1;
    SimulatedDataSource source(params.numZmw,
                               simConf,
                               std::move(cfg),
                               std::make_unique<SawtoothGenerator>(sawConfig));

    auto Expected = [&](size_t zmw, size_t frame) -> int16_t {
        auto expectedSignals = (params.numSignals + laneSize - 1) / laneSize * laneSize;
        auto expectedSimFrames = (params.signalFrames + params.framesPerBlock - 1) / params.framesPerBlock * params.framesPerBlock;

        auto wrappedZmw = zmw % expectedSignals;
        auto wrappedFrame = frame % expectedSimFrames;

        auto range = sawConfig.maxAmp - sawConfig.minAmp + 1;
        auto slope = range / static_cast<double>(sawConfig.periodFrames);
        auto ret = sawConfig.minAmp + static_cast<int16_t>((wrappedFrame + wrappedZmw * sawConfig.startFrameStagger) * slope) % range;
        return static_cast<int16_t>(ret);
    };

    const auto expectedChunks = params.totalFrames / params.framesPerBlock;
    const auto expectedPools = params.numZmw / (params.lanesPerPool * laneSize);
    EXPECT_EQ(ValidateData(Expected, source), expectedChunks * expectedPools);
    EXPECT_EQ("SimulatedDataSource:SawtoothGenerator", source.InstrumentName());
    EXPECT_EQ(Platform::DONT_CARE, source.Platform());
}

//-----------------------------------------Testing parameters---------------------------//

// Start with a single block
INSTANTIATE_TEST_SUITE_P(SingleBlock,
                         SimDataSource,
                         testing::Values(TestingParams{
                                 512,  /* framesPerBlock */
                                 1,    /* lanesPerPool */
                                 512,  /* totalFrames */
                                 64,   /* numZmw */
                                 64,   /* numSignals */
                                 512,  /* signalFrames */}));

// upgrade to a single batch, which will contain 4 blocks
INSTANTIATE_TEST_SUITE_P(SingleBatch,
                         SimDataSource,
                         testing::Values(TestingParams{
                                 512,  /* framesPerBlock */
                                 4,    /* lanesPerPool */
                                 512,  /* totalFrames */
                                 256 , /* numZmw */
                                 256,   /* numSignals */
                                 512,  /* signalFrames */}));

// Increate the number of zmw so that we have two batches
INSTANTIATE_TEST_SUITE_P(SingleChunkMultiBatch,
                         SimDataSource,
                         testing::Values(TestingParams{
                                 512,  /* framesPerBlock */
                                 4,    /* lanesPerPool */
                                 512,  /* totalFrames */
                                 512 , /* numZmw */
                                 512,  /* numSignals */
                                 512,  /* signalFrames */}));

// Finally increase the number of frames so we have two chunks
INSTANTIATE_TEST_SUITE_P(MultiChunkMultiBatch,
                         SimDataSource,
                         testing::Values(TestingParams{
                                 512,  /* framesPerBlock */
                                 4,    /* lanesPerPool */
                                 1024, /* totalFrames */
                                 512 , /* numZmw */
                                 512,  /* numSignals */
                                 1024, /* signalFrames */}));

// Now use fewer simulated signals than zmw to check space replication
INSTANTIATE_TEST_SUITE_P(EvenSpaceReplication,
                         SimDataSource,
                         testing::Values(TestingParams{
                                 512,  /* framesPerBlock */
                                 4,    /* lanesPerPool */
                                 1024, /* totalFrames */
                                 512 , /* numZmw */
                                 128,  /* numSignals */
                                 1024, /* signalFrames */}));

// Now use fewer simulated frames than the full run to check time replication
INSTANTIATE_TEST_SUITE_P(EvenTimeReplication,
                         SimDataSource,
                         testing::Values(TestingParams{
                                 512,  /* framesPerBlock */
                                 4,    /* lanesPerPool */
                                 1024, /* totalFrames */
                                 512 , /* numZmw */
                                 512,  /* numSignals */
                                 512,  /* signalFrames */}));

// Replicate in both time and space
INSTANTIATE_TEST_SUITE_P(FullReplication,
                         SimDataSource,
                         testing::Values(TestingParams{
                                 512,  /* framesPerBlock */
                                 4,    /* lanesPerPool */
                                 1024, /* totalFrames */
                                 512 , /* numZmw */
                                 128,  /* numSignals */
                                 512,  /* signalFrames */}));

// Check replication that does not evenly divide other parameters.
// It's expected that both simulated signals and frames get rounded
// up to an ingregal block size
INSTANTIATE_TEST_SUITE_P(UnevenReplication,
                         SimDataSource,
                         testing::Values(TestingParams{
                                 512,  /* framesPerBlock */
                                 4,    /* lanesPerPool */
                                 2048, /* totalFrames */
                                 512 , /* numZmw */
                                 100,  /* numSignals */
                                 800,  /* signalFrames */}));

// Same as last, this time with the request not
// even filling out a full block.
INSTANTIATE_TEST_SUITE_P(TinyReplication,
                         SimDataSource,
                         testing::Values(TestingParams{
                                 512,  /* framesPerBlock */
                                 4,    /* lanesPerPool */
                                 2048, /* totalFrames */
                                 512 , /* numZmw */
                                 1,    /* numSignals */
                                 10 ,  /* signalFrames */}));

//----------------------------Standalone tests for DataSource API---------------------------//

TEST(SimDataSourceAPI, Layout)
{
    uint32_t lanesPerPool = 8;
    uint32_t framesPerBlock = 512;
    uint32_t totalFrames = 2048;
    uint32_t totalZmw = 4096;

    PacketLayout layout(PacketLayout::BLOCK_LAYOUT_DENSE, PacketLayout::INT16,
                        {lanesPerPool, framesPerBlock, laneSize});
    // don't care how data is generated, this particular test won't even look at it
    SimulatedDataSource::SimConfig simConf(1, 1);

    DataSourceBase::Configuration cfg(layout, std::make_unique<MallocAllocator>());
    cfg.numFrames = totalFrames;
    SimulatedDataSource source(totalZmw,
                               simConf,
                               std::move(cfg),
                               std::make_unique<ConstantGenerator>());

    const auto& layouts = source.PacketLayouts();
    EXPECT_EQ(layouts.size(), 8);
    std::set<size_t> seenIds;
    for (const auto& kv: layouts)
    {
        seenIds.insert(kv.first);
        const auto& providedLayout = kv.second;
        EXPECT_EQ(providedLayout.BlockWidth(), laneSize);
        EXPECT_EQ(providedLayout.NumFrames(), framesPerBlock);
        EXPECT_EQ(providedLayout.NumBlocks(), lanesPerPool);
    }
    ASSERT_EQ(seenIds.size(), layouts.size());
    ASSERT_EQ(*seenIds.rbegin(), layouts.size() - 1);

    EXPECT_EQ(source.NumFrames(), totalFrames);
    EXPECT_EQ(source.NumZmw(), totalZmw);

    EXPECT_EQ("SimulatedDataSource:ConstantGenerator", source.InstrumentName());
    EXPECT_EQ(Platform::DONT_CARE, source.Platform());

    // The current implementation rounds up the requested ZMW to fill an integral
    // number of lanes, but otherwise will have a "runt" batch at the end if necessary
    cfg = DataSourceBase::Configuration(layout, std::make_unique<MallocAllocator>());
    // The zmw count should be rounded up to full up a lane
    SimulatedDataSource source2(totalZmw - 260,
                                simConf,
                                std::move(cfg),
                                std::make_unique<ConstantGenerator>());

    const auto& layouts2 = source2.PacketLayouts();
    // Should have the same number of batches, just the last one
    // is now smaller
    ASSERT_EQ(layouts2.size(), layouts.size());
    for (size_t i = 0; i < layouts2.size(); ++i)
    {
        ASSERT_TRUE(layouts2.count(i));
        const auto& l1 = layouts.at(i);
        const auto& l2 = layouts2.at(i);
        EXPECT_EQ(l1.BlockWidth(), l2.BlockWidth());
        EXPECT_EQ(l1.NumFrames(), l2.NumFrames());
        if (i < layouts2.size() - 1)
        {
            EXPECT_EQ(l1.NumBlocks(), l2.NumBlocks());
        } else
        {
            EXPECT_EQ(l1.NumBlocks() - 4, l2.NumBlocks());
        }

    }

    EXPECT_EQ(source2.NumZmw(), totalZmw - 256);
    EXPECT_EQ("SimulatedDataSource:ConstantGenerator", source2.InstrumentName());
    EXPECT_EQ(Platform::DONT_CARE, source2.Platform());
}

TEST(SimDataSourceAPI, ZmwInfo)
{
    uint32_t lanesPerPool = 8;
    uint32_t framesPerBlock = 512;
    uint32_t totalFrames = 2048;
    uint32_t totalZmw = 4096;

    PacketLayout layout(PacketLayout::BLOCK_LAYOUT_DENSE, PacketLayout::INT16,
                        {lanesPerPool, framesPerBlock, laneSize});
    // don't care how data is generated, this particular test won't even look at it
    SimulatedDataSource::SimConfig simConf(1, 1);

    DataSourceBase::Configuration cfg(layout, std::make_unique<MallocAllocator>());
    cfg.numFrames = totalFrames;
    SimulatedDataSource source(totalZmw,
                               simConf,
                               std::move(cfg),
                               std::make_unique<ConstantGenerator>());
    EXPECT_EQ(source.NumZmw(), totalZmw);
    EXPECT_EQ(source.NumFrames(), totalFrames);

    // Should check the values, but we've not actually defined what the UnitCellFeature values are yet
    EXPECT_EQ(source.GetUnitCellProperties().size(), totalZmw);
    EXPECT_TRUE(std::all_of(source.GetUnitCellProperties().cbegin(),
                            source.GetUnitCellProperties().cend(),
                            [](const auto& uc){ return uc.flags == static_cast<uint32_t>(ZmwFeatures::Sequencing); }));

    // ID's should have one unique value for each zmw
    auto ids = source.UnitCellIds();
    std::set<uint32_t> uniqueIds;
    for (auto& val : ids) uniqueIds.insert(val);

    EXPECT_EQ(ids.size(), totalZmw);
    EXPECT_EQ(uniqueIds.size(), totalZmw);

    EXPECT_EQ("SimulatedDataSource:ConstantGenerator", source.InstrumentName());
    EXPECT_EQ(Platform::DONT_CARE, source.Platform());
}
