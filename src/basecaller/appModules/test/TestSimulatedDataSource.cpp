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

#include <pacbio/logging/Logger.h>

#include <common/MongoConstants.h>

#include <appModules/SimulatedDataSource.h>
#include <pacbio/datasource/MallocAllocator.h>

using namespace PacBio::Application;
using namespace PacBio::DataSource;
using namespace PacBio::Memory;
using namespace PacBio::Mongo;

template <typename ExpectedFunc>
bool ValidatePacket(const ExpectedFunc& expected, const SensorPacket& packet)
{
    const auto& layout = packet.Layout();
    auto blockZmw = packet.StartZmw();
    size_t errorCount = 0;
    constexpr size_t maxErrors = 10;
    for (size_t i = 0; i < layout.NumBlocks(); ++i)
    {
        auto data = reinterpret_cast<const int16_t*>(packet.BlockData(i).Data());
        auto chunkFrame = packet.StartFrame();
        for (size_t frame = 0; frame < layout.NumFrames(); ++frame)
        {
            for (size_t zmw = 0; zmw < layout.BlockWidth(); ++zmw)
            {
                auto e = expected(blockZmw + zmw, chunkFrame + frame);
                EXPECT_EQ(e, *data) << "zmw, frame: " << blockZmw + zmw << ", " << chunkFrame + frame;
                if (e != *data) errorCount++;
                if (errorCount == maxErrors)
                {
                    EXPECT_TRUE(false) << "Reached max errors per packet";
                    return false;
                }
                ++data;
            }
        }
        blockZmw += layout.BlockWidth();
    }
    return errorCount == 0;
}

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
            const bool validLayout = (chunk.Valid() && expectedNextStartFrame == chunk.StartFrame());
            expectedNextStartFrame = chunk.StopFrame();
            EXPECT_TRUE(validLayout) << "Failed layout validation, skipping data validation";
            if (!validLayout) continue;

            for (const auto& packet : chunk)
            {
                auto packetValid = ValidatePacket(expected, packet);
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

    auto Expected = [&](size_t zmw, size_t frame) -> int16_t {
        auto expectedSignals = (params.numSignals + laneSize - 1) / laneSize * laneSize;
        return zmw % expectedSignals;
    };
    const auto expectedChunks = params.totalFrames / params.framesPerBlock;
    const auto expectedPools = params.numZmw / (params.lanesPerPool * laneSize);
    EXPECT_EQ(ValidateData(Expected, source), expectedChunks * expectedPools);
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

        auto range = sawConfig.maxAmp - sawConfig.minAmp;
        auto slope = range / static_cast<double>(sawConfig.periodFrames);
        return sawConfig.minAmp + static_cast<int16_t>((wrappedFrame + wrappedZmw * sawConfig.startFrameStagger) * slope) % range;
    };

    const auto expectedChunks = params.totalFrames / params.framesPerBlock;
    const auto expectedPools = params.numZmw / (params.lanesPerPool * laneSize);
    EXPECT_EQ(ValidateData(Expected, source), expectedChunks * expectedPools);
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

        auto range = sawConfig.maxAmp - sawConfig.minAmp;
        auto slope = range / static_cast<double>(sawConfig.periodFrames);
        return sawConfig.minAmp + static_cast<int16_t>((wrappedFrame + wrappedZmw * sawConfig.startFrameStagger) * slope) % range;
    };

    const auto expectedChunks = params.totalFrames / params.framesPerBlock;
    const auto expectedPools = params.numZmw / (params.lanesPerPool * laneSize);
    EXPECT_EQ(ValidateData(Expected, source), expectedChunks * expectedPools);
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
INSTANTIATE_TEST_SUITE_P(MuliChunkMultiBatch,
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
