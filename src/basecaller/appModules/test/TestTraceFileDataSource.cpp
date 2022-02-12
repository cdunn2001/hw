// Copyright (c) 2021, Pacific Biosciences of California, Inc.
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

#include <pacbio/datasource/MallocAllocator.h>
#include <pacbio/datasource/ZmwFeatures.h>
#include <pacbio/dev/TemporaryDirectory.h>

#include <appModules/TraceFileDataSource.h>
#include <appModules/TraceSaver.h>
#include <dataTypes/configs/AnalysisConfig.h>

#include "MockExperimentData.h"

using namespace PacBio::Application;
using namespace PacBio::DataSource;
using namespace PacBio::File;
using namespace PacBio::Memory;
using namespace PacBio::Mongo;

namespace {

struct GeneratedTraceInfo
{
    std::vector<uint32_t> holeNumbers;
    std::vector<DataSourceBase::UnitCellProperties> properties;
    std::vector<uint32_t> batchIds;
};

GeneratedTraceInfo GenerateTraceFile(const std::string& name)
{
    const uint32_t numLanes = 12;
    const uint32_t numZmw = numLanes * laneSize;
    // We're going to hijack the TraceSaver to create a tracefile
    // for us, so we need a vector containing all lane indexes, so
    // that the TraceSaver is convinced to save the "full chip"
    std::vector<uint32_t> lanes(numLanes);
    std::iota(lanes.begin(), lanes.end(), 0);

    GeneratedTraceInfo params;

    params.holeNumbers.resize(numZmw);
    params.properties.resize(numZmw);
    for (size_t i = 0; i < numZmw; ++i)
    {
        params.holeNumbers[i] = i*3;
        params.properties[i].flags = ZmwFeatures::Sequencing;
        params.properties[i].type = 0;
        params.properties[i].x = i;
        params.properties[i].y = 2*i;
    }

    // Set up a fairly arbitrary grouping of ZMW into batches.
    params.batchIds.reserve(numZmw);
    // First pool has one batch, total 1 lanes so far
    uint32_t currPool = 1;
    params.batchIds.insert(params.batchIds.end(), laneSize, currPool);

    // second pool has two batches, total 3 lanes so far
    currPool += 2;
    params.batchIds.insert(params.batchIds.end(), laneSize*2, currPool);

    // next pool has three batches, total 6 lanes so far
    currPool += 2;
    params.batchIds.insert(params.batchIds.end(), laneSize*3, currPool);

    // final pool has 6 batches, total 12 lanes in all
    currPool += 2;
    params.batchIds.insert(params.batchIds.end(), laneSize*6, currPool);

    assert(params.batchIds.size() == numZmw);

    const size_t framesPerHdf5Chunk = 512;
    const size_t zmwsPerHdf5Chunk = laneSize;
    // Instantiate this, to force the file's creation on disk
    // We don't really care that the trace data won't be
    // populated, we mostly just want something with batchIds
    TraceSaverBody tmp(name,
                       512,
                       DataSourceBase::LaneSelector(lanes),
                       framesPerHdf5Chunk,
                       zmwsPerHdf5Chunk,
                       TraceDataType::INT16,
                       params.holeNumbers,
                       params.properties,
                       params.batchIds,
                       MockExperimentData(),
                       Data::MockAnalysisConfig());
    (void)tmp;

    return params;
}

}

TEST(TraceFileDataSourceMisc, Replication)
{
    const uint32_t numZmwLanes = 275;
    const uint32_t numZmw = numZmwLanes * laneSize;
    const uint32_t lanesPerPool = 64;
    const uint32_t framesPerBlock = 512;
    const uint32_t numFrames = 1024;

    PacBio::Dev::TemporaryDirectory tempDir;
    const std::string traceFileName = tempDir.DirName()+"/test.trc.h5";
    const auto params = GenerateTraceFile(traceFileName);

    // We specifically want to read in a trace file with fewer ZMW
    // than we want, to utilize the replication mechanisms.
    EXPECT_NE(params.holeNumbers.size(), numZmw);

    Data::TraceReplication trcConfig;
    trcConfig.traceFile = traceFileName;
    trcConfig.numFrames = numFrames;
    trcConfig.numZmwLanes = numZmwLanes;

    PacketLayout layout(PacketLayout::BLOCK_LAYOUT_DENSE,
                        PacketLayout::INT16,
                        {lanesPerPool, framesPerBlock, laneSize});
    DataSourceBase::Configuration sourceConfig(layout, std::make_unique<MallocAllocator>());

    TraceFileDataSource source(std::move(sourceConfig), trcConfig);

    const auto& expMetadata = MockExperimentData();

    EXPECT_EQ(source.NumZmw(), numZmw);
    EXPECT_EQ(source.NumFrames(), numFrames);
    EXPECT_EQ(source.ImagePsfMatrix().num_elements(), expMetadata.chipInfo.imagePsf.num_elements());
    const size_t ip = source.ImagePsfMatrix().shape()[0]/2;
    EXPECT_FLOAT_EQ(source.ImagePsfMatrix()[ip][ip],
                    expMetadata.chipInfo.imagePsf[ip][ip]);
    EXPECT_EQ(source.CrosstalkFilterMatrix().num_elements(), expMetadata.chipInfo.xtalkCorrection.num_elements());
    const size_t cf = source.CrosstalkFilterMatrix().shape()[0]/2;
    EXPECT_FLOAT_EQ(source.CrosstalkFilterMatrix()[cf][cf],
                    expMetadata.chipInfo.xtalkCorrection[cf][cf]);
    EXPECT_EQ(source.Platform(), expMetadata.runInfo.Platform());
    EXPECT_EQ(source.InstrumentName(), expMetadata.runInfo.instrumentName);

    {
        // TraceReplication ignores the incoming hole numbers and just
        // sets them to 0-N (to make sure they all remain unique)
        const auto& ids = source.UnitCellIds();
        EXPECT_EQ(ids.size(), numZmw);
        auto expectIds = std::vector<uint32_t>(ids.size());
        std::iota(expectIds.begin(), expectIds.end(), 0);
        EXPECT_TRUE(std::equal(ids.begin(), ids.end(), expectIds.begin()));

        const auto& props = source.GetUnitCellProperties();
        EXPECT_TRUE(std::all_of(props.begin(), props.end(),
                                [](const auto& uc) { return uc.flags == ZmwFeatures::Sequencing &&
                                                            uc.type == 0; }));
    }

    // The layouts need to accurately span the specified number of ZMW
    const auto& layouts = source.PacketLayouts();
    EXPECT_EQ(numZmw, std::accumulate(layouts.begin(), layouts.end(), 0,
                                      [](size_t count, const auto& kv)
                                      {
                                          return count + kv.second.NumZmw();
                                      }));

    // The layouts for replication mode also should be "dense", that is each
    // batch is the requested size, save perhaps for a short runt at the end
    for (size_t i = 0; i < layouts.size(); ++i)
    {
        if (i == layouts.size() - 1)
            EXPECT_EQ(layouts.at(i).NumBlocks(), numZmwLanes % lanesPerPool);
        else
            EXPECT_EQ(layouts.at(i).NumBlocks(), lanesPerPool);
        EXPECT_EQ(layouts.at(i).NumFrames(), framesPerBlock);
        EXPECT_EQ(layouts.at(i).BlockWidth(), laneSize);
        EXPECT_EQ(layouts.at(i).Encoding(), PacketLayout::EncodingFormat::INT16);
        EXPECT_EQ(layouts.at(i).Type(), PacketLayout::PacketLayout::BLOCK_LAYOUT_DENSE);
    }

    // Now loop through all the generated data.  The actual data values are
    // trash (probably 0), but we just want to make sure the produced batches
    // match the layouts we promised to deliver
    for (const auto& chunk : source.AllChunks<int16_t>())
    {
        EXPECT_EQ(chunk.size(), layouts.size());
        for (const auto& batch : chunk)
        {
            EXPECT_EQ(batch.StorageDims().lanesPerBatch,
                      layouts.at(batch.Metadata().PoolId()).NumBlocks());
            EXPECT_EQ(batch.StorageDims().framesPerBatch, framesPerBlock);
            EXPECT_EQ(batch.StorageDims().laneWidth, laneSize);
        }
    }

    // Intentionally odd roi.  Every selected ZMW gets us a full lane,
    // which means the 13 gets us a lane, and 184-271 overlaps three lanes
    auto selected = source.SelectedLanesWithinROI({{13},{184,88}});
    ASSERT_EQ(selected.size(), 4);
    auto itr = selected.begin();
    EXPECT_EQ(*itr, 0);
    itr++;
    EXPECT_EQ(*itr, 2);
    itr++;
    EXPECT_EQ(*itr, 3);
    itr++;
    EXPECT_EQ(*itr, 4);
    itr++;
    EXPECT_EQ(itr, selected.end());

}

TEST(TraceFileDataSourceMisc, Reanalysis)
{
    PacBio::Dev::TemporaryDirectory tempDir;
    const std::string traceFileName = tempDir.DirName()+"/test.trc.h5";
    const auto params = GenerateTraceFile(traceFileName);

    Data::TraceReanalysis trcConfig;
    trcConfig.traceFile = traceFileName;

    // lanesPerPool should ultimately get rejected, the value doens't matter
    const size_t framesPerBlock = 512;
    PacketLayout layout(PacketLayout::BLOCK_LAYOUT_DENSE, PacketLayout::INT16,
                        {1, framesPerBlock, laneSize});
    DataSourceBase::Configuration sourceConfig(layout, std::make_unique<MallocAllocator>());

    TraceFileDataSource source(std::move(sourceConfig), trcConfig);

    const auto numZmw = source.NumZmw();
    EXPECT_EQ(numZmw, params.batchIds.size());

    const auto& expMetadata = MockExperimentData();

    EXPECT_EQ(source.ImagePsfMatrix().num_elements(), expMetadata.chipInfo.imagePsf.num_elements());
    const size_t ip = source.ImagePsfMatrix().shape()[0]/2;
    EXPECT_FLOAT_EQ(source.ImagePsfMatrix()[ip][ip],
                    expMetadata.chipInfo.imagePsf[ip][ip]);
    EXPECT_EQ(source.CrosstalkFilterMatrix().num_elements(), expMetadata.chipInfo.xtalkCorrection.num_elements());
    const size_t cf = source.CrosstalkFilterMatrix().shape()[0]/2;
    EXPECT_FLOAT_EQ(source.CrosstalkFilterMatrix()[cf][cf],
                    expMetadata.chipInfo.xtalkCorrection[cf][cf]);
    EXPECT_EQ(source.Platform(), expMetadata.runInfo.Platform());
    EXPECT_EQ(source.InstrumentName(), expMetadata.runInfo.instrumentName);

    {
        // Hole numbers should be preserved in reanalysis mode
        const auto& ids = source.UnitCellIds();
        EXPECT_EQ(ids.size(), numZmw);
        EXPECT_TRUE(std::equal(ids.begin(), ids.end(), params.holeNumbers.begin()));
        const auto& props = source.GetUnitCellProperties();
        EXPECT_TRUE(std::equal(props.begin(), props.end(), params.properties.begin(),
                               [](const auto& v1, const auto& v2) { return v1.x == v2.x; }));
        EXPECT_TRUE(std::equal(props.begin(), props.end(), params.properties.begin(),
                               [](const auto& v1, const auto& v2) { return v1.y == v2.y; }));
        EXPECT_TRUE(std::all_of(props.begin(), props.end(),
                                [](const auto& uc) { return uc.flags == static_cast<uint32_t>(ZmwFeatures::Sequencing) &&
                                                            uc.type == 0; }));
    }

    // Make sure the reported layouts span all ZMW
    const auto& layouts = source.PacketLayouts();
    EXPECT_EQ(numZmw, std::accumulate(layouts.begin(), layouts.end(), 0,
                                      [](size_t count, const auto& kv) { return count + kv.second.NumZmw(); }));

    {
        // Make sure that the layouts we're using have the
        // batchIds that were stored in the tracefile.  This is
        // one of the more important pieces of trace re-analysis,
        // because this allows us to reproduce the original DME
        // estimation schedule
        auto expectedBatches = params.batchIds;
        auto itr = std::unique(expectedBatches.begin(), expectedBatches.end());
        expectedBatches.erase(itr, expectedBatches.end());

        ASSERT_EQ(layouts.size(), expectedBatches.size());
        for (const auto batch : expectedBatches)
        {
            ASSERT_NO_THROW(layouts.at(batch));
        }
    }

    for (const auto& kv : layouts)
    {
        // Make sure the layout dimensions are as expected. In particular
        // we've already ensured the expected batchIDs are present, now
        // make sure that the ZMW counts with each ID is as expected
        EXPECT_EQ(kv.second.NumBlocks(), std::count(params.batchIds.begin(),
                                                    params.batchIds.end(),
                                                    kv.first) / laneSize);
        EXPECT_EQ(kv.second.NumFrames(), framesPerBlock);
        EXPECT_EQ(kv.second.BlockWidth(), laneSize);
        EXPECT_EQ(kv.second.Encoding(), PacketLayout::EncodingFormat::INT16);
        EXPECT_EQ(kv.second.Type(), PacketLayout::PacketLayout::BLOCK_LAYOUT_SPARSE);
    }

    // Loop through the (trash) data to make sure it has dimensions
    // that agree with the promised layouts.
    for (const auto& chunk : source.AllChunks<int16_t>())
    {
        EXPECT_EQ(chunk.size(), layouts.size());
        for (const auto& batch : chunk)
        {
            EXPECT_EQ(batch.StorageDims().lanesPerBatch,
                      layouts.at(batch.Metadata().PoolId()).NumBlocks());
            EXPECT_EQ(batch.StorageDims().framesPerBatch, framesPerBlock);
            EXPECT_EQ(batch.StorageDims().laneWidth, laneSize);
        }
    }

    // ROI for reanalysis is based off a list of hole numbers
    ASSERT_GT(params.holeNumbers.back(), params.holeNumbers.size());
    auto lane1 = static_cast<int>(params.holeNumbers.front());
    auto lane2 = static_cast<int>(params.holeNumbers.back());
    auto selected = source.SelectedLanesWithinROI({{lane1},{lane2}});
    ASSERT_EQ(selected.size(), 2);
    auto itr = selected.begin();
    EXPECT_EQ(*itr, 0);
    itr++;
    EXPECT_EQ(*itr, numZmw/laneSize-1);
    itr++;
    EXPECT_EQ(itr, selected.end());

}

TEST(TraceFileDataSourceMisc, ReanalysisWithWhitelist)
{
    PacBio::Dev::TemporaryDirectory tempDir;
    const std::string traceFileName = tempDir.DirName()+"/test.trc.h5";
    const auto params = GenerateTraceFile(traceFileName);

    Data::TraceReanalysis trcConfig;
    trcConfig.traceFile = traceFileName;
    // We only want two lanes
    trcConfig.whitelist.push_back(params.holeNumbers[4]);
    trcConfig.whitelist.push_back(params.holeNumbers.back());

    // lanesPerPool should ultimately get rejected, the value doens't matter
    PacketLayout layout(PacketLayout::BLOCK_LAYOUT_DENSE, PacketLayout::INT16,
                        {1, 512, laneSize});
    DataSourceBase::Configuration sourceConfig(layout, std::make_unique<MallocAllocator>());

    TraceFileDataSource source(std::move(sourceConfig), trcConfig);

    // Only the two requested lanes should be present now
    const size_t framesPerBlock = 512;
    const auto numZmw = source.NumZmw();
    EXPECT_NE(numZmw, params.batchIds.size());
    EXPECT_EQ(numZmw, laneSize*2);

    const auto& expMetadata = MockExperimentData();

    {
        EXPECT_EQ(source.ImagePsfMatrix().num_elements(), expMetadata.chipInfo.imagePsf.num_elements());
        const size_t ip = source.ImagePsfMatrix().shape()[0]/2;
        EXPECT_FLOAT_EQ(source.ImagePsfMatrix()[ip][ip],
                        expMetadata.chipInfo.imagePsf[ip][ip]);
        EXPECT_EQ(source.CrosstalkFilterMatrix().num_elements(), expMetadata.chipInfo.xtalkCorrection.num_elements());
        const size_t cf = source.CrosstalkFilterMatrix().shape()[0]/2;
        EXPECT_FLOAT_EQ(source.CrosstalkFilterMatrix()[cf][cf],
                        expMetadata.chipInfo.xtalkCorrection[cf][cf]);
        EXPECT_EQ(source.Platform(), expMetadata.runInfo.Platform());
        EXPECT_EQ(source.InstrumentName(), expMetadata.runInfo.instrumentName);
    }

    {
        // Make sure hole numbers are preserved, even though we are reading
        // a subset of the tracefile
        const auto& ids = source.UnitCellIds();
        EXPECT_EQ(ids.size(), numZmw);
        EXPECT_TRUE(std::equal(ids.begin(), ids.begin() + laneSize, params.holeNumbers.begin()));
        EXPECT_TRUE(std::equal(ids.end() - laneSize, ids.end(), params.holeNumbers.end() - laneSize));

        const auto& props = source.GetUnitCellProperties();
        EXPECT_EQ(props.size(), numZmw);
        EXPECT_TRUE(std::equal(props.begin(), props.begin() + laneSize, params.properties.begin(),
                               [](const auto& v1, const auto& v2) {return v1.x == v2.x;}));
        EXPECT_TRUE(std::equal(props.begin(), props.begin() + laneSize, params.properties.begin(),
                               [](const auto& v1, const auto& v2) {return v1.y == v2.y;}));
        EXPECT_TRUE(std::equal(props.end() - laneSize, props.end(), params.properties.end() - laneSize,
                               [](const auto& v1, const auto& v2) {return v1.x == v2.x;}));
        EXPECT_TRUE(std::equal(props.end() - laneSize, props.end(), params.properties.end() - laneSize,
                               [](const auto& v1, const auto& v2) {return v1.y == v2.y;}));
        EXPECT_TRUE(std::all_of(props.begin(), props.end(),
                                [](const auto& uc) { return uc.flags == ZmwFeatures::Sequencing &&
                                                            uc.type == 0; }));
    }

    const auto& layouts = source.PacketLayouts();
    EXPECT_EQ(numZmw, std::accumulate(layouts.begin(), layouts.end(), 0,
                                      [](size_t count, const auto& kv) { return count + kv.second.NumZmw(); }));

    {
        auto expectedBatches = params.batchIds;
        auto itr = std::unique(expectedBatches.begin(), expectedBatches.end());
        expectedBatches.erase(itr, expectedBatches.end());

        ASSERT_EQ(layouts.size(), 2);
        ASSERT_NO_THROW(layouts.at(params.batchIds.front()));
        ASSERT_NO_THROW(layouts.at(params.batchIds.back()));
    }

    for (const auto& kv : layouts)
    {
        EXPECT_EQ(kv.second.NumBlocks(), 1);
        EXPECT_EQ(kv.second.NumFrames(), framesPerBlock);
        EXPECT_EQ(kv.second.BlockWidth(), laneSize);
        EXPECT_EQ(kv.second.Encoding(), PacketLayout::EncodingFormat::INT16);
        EXPECT_EQ(kv.second.Type(), PacketLayout::PacketLayout::BLOCK_LAYOUT_SPARSE);
    }

    for (const auto& chunk : source.AllChunks<int16_t>())
    {
        EXPECT_EQ(chunk.size(), layouts.size());
        for (const auto& batch : chunk)
        {
            EXPECT_EQ(batch.StorageDims().lanesPerBatch,
                      layouts.at(batch.Metadata().PoolId()).NumBlocks());
            EXPECT_EQ(batch.StorageDims().framesPerBatch, framesPerBlock);
            EXPECT_EQ(batch.StorageDims().laneWidth, laneSize);
        }
    }
}

struct TestParams
{
    bool cacheInput;
    std::string traceFile;
    // The number of values the test expects to skip because the value doesn't fit
    // in a uint8_t
    uint32_t expectedSkip;
};
struct TestTraceFileDataSource : testing::TestWithParam<TestParams> {};


TEST_P(TestTraceFileDataSource, Read8And16)
{
    PacBio::Logging::LogSeverityContext severity(PacBio::Logging::LogLevel::ERROR);

    const uint32_t lanesPerPool = 64;
    const uint32_t framesPerBlock = 512;

    PacketLayout layout16(PacketLayout::BLOCK_LAYOUT_DENSE, PacketLayout::INT16,
                        {lanesPerPool, framesPerBlock, laneSize});
    PacketLayout layout8(PacketLayout::BLOCK_LAYOUT_DENSE, PacketLayout::UINT8,
                        {lanesPerPool, framesPerBlock, laneSize});

    DataSourceBase::Configuration cfg16(layout16, std::make_unique<MallocAllocator>());
    DataSourceBase::Configuration cfg8(layout8, std::make_unique<MallocAllocator>());

    Data::TraceReplication trcConfig;
    trcConfig.traceFile = GetParam().traceFile;
    trcConfig.numFrames = framesPerBlock;
    trcConfig.numZmwLanes = lanesPerPool;
    trcConfig.cache = GetParam().cacheInput;
    trcConfig.inputType = Data::TraceInputType::INT16;

    TraceFileDataSource source16(std::move(cfg16), trcConfig);

    trcConfig.inputType = Data::TraceInputType::UINT8;
    TraceFileDataSource source8(std::move(cfg8), trcConfig);

    while (!source16.ChunksReady()) source16.ContinueProcessing();
    while (!source8.ChunksReady()) source8.ContinueProcessing();

    SensorPacketsChunk chunk16;
    SensorPacketsChunk chunk8;

    source16.PopChunk(chunk16, std::chrono::milliseconds{100});
    source8.PopChunk(chunk8, std::chrono::milliseconds{100});

    ASSERT_EQ(chunk16.NumPackets(), 1);
    ASSERT_EQ(chunk8.NumPackets(), 1);

    SensorPacket& packet16 = *chunk16.begin();
    SensorPacket& packet8 = *chunk8.begin();

    ASSERT_EQ(packet8.Layout().BlockWidth(), packet16.Layout().BlockWidth());
    ASSERT_EQ(packet8.Layout().NumBlocks(), packet16.Layout().NumBlocks());
    ASSERT_EQ(packet8.Layout().NumFrames(), packet16.Layout().NumFrames());
    ASSERT_EQ(packet8.Layout().Type(), PacketLayout::BLOCK_LAYOUT_DENSE);
    ASSERT_EQ(packet16.Layout().Type(), PacketLayout::BLOCK_LAYOUT_DENSE);
    ASSERT_EQ(packet8.Layout().Encoding(), PacketLayout::UINT8);
    ASSERT_EQ(packet16.Layout().Encoding(), PacketLayout::INT16);

    ASSERT_EQ(packet16.BytesInBlock(), 2*packet8.BytesInBlock());

    size_t skipped = 0;
    size_t checked = 0;
    for (uint32_t i = 0; i < packet16.Layout().NumBlocks(); ++i)
    {
        uint8_t* data8 = packet8.BlockData(i).Data();
        int16_t* data16 = reinterpret_cast<int16_t*>(packet16.BlockData(i).Data());

        for (uint32_t j = 0; j < packet8.BytesInBlock(); ++j)
        {
            if (data16[j] < 0)
            {
                EXPECT_EQ(data8[j], 0);
                skipped++;
            } else if (data16[j] > 255)
            {
                EXPECT_EQ(data8[j], 255);
                skipped++;
            } else {
                EXPECT_EQ(data8[j], static_cast<uint8_t>(data16[j]))
                    << j << " " << data8[j] << " " << data16[j];
                checked++;
            }
        }
    }
    ASSERT_GT(checked, skipped);
    // Make sure the expected number of values were checked
    EXPECT_EQ(skipped, GetParam().expectedSkip);
}

INSTANTIATE_TEST_SUITE_P(,
                         TestTraceFileDataSource,
                         ::testing::Values(TestParams{true,std::string{"/pbi/dept/primary/unitTestInput/mongo/test.trc.h5"}, 132784},
                                           TestParams{false,std::string{"/pbi/dept/primary/unitTestInput/mongo/test.trc.h5"}, 132784}),
                                           [](const testing::TestParamInfo<TestParams>& info) {
                             std::string ret;
                             if (info.param.cacheInput) ret = "Cached";
                             else ret = "NotCached";
                             if (info.param.traceFile.find("test.trc.h5") != std::string::npos) ret += "16BitInput";
                             else ret += "8BitInput";
                             return ret;
                         });
