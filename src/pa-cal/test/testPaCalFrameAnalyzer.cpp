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

#include <random>
#include <vector>
#include <algorithm>
#include <functional>

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <Eigen/Core>

#include <pacbio/datasource/DataSourceBase.h>
#include <pacbio/datasource/ZmwFeatures.h>
#include <pacbio/datasource/MallocAllocator.h>

#include <pa-cal/PaCalConfig.h>
#include <pa-cal/PaCalConstants.h>
#include <pa-cal/FrameAnalyzer.h>
#include <pa-cal/SignalSimulator.h>


using namespace PacBio::Calibration;
using namespace PacBio::DataSource;


static void testDist(size_t z, float expMean, float actMean, float expVar, float actVar, size_t n)
{
    float confSigma;
   
    confSigma = 4.5;
    auto stt = (expMean - actMean) / std::sqrt(actVar / n);
    EXPECT_LE(std::abs(stt), confSigma) << "z: " << z << std::endl;

    confSigma = 3.5;
    auto varTest =  (expVar - actVar) / actVar / sqrt(2.0 / n);
    EXPECT_LT(std::abs(varTest), confSigma) << "z: " << z << std::endl;
}


template<typename T>
void TestAnalyzeChunk(PacketLayout::EncodingFormat dataType, int16_t pedestal)
{
    size_t numBlocks       = 3;
    size_t framesPerBlock  = 512;
    size_t chipWidth       = 16;  // 16 * 4 = 64 == laneSize

                                          /*  lanesPerPool  framesPerBlock     laneSize  */
    std::array<size_t, 3> layoutDims =    { numBlocks,      framesPerBlock,    laneSize };
    PacketLayout layout{PacketLayout::BLOCK_LAYOUT_DENSE, dataType, layoutDims};

    auto numZmw = layout.NumZmw();
    DataSourceBase::UnitCellProperties prop = { ZmwFeatures::Sequencing, 0, 0, 0 };
    std::vector<DataSourceBase::UnitCellProperties> cellProps(numZmw, prop);

    for (size_t z = 0; z < numZmw; ++z)
    {
        auto lda = chipWidth;    // Same as SignalSimulator::Id2Coords
        auto coord = std::make_pair(z / lda, z % lda);
        cellProps[z].x = coord.first;
        cellProps[z].y = coord.second;
    }

    size_t chunkIdx       = 0;
    size_t batchIdx       = 0;
    size_t startZmw       = 0;
    size_t startFrame     = 0;
    auto alloc = std::make_unique<MallocAllocator>();
    SensorPacket batchData(layout, batchIdx, startZmw, startFrame, *alloc);

    std::default_random_engine gnr;
    std::vector<std::pair<int16_t, int16_t>> cellDists;
    for (size_t i = 0; i < numBlocks; ++i)
    {
        auto zmwPerBlock    = layout.BlockWidth();
        boost::multi_array_ref<T, 2> blkData(
            reinterpret_cast<T*>(batchData.BlockData(i).Data()),
            boost::extents[zmwPerBlock][framesPerBlock],
            boost::fortran_storage_order());

        for (size_t z = 0; z < zmwPerBlock; ++z)
        {
            std::uniform_int_distribution<int16_t> metadist(0, 64);
            auto mdist = std::make_pair(96 + metadist(gnr), metadist(gnr) % 13 + 4);
            cellDists.emplace_back(mdist);

            std::normal_distribution<> dist(mdist.first, mdist.second);
            std::generate(blkData[z].begin(), blkData[z].end(), std::bind(dist, gnr));
            std::for_each(blkData[z].begin(), blkData[z].end(), [=](auto& x) { x += pedestal; });
        }
    }

    auto chunk = SensorPacketsChunk(chunkIdx * layout.NumFrames(),
                                    (chunkIdx + 1) * layout.NumFrames());
    chunk.SetZmwRange(0, numZmw);
    chunk.AddPacket(std::move(batchData));
    ASSERT_TRUE(chunk.HasFullPacketCoverage());

    const auto& stats = AnalyzeChunk(chunk, pedestal, cellProps);

    for (size_t z = 0; z < numZmw; ++z)
    {
        auto [ x, y ]            = std::make_pair(cellProps[z].x, cellProps[z].y);
        auto [ expMean, expStd ] = cellDists[z];
        auto [ actMean, actVar ] = std::make_pair(stats.mean[x][y], stats.variance[x][y]);

        testDist(z, expMean, actMean, expStd*expStd, actVar, framesPerBlock);
    }
}

void TestSimAnalyze(PacketLayout::EncodingFormat dataType, int16_t pedestal)
{
    size_t numBlocks       = 4;
    size_t framesPerBlock  = 512;
                                          /*  lanesPerPool  framesPerBlock     laneSize  */
    std::array<size_t, 3> layoutDims =    { numBlocks,      framesPerBlock,    laneSize };
    PacketLayout layout{PacketLayout::BLOCK_LAYOUT_DENSE, dataType, layoutDims};

    SimInputConfig simConf;
    simConf.nRows = 36;
    simConf.nCols = 32;
    simConf.Pedestal = pedestal;

    DataSourceBase::Configuration cfg(layout, std::make_unique<MallocAllocator>());
    cfg.numFrames = framesPerBlock;

    DataSourceSimulator source(std::move(cfg), std::move(simConf));

    source.Start();
    auto cellProps = source.GetUnitCellProperties();

    while (source.IsRunning())
    {
        source.ContinueProcessing();
        SensorPacketsChunk chunk;
        if (source.PopChunk(chunk, std::chrono::milliseconds(10)))
        {
            const auto& stats = AnalyzeChunk(chunk, pedestal, cellProps);

            for (size_t z = chunk.StartZmw(); z < chunk.StopZmw(); ++z)
            {
                auto [ x, y ]            = std::make_pair(cellProps[z].x, cellProps[z].y);
                auto [ expMean, expStd ] = source.Id2Norm(z);
                auto [ actMean, actVar ] = std::make_pair(stats.mean[x][y], stats.variance[x][y]);

                testDist(z, expMean, actMean + pedestal, expStd*expStd, actVar, framesPerBlock);
            }
        }
    }
}


class AnalyzeChunkTest : public testing::TestWithParam<int16_t> {};

TEST_P(AnalyzeChunkTest, OneBatch16)
{
    TestAnalyzeChunk<int16_t>(PacketLayout::INT16, GetParam());
}

TEST_P(AnalyzeChunkTest, OneBatch8)
{
    TestAnalyzeChunk<uint8_t>(PacketLayout::UINT8, GetParam());
}

TEST_P(AnalyzeChunkTest, SimPackets16)
{
    TestSimAnalyze(PacketLayout::INT16, GetParam());
}

TEST_P(AnalyzeChunkTest, SimPackets8)
{
    TestSimAnalyze(PacketLayout::UINT8, GetParam());
}

INSTANTIATE_TEST_SUITE_P(FrameAnalyzerSuite, AnalyzeChunkTest, testing::Values(-10, 0, 7));
