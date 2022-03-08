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

#include "FrameAnalyzer.h"

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

#include "PaCalConfig.h"
#include "PaCalConstants.h"
#include "SignalSimulator.h"


using namespace PacBio::Calibration;
using namespace PacBio::DataSource;


TEST(FrameAnalyzer, OneBatch)
{
    typedef int16_t T;

    size_t numBlocks       = 1;
    size_t framesPerBlock  = 512;
    size_t chipWidth       = 16;  // 16 * 4 = 64 == laneSize

                                /*  lanesPerPool  framesPerBlock     laneSize  */
    std::array<size_t, 3> layoutDims =    { numBlocks,      framesPerBlock,    laneSize };
    PacketLayout layout{PacketLayout::BLOCK_LAYOUT_DENSE, PacketLayout::INT16, layoutDims};

    auto numZmw = layout.NumZmw();
    DataSourceBase::UnitCellProperties prop = { ZmwFeatures::Sequencing, 0, 0, 0 };
    std::vector<DataSourceBase::UnitCellProperties> cellProps(numZmw, prop);

    for (size_t z = 0; z < numZmw; ++z)
    {
        auto lda = chipWidth;
        auto coord = std::make_pair(z / lda, z % lda);  // Same as SignalSimulator::Id2Coords
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
        boost::multi_array_ref<T, 2> blockData(
            reinterpret_cast<T*>(batchData.BlockData(i).Data()),
            boost::extents[zmwPerBlock][framesPerBlock],
            boost::fortran_storage_order());

        for (size_t z = 0; z < zmwPerBlock; ++z)
        {
            std::uniform_int_distribution<int16_t> metadist(0, numZmw);
            cellDists.emplace_back(std::make_pair(metadist(gnr), metadist(gnr) / 5 + 2));

            std::normal_distribution<> dist(cellDists.back().first, cellDists.back().second);
            std::generate(blockData[z].begin(), blockData[z].end(), std::bind(dist, gnr));
        }
    }

    auto chunk = SensorPacketsChunk(chunkIdx * layout.NumFrames(),
                                    (chunkIdx + 1) * layout.NumFrames());
    chunk.SetZmwRange(0, numZmw);
    chunk.AddPacket(std::move(batchData));
    assert(chunk.HasFullPacketCoverage());

    const auto& stats = AnalyzeChunk(chunk, cellProps);

    for (size_t z = 0; z < numZmw; ++z)
    {
        auto [ x, y ] = std::make_pair(cellProps[z].x, cellProps[z].y);
        auto [ expMean, expStd ] = cellDists[z];
        auto [ actMean, actVar ] = std::make_pair(stats.mean[x][y], stats.variance[x][y]);
        
        EXPECT_NEAR(actMean, expMean,        6 * expStd / std::sqrt(framesPerBlock) + expStd / 2);
        EXPECT_NEAR(actVar,  expStd*expStd,  6 * expStd);
    }

}