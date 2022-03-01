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

#include <vector>

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <Eigen/Core>

#include <pacbio/datasource/DataSourceBase.h>
#include <pacbio/datasource/ZmwFeatures.h>
#include <pacbio/datasource/MallocAllocator.h>

#include "PaCalConfig.h"
#include "PaCalConstants.h"
#include "SignalSimulator.h"

typedef Eigen::Matrix<int16_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXs;

using namespace testing;

using namespace PacBio::Calibration;
using namespace PacBio::DataSource;

struct TestingParams
{
    size_t framesPerBlock;
    size_t lanesPerPool;
    size_t totalFrames;
    size_t numZmw;
    size_t nRows = 0;
    size_t nCols = 0;

    // Need this so gtest prints something sensible on test failure
    friend std::ostream& operator<<(std::ostream& os, const TestingParams& param)
    {
        os << std::endl;
        os << std::endl << "chip size:         " << param.nRows << " x " << param.nCols;
        os << std::endl << "framesPerBlock:    " << param.framesPerBlock;
        os << std::endl << "lanesPerPool:      " << param.lanesPerPool;
        os << std::endl << "numZmw:            " << param.numZmw;
        os << std::endl << "totalFrames:       " << param.totalFrames;
        os << std::endl;
        return os;
    }

};

class SimDataSourceTest : public testing::TestWithParam<TestingParams> {};

TEST_P(SimDataSourceTest, Constant)
{
    SimInputConfig simConf;

    auto params = GetParam();
    params.nRows = params.nRows != 0 ? params.nRows : simConf.nRows;
    params.nCols = params.nCols != 0 ? params.nCols : simConf.nCols;

    simConf.nRows = params.nRows;
    simConf.nCols = params.nCols;

    PacketLayout layout(PacketLayout::BLOCK_LAYOUT_DENSE, PacketLayout::INT16,
                        {params.lanesPerPool, params.framesPerBlock, laneSize});

    DataSourceBase::Configuration cfg(layout, std::make_unique<MallocAllocator>());
    cfg.numFrames = params.totalFrames;
    DataSourceSimulator source(std::move(cfg), std::move(simConf));

    SensorPacketsChunk chunk;
    size_t expectedNextStartFrame = 0;
    size_t numValidChunks = 0;
    while (source.IsRunning())
    {
        source.ContinueProcessing();
        if (source.PopChunk(chunk, std::chrono::milliseconds(10)))
        {
            const bool validLayout = (chunk.IsValid() && expectedNextStartFrame == chunk.StartFrame());
            expectedNextStartFrame = chunk.StopFrame();
            EXPECT_TRUE(validLayout) << "Failed layout validation, skipping data validation";
            if (validLayout) numValidChunks++;

            size_t numValidPackets = 0;
            size_t numPackets = chunk.NumPackets();
            for (SensorPacket& packet : chunk)
            {
                size_t startZmw       = packet.StartZmw();
                PacketLayout pkLayout = packet.Layout();

                size_t blkFrames     = pkLayout.NumFrames();
                size_t zmwPerBlock   = pkLayout.BlockWidth();
                for (size_t i = 0; i < pkLayout.NumBlocks(); ++i)
                {
                    SensorPacket::DataView blockView = packet.BlockData(i);
                    auto dataPtr = reinterpret_cast<int16_t*>(blockView.Data());
                    Eigen::Map<MatrixXs> mapView(dataPtr, blkFrames, zmwPerBlock);
                    Eigen::MatrixXf blkm = mapView.cast<float>();

                    auto blkMean = blkm.colwise().mean();
                    auto blkVar1  = (blkm.rowwise() - blkMean).array().square().colwise().sum();
                    auto blkVar = blkVar1 / (blkFrames - 1);
                    assert(size_t(blkMean.cols()) == zmwPerBlock);

                    for (size_t z = 0; z < zmwPerBlock; ++z)
                    {
                        size_t zmwIdx = startZmw + i * zmwPerBlock + z;
                        auto [ expMean, expStd ] = source.Id2Norm(zmwIdx);
                        auto [ actMean, actVar ] = std::make_pair(blkMean(z), blkVar(z));

                        EXPECT_NEAR(actMean, expMean, 3*std::sqrt(actVar / blkFrames));
                        EXPECT_NEAR(actVar,  expStd*expStd, 3*expStd);
                    }
                }

                numValidPackets++;
            }

            EXPECT_TRUE(numPackets == numValidPackets) << "Failed packet validation.";
        }
    }
}

#if 0
// For debugging purposes
INSTANTIATE_TEST_SUITE_P(TinyBlock,
                         SimDataSourceTest,
                         testing::Values(TestingParams{
                                 48,   /* framesPerBlock */
                                 1,    /* lanesPerPool   */
                                 48,   /* totalFrames    */
                                 24,   /* numZmw         */
                                 8, 8  /* nRown x nCols */
}));
#endif

INSTANTIATE_TEST_SUITE_P(SingleBlock,
                         SimDataSourceTest,
                         testing::Values(TestingParams{
                                 512,  /* framesPerBlock */
                                 1,    /* lanesPerPool   */
                                 512,  /* totalFrames    */
                                 64,   /* numZmw         */
}));

INSTANTIATE_TEST_SUITE_P(QuadBlocks,
                         SimDataSourceTest,
                         testing::Values(TestingParams{
                                 512,  /* framesPerBlock */
                                 1,    /* lanesPerPool   */
                                 512,  /* totalFrames    */
                                 256,  /* numZmw         */
}));

// upgrade to a single batch, which will contain 4 blocks
INSTANTIATE_TEST_SUITE_P(SingleBatch,
                         SimDataSourceTest,
                         testing::Values(TestingParams{
                                 512,  /* framesPerBlock */
                                 4,    /* lanesPerPool   */
                                 512,  /* totalFrames    */
                                 256 , /* numZmw         */
}));

INSTANTIATE_TEST_SUITE_P(MultiChunkMultiBatch,
                         SimDataSourceTest,
                         testing::Values(TestingParams{
                                 512,  /* framesPerBlock */
                                 4,    /* lanesPerPool */
                                 1024, /* totalFrames */
                                 256 , /* numZmw */
}));
