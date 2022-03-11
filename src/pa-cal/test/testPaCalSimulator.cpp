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

#include <pa-cal/PaCalConfig.h>
#include <pa-cal/PaCalConstants.h>
#include <pa-cal/SignalSimulator.h>


// Eigen::Matrix has fortran memory layout by default
// Unsigned one byte data is interpreted as _signed_ int8 to simplify 
// conversion to and comparison with floating point results
typedef Eigen::Matrix<int16_t, Eigen::Dynamic, Eigen::Dynamic> MatrixXs;
typedef Eigen::Matrix<int8_t, Eigen::Dynamic, Eigen::Dynamic>  MatrixXb;

using namespace testing;

using namespace PacBio::Calibration;
using namespace PacBio::DataSource;

struct TestingParams
{
    size_t framesPerBlock;
    size_t lanesPerPool;
    size_t totalFrames;
    size_t nRows = 0;
    size_t nCols = 0;
    size_t numBits = 16;
    double delay = 0.001;

    // Need this so gtest prints something sensible on test failure
    friend std::ostream& operator<<(std::ostream& os, const TestingParams& param)
    {
        os << std::endl;
        os << std::endl << "chip size:         " << param.nRows << " x " << param.nCols;
        os << std::endl << "delay:             " << param.delay;
        os << std::endl << "framesPerBlock:    " << param.framesPerBlock;
        os << std::endl << "lanesPerPool:      " << param.lanesPerPool;
        os << std::endl << "totalFrames:       " << param.totalFrames;
        os << std::endl;
        return os;
    }
};

Eigen::MatrixXf convM2Float(Eigen::Map<const MatrixXs>& map)
{
    return map.cast<float>();
}
Eigen::MatrixXf convM2Float(Eigen::Map<const MatrixXb>& map)
{
    MatrixXb map2 = (map.array() - UINT8_MAX - 1);
    return map2.cast<float>();
}

template<typename M, typename S> // MatrixXs or MatrixXb, scalar has to be separate
void ValidatePacket(const DataSourceSimulator& source, const SensorPacket& packet)
{
    size_t startZmw       = packet.StartZmw();
    PacketLayout pkLayout = packet.Layout();
    size_t framesPerBlock = pkLayout.NumFrames();
    size_t zmwPerBlock    = pkLayout.BlockWidth();

    for (size_t i = 0; i < pkLayout.NumBlocks(); ++i)
    {
        // Setup source data
        SensorPacket::ConstDataView blockView = packet.BlockData(i);
        auto srcDataPtr = reinterpret_cast<const S*>(blockView.Data());
        Eigen::Map<const M> srcBlockMap(srcDataPtr, zmwPerBlock, framesPerBlock);

        // Find moments
        Eigen::MatrixXf blkm = convM2Float(srcBlockMap);
        auto blkMean = blkm.rowwise().mean();
        auto blkVar1 = (blkm.colwise() - blkMean).array().square().rowwise().sum();
        auto blkVar = blkVar1 / (framesPerBlock - 1);
        EXPECT_EQ(blkMean.rows(), zmwPerBlock);

        for (size_t z = 0; z < zmwPerBlock; ++z)
        {
            size_t zmwIdx = startZmw + i * zmwPerBlock + z;
            auto [ expMean, expStd ] = source.Id2Norm(zmwIdx);
            auto [ actMean, actStd ] = std::make_pair(blkMean(z), sqrt(blkVar(z)));

            EXPECT_NEAR(actMean, expMean, 3*expStd*std::sqrt(framesPerBlock));
            EXPECT_NEAR(actStd*actStd,  expStd*expStd, 3*expStd);
        }
    }
}


class DataSourceSimTest : public testing::TestWithParam<TestingParams> {};

TEST_P(DataSourceSimTest, Chip)
{
    auto params = GetParam();

    SimInputConfig simConf;
    simConf.nRows = params.nRows != 0 ? params.nRows : simConf.nRows;
    simConf.nCols = params.nCols != 0 ? params.nCols : simConf.nCols;
    simConf.minInputDelaySeconds
                  = params.delay != 0 ? params.delay : simConf.minInputDelaySeconds;
    auto dataType = params.numBits == 16 ? PacketLayout::INT16 : PacketLayout::UINT8;

    PacketLayout layout(PacketLayout::BLOCK_LAYOUT_DENSE, dataType,
                        {params.lanesPerPool, params.framesPerBlock, laneSize});

    DataSourceBase::Configuration cfg(layout, std::make_unique<MallocAllocator>());
    cfg.numFrames = params.totalFrames;
    DataSourceSimulator source(std::move(cfg), std::move(simConf));

    auto validateBatch = (layout.Encoding() == PacketLayout::INT16) ?
        ValidatePacket<MatrixXs, MatrixXs::Scalar> : ValidatePacket<MatrixXb, MatrixXb::Scalar>;

    SensorPacketsChunk chunk;
    size_t expNextStartFrame = 0;

    source.Start();
    while (source.IsRunning())
    {
        source.ContinueProcessing();
        if (source.PopChunk(chunk, std::chrono::milliseconds(10)))
        {
            EXPECT_TRUE(chunk.IsValid()) << "Chunk is not valid";
            EXPECT_TRUE(expNextStartFrame == chunk.StartFrame()) << "Unexpected start frame";

            expNextStartFrame = chunk.StopFrame();
            
            for (SensorPacket& packet : chunk)
            {
                validateBatch(source, packet);
            }
        }
    }
}

#if 0
// For debugging purposes
INSTANTIATE_TEST_SUITE_P(TinyBlock,
                         DataSourceSimTest,
                         testing::Values(TestingParams{
                                32,             /* framesPerBlock */
                                1,              /* lanesPerPool   */
                                32,             /* totalFrames    */
                                8, 8,           /* nRown x nCols  */
                                8
}));
#endif // 0

const auto simSweep08 = ::testing::Values(
    /* framesPerBlock  lanesPerPool  totalFrames  nRown x nCols bits */
    TestingParams { 512,     1,      512,      4, 16,     8 },
    TestingParams { 512,     1,      512,     12, 16,     8 },
    TestingParams { 512,     4,      512,     32,  8,     8 },
    TestingParams { 512,    32,     1024,     64, 64,     8 },
    TestingParams { 512,   2048,    2048,   256, 256,     8 }
);
INSTANTIATE_TEST_SUITE_P(BlockSuite08, DataSourceSimTest, simSweep08);

const auto simSweep16 = ::testing::Values(
    /* framesPerBlock  lanesPerPool  totalFrames  nRown x nCols bits */
    TestingParams { 512,     1,      512,      4, 16,    16 },
    TestingParams { 512,     1,      512,     12, 16,    16 },
    TestingParams { 512,     4,      512,     32,  8,    16 },
    TestingParams { 512,    32,     1024,     64, 64,    16 },
    TestingParams { 512,   4096,    2048,   256, 256,    16 }
);
INSTANTIATE_TEST_SUITE_P(BlockSuite16, DataSourceSimTest, simSweep16);

const auto simSweepTO = ::testing::Values(
    /* framesPerBlock  lanesPerPool  totalFrames  nRown x nCols bits sec */
    TestingParams { 512,     1,      512,      4, 16,    16,    2 }
);
INSTANTIATE_TEST_SUITE_P(SimulateTimeout, DataSourceSimTest, simSweepTO);
