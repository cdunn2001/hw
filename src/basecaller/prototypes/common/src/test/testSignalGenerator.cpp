
#include <gtest/gtest.h>
#include <random>
#include <thread>

#include <common/ZmwDataManager.h>
#include <common/DataGenerators/SignalGenerator.h>
#include <pacbio/primary/SequelTraceFile.h>

using namespace PacBio::Cuda;
using namespace PacBio::Cuda::Data;
using namespace PacBio::Cuda::Memory;

TEST(SignalGeneratorTest, Construct)
{
    static constexpr size_t zmwLaneWidth = 64;
    const std::string traceFileName = "/pbi/dept/primary/sim/spider/designer_spider1p0NTO_fv2p4_SNR-50.trc.h5";

    auto dataParams = DataManagerParams()
        .LaneWidth(zmwLaneWidth);

    auto traceParams = TraceFileParams()
            .TraceFileName(traceFileName);

    SignalGenerator sg(dataParams, traceParams);
}

TEST(SignalGeneratorTest, CompareData)
{
    using PacBio::Primary::SequelTraceFileHDF5;
    using PacBio::Primary::Tile;
    using PacBio::Primary::zmwsPerTranche;

    static constexpr size_t zmwLaneWidth = 64;

    auto dataParams = DataManagerParams()
            .NumBlocks(4)
            .BlockLength(64)
            .NumZmwLanes(20000)
            .KernelLanes(5000)
            .LaneWidth(zmwLaneWidth);

    auto traceParams = TraceFileParams()
            .TraceFileName("/pbi/dept/primary/sim/spider/designer_spider1p0NTO_fv2p4_SNR-50.trc.h5");

    ZmwDataManager<int16_t> manager(dataParams,
                                    std::make_unique<SignalGenerator>(dataParams, traceParams),
                                    true);

    static constexpr size_t numLanesToCheck = 16;
    std::random_device rd;
    std::mt19937 eng(rd());
    std::uniform_int_distribution<> distr(0, dataParams.kernelLanes);

    SequelTraceFileHDF5 traceIn(traceParams.traceFileName);
    std::vector<int16_t> truth(dataParams.laneWidth * dataParams.blockLength);

    while (manager.MoreData())
    {
        auto data = manager.NextBatch();
        auto frameOffset = data.FirstFrame();
        auto batchIdx = data.Batch();
        auto& in = data.KernelInput();

        for (size_t laneCheck = 0; laneCheck < numLanesToCheck; laneCheck++)
        {
            size_t lane = distr(eng);
            auto blockView = in.GetBlockView(lane);
            std::vector<int16_t> expectedPixelSums(zmwLaneWidth, 0);
            std::vector<int16_t> pixelSums(zmwLaneWidth, 0);

            const size_t zmwOffset = batchIdx * dataParams.laneWidth * dataParams.kernelLanes + lane * dataParams.laneWidth;
            traceIn.Read1CZmwLaneSegment(zmwOffset, dataParams.laneWidth, frameOffset, dataParams.blockLength, truth.data());
            for (size_t frame = 0; frame < dataParams.blockLength; ++frame)
            {
                auto rowOffset = frame * dataParams.laneWidth;
                for (size_t k = 0; k < dataParams.laneWidth; ++k)
                {
                    pixelSums[k] += blockView(frame, k);
                    expectedPixelSums[k] += truth[rowOffset + k];
                }
            }

            for (size_t zmw = 0; zmw < expectedPixelSums.size(); zmw++)
            {
                EXPECT_EQ(expectedPixelSums[zmw], pixelSums[zmw])
                    << "batchIdx=" << batchIdx
                    << " frameOffset=" << frameOffset
                    << " expectedPixelSums[" << zmw << "]=" << expectedPixelSums[zmw]
                    << " pixelSums[" << zmw << "]=" << pixelSums[zmw];
            }
        }
        manager.ReturnBatch(std::move(data));
    }
}
