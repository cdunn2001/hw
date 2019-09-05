#include <gtest/gtest.h>

#include <common/cuda/memory/DeviceOnlyArray.cuh>
#include <common/cuda/streams/LaunchManager.cuh>

#include <common/ZmwDataManager.h>
#include <common/DataGenerators/SawtoothGenerator.h>

#include <BlockCircularBuffer.cuh>
#include <LocalCircularBuffer.cuh>
#include <CircularBufferKernels.cuh>


using namespace PacBio::Cuda;
using namespace PacBio::Cuda::Data;
using namespace PacBio::Cuda::Memory;
using namespace PacBio::Mongo::Data;


static constexpr int laneWidth = 64;
static constexpr int gpuBlockThreads = laneWidth/2;
static constexpr int lag = 4;

auto params = DataManagerParams()
        .LaneWidth(laneWidth)
        .ImmediateCopy(true)
        .FrameRate(100)
        .NumZmwLanes(16)
        .KernelLanes(4)
        .NumBlocks(4)
        .BlockLength(128);

namespace
{

    static bool validateData = true;

    void ValidateData(TraceBatch <int16_t>& input, TraceBatch <int16_t>& output)
    {
        if (!validateData) return;

        for (size_t i = 0; i < input.LanesPerBatch(); ++i)
        {
            auto inBlock = input.GetBlockView(i);
            auto outBlock = output.GetBlockView(i);

            for (size_t zmw = 0; zmw < inBlock.LaneWidth(); ++zmw)
            {

                for (size_t frame = 0; frame < inBlock.NumFrames() - lag; ++frame)
                {
                    EXPECT_EQ(inBlock(frame, zmw), outBlock(frame + lag, zmw));
                }
            }
        }
    }
}

TEST(BlockCircularBuffer, GlobalMemory)
{
    std::vector<DeviceOnlyArray<BlockCircularBuffer<gpuBlockThreads,lag>>> circularBuffers;
    for (int i = 0; i < params.numZmwLanes / params.kernelLanes; i++)
    {
        circularBuffers.emplace_back(SOURCE_MARKER(), params.kernelLanes, 0);
    }

    ZmwDataManager<short> manager(params, std::make_unique<SawtoothGenerator>(params), true);
    while (manager.MoreData())
    {
        auto data = manager.NextBatch();
        auto firstFrame = data.FirstFrame();
        auto batchIdx = data.Batch();
        auto& in = data.KernelInput();
        auto& out = data.KernelOutput();

        const auto& launcher = PBLauncher(GlobalMemCircularBuffer<gpuBlockThreads,lag,BlockCircularBuffer>,
                                          params.kernelLanes, gpuBlockThreads);

        launcher(in, circularBuffers[batchIdx], out);

        ValidateData(in, out);

        manager.ReturnBatch(std::move(data));
    }
}

TEST(CircularBuffer, SharedMemory)
{
    std::vector<DeviceOnlyArray<BlockCircularBuffer<gpuBlockThreads,lag>>> circularBuffers;
    for (int i = 0; i < params.numZmwLanes / params.kernelLanes; i++)
    {
        circularBuffers.emplace_back(SOURCE_MARKER(), params.kernelLanes, 0);
    }

    ZmwDataManager<short> manager(params, std::make_unique<SawtoothGenerator>(params), true);
    while (manager.MoreData())
    {
        auto data = manager.NextBatch();
        auto firstFrame = data.FirstFrame();
        auto batchIdx = data.Batch();
        auto& in = data.KernelInput();
        auto& out = data.KernelOutput();

        const auto& launcher = PBLauncher(SharedMemCircularBuffer<gpuBlockThreads,lag,BlockCircularBuffer>,
                                          params.kernelLanes, gpuBlockThreads);

        launcher(in, circularBuffers[batchIdx], out);

        ValidateData(in, out);

        manager.ReturnBatch(std::move(data));
    }
}

TEST(CircularBufferShift, LocalMemory)
{
    std::vector<DeviceOnlyArray<LocalCircularBuffer<gpuBlockThreads,lag>>> circularBuffers;
    for (int i = 0; i < params.numZmwLanes / params.kernelLanes; i++)
    {
        circularBuffers.emplace_back(SOURCE_MARKER(), params.kernelLanes);
    }

    ZmwDataManager<short> manager(params, std::make_unique<SawtoothGenerator>(params), true);
    while (manager.MoreData())
    {
        auto data = manager.NextBatch();
        auto firstFrame = data.FirstFrame();
        auto batchIdx = data.Batch();
        auto& in = data.KernelInput();
        auto& out = data.KernelOutput();

        const auto& launcher = PBLauncher(LocalMemCircularBuffer<gpuBlockThreads,lag,LocalCircularBuffer>,
                                          params.kernelLanes, gpuBlockThreads);

        launcher(in, circularBuffers[batchIdx], out);

        ValidateData(in, out);

        manager.ReturnBatch(std::move(data));
    }
}