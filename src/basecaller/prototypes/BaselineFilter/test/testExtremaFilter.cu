#include <memory>

#include <gtest/gtest.h>

#include <common/cuda/memory/DeviceOnlyArray.cuh>
#include <common/cuda/streams/LaunchManager.cuh>

#include <common/ZmwDataManager.h>
#include <common/DataGenerators/SawtoothGenerator.h>

#include <ExtremaFilter.cuh>
#include <ExtremaFilterKernels.cuh>

using namespace PacBio::Cuda;
using namespace PacBio::Cuda::Data;
using namespace PacBio::Cuda::Memory;
using namespace PacBio::Mongo::Data;

namespace {

void ValidateData(TraceBatch<int16_t>& batch,
                  size_t filterWidth,
                  size_t sawtoothHeight,
                  size_t startFrame,
                  const DataManagerParams& params)
{
    for (size_t i = 0; i < batch.LanesPerBatch(); ++i)
    {
        auto block = batch.GetBlockView(i);
        for (size_t frame = 0; frame < block.NumFrames(); ++frame)
        {
            if (startFrame+frame < filterWidth) continue;

            short expectedVal = (startFrame + frame - 1) % sawtoothHeight;
            if (expectedVal < filterWidth-1)
                expectedVal = sawtoothHeight-1;

            for (size_t zmw = 0; zmw < block.LaneWidth(); ++zmw)
            {
                EXPECT_EQ(block(frame, zmw), expectedVal);
            }
        }
    }

}

}

TEST(MaxFilterTest, GlobalMemoryMax)
{
    static constexpr int laneWidth = 64;
    static constexpr int gpuBlockThreads = laneWidth/2;
    static constexpr int FilterWidth = 7;
    static constexpr int SawtoothHeight = 25;

    auto params = DataManagerParams()
            .LaneWidth(laneWidth)
            .ImmediateCopy(true)
            .FrameRate(1000)
            .NumZmwLanes(16)
            .KernelLanes(4)
            .NumBlocks(4)
            .BlockLength(64);

    using Filter = ExtremaFilter<gpuBlockThreads, FilterWidth>;
    std::vector<DeviceOnlyArray<Filter>> filterData;
    for (int i = 0; i < params.numZmwLanes / params.kernelLanes; ++i)
    {
        filterData.emplace_back(SOURCE_MARKER(), params.kernelLanes, 0);
    }

    ZmwDataManager<short> manager(params, std::make_unique<SawtoothGenerator>(params), true);
    while (manager.MoreData())
    {
        auto data = manager.NextBatch();
        auto firstFrame = data.FirstFrame();
        auto batchIdx = data.Batch();
        auto& in = data.KernelInput();
        auto& out = data.KernelOutput();

        const auto& launcher = PBLauncher(MaxGlobalFilter<gpuBlockThreads, FilterWidth>,
                                        params.kernelLanes, gpuBlockThreads);
        launcher(in,
                 filterData[batchIdx],
                 out);

        ValidateData(out, FilterWidth, SawtoothHeight, firstFrame, params);

        manager.ReturnBatch(std::move(data));
    }
}

TEST(MaxFilterTest, SharedMemoryMax)
{
    static constexpr int laneWidth = 64;
    static constexpr int gpuBlockThreads = laneWidth/2;
    static constexpr int FilterWidth = 7;
    static constexpr int SawtoothHeight = 25;

    auto params = DataManagerParams()
            .LaneWidth(laneWidth)
            .ImmediateCopy(true)
            .FrameRate(1000)
            .NumZmwLanes(16)
            .KernelLanes(4)
            .NumBlocks(4)
            .BlockLength(64);

    using Filter = ExtremaFilter<gpuBlockThreads, FilterWidth>;
    std::vector<DeviceOnlyArray<Filter>> filterData;
    for (int i = 0; i < params.numZmwLanes / params.kernelLanes; ++i)
    {
        filterData.emplace_back(SOURCE_MARKER(), params.kernelLanes, 0);
    }

    ZmwDataManager<int16_t> manager(params, std::make_unique<SawtoothGenerator>(params), true);
    while (manager.MoreData())
    {
        auto data = manager.NextBatch();
        auto firstFrame = data.FirstFrame();
        auto batchIdx = data.Batch();
        auto& in = data.KernelInput();
        auto& out = data.KernelOutput();

        const auto& launcher = PBLauncher(MaxSharedFilter<gpuBlockThreads, FilterWidth>,
                                        params.kernelLanes, gpuBlockThreads);
        launcher(in,
                 filterData[batchIdx],
                 out);

        ValidateData(out, FilterWidth, SawtoothHeight, firstFrame, params);

        manager.ReturnBatch(std::move(data));
    }
}

TEST(MaxFilterTest, LocalMemoryMax)
{
    static constexpr int laneWidth = 64;
    static constexpr int gpuBlockThreads = laneWidth/2;
    static constexpr int FilterWidth = 7;
    static constexpr int SawtoothHeight = 25;

    auto params = DataManagerParams()
            .LaneWidth(laneWidth)
            .ImmediateCopy(true)
            .FrameRate(1000)
            .NumZmwLanes(16)
            .KernelLanes(4)
            .NumBlocks(4)
            .BlockLength(64);

    using Filter = ExtremaFilter<gpuBlockThreads, FilterWidth>;
    std::vector<DeviceOnlyArray<Filter>> filterData;
    for (int i = 0; i < params.numZmwLanes / params.kernelLanes; ++i)
    {
        filterData.emplace_back(SOURCE_MARKER(), params.kernelLanes, 0);
    }

    ZmwDataManager<int16_t> manager(params, std::make_unique<SawtoothGenerator>(params), true);
    while (manager.MoreData())
    {
        auto data = manager.NextBatch();
        auto firstFrame = data.FirstFrame();
        auto batchIdx = data.Batch();
        auto& in = data.KernelInput();
        auto& out = data.KernelOutput();

        const auto& launcher = PBLauncher(MaxLocalFilter<gpuBlockThreads, FilterWidth>,
                                        params.kernelLanes, gpuBlockThreads);
        launcher(in,
                 filterData[batchIdx],
                 out);

        ValidateData(out, FilterWidth, SawtoothHeight, firstFrame, params);

        manager.ReturnBatch(std::move(data));
    }
}
