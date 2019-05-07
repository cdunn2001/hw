#include <memory>

#include <gtest/gtest.h>

#include <common/cuda/memory/DeviceOnlyArray.cuh>

#include <common/ZmwDataManager.h>
#include <common/DataGenerators/SawtoothGenerator.h>

#include <ExtremaFilter.cuh>
#include <ExtremaFilterKernels.cuh>

using namespace PacBio::Cuda;
using namespace PacBio::Cuda::Data;
using namespace PacBio::Cuda::Memory;
using namespace PacBio::Mongo::Data;

namespace {

void ValidateData(TraceBatch<short2>& batch,
                  size_t filterWidth,
                  size_t sawtoothHeight,
                  size_t startFrame,
                  const DataManagerParams& params)
{
    for (size_t i = 0; i < batch.LanesPerBatch(); ++i)
    {
        auto block = batch.GetBlockView(i);
        for (size_t frame = 0; frame < block.BlockLen(); ++frame)
        {
            if (startFrame+frame < filterWidth) continue;

            short expectedVal = (startFrame + frame - 1) % sawtoothHeight;
            if (expectedVal < filterWidth-1)
                expectedVal = sawtoothHeight-1;

            for (size_t zmw = 0; zmw < block.LaneWidth(); ++zmw)
            {
                EXPECT_EQ(block(frame, zmw).x, expectedVal);
                EXPECT_EQ(block(frame, zmw).y, expectedVal);
            }
        }
    }

}

}

TEST(MaxFilterTest, GlobalMemoryMax)
{
    static constexpr int zmwLaneWidth = 64;
    static constexpr int gpuLaneWidth = zmwLaneWidth/2;
    static constexpr int FilterWidth = 7;
    static constexpr int SawtoothHeight = 25;

    auto params = DataManagerParams()
            .ZmwLaneWidth(zmwLaneWidth)
            .ImmediateCopy(true)
            .FrameRate(1000)
            .NumZmwLanes(16)
            .KernelLanes(4)
            .NumBlocks(4)
            .BlockLength(64);

    using Filter = ExtremaFilter<gpuLaneWidth, FilterWidth>;
    DeviceOnlyArray<Filter> filterData(params.numZmwLanes, 0);

    ZmwDataManager<short2> manager(params, std::make_unique<SawtoothGenerator>(params), true);
    while (manager.MoreData())
    {
        auto data = manager.NextBatch();
        auto firstFrame = data.FirstFrame();
        auto batchIdx = data.Batch();
        auto& in = data.KernelInput();
        auto& out = data.KernelOutput();

        MaxGlobalFilter<gpuLaneWidth, FilterWidth><<<params.kernelLanes, gpuLaneWidth>>>(
            in,
            filterData.GetDeviceView(batchIdx * params.kernelLanes, params.kernelLanes),
            out);

        ValidateData(out, FilterWidth, SawtoothHeight, firstFrame, params);

        manager.ReturnBatch(std::move(data));
    }
}

TEST(MaxFilterTest, SharedMemoryMax)
{
    static constexpr int zmwLaneWidth = 64;
    static constexpr int gpuLaneWidth = zmwLaneWidth/2;
    static constexpr int FilterWidth = 7;
    static constexpr int SawtoothHeight = 25;

    auto params = DataManagerParams()
            .ZmwLaneWidth(zmwLaneWidth)
            .ImmediateCopy(true)
            .FrameRate(1000)
            .NumZmwLanes(16)
            .KernelLanes(4)
            .NumBlocks(4)
            .BlockLength(64);

    using Filter = ExtremaFilter<gpuLaneWidth, FilterWidth>;
    DeviceOnlyArray<Filter> filterData(params.numZmwLanes, 0);

    ZmwDataManager<short2> manager(params, std::make_unique<SawtoothGenerator>(params), true);
    while (manager.MoreData())
    {
        auto data = manager.NextBatch();
        auto firstFrame = data.FirstFrame();
        auto batchIdx = data.Batch();
        auto& in = data.KernelInput();
        auto& out = data.KernelOutput();

        MaxSharedFilter<gpuLaneWidth, FilterWidth><<<params.kernelLanes, gpuLaneWidth>>>(
            in,
            filterData.GetDeviceView(batchIdx * params.kernelLanes, params.kernelLanes),
            out);

        ValidateData(out, FilterWidth, SawtoothHeight, firstFrame, params);

        manager.ReturnBatch(std::move(data));
    }
}

TEST(MaxFilterTest, LocalMemoryMax)
{
    static constexpr int zmwLaneWidth = 64;
    static constexpr int gpuLaneWidth = zmwLaneWidth/2;
    static constexpr int FilterWidth = 7;
    static constexpr int SawtoothHeight = 25;

    auto params = DataManagerParams()
            .ZmwLaneWidth(zmwLaneWidth)
            .ImmediateCopy(true)
            .FrameRate(1000)
            .NumZmwLanes(16)
            .KernelLanes(4)
            .NumBlocks(4)
            .BlockLength(64);

    using Filter = ExtremaFilter<gpuLaneWidth, FilterWidth>;
    DeviceOnlyArray<Filter> filterData(params.numZmwLanes, 0);

    ZmwDataManager<short2> manager(params, std::make_unique<SawtoothGenerator>(params), true);
    while (manager.MoreData())
    {
        auto data = manager.NextBatch();
        auto firstFrame = data.FirstFrame();
        auto batchIdx = data.Batch();
        auto& in = data.KernelInput();
        auto& out = data.KernelOutput();

        MaxLocalFilter<gpuLaneWidth, FilterWidth><<<params.kernelLanes, gpuLaneWidth>>>(
            in,
            filterData.GetDeviceView(batchIdx * params.kernelLanes, params.kernelLanes),
            out);

        ValidateData(out, FilterWidth, SawtoothHeight, firstFrame, params);

        manager.ReturnBatch(std::move(data));
    }
}
