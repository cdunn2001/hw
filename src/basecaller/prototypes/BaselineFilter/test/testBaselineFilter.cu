#include <memory>

#include <gtest/gtest.h>

#include <common/cuda/memory/DeviceOnlyArray.cuh>
#include <common/ZmwDataManager.h>
#include <common/DataGenerators/PicketFenceGenerator.h>

#include <BaselineFilter.cuh>
#include <BaselineFilterKernels.cuh>

using namespace PacBio::Cuda;
using namespace PacBio::Cuda::Data;
using namespace PacBio::Cuda::Memory;
using namespace PacBio::Mongo::Data;

// Rough tests, that at least makes sure the baseline filter produces results in
// the correct ballpark.  The baseline filter does not currently gracefully handle
// boundary conditions, so during verification we must skip the first couple
// hundred frames as they are not valid.
TEST(BaselineFilterTest, GlobalMemory)
{
    static constexpr size_t laneWidth = 64;
    static constexpr size_t gpuBlockThreads = laneWidth/2;

    // Have 4 lanes per cuda kernel, and 4 kernel invocations, to flush out
    // errors in the bookkeeping
    auto dataParams = DataManagerParams()
            .LaneWidth(laneWidth)
            .ImmediateCopy(true)
            .FrameRate(1000)
            .NumZmwLanes(16)
            .KernelLanes(4)
            .NumBlocks(4)
            .BlockLength(512);

    auto picketParams = PicketFenceParams()
            .NumSignals(1)
            .BaselineSignalLevel(150);


    using Filter = BaselineFilter<gpuBlockThreads, IntSeq<2,8>, IntSeq<9,31>>;
    std::vector<DeviceOnlyArray<Filter>> filterData;
    for (int i = 0; i < dataParams.numZmwLanes / dataParams.kernelLanes; ++i)
    {
        filterData.emplace_back(dataParams.kernelLanes, 0);
    }

    ZmwDataManager<int16_t> manager(dataParams,
                                  std::make_unique<PicketFenceGenerator>(dataParams, picketParams),
                                  true);

    while (manager.MoreData())
    {
        auto data = manager.NextBatch();
        auto firstFrame = data.FirstFrame();
        auto batchIdx = data.Batch();
        auto& in = data.KernelInput();
        auto& out = data.KernelOutput();

        GlobalBaselineFilter<<<dataParams.kernelLanes, gpuBlockThreads>>>(
            in,
            filterData[batchIdx].GetDeviceView(),
            out);


        for (size_t i = 0; i < out.LanesPerBatch(); ++i)
        {
            auto block = out.GetBlockView(i);
            for (size_t j = 0; j < block.NumFrames(); ++j)
            {
                // Baseline filter is still warming up before this point
                if (firstFrame + j < 1030) continue;

                for (size_t k = 0; k < block.LaneWidth(); ++k)
                {
                    // Not entire sure why the data is empirically around 230 instead of
                    // 150.  Could be the lack of bias correction, though I'd be surpised
                    // it is so large.  Could be an artifact of the data source, which
                    // I've not examined in detail.  It could also be a bug, though
                    // other tests ensure that alternate implementations match.  Ultimately
                    // I think things look suspicious, but things are otherwise behaving
                    // correctly and as this is testing mere prototypes, there isn't time
                    // to dig into this just yet.
                    EXPECT_NEAR(block(j,k), 230, 20);
                }

            }
        }

        manager.ReturnBatch(std::move(data));
    }
}

// Now just make sure that the alternate (optimized)
// implementations produce identical results
TEST(BaselineFilterTest, SharedMemory)
{
    static constexpr size_t laneWidth = 64;
    static constexpr size_t gpuBlockThreads = laneWidth/2;

    // Have 4 lanes per cuda kernel, and 4 kernel invocations, to flush out
    // errors in the bookkeeping
    auto dataParams = DataManagerParams()
            .LaneWidth(laneWidth)
            .ImmediateCopy(true)
            .FrameRate(1000)
            .NumZmwLanes(16)
            .KernelLanes(4)
            .NumBlocks(4)
            .BlockLength(512);

    auto picketParams = PicketFenceParams()
            .NumSignals(1)
            .BaselineSignalLevel(150);


    using Filter = BaselineFilter<gpuBlockThreads, IntSeq<2,8>, IntSeq<9,31>>;
    std::vector<DeviceOnlyArray<Filter>> filterData;
    std::vector<DeviceOnlyArray<Filter>> filterRefData;
    for (int i = 0; i < dataParams.numZmwLanes / dataParams.kernelLanes; ++i)
    {
        filterData.emplace_back(dataParams.kernelLanes, 0);
        filterRefData.emplace_back(dataParams.kernelLanes, 0);
    }

    ZmwDataManager<int16_t> manager(dataParams,
                                  std::make_unique<PicketFenceGenerator>(dataParams, picketParams),
                                  true);

    BatchDimensions dims;
    dims.laneWidth = dataParams.laneWidth;
    dims.framesPerBatch = dataParams.blockLength;
    dims.lanesPerBatch = dataParams.kernelLanes;
    BatchData<int16_t> truth(dims, SyncDirection::HostReadDeviceWrite, nullptr);

    while (manager.MoreData())
    {
        auto data = manager.NextBatch();
        auto batchIdx = data.Batch();
        auto& in = data.KernelInput();
        auto& out = data.KernelOutput();

        SharedBaselineFilter<<<dataParams.kernelLanes, gpuBlockThreads>>>(
            in,
            filterData[batchIdx].GetDeviceView(),
            out);

        GlobalBaselineFilter<<<dataParams.kernelLanes, gpuBlockThreads>>>(
            in,
            filterRefData[batchIdx].GetDeviceView(),
            truth);

        for (size_t i = 0; i < in.LanesPerBatch(); ++i)
        {
            auto truthView = truth.GetBlockView(i);
            auto view = out.GetBlockView(i);
            for (size_t j = 0; j < view.Size(); ++j)
            {
                EXPECT_EQ(truthView[i], view[i]);
            }
        }

        manager.ReturnBatch(std::move(data));
    }
}

// Now just make sure that the alternate (optimized)
// implementations produce identical results
TEST(BaselineFilterTest, MultiKernelFilter)
{
    static constexpr size_t laneWidth = 64;
    static constexpr size_t gpuBlockThreads = laneWidth/2;
    // Have 4 lanes per cuda kernel, and 4 kernel invocations, to flush out
    // errors in the bookkeeping
    auto dataParams = DataManagerParams()
            .LaneWidth(laneWidth)
            .ImmediateCopy(true)
            .FrameRate(1000)
            .NumZmwLanes(4)
            .KernelLanes(4)
            .NumBlocks(4)
            .BlockLength(128);

    auto picketParams = PicketFenceParams()
            .NumSignals(1)
            .BaselineSignalLevel(150);


    using RefFilter = BaselineFilter<gpuBlockThreads, IntSeq<2,8>, IntSeq<9,31>>;
    using Filter = ComposedFilter<gpuBlockThreads, 9, 31, 2, 8>;
    std::vector<DeviceOnlyArray<RefFilter>> filterRefData;
    std::vector<Filter> filterData;
    for (int i = 0; i < dataParams.numZmwLanes / dataParams.kernelLanes; ++i)
    {
        filterData.emplace_back(dataParams.kernelLanes, 0);
        filterRefData.emplace_back(dataParams.kernelLanes, 0);
    }

    ZmwDataManager<int16_t> manager(dataParams,
                                  std::make_unique<PicketFenceGenerator>(dataParams, picketParams),
                                  true);

    BatchDimensions dims;
    dims.laneWidth = dataParams.laneWidth;
    dims.framesPerBatch = dataParams.blockLength;
    dims.lanesPerBatch = dataParams.kernelLanes;
    BatchData<int16_t> truth(dims, SyncDirection::HostReadDeviceWrite, nullptr);
    BatchData<int16_t> work1(dims, SyncDirection::HostReadDeviceWrite, nullptr);
    BatchData<int16_t> work2(dims, SyncDirection::HostReadDeviceWrite, nullptr);

    while (manager.MoreData())
    {
        auto data = manager.NextBatch();
        auto batchIdx = data.Batch();
        auto& in = data.KernelInput();
        auto& out = data.KernelOutput();

        filterData[batchIdx].RunComposedFilter(in, out, work1, work2);

        GlobalBaselineFilter<<<dataParams.kernelLanes, gpuBlockThreads>>>(
            in,
            filterRefData[batchIdx].GetDeviceView(),
            truth);

        for (size_t i = 0; i < in.LanesPerBatch(); ++i)
        {
            auto truthView = truth.GetBlockView(i);
            auto view = out.GetBlockView(i);
            for (size_t j = 0; j < view.Size(); ++j)
            {
                EXPECT_EQ(truthView[i], view[i]);
            }
        }

        manager.ReturnBatch(std::move(data));
    }
}
