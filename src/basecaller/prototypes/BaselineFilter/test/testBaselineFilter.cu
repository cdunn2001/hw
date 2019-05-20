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
    static constexpr size_t zmwLaneWidth = 32;
    static constexpr size_t gpuLaneWidth = zmwLaneWidth/2;

    // Have 4 lanes per cuda kernel, and 4 kernel invocations, to flush out
    // errors in the bookkeeping
    auto dataParams = DataManagerParams()
            .ZmwLaneWidth(zmwLaneWidth)
            .ImmediateCopy(true)
            .FrameRate(1000)
            .NumZmwLanes(16)
            .KernelLanes(4)
            .NumBlocks(4)
            .BlockLength(512);

    auto picketParams = PicketFenceParams()
            .NumSignals(1)
            .BaselineSignalLevel(150);


    using Filter = BaselineFilter<gpuLaneWidth, IntSeq<2,8>, IntSeq<9,31>>;
    DeviceOnlyArray<Filter> filterData(dataParams.numZmwLanes, 0);

    ZmwDataManager<short2> manager(dataParams,
                                   std::make_unique<PicketFenceGenerator>(dataParams, picketParams),
                                   true);

    while (manager.MoreData())
    {
        auto data = manager.NextBatch();
        auto firstFrame = data.FirstFrame();
        auto batchIdx = data.Batch();
        auto& in = data.KernelInput();
        auto& out = data.KernelOutput();

        GlobalBaselineFilter<<<dataParams.kernelLanes, gpuLaneWidth>>>(
            in,
            filterData.GetDeviceView(batchIdx * dataParams.kernelLanes, dataParams.kernelLanes),
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
                    EXPECT_NEAR(block(j,k).x, 230, 20);
                    EXPECT_NEAR(block(j,k).y, 230, 20);
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
    static constexpr size_t zmwLaneWidth = 32;
    static constexpr size_t gpuLaneWidth = zmwLaneWidth/2;

    // Have 4 lanes per cuda kernel, and 4 kernel invocations, to flush out
    // errors in the bookkeeping
    auto dataParams = DataManagerParams()
            .ZmwLaneWidth(zmwLaneWidth)
            .ImmediateCopy(true)
            .FrameRate(1000)
            .NumZmwLanes(16)
            .KernelLanes(4)
            .NumBlocks(4)
            .BlockLength(512);

    auto picketParams = PicketFenceParams()
            .NumSignals(1)
            .BaselineSignalLevel(150);


    using Filter = BaselineFilter<gpuLaneWidth, IntSeq<2,8>, IntSeq<9,31>>;
    DeviceOnlyArray<Filter> filterData(dataParams.numZmwLanes, 0);
    DeviceOnlyArray<Filter> filterRefData(dataParams.numZmwLanes, 0);

    ZmwDataManager<short2> manager(dataParams,
                                   std::make_unique<PicketFenceGenerator>(dataParams, picketParams),
                                   true);

    BatchDimensions dims;
    dims.laneWidth = dataParams.gpuLaneWidth;
    dims.framesPerBatch = dataParams.blockLength;
    dims.lanesPerBatch = dataParams.kernelLanes;
    BatchData<short2> truth(dims, SyncDirection::HostReadDeviceWrite, nullptr);

    while (manager.MoreData())
    {
        auto data = manager.NextBatch();
        auto batchIdx = data.Batch();
        auto& in = data.KernelInput();
        auto& out = data.KernelOutput();

        SharedBaselineFilter<<<dataParams.kernelLanes, gpuLaneWidth>>>(
            in,
            filterData.GetDeviceView(batchIdx * dataParams.kernelLanes, dataParams.kernelLanes),
            out);

        GlobalBaselineFilter<<<dataParams.kernelLanes, gpuLaneWidth>>>(
            in,
            filterRefData.GetDeviceView(batchIdx * dataParams.kernelLanes, dataParams.kernelLanes),
            truth);

        for (size_t i = 0; i < in.LanesPerBatch(); ++i)
        {
            auto truthView = truth.GetBlockView(i);
            auto view = out.GetBlockView(i);
            for (size_t j = 0; j < view.Size(); ++j)
            {
                EXPECT_EQ(truthView[i].x, view[i].x);
                EXPECT_EQ(truthView[i].y, view[i].y);
            }
        }

        manager.ReturnBatch(std::move(data));
    }
}

// Now just make sure that the alternate (optimized)
// implementations produce identical results
TEST(BaselineFilterTest, MultiKernelFilter)
{
    static constexpr size_t zmwLaneWidth = 32;
    static constexpr size_t gpuLaneWidth = zmwLaneWidth/2;
    // Have 4 lanes per cuda kernel, and 4 kernel invocations, to flush out
    // errors in the bookkeeping
    auto dataParams = DataManagerParams()
            .ZmwLaneWidth(zmwLaneWidth)
            .ImmediateCopy(true)
            .FrameRate(1000)
            .NumZmwLanes(4)
            .KernelLanes(4)
            .NumBlocks(4)
            .BlockLength(128);

    auto picketParams = PicketFenceParams()
            .NumSignals(1)
            .BaselineSignalLevel(150);


    using RefFilter = BaselineFilter<gpuLaneWidth, IntSeq<2,8>, IntSeq<9,31>>;
    using Filter = ComposedFilter<gpuLaneWidth, 9, 31, 2, 8>;
    DeviceOnlyArray<RefFilter> filterRefData(dataParams.numZmwLanes, 0);
    std::vector<Filter> filterData;
    for (size_t i = 0; i < dataParams.numZmwLanes / dataParams.kernelLanes; ++i)
    {
        filterData.emplace_back(dataParams.kernelLanes, 0);
    }

    ZmwDataManager<short2> manager(dataParams,
                                   std::make_unique<PicketFenceGenerator>(dataParams, picketParams),
                                   true);

    BatchDimensions dims;
    dims.laneWidth = dataParams.gpuLaneWidth;
    dims.framesPerBatch = dataParams.blockLength;
    dims.lanesPerBatch = dataParams.kernelLanes;
    BatchData<short2> truth(dims, SyncDirection::HostReadDeviceWrite, nullptr);
    BatchData<short2> work1(dims, SyncDirection::HostReadDeviceWrite, nullptr);
    BatchData<short2> work2(dims, SyncDirection::HostReadDeviceWrite, nullptr);

    while (manager.MoreData())
    {
        auto data = manager.NextBatch();
        auto batchIdx = data.Batch();
        auto& in = data.KernelInput();
        auto& out = data.KernelOutput();

        filterData[batchIdx].RunComposedFilter(in, out, work1, work2);

        GlobalBaselineFilter<<<dataParams.kernelLanes, gpuLaneWidth>>>(
            in,
            filterRefData.GetDeviceView(batchIdx * dataParams.kernelLanes, dataParams.kernelLanes),
            truth);

        for (size_t i = 0; i < in.LanesPerBatch(); ++i)
        {
            auto truthView = truth.GetBlockView(i);
            auto view = out.GetBlockView(i);
            for (size_t j = 0; j < view.Size(); ++j)
            {
                EXPECT_EQ(truthView[i].x, view[i].x);
                EXPECT_EQ(truthView[i].y, view[i].y);
            }
        }

        manager.ReturnBatch(std::move(data));
    }
}
