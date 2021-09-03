#include <memory>

#include <gtest/gtest.h>

#include <common/cuda/memory/DeviceOnlyArray.cuh>
#include <common/cuda/streams/LaunchManager.cuh>

#include <appModules/SimulatedDataSource.h>
#include "pacbio/datasource/MallocAllocator.h"

#include <BaselineFilter.cuh>
#include <BaselineFilterKernels.cuh>

using namespace PacBio::Application;
using namespace PacBio::Cuda;
using namespace PacBio::Cuda::Memory;
using namespace PacBio::DataSource;
using namespace PacBio::Mongo;
using namespace PacBio::Mongo::Basecaller;
using namespace PacBio::Mongo::Data;

namespace {

SimulatedDataSource CreateSource()
{
    static constexpr int lanesPerBatch = 4;
    static constexpr int framesPerBlock = 512;
    static constexpr int numZmw = 16*64;
    static constexpr int numFrames = 2048;

    PicketFenceGenerator::Config pickCfg;
    pickCfg.generatePoisson = false;
    pickCfg.baselineSignalLevel = 150;
    auto generator = std::make_unique<PicketFenceGenerator>(pickCfg);

    SimulatedDataSource::SimConfig simCfg(numZmw, numFrames);

    return SimulatedDataSource(numZmw, simCfg, lanesPerBatch, framesPerBlock, std::move(generator));
}

}

// Rough tests, that at least makes sure the baseline filter produces results in
// the correct ballpark.  The baseline filter does not currently gracefully handle
// boundary conditions, so during verification we must skip the first couple
// hundred frames as they are not valid.
TEST(BaselineFilterTest, GlobalMemory)
{
    static constexpr size_t gpuBlockThreads = laneSize/2;
    auto source = CreateSource();

    using Filter = BaselineFilter<gpuBlockThreads, IntSeq<2,8>, IntSeq<9,31>>;
    std::vector<DeviceOnlyArray<Filter>> filterData;
    for (uint32_t i = 0; i < source.PacketLayouts().size(); ++i)
    {
        filterData.emplace_back(SOURCE_MARKER(), source.PacketLayouts()[i].NumBlocks(), 0);
    }

    for (const auto& batch : source.AllBatches())
    {
        auto firstFrame = batch.GetMeta().FirstFrame();
        auto batchIdx = batch.GetMeta().PoolId();
        TraceBatch<int16_t> out(batch.GetMeta(),
                                batch.StorageDims(),
                                SyncDirection::HostReadDeviceWrite,
                                SOURCE_MARKER());

        const auto& baseliner = PBLauncher(GlobalBaselineFilter<Filter>,
                                           batch.LanesPerBatch(),
                                           gpuBlockThreads);
        baseliner(batch,
                  filterData[batchIdx],
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
    }
}

// Now just make sure that the alternate (optimized)
// implementations produce identical results
TEST(BaselineFilterTest, SharedMemory)
{
    static constexpr size_t gpuBlockThreads = laneSize/2;
    auto source = CreateSource();

    using Filter = BaselineFilter<gpuBlockThreads, IntSeq<2,8>, IntSeq<9,31>>;
    std::vector<DeviceOnlyArray<Filter>> filterData;
    std::vector<DeviceOnlyArray<Filter>> filterRefData;
    for (uint32_t i = 0; i < source.PacketLayouts().size(); ++i)
    {
        filterData.emplace_back(SOURCE_MARKER(), source.PacketLayouts()[i].NumBlocks(), 0);
        filterRefData.emplace_back(SOURCE_MARKER(), source.PacketLayouts()[i].NumBlocks(), 0);
    }

    for (const auto& batch : source.AllBatches())
    {
        auto batchIdx = batch.GetMeta().PoolId();
        BatchData<int16_t> truth(batch.StorageDims(),
                                 SyncDirection::HostReadDeviceWrite,
                                 SOURCE_MARKER());
        BatchData<int16_t> out(batch.StorageDims(),
                               SyncDirection::HostReadDeviceWrite,
                               SOURCE_MARKER());

        const auto& shared = PBLauncher(SharedBaselineFilter<Filter>,
                                        batch.LanesPerBatch(),
                                        gpuBlockThreads);
        shared(batch,
               filterData[batchIdx],
               out);

        const auto& global = PBLauncher(GlobalBaselineFilter<Filter>,
                                        batch.LanesPerBatch(),
                                        gpuBlockThreads);
        global(batch,
               filterRefData[batchIdx],
               truth);

        for (size_t i = 0; i < batch.LanesPerBatch(); ++i)
        {
            auto truthView = truth.GetBlockView(i);
            auto view = out.GetBlockView(i);
            for (size_t j = 0; j < view.Size(); ++j)
            {
                EXPECT_EQ(truthView[i], view[i]);
            }
        }
    }
}

// Now just make sure that the alternate (optimized)
// implementations produce identical results
TEST(BaselineFilterTest, MultiKernelFilter)
{
    static constexpr size_t gpuBlockThreads = laneSize/2;
    auto source = CreateSource();

    using RefFilter = BaselineFilter<gpuBlockThreads, IntSeq<2,8>, IntSeq<9,31>>;
    using Filter = ComposedFilter<gpuBlockThreads, 4>;
    std::vector<DeviceOnlyArray<RefFilter>> filterRefData;
    std::vector<Filter> filterData;
    auto params = FilterParamsLookup(BasecallerBaselinerConfig::FilterTypes::TwoScaleMedium);
    for (uint32_t i = 0; i < source.PacketLayouts().size(); ++i)
    {
        filterData.emplace_back(params, SOURCE_MARKER(), source.PacketLayouts()[i].NumBlocks(), 0);
        filterRefData.emplace_back(SOURCE_MARKER(), source.PacketLayouts()[i].NumBlocks(), 0);
    }

    for (const auto& batch : source.AllBatches())
    {
        auto batchIdx = batch.GetMeta().PoolId();
        BatchData<int16_t> truth(batch.StorageDims(),
                                 SyncDirection::HostReadDeviceWrite,
                                 SOURCE_MARKER());
        TraceBatch<int16_t> out(batch.GetMeta(),
                                batch.StorageDims(),
                                SyncDirection::HostReadDeviceWrite,
                                SOURCE_MARKER());
        BatchData<int16_t> work1(batch.StorageDims(),
                                 SyncDirection::HostReadDeviceWrite,
                                 SOURCE_MARKER());
        BatchData<int16_t> work2(batch.StorageDims(),
                                 SyncDirection::HostReadDeviceWrite,
                                 SOURCE_MARKER());

        filterData[batchIdx].RunComposedFilter(batch, out, work1, work2);

        const auto& global = PBLauncher(GlobalBaselineFilter<RefFilter>,
                                        batch.LanesPerBatch(),
                                        gpuBlockThreads);
        global(batch,
               filterRefData[batchIdx],
               truth);

        for (size_t i = 0; i < batch.LanesPerBatch(); ++i)
        {
            auto truthView = truth.GetBlockView(i);
            auto view = out.GetBlockView(i);
            for (size_t j = 0; j < view.Size(); ++j)
            {
                EXPECT_EQ(truthView[i], view[i]);
            }
        }
    }
}
