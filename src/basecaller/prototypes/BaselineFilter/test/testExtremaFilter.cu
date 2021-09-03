#include <memory>

#include <gtest/gtest.h>

#include <common/cuda/memory/DeviceOnlyArray.cuh>
#include <common/cuda/streams/LaunchManager.cuh>

#include <appModules/SimulatedDataSource.h>

#include <ExtremaFilter.cuh>
#include <ExtremaFilterKernels.cuh>

using namespace PacBio::Application;
using namespace PacBio::Cuda;
using namespace PacBio::Cuda::Memory;
using namespace PacBio::DataSource;
using namespace PacBio::Mongo::Data;

namespace {

void ValidateData(TraceBatch<int16_t>& batch,
                  size_t filterWidth,
                  size_t sawtoothHeight,
                  size_t startFrame)
{
    for (size_t i = 0; i < batch.LanesPerBatch(); ++i)
    {
        auto block = batch.GetBlockView(i);
        for (size_t frame = 0; frame < block.NumFrames(); ++frame)
        {
            if (startFrame+frame < filterWidth) continue;

            uint16_t expectedVal = (startFrame + frame - 1) % sawtoothHeight;
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

enum class TestTypes { Global, Shared, Local };

struct ExtremaFilterTests : testing::TestWithParam<TestTypes> {};

TEST_P(ExtremaFilterTests, Max)
{
    static constexpr int laneWidth = 64;
    static constexpr int framesPerBlock = 64;
    static constexpr int lanesPerBatch = 4;
    static constexpr int numZmw = 16*64;
    static constexpr int numFrames = 256;
    static constexpr int gpuBlockThreads = laneWidth/2;
    static constexpr int FilterWidth = 7;
    static constexpr int SawtoothHeight = 25;

    SawtoothGenerator::Config sawCfg;
    sawCfg.minAmp = 0;
    sawCfg.maxAmp = SawtoothHeight;
    sawCfg.periodFrames = SawtoothHeight;
    auto generator = std::make_unique<SawtoothGenerator>(sawCfg);

    SimulatedDataSource::SimConfig simCfg(numZmw, numFrames);

    SimulatedDataSource source(numZmw,
                               simCfg,
                               lanesPerBatch,
                               framesPerBlock,
                               std::move(generator));

    using Filter = ExtremaFilter<gpuBlockThreads, FilterWidth>;
    std::vector<DeviceOnlyArray<Filter>> filterData;
    for (uint32_t i = 0; i < source.PacketLayouts().size(); ++i)
    {
        filterData.emplace_back(SOURCE_MARKER(), lanesPerBatch, 0);
    }

    for (const auto& batch : source.AllBatches())
    {
        auto firstFrame = batch.GetMeta().FirstFrame();
        auto batchIdx = batch.GetMeta().PoolId();

        auto testFunc = [](){
            switch (GetParam())
            {
            case TestTypes::Global:
                return MaxGlobalFilter<gpuBlockThreads, FilterWidth>;
            case TestTypes::Shared:
                return MaxSharedFilter<gpuBlockThreads, FilterWidth>;
            case TestTypes::Local:
                return MaxLocalFilter<gpuBlockThreads, FilterWidth>;
            default:
                throw PBException("Missing handler for test case enum");
            }
        }();

        const auto& launcher = PBLauncher(testFunc,lanesPerBatch, gpuBlockThreads);

        auto out = TraceBatch<int16_t>(batch.GetMeta(),
                                       batch.StorageDims(),
                                       SyncDirection::HostReadDeviceWrite,
                                       SOURCE_MARKER());
        launcher(batch,
                 filterData[batchIdx],
                 out);

        ValidateData(out, FilterWidth, SawtoothHeight, firstFrame);
    }
}

INSTANTIATE_TEST_SUITE_P(,
                         ExtremaFilterTests,
                         testing::Values(TestTypes::Global, TestTypes::Shared, TestTypes::Local),
                         [](const testing::TestParamInfo<TestTypes>& info) {
                             if (info.param == TestTypes::Global) return "Global";
                             if (info.param == TestTypes::Shared) return "Shared";
                             if (info.param == TestTypes::Local) return "Local";
                             return "NameError";
                         });
