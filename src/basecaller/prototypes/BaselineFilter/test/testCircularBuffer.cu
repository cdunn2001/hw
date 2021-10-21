#include <gtest/gtest.h>

#include <common/cuda/memory/DeviceOnlyArray.cuh>
#include <common/cuda/streams/LaunchManager.cuh>

#include <appModules/SimulatedDataSource.h>

#include <BlockCircularBuffer.cuh>
#include <LocalCircularBuffer.cuh>
#include <CircularBufferKernels.cuh>

using namespace PacBio::Application;
using namespace PacBio::Cuda;
using namespace PacBio::Cuda::Memory;
using namespace PacBio::DataSource;
using namespace PacBio::Mongo::Data;

static constexpr int laneWidth = 64;
static constexpr int gpuBlockThreads = laneWidth/2;
static constexpr int lag = 4;

namespace
{

    static bool validateData = true;

    void ValidateData(const TraceBatch <int16_t>& input, const TraceBatch <int16_t>& output)
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

enum class TestTypes { Global, Shared, Local };

struct BlockCircularBufferTest : testing::TestWithParam<TestTypes> {};

TEST_P(BlockCircularBufferTest, Basic)
{
    static constexpr int SawtoothHeight = 25;
    static constexpr int lanesPerBatch = 4;
    static constexpr int framesPerBlock = 128;
    static constexpr int numZmw = 16*64;
    static constexpr int numLanes = 16;
    static constexpr int numFrames = 512;

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

    std::vector<DeviceOnlyArray<BlockCircularBuffer<gpuBlockThreads,lag>>> circularBuffers;
    for (uint32_t i = 0; i < numLanes / lanesPerBatch; i++)
    {
        circularBuffers.emplace_back(SOURCE_MARKER(), lanesPerBatch, 0);
    }

    for (const auto& batch: source.AllBatches<int16_t>())
    {
        auto firstFrame = batch.GetMeta().FirstFrame();
        auto batchIdx = batch.Metadata().PoolId();
        TraceBatch<int16_t> out(batch.GetMeta(),
                                batch.StorageDims(),
                                SyncDirection::HostReadDeviceWrite,
                                SOURCE_MARKER());
        auto testFunc = [](){
            switch (GetParam())
            {
            case TestTypes::Global:
                return GlobalMemCircularBuffer<gpuBlockThreads, lag, BlockCircularBuffer>;
            case TestTypes::Shared:
                return SharedMemCircularBuffer<gpuBlockThreads, lag, BlockCircularBuffer>;
            case TestTypes::Local:
                return LocalMemCircularBuffer<gpuBlockThreads, lag, BlockCircularBuffer>;
            default:
                throw PBException("Missing handler for test case enum");
            }
        }();

        const auto& launcher = PBLauncher(testFunc, lanesPerBatch, gpuBlockThreads);

        launcher(batch, circularBuffers[batchIdx], out);

        ValidateData(batch, out);
    }
}

INSTANTIATE_TEST_SUITE_P(,
                         BlockCircularBufferTest,
                         testing::Values(TestTypes::Global, TestTypes::Shared, TestTypes::Local),
                         [](const testing::TestParamInfo<TestTypes>& info) {
                             if (info.param == TestTypes::Global) return "Global";
                             if (info.param == TestTypes::Shared) return "Shared";
                             if (info.param == TestTypes::Local) return "Local";
                             return "NameError";
                         });
