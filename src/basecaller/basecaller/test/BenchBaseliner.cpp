#include <random>

#include <boost/multi_array.hpp>

#include <benchmark/benchmark.h>

#include <common/LaneArray.h>
#include <common/cuda/memory/ManagedAllocations.h>

#include <dataTypes/configs/MovieConfig.h>
#include <dataTypes/configs/BatchLayoutConfig.h>
#include <dataTypes/configs/BasecallerBaselinerConfig.h>

#include <basecaller/traceAnalysis/HostMultiScaleBaseliner.h>
#include <basecaller/traceAnalysis/DeviceMultiScaleBaseliner.h>

using namespace PacBio::DataSource;
using namespace PacBio::Cuda::Memory;

using namespace PacBio::Mongo;
using namespace PacBio::Mongo::Data;
using namespace PacBio::Mongo::Basecaller;

using LaneArray = PacBio::Mongo::LaneArray<int16_t>;
using FloatArray = PacBio::Mongo::LaneArray<float>;

namespace PacBio {
namespace Mongo {
namespace Basecaller {

namespace {

struct HostTestConfig : public Configuration::PBConfig<HostTestConfig>
{
    PB_CONFIG(HostTestConfig);

    PB_CONFIG_OBJECT(Data::BasecallerBaselinerConfig, baselineConfig);

    PB_CONFIG_PARAM(ComputeDevices, analyzerHardware, ComputeDevices::Host);

    static BasecallerBaselinerConfig BaselinerConfig(BasecallerBaselinerConfig::FilterTypes type)
    {
        Json::Value json;
        json["baselineConfig"]["Method"] = "HostMultiScale";
        json["baselineConfig"]["Filter"] = type.toString();
        HostTestConfig cfg{json};

        return cfg.baselineConfig;
    }
};

struct DeviceTestConfig : public Configuration::PBConfig<DeviceTestConfig>
{
    PB_CONFIG(DeviceTestConfig);

    PB_CONFIG_OBJECT(Data::BasecallerBaselinerConfig, baselineConfig);

    PB_CONFIG_PARAM(ComputeDevices, analyzerHardware, ComputeDevices::V100);

    static BasecallerBaselinerConfig BaselinerConfig(BasecallerBaselinerConfig::FilterTypes type)
    {
        Json::Value json;
        json["baselineConfig"]["Method"] = "DeviceMultiScale";
        json["baselineConfig"]["Filter"] = type.toString();
        DeviceTestConfig cfg{json};

        return cfg.baselineConfig;
    }
};



static TraceBatch<int16_t> GenerateBatch(uint32_t lanesPerPool, uint32_t framesPerChunk)
{
    PacketLayout layout(PacketLayout::BLOCK_LAYOUT_DENSE, PacketLayout::INT16,
                            //     blocks          frames     width
                            {lanesPerPool, framesPerChunk, laneSize});
    size_t framesPerBlock = layout.NumFrames(), zmwPerBlock = layout.BlockWidth();

    std::default_random_engine gnr;
    std::normal_distribution<> dist(200, 20);
    boost::multi_array<int16_t, 3> dataBuf(boost::extents[lanesPerPool][framesPerBlock][zmwPerBlock]);
    auto rangeStart = dataBuf.data(), rangeEnd = (dataBuf.data() + dataBuf.num_elements());
    std::generate(rangeStart, rangeEnd, std::bind(dist, gnr));

    auto batchIdx = 0; auto startFrame = 0; auto endFrame = framesPerBlock; auto startZmw = 0;
    BatchMetadata meta(batchIdx, startFrame, endFrame, startZmw);
    BatchDimensions dims {(uint32_t)layout.NumBlocks(), (uint32_t)layout.NumFrames(), (uint32_t)layout.BlockWidth()};
    TraceBatch<int16_t> batch(meta, dims, SyncDirection::HostWriteDeviceRead, SOURCE_MARKER());

    auto li = 0 /* laneIdx */;
    std::memcpy(batch.GetBlockView(li).Data(), dataBuf[li].origin(), dataBuf.num_elements()*sizeof(int16_t));

    return batch;
}

}

class BenchBaseliner : public benchmark::Fixture {
protected:
    uint32_t numZmwLanes = 4;
    uint32_t numPools = 2;
    uint32_t lanesPerPool = numZmwLanes / numPools;
    uint32_t framesPerChunk = 512;

public:
    void SetUp(const ::benchmark::State& state) {
        framesPerChunk *= state.range(0); // mul by blocks
    }

    void TearDown(const ::benchmark::State& state) {
    }
};

BENCHMARK_DEFINE_F(BenchBaseliner, Host)(benchmark::State& st)
{
    Data::MovieConfig movConfig;
    auto baselinerConfig = HostTestConfig::BaselinerConfig(BasecallerBaselinerConfig::FilterTypes::TwoScaleMedium);
    HostMultiScaleBaseliner::Configure(baselinerConfig, movConfig);

    BaselinerParams blp({2, 8}, {9, 31}, 2.44f, 0.50f); // strides, widths, sigma, mean
    HostMultiScaleBaseliner baseliner(0, 1.1f, blp, lanesPerPool);

    TraceBatch<int16_t> batch = GenerateBatch(lanesPerPool, framesPerChunk);
    
    for (auto _ : st) {
        // This code gets timed
        auto res = baseliner(batch);

        benchmark::DoNotOptimize(res);  // Allow res to be clobbered.
        benchmark::ClobberMemory();     // and written to memory.
    }

    HostMultiScaleBaseliner::Finalize();
}

BENCHMARK_DEFINE_F(BenchBaseliner, Device)(benchmark::State& st)
{
    Data::MovieConfig movConfig;
    auto baselinerConfig = DeviceTestConfig::BaselinerConfig(BasecallerBaselinerConfig::FilterTypes::TwoScaleMedium);
    DeviceMultiScaleBaseliner::Configure(baselinerConfig, movConfig);

    BaselinerParams blp({2, 8}, {9, 31}, 2.44f, 0.50f); // strides, widths, sigma, mean
    DeviceMultiScaleBaseliner baseliner(0, 1.1f, blp, lanesPerPool);

    TraceBatch<int16_t> batch = GenerateBatch(lanesPerPool, framesPerChunk);
    
    for (auto _ : st) {
        // This code gets timed
        auto res = baseliner(batch);

        benchmark::DoNotOptimize(res);  // Allow res to be clobbered.
        benchmark::ClobberMemory();     // and written to memory.
    }

    DeviceMultiScaleBaseliner::Finalize();
}


BENCHMARK_REGISTER_F(BenchBaseliner, Host)->Unit(benchmark::kMicrosecond)
                                        ->RangeMultiplier(4)->Range(1, 16);

BENCHMARK_REGISTER_F(BenchBaseliner, Device)->Unit(benchmark::kMicrosecond)
                                        ->RangeMultiplier(4)->Range(1, 16);

// Benchmark entry point
BENCHMARK_MAIN();

}}} // PacBio::Mongo::Basecaller
