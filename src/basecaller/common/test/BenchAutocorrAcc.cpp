
#if 1
#include "common/AutocorrAccumulator.h"

#include <chrono>
#include <vector>
#include <random>
#include <algorithm>

#include <Eigen/Core>

#include <benchmark/benchmark.h>

#include <common/LaneArray.h>

using namespace PacBio::Mongo;


std::vector<float> bench_data(size_t w, size_t n)
{
  std::vector<float> rarr(w*n);
  std::default_random_engine gnr;
  std::normal_distribution<float> dist(5.0, 2.0);
  std::generate(rarr.begin(), rarr.end(), std::bind(dist, gnr));
  return rarr;
}

static void BM_AutoCorrVarLaneArray(benchmark::State& state)
{
  using FloatArray = LaneArray<float>;

  auto w = laneSize;
  auto n = state.range(0);
  auto rarr = bench_data(w, n);

  std::vector<FloatArray> farr(n);
  for (decltype(n) i = 0; i < n; ++i)
  {
    farr[i] = FloatArray(MemoryRange<float, laneSize>(&rarr[i*w]));
  }

  for (auto _ : state) {
    // This code gets timed
    AutocorrAccumulator<FloatArray> aca(FloatArray(0.0f));

    for (auto &r_ : farr)
    {
        aca.AddSample(r_);
    }
    auto var = aca.Variance();
    auto var3 = var.data()[0][3];
    benchmark::DoNotOptimize(var3); // Allow var...data() to be clobbered.
    benchmark::ClobberMemory();     // and written to memory.
  }
}

// Scan array in direction of axis 0, variance is equal to BM_AutoCorrVarLaneArray
static void BM_AutoCorrVarEigenViewAxis0(benchmark::State& state)
{
  // Perform setup here
  auto w = laneSize;
  auto n = state.range(0);
  auto rarr = bench_data(w, n);

  for (auto _ : state) {
    // This code gets timed
    Eigen::Map<Eigen::MatrixXf> xr(rarr.data(), w, n);
    auto var = (xr.colwise() - xr.rowwise().mean()).array().square().rowwise().sum() / (n - 1);
    auto var3 = var[3];

    benchmark::DoNotOptimize(var3); // Allow var...data() to be clobbered.
    benchmark::ClobberMemory();     // and written to memory.
  }
}

// Scan array in direction of axis 1, variance is NOT equal to BM_AutoCorrVarLaneArray
static void BM_AutoCorrVarEigenViewAxis1(benchmark::State& state)
{
  // Perform setup here
  auto w = laneSize;
  auto n = state.range(0);
  auto rarr = bench_data(w, n);

  for (auto _ : state) {
    // This code gets timed
    Eigen::Map<Eigen::MatrixXf> xr(&rarr[0], n, w);  // transpose dimensions
    auto var = (xr.rowwise() - xr.colwise().mean()).array().square().colwise().sum() / (n - 1);
    auto var3 = var[3];

    benchmark::DoNotOptimize(var3); // Allow var...data() to be clobbered.
    benchmark::ClobberMemory();     // and written to memory.
  }
}

// Register the function as a benchmark
BENCHMARK(BM_AutoCorrVarLaneArray)->Unit(benchmark::kMicrosecond)
                                  ->RangeMultiplier(16)->Range(16, 4096);
BENCHMARK(BM_AutoCorrVarEigenViewAxis0)->Unit(benchmark::kMicrosecond)
                                  ->RangeMultiplier(16)->Range(16, 4096);
BENCHMARK(BM_AutoCorrVarEigenViewAxis1)->Unit(benchmark::kMicrosecond)
                                  ->RangeMultiplier(16)->Range(16, 4096);

// Run the benchmark
BENCHMARK_MAIN();

#endif // 0