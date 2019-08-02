// Copyright (c) 2019, Pacific Biosciences of California, Inc.
//
// All rights reserved.
//
// THIS SOFTWARE CONSTITUTES AND EMBODIES PACIFIC BIOSCIENCES' CONFIDENTIAL
// AND PROPRIETARY INFORMATION.
//
// Disclosure, redistribution and use of this software is subject to the
// terms and conditions of the applicable written agreement(s) between you
// and Pacific Biosciences, where "you" refers to you or your company or
// organization, as applicable.  Any other disclosure, redistribution or
// use is prohibited.
//
// THIS SOFTWARE IS PROVIDED BY PACIFIC BIOSCIENCES AND ITS CONTRIBUTORS "AS
// IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
// THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL PACIFIC BIOSCIENCES OR ITS
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
// OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
// WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
// OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
// ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
//  Description:
//  Defines unit tests for the strategies for estimation and subtraction of
//  baseline and estimation of associated statistics.

#include <basecaller/traceAnalysis/HostMultiScaleBaseliner.h>
#include <basecaller/traceAnalysis/DeviceMultiScaleBaseliner.h>
#include <basecaller/traceAnalysis/DeviceSGCFrameLabeler.h>
#include <basecaller/traceAnalysis/PulseAccumulator.h>
#include <basecaller/traceAnalysis/DetectionModelEstimator.h>
#include <basecaller/traceAnalysis/TraceHistogramAccumulator.h>
#include <basecaller/traceAnalysis/HFMetricsFilter.h>
#include <basecaller/analyzer/BatchAnalyzer.h>

#include <common/DataGenerators/BatchGenerator.h>
#include <dataTypes/BasecallerConfig.h>
#include <dataTypes/BaselineStats.h>
#include <dataTypes/LaneDetectionModel.h>
#include <dataTypes/MovieConfig.h>
#include <dataTypes/PrimaryConfig.h>
#include <common/MongoConstants.h>

#include <gtest/gtest.h>

namespace PacBio {
namespace Mongo {
namespace Basecaller {

namespace {

// We simulate each base per zmw per block according to the following scheme:
struct BaseSimConfig
{
    unsigned int numBases = 10;
    unsigned int ipd = 1;
    unsigned int baseWidth = 3;
    unsigned int meanSignal = 50;
    unsigned int midSignal = 55;
    unsigned int maxSignal = 72;
};

Data::PulseBatch GenerateBases(BaseSimConfig config, size_t batchNo=0)
{
    // TODO: replace with PulseBatch
    Cuda::Data::BatchGenerator batchGenerator(Data::GetPrimaryConfig().framesPerChunk,
                                              Data::GetPrimaryConfig().zmwsPerLane,
                                              Data::GetPrimaryConfig().lanesPerPool,
                                              8192,
                                              Data::GetPrimaryConfig().lanesPerPool);
    auto chunk = batchGenerator.PopulateChunk();

    unsigned int poolSize = Data::GetPrimaryConfig().lanesPerPool;
    unsigned int chunkSize = Data::GetPrimaryConfig().framesPerChunk;

    Data::BatchDimensions dims;
    dims.framesPerBatch = chunkSize;
    dims.laneWidth = laneSize;
    dims.lanesPerBatch = poolSize;

    Data::BasecallerAlgorithmConfig basecallerConfig;
    Data::PulseBatchFactory batchFactory(
        basecallerConfig.pulseAccumConfig.maxCallsPerZmw,
        dims,
        Cuda::Memory::SyncDirection::HostWriteDeviceRead,
        true);

    auto pulses = batchFactory.NewBatch(chunk.front().Metadata());

    auto LabelConv = [](size_t index) {
        switch (index % 4)
        {
        case 0:
            return Data::Pulse::NucleotideLabel::A;
        case 1:
            return Data::Pulse::NucleotideLabel::C;
        case 2:
            return Data::Pulse::NucleotideLabel::G;
        case 3:
            return Data::Pulse::NucleotideLabel::T;
        case 4:
            return Data::Pulse::NucleotideLabel::N;
        default:
            return Data::Pulse::NucleotideLabel::NONE;
        }
    };
    for (size_t lane = 0; lane < pulses.Dims().lanesPerBatch; ++lane)
    {
        auto pulseView = pulses.Pulses().LaneView(lane);

        pulseView.Reset();
        for (size_t zmw = 0; zmw < laneSize; ++zmw)
        {
            for (size_t b = 0; b < config.numBases; ++b)
            {
                Data::Pulse pulse;

                auto label = LabelConv(b);

                // Populate pulse data
                pulse.Label(label);
                pulse.Start(b * config.baseWidth + b * config.ipd
                                    + batchNo * chunkSize)
                             .Width(config.baseWidth);
                pulse.MeanSignal(config.meanSignal)
                             .MidSignal(config.midSignal)
                             .MaxSignal(config.maxSignal);
                pulseView.push_back(zmw, pulse);
            }
        }
    }
    return pulses;
}

Cuda::Memory::UnifiedCudaArray<Data::BaselineStats<laneSize>> GenerateBaselineStats(BaseSimConfig config)
{
    unsigned int poolSize = Data::GetPrimaryConfig().lanesPerPool;
    Cuda::Memory::UnifiedCudaArray<Data::BaselineStats<laneSize>> ret(
            poolSize,
            Cuda::Memory::SyncDirection::HostWriteDeviceRead,
            true);
    for (size_t lane = 0; lane < poolSize; ++lane)
    {
        auto baselineStats = ret.GetHostView()[lane];
        for (size_t zmw = 0; zmw < laneSize; ++zmw)
        {
            baselineStats.rawBaselineSum_[zmw] = 100;
            baselineStats.m0_[zmw] = 98;
            baselineStats.m1_[zmw] = 1;
            baselineStats.m2_[zmw] = 100;
        }
    }
    return ret;
}

} // anonymous namespace

TEST(TestHFMetricsFilter, Populated)
{
    {
        Data::BasecallerAlgorithmConfig basecallerConfig{};
        HostHFMetricsFilter::Configure(basecallerConfig.Metrics);
    }

    Cuda::Memory::UnifiedCudaArray<Data::LaneModelParameters<Cuda::PBHalf,
                                                             laneSize>> models(
        Data::GetPrimaryConfig().lanesPerPool,
        Cuda::Memory::SyncDirection::Symmetric,
        true);

    int poolId = 0;
    HostHFMetricsFilter hfMetrics(poolId);

    // TODO: test that the last block is finalized regardless of condition?

    size_t numFramesPerBatch = 128;
    size_t numBatchesPerHFMB = Data::GetPrimaryConfig().framesPerHFMetricBlock
                             / numFramesPerBatch; // = 32, for 4096 frame HFMBs

    BaseSimConfig config;
    config.ipd = 0;
    const auto& baselineStats = GenerateBaselineStats(config);

    int blocks_tested = 0;

    for (size_t batchIdx = 0; batchIdx < numBatchesPerHFMB; ++batchIdx)
    {
        auto pulses = GenerateBases(config, batchIdx);
        auto basecallingMetrics = hfMetrics(pulses, baselineStats, models);
        // TODO: add a check that the following actually runs exactly once:
        if (basecallingMetrics)
        {
            ASSERT_EQ(numBatchesPerHFMB - 1, batchIdx); // = 31, HFMB is complete
            for (uint32_t l = 0; l < pulses.Dims().lanesPerBatch; l++)
            {
                const auto& mb = basecallingMetrics->GetHostView()[l];
                for (uint32_t z = 0; z < laneSize; ++z)
                {
                    EXPECT_EQ(numBatchesPerHFMB
                                * config.numBases
                                * config.baseWidth,
                              mb.numPulseFrames[z]);
                    EXPECT_EQ(numBatchesPerHFMB
                                * config.numBases
                                * config.baseWidth,
                              mb.numBaseFrames[z]);
                    // The pulses don't run to the end of each block, so all
                    // but one pulse is abutted
                    ASSERT_EQ((numBatchesPerHFMB) * (config.numBases - 1),
                              mb.numHalfSandwiches[z]);
                    // If numBases isn't evenly divisible by numAnalogs, the
                    // first analogs will be padded by the remainder
                    // e.g. 10 pulses per chunk means 2 for each analog, then
                    // three for the first two analogs:
                    EXPECT_EQ(numBatchesPerHFMB
                                * (config.numBases/numAnalogs
                                   + (config.numBases % numAnalogs > 0 ? 1 : 0)),
                              mb.numPulsesByAnalog[0][z]);
                    EXPECT_EQ(numBatchesPerHFMB
                                * (config.numBases/numAnalogs
                                   + (config.numBases % numAnalogs > 1 ? 1 : 0)),
                              mb.numPulsesByAnalog[1][z]);
                    EXPECT_EQ(numBatchesPerHFMB
                                * (config.numBases/numAnalogs
                                   + (config.numBases % numAnalogs > 2 ? 1 : 0)),
                              mb.numPulsesByAnalog[2][z]);
                    EXPECT_EQ(numBatchesPerHFMB
                                * (config.numBases/numAnalogs
                                   + (config.numBases % numAnalogs > 3 ? 1 : 0)),
                              mb.numPulsesByAnalog[3][z]);
                    ASSERT_EQ(numBatchesPerHFMB * config.numBases,
                              mb.numPulses[z]);
                    ASSERT_EQ(numBatchesPerHFMB * config.numBases,
                              mb.numBases[z]);
                }
            }
            ++blocks_tested;
        }
    }
    EXPECT_EQ(1, blocks_tested);
    hfMetrics.Finalize();
}

TEST(TestHFMetricsFilter, Noop)
{
    {
        Data::BasecallerAlgorithmConfig basecallerConfig{};
        NoHFMetricsFilter::Configure(basecallerConfig.Metrics);
    }
    Cuda::Memory::UnifiedCudaArray<Data::LaneModelParameters<Cuda::PBHalf,
                                                             laneSize>> models(
        Data::GetPrimaryConfig().lanesPerPool,
        Cuda::Memory::SyncDirection::Symmetric,
        true);


    int poolId = 0;
    NoHFMetricsFilter hfMetrics(poolId);
    size_t numFramesPerBatch = 128;
    size_t numBatchesPerHFMB = Data::GetPrimaryConfig().framesPerHFMetricBlock
                             / numFramesPerBatch; // = 32, for 4096 frame HFMBs
    BaseSimConfig config;
    config.ipd = 0;
    const auto& baselineStats = GenerateBaselineStats(config);

    for (size_t batchIdx = 0; batchIdx < numBatchesPerHFMB; ++batchIdx)
    {
        auto pulses = GenerateBases(config, batchIdx);
        auto basecallingMetrics = hfMetrics(pulses, baselineStats, models);
        ASSERT_FALSE(basecallingMetrics);
    }
    hfMetrics.Finalize();
}


}}} // PacBio::Mongo::Basecaller
