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
#include <common/BlockActivityLabels.h>

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
    std::map<Data::Pulse::NucleotideLabel, unsigned int> midSignal = {
        {Data::Pulse::NucleotideLabel::G, 10},
        {Data::Pulse::NucleotideLabel::A, 20},
        {Data::Pulse::NucleotideLabel::T, 30},
        {Data::Pulse::NucleotideLabel::C, 40}};
    std::map<Data::Pulse::NucleotideLabel, unsigned int> maxSignal = {
        {Data::Pulse::NucleotideLabel::G, 15},
        {Data::Pulse::NucleotideLabel::A, 25},
        {Data::Pulse::NucleotideLabel::T, 35},
        {Data::Pulse::NucleotideLabel::C, 45}};
    std::string pattern = "ACGT";
};

Data::PulseBatch GenerateBases(BaseSimConfig config, size_t batchNo = 0)
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

    auto LabelConv = [&](size_t index) {
        switch (config.pattern[index % config.pattern.size()])
        {
        case 'A':
            return Data::Pulse::NucleotideLabel::A;
        case 'C':
            return Data::Pulse::NucleotideLabel::C;
        case 'G':
            return Data::Pulse::NucleotideLabel::G;
        case 'T':
            return Data::Pulse::NucleotideLabel::T;
        case 'N':
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
                     .MidSignal(config.midSignal[label])
                     .MaxSignal(config.maxSignal[label]);
                pulse.SignalM2(pulse.MeanSignal()
                               * pulse.MeanSignal()
                               * pulse.Width());
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
            baselineStats.m1_[zmw] = 10;
            baselineStats.m2_[zmw] = 100;
        }
    }
    /*
    for (size_t lane = 0; lane < poolSize; ++lane)
    {
        auto baselineStats = ret.GetHostView()[lane];
        for (size_t zmw = 0; zmw < laneSize; ++zmw)
        {
            std::cout << baselineStats.rawBaselineSum_[zmw] << " ";
            std::cout << baselineStats.m0_[zmw] << " ";
            std::cout << baselineStats.m1_[zmw] << " ";
            std::cout << baselineStats.m2_[zmw] << " ";
            std::cout << std::endl;
        }
    }
    */
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
    config.pattern = "ACGTGG";
    config.ipd = 0;
    const auto& baselineStats = GenerateBaselineStats(config);
    /*
    for (size_t l = 0; l < Data::GetPrimaryConfig().lanesPerPool; l++)
    {
        const auto& laneStats = baselineStats.GetHostView()[l];
        for (size_t zi = 0; zi < laneSize; ++zi)
        {
            std::cout << laneStats.m0_[zi] << " " << laneStats.m1_[zi] << " " << laneStats.m2_[zi] << std::endl;
        }
    }
    return;
    */

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
                ASSERT_EQ(sizeof(mb), 8128);
                for (uint32_t z = 0; z < laneSize; ++z)
                {
                    ASSERT_EQ(numBatchesPerHFMB
                                * config.numBases
                                * config.baseWidth,
                              mb.numPulseFrames[z]);
                    ASSERT_EQ(numBatchesPerHFMB
                                * config.numBases
                                * config.baseWidth,
                              mb.numBaseFrames[z]);
                    // The pulses don't run to the end of each block, so all
                    // but one pulse is abutted. Plus we have the GG, which
                    // doesn't count as a sandwich.
                    ASSERT_EQ((numBatchesPerHFMB) * (config.numBases - 2),
                              mb.numHalfSandwiches[z]);
                    // Sandwiches are one per block, on the 'GTG'
                    ASSERT_EQ(numBatchesPerHFMB,
                              mb.numSandwiches[z]);
                    // Stutters are one per block, on the 'GG'
                    ASSERT_EQ(numBatchesPerHFMB,
                              mb.numPulseLabelStutters[z]);
                    ASSERT_EQ(numBatchesPerHFMB * 2,
                              mb.numPulsesByAnalog[0][z]);
                    ASSERT_EQ(numBatchesPerHFMB * 2,
                              mb.numPulsesByAnalog[1][z]);
                    // G is over represented in the pattern above
                    ASSERT_EQ(numBatchesPerHFMB * 4,
                              mb.numPulsesByAnalog[2][z]);
                    ASSERT_EQ(numBatchesPerHFMB * 2,
                              mb.numPulsesByAnalog[3][z]);
                    ASSERT_EQ(numBatchesPerHFMB * config.numBases,
                              mb.numPulses[z]);
                    ASSERT_EQ(numBatchesPerHFMB * config.numBases,
                              mb.numBases[z]);
                    // This will always be empty, unless features like
                    // autocorrelation are populated
                    ASSERT_EQ(ActivityLabeler::HQRFPhysicalState::EMPTY,
                              mb.activityLabel[z]);
                    // We've already checked that numpulsesbyanalog is correct,
                    // so we'll just use it here for readability:
                    // Also note that we are subtracting out the partial frames
                    // (thus -2).
                    ASSERT_EQ(20 * (config.baseWidth - 2) * mb.numPulsesByAnalog[0][z],
                              mb.pkMidSignal[0][z]);
                    ASSERT_EQ(40 * (config.baseWidth - 2) * mb.numPulsesByAnalog[1][z],
                              mb.pkMidSignal[1][z]);
                    ASSERT_EQ(10 * (config.baseWidth - 2) * mb.numPulsesByAnalog[2][z],
                              mb.pkMidSignal[2][z]);
                    ASSERT_EQ(30 * (config.baseWidth - 2) * mb.numPulsesByAnalog[3][z],
                              mb.pkMidSignal[3][z]);
                    ASSERT_EQ(25,
                              mb.pkMax[0][z]);
                    ASSERT_EQ(45,
                              mb.pkMax[1][z]);
                    ASSERT_EQ(15,
                              mb.pkMax[2][z]);
                    ASSERT_EQ(35,
                              mb.pkMax[3][z]);
                    ASSERT_EQ(25,
                              mb.bpZvar[0][z]);
                    ASSERT_EQ(45,
                              mb.bpZvar[1][z]);
                    ASSERT_EQ(15,
                              mb.bpZvar[2][z]);
                    ASSERT_EQ(35,
                              mb.bpZvar[3][z]);
                    ASSERT_EQ(25,
                              mb.pkZvar[0][z]);
                    ASSERT_EQ(45,
                              mb.pkZvar[1][z]);
                    ASSERT_EQ(15,
                              mb.pkZvar[2][z]);
                    ASSERT_EQ(35,
                              mb.pkZvar[3][z]);
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
