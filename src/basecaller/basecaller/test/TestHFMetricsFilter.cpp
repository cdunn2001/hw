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

#include <basecaller/traceAnalysis/DeviceHFMetricsFilter.h>
#include <basecaller/traceAnalysis/PulseAccumulator.h>
#include <basecaller/traceAnalysis/DetectionModelEstimator.h>
#include <basecaller/traceAnalysis/TraceHistogramAccumulator.h>
#include <basecaller/traceAnalysis/HFMetricsFilter.h>
#include <basecaller/traceAnalysis/HostHFMetricsFilter.h>
#include <basecaller/analyzer/BatchAnalyzer.h>

#include <common/DataGenerators/BatchGenerator.h>
#include <dataTypes/BasicTypes.h>
#include <dataTypes/BaselinerStatAccumState.h>
#include <dataTypes/BatchMetrics.h>
#include <dataTypes/LaneDetectionModel.h>
#include <dataTypes/configs/BasecallerPulseAccumConfig.h>
#include <dataTypes/configs/BasecallerMetricsConfig.h>
#include <dataTypes/configs/BatchLayoutConfig.h>
#include <dataTypes/configs/MovieConfig.h>
#include <common/MongoConstants.h>
#include <dataTypes/HQRFPhysicalStates.h>

#include <gtest/gtest.h>

namespace PacBio {
namespace Mongo {
namespace Basecaller {

namespace {

class TestConfig : public Configuration::PBConfig<TestConfig>
{
public:
    PB_CONFIG(TestConfig);

    PB_CONFIG_OBJECT(Data::BatchLayoutConfig, layout);
    PB_CONFIG_OBJECT(Data::BasecallerMetricsConfig, metrics);
    PB_CONFIG_OBJECT(Data::BasecallerPulseAccumConfig, pulses);
    PB_CONFIG_PARAM(ComputeDevices, analyzerHardware, ComputeDevices::Host);
};

// We simulate each base per zmw per block according to the following scheme:
struct BaseSimConfig
{
    BaseSimConfig()
    {
        config = TestConfig{};
        dims.framesPerBatch = config.layout.framesPerChunk;
        dims.laneWidth = laneSize;
        dims.lanesPerBatch = config.layout.lanesPerPool;
    };

    Data::BatchDimensions dims;
    unsigned int numBases = 10;
    unsigned int ipd = 1;
    unsigned int baseWidth = 4;
    double frameRate = 100;
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
    TestConfig config;
};

Data::PulseBatch GenerateBases(BaseSimConfig sim, size_t batchNo = 0)
{
    const auto& layoutConfig = sim.config.layout;
    const auto& pulseConfig = sim.config.pulses;

    Cuda::Data::BatchGenerator batchGenerator(layoutConfig.framesPerChunk,
                                              layoutConfig.zmwsPerLane,
                                              layoutConfig.lanesPerPool,
                                              8192,
                                              layoutConfig.lanesPerPool);
    auto chunk = batchGenerator.PopulateChunk();

    Data::PulseBatchFactory batchFactory(
        pulseConfig.maxCallsPerZmw,
        Cuda::Memory::SyncDirection::HostWriteDeviceRead);

    auto pulses = batchFactory.NewBatch(
            chunk.front().Metadata(),
            chunk.front().StorageDims()).first;

    auto LabelConv = [&](size_t index) {
        switch (sim.pattern[index % sim.pattern.size()])
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
            for (size_t b = 0; b < sim.numBases; ++b)
            {
                Data::Pulse pulse;

                auto label = LabelConv(b);

                // Populate pulse data
                pulse.Label(label);
                pulse.Start(b * sim.baseWidth + b * sim.ipd
                                    + batchNo * sim.dims.framesPerBatch)
                     .Width(sim.baseWidth);
                pulse.MidSignal(sim.midSignal[label] + b)
                     .MaxSignal(sim.maxSignal[label]);
                pulse.MeanSignal(pulse.MidSignal());
                float m2modifier = (b % 2 == 0) ? 1.1f : 1.01f;
                pulse.SignalM2(pulse.MidSignal()
                               * pulse.MidSignal()
                               * (pulse.Width() - 2) * m2modifier);
                pulse.IsReject(false);
                pulseView.push_back(zmw, pulse);
            }
        }
    }
    return pulses;
}

Data::BaselinerMetrics GenerateBaselineMetrics(BaseSimConfig config)
{
    Data::BaselinerMetrics ret(
            config.dims.lanesPerBatch,
            Cuda::Memory::SyncDirection::HostWriteDeviceRead,
            SOURCE_MARKER());
    Data::BaselinerStatAccumulator<Data::BaselinedTraceElement> bsa{};
    for (size_t lane = 0; lane < config.dims.lanesPerBatch; ++lane)
    {
        ret.baselinerStats.GetHostView()[lane] = bsa.GetState();
        auto& baselinerStats = ret.baselinerStats.GetHostView()[lane];
        for (size_t zmw = 0; zmw < laneSize; ++zmw)
        {
            baselinerStats.rawBaselineSum[zmw] = 100;
            baselinerStats.baselineStats.moment0[zmw] = 98;
            baselinerStats.baselineStats.moment1[zmw] = 10;
            baselinerStats.baselineStats.moment2[zmw] = 100;
            baselinerStats.fullAutocorrState.moment1First[zmw] = 10;
            baselinerStats.fullAutocorrState.moment1Last[zmw] = 20;
            baselinerStats.fullAutocorrState.moment2[zmw] = 120;
            baselinerStats.fullAutocorrState.basicStats.moment0[zmw] = 500;
            baselinerStats.fullAutocorrState.basicStats.moment1[zmw] = 1000;
            baselinerStats.fullAutocorrState.basicStats.moment2[zmw] = 10000;
        }
    }
    return ret;
}

Data::FrameLabelerMetrics GenerateFrameLabelerMetrics(BaseSimConfig config)
{
    Data::FrameLabelerMetrics ret(
            config.dims,
            Cuda::Memory::SyncDirection::HostWriteDeviceRead,
            SOURCE_MARKER());
    for (size_t lane = 0; lane < config.dims.lanesPerBatch; ++lane)
    {
        for (size_t zmw = 0; zmw < laneSize; ++zmw)
        {
            ret.viterbiScore.GetHostView()[lane][zmw] = 1.09;
        }
    }
    return ret;
}

Data::PulseDetectorMetrics GeneratePulseDetectorMetrics(BaseSimConfig config)
{
    Data::PulseDetectorMetrics ret(
            config.dims,
            Cuda::Memory::SyncDirection::HostWriteDeviceRead,
            SOURCE_MARKER());
    Data::BaselinerStatAccumulator<Data::BaselinedTraceElement> bsa{};
    for (size_t lane = 0; lane < config.dims.lanesPerBatch; ++lane)
    {
        ret.baselineStats.GetHostView()[lane] = bsa.GetState().baselineStats;
        auto& baselinerStats = ret.baselineStats.GetHostView()[lane];
        for (size_t zmw = 0; zmw < laneSize; ++zmw)
        {
            baselinerStats.moment0[zmw] = 98;
            baselinerStats.moment1[zmw] = 10;
            baselinerStats.moment2[zmw] = 100;
        }
    }
    return ret;
}

Cuda::Memory::UnifiedCudaArray<Data::LaneModelParameters<Cuda::PBHalf, laneSize>>
GenerateModels(BaseSimConfig sim)
{
    const auto& layoutConfig = sim.config.layout;
    Cuda::Memory::UnifiedCudaArray<Data::LaneModelParameters<Cuda::PBHalf,
                                                             laneSize>> models(
        layoutConfig.lanesPerPool,
        Cuda::Memory::SyncDirection::Symmetric,
        SOURCE_MARKER());
    Data::LaneModelParameters<Cuda::PBHalf, laneSize> model;
    model.AnalogMode(0).SetAllMeans(227.13);
    model.AnalogMode(1).SetAllMeans(154.45);
    model.AnalogMode(2).SetAllMeans(97.67);
    model.AnalogMode(3).SetAllMeans(61.32);

    model.AnalogMode(0).SetAllVars(776);
    model.AnalogMode(1).SetAllVars(426);
    model.AnalogMode(2).SetAllVars(226);
    model.AnalogMode(3).SetAllVars(132);

    // Need a new trace file to target, these values come from a file with
    // zero baseline mean
    model.BaselineMode().SetAllMeans(0);
    model.BaselineMode().SetAllVars(33);

    auto view = models.GetHostView();
    for (size_t i = 0; i < view.Size(); ++i)
    {
        view[i] = model;
    }
    return models;
}

} // anonymous namespace

template <typename HFT>
void testPopulated(HFT& hfMetrics, BaseSimConfig& sim)
{

    // TODO: test that the last block is finalized regardless of condition?

    size_t numFramesPerBatch = sim.config.layout.framesPerChunk;
    size_t numBatchesPerHFMB = sim.config.metrics.framesPerHFMetricBlock
                             / numFramesPerBatch; // = 32, for 4096 frame HFMBs

    sim.pattern = "ACGTGG";
    sim.ipd = 0;
    const auto& baselinerStats = GenerateBaselineMetrics(sim);
    const auto& models = GenerateModels(sim);
    const auto& flMetrics = GenerateFrameLabelerMetrics(sim);
    const auto& pdMetrics = GeneratePulseDetectorMetrics(sim);

    int blocks_tested = 0;

    for (size_t batchIdx = 0; batchIdx < numBatchesPerHFMB; ++batchIdx)
    {
        auto pulses = GenerateBases(sim, batchIdx);
        auto basecallingMetrics = hfMetrics(
                pulses, baselinerStats, models, flMetrics, pdMetrics);
        if (basecallingMetrics)
        {
            ASSERT_EQ(numBatchesPerHFMB - 1, batchIdx);
            for (uint32_t l = 0; l < pulses.Dims().lanesPerBatch; l++)
            {
                const auto& mb = basecallingMetrics->GetHostView()[l];
                const auto& bs = pdMetrics.baselineStats.GetHostView()[l];
                ASSERT_EQ(sizeof(mb), 8768);
                for (uint32_t z = 0; z < laneSize; ++z)
                {
                    ASSERT_EQ(numBatchesPerHFMB
                                * sim.numBases
                                * sim.baseWidth,
                              mb.numPulseFrames[z]);
                    ASSERT_EQ(numBatchesPerHFMB
                                * sim.numBases
                                * sim.baseWidth,
                              mb.numBaseFrames[z]);
                    // The pulses don't run to the end of each block, so all
                    // but one pulse is abutted. Plus we have the GG, which
                    // doesn't count as a sandwich.
                    ASSERT_EQ((numBatchesPerHFMB) * (sim.numBases - 2),
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
                    ASSERT_EQ(numBatchesPerHFMB * 2,
                              mb.numBasesByAnalog[0][z]);
                    ASSERT_EQ(numBatchesPerHFMB * 2,
                              mb.numBasesByAnalog[1][z]);
                    // G is over represented in the pattern above
                    ASSERT_EQ(numBatchesPerHFMB * 4,
                              mb.numBasesByAnalog[2][z]);
                    ASSERT_EQ(numBatchesPerHFMB * 2,
                              mb.numBasesByAnalog[3][z]);
                    ASSERT_EQ(numBatchesPerHFMB * sim.numBases,
                              mb.numPulses[z]);
                    ASSERT_EQ(numBatchesPerHFMB * sim.numBases,
                              mb.numBases[z]);
                    // This will always something random, doesn't matter at the
                    // moment
                    EXPECT_EQ(Data::HQRFPhysicalStates::MULTI,
                              mb.activityLabel[z]);
                    EXPECT_EQ(0,
                              mb.startFrame[z]);
                    EXPECT_EQ(numBatchesPerHFMB * numFramesPerBatch,
                              mb.startFrame[z] + mb.numFrames[z]);
                    EXPECT_EQ(numBatchesPerHFMB * numFramesPerBatch,
                              mb.numFrames[z]);
                    // We've already checked that numpulsesbyanalog is correct,
                    // so we'll just use it here for readability:
                    // Also note that we are subtracting out the partial frames
                    // (thus -2).
                    EXPECT_EQ(736, mb.pkMidSignal[0][z]);
                    EXPECT_EQ(1408, mb.pkMidSignal[1][z]);
                    EXPECT_EQ(944, mb.pkMidSignal[2][z]);
                    EXPECT_EQ(1152, mb.pkMidSignal[3][z]);
                    ASSERT_EQ(25, mb.pkMax[0][z]);
                    ASSERT_EQ(45, mb.pkMax[1][z]);
                    ASSERT_EQ(15, mb.pkMax[2][z]);
                    ASSERT_EQ(35, mb.pkMax[3][z]);
                    EXPECT_NEAR(0.0171916, mb.bpZvar[0][z], 0.001);
                    EXPECT_NEAR(0.0043878, mb.bpZvar[1][z], 0.001);
                    EXPECT_NEAR(0.0192236, mb.bpZvar[2][z], 0.001);
                    EXPECT_NEAR(0.0065546, mb.bpZvar[3][z], 0.001);
                    EXPECT_NEAR(0.0706076, mb.pkZvar[0][z], 0.001);
                    EXPECT_NEAR(0.0451073, mb.pkZvar[1][z], 0.001);
                    EXPECT_NEAR(0.0744171, mb.pkZvar[2][z], 0.001);
                    EXPECT_NEAR(0.0960086, mb.pkZvar[3][z], 0.001);
                    ASSERT_EQ(numBatchesPerHFMB * 2 * (sim.baseWidth - 2),
                              mb.numPkMidFrames[0][z]);
                    ASSERT_EQ(numBatchesPerHFMB * 2 * (sim.baseWidth - 2),
                              mb.numPkMidFrames[1][z]);
                    ASSERT_EQ(numBatchesPerHFMB * 4 * (sim.baseWidth - 2),
                              mb.numPkMidFrames[2][z]);
                    ASSERT_EQ(numBatchesPerHFMB * 2 * (sim.baseWidth - 2),
                              mb.numPkMidFrames[3][z]);
                    ASSERT_EQ(numBatchesPerHFMB * 2,
                              mb.numPkMidBasesByAnalog[0][z]);
                    ASSERT_EQ(numBatchesPerHFMB * 2,
                              mb.numPkMidBasesByAnalog[1][z]);
                    ASSERT_EQ(numBatchesPerHFMB * 4,
                              mb.numPkMidBasesByAnalog[2][z]);
                    ASSERT_EQ(numBatchesPerHFMB * 2,
                              mb.numPkMidBasesByAnalog[3][z]);
                    EXPECT_NEAR(0.0150028, mb.autocorrelation[z], 0.001);
                    EXPECT_NEAR(0.002128, mb.pulseDetectionScore[z], 0.0001);
                    EXPECT_NEAR(bs.moment0[z] * numBatchesPerHFMB, mb.numFramesBaseline[z], 0.0001);
                    EXPECT_NEAR(bs.moment1[z] / bs.moment0[z], mb.frameBaselineDWS[z], 0.0001);
                    EXPECT_NEAR((bs.moment2[z] - (bs.moment1[z] * bs.moment1[z] / bs.moment0[z])) / (bs.moment0[z] - 1.0f), mb.frameBaselineVarianceDWS[z], 0.01);
                    // TODO: These aren't expected to be "correct", and should
                    // be replaced when these metrics are expected to be
                    // correct. The values themselves may need to be helped
                    // with some simulation in the above functions
                    EXPECT_EQ(0, mb.pixelChecksum[z]);
                }
            }
            ++blocks_tested;
        }
    }
    EXPECT_EQ(1, blocks_tested);
    hfMetrics.Finalize();
}

TEST(TestHFMetricsFilter, Populated_Device)
{
    BaseSimConfig sim;
    {
        const auto& metricsConfig = sim.config.metrics;
        DeviceHFMetricsFilter::Configure(metricsConfig.sandwichTolerance,
                                         metricsConfig.framesPerHFMetricBlock,
                                         sim.frameRate,
                                         metricsConfig.realtimeActivityLabels);
    }


    int poolId = 0;
    DeviceHFMetricsFilter hfMetrics(poolId, sim.dims.lanesPerBatch);
    testPopulated(hfMetrics, sim);
}

TEST(TestHFMetricsFilter, Populated)
{
    BaseSimConfig sim;
    {
        const auto& metricsConfig = sim.config.metrics;
        HFMetricsFilter::Configure(metricsConfig.sandwichTolerance,
                                   metricsConfig.framesPerHFMetricBlock,
                                   sim.frameRate,
                                   metricsConfig.realtimeActivityLabels);
    }


    int poolId = 0;
    HostHFMetricsFilter hfMetrics(poolId, sim.dims.lanesPerBatch);
    testPopulated(hfMetrics, sim);
}

TEST(TestHFMetricsFilter, Noop)
{
    BaseSimConfig sim;
    const auto& layoutConfig = sim.config.layout;
    const auto& metricsConfig = sim.config.metrics;
    {
        NoHFMetricsFilter::Configure(metricsConfig.sandwichTolerance,
                                     metricsConfig.framesPerHFMetricBlock,
                                     sim.frameRate,
                                     metricsConfig.realtimeActivityLabels);
    }
    Cuda::Memory::UnifiedCudaArray<Data::LaneModelParameters<Cuda::PBHalf,
                                                             laneSize>> models(
        layoutConfig.lanesPerPool,
        Cuda::Memory::SyncDirection::Symmetric,
        SOURCE_MARKER());


    int poolId = 0;
    NoHFMetricsFilter hfMetrics(poolId);
    size_t numFramesPerBatch = layoutConfig.framesPerChunk;
    size_t numBatchesPerHFMB = metricsConfig.framesPerHFMetricBlock
                             / numFramesPerBatch; // = 32, for 4096 frame HFMBs
    sim.ipd = 0;
    const auto& baselinerStats = GenerateBaselineMetrics(sim);
    const auto& flMetrics = GenerateFrameLabelerMetrics(sim);
    const auto& pdMetrics = GeneratePulseDetectorMetrics(sim);

    for (size_t batchIdx = 0; batchIdx < numBatchesPerHFMB; ++batchIdx)
    {
        auto pulses = GenerateBases(sim, batchIdx);
        auto basecallingMetrics = hfMetrics(pulses, baselinerStats, models, flMetrics, pdMetrics);
        ASSERT_FALSE(basecallingMetrics);
    }
    hfMetrics.Finalize();
}


}}} // PacBio::Mongo::Basecaller
