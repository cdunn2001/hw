
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
//  Defines unit tests for the strategies for pulse accumulation

#include <basecaller/traceAnalysis/PulseAccumulator.h>
#include <basecaller/traceAnalysis/HostPulseAccumulator.h>
#include <basecaller/traceAnalysis/HostSimulatedPulseAccumulator.h>
#include <basecaller/traceAnalysis/DevicePulseAccumulator.h>

#include <common/DataGenerators/BatchGenerator.h>
#include <common/StatAccumulator.h>

#include <dataTypes/configs/BasecallerPulseAccumConfig.h>
#include <dataTypes/configs/BatchLayoutConfig.h>
#include <dataTypes/configs/MovieConfig.h>
#include <dataTypes/CameraTraceBatch.h>
#include <dataTypes/LabelsBatch.h>

#include <gtest/gtest.h>
#include <random>

namespace PacBio {
namespace Mongo {
namespace Basecaller {

namespace {

// Creates a run configuration, with the number of lanes
// easily configurable on construction
struct TestConfig : Configuration::PBConfig<TestConfig>
{
    PB_CONFIG(TestConfig);

    PB_CONFIG_OBJECT(Data::BatchLayoutConfig, layout);
    PB_CONFIG_OBJECT(Data::BasecallerPulseAccumConfig, pulses);

    TestConfig(size_t numLanes)
        : TestConfig(InitJson(numLanes))
    {
        dims_.framesPerBatch = layout.framesPerChunk;
        dims_.laneWidth = layout.zmwsPerLane;
        dims_.lanesPerBatch = layout.lanesPerPool;
    }

    const Data::BatchDimensions& Dims() const
    {
        return dims_;
    }
private:
    static Json::Value InitJson(size_t numLanes)
    {
        Json::Value ret;
        ret["layout"]["lanesPerPool"] = numLanes;
        return ret;
    }

    Data::BatchDimensions dims_;
};

}

TEST(TestNoOpPulseAccumulator, Run)
{
    TestConfig config{1};
    const auto framesPerChunk = config.layout.framesPerChunk;

    PulseAccumulator::Configure(config.pulses);

    auto cameraBatchFactory = std::make_unique<Data::CameraBatchFactory>(
            Cuda::Memory::SyncDirection::HostWriteDeviceRead);

    auto labelsBatchFactory = std::make_unique<Data::LabelsBatchFactory>(
            16u,    // NOTE: Viterbi frame latency lookback, eventually this should not be hard-coded.
            Cuda::Memory::SyncDirection::HostWriteDeviceRead);

    uint32_t poolId = 0;
    auto cameraBatch = cameraBatchFactory->NewBatch(Data::BatchMetadata(0, 0, framesPerChunk, 0), config.Dims());
    auto labelsBatch = labelsBatchFactory->NewBatch(std::move(cameraBatch.first));

    PulseAccumulator pulseAccumulator(poolId);

    auto pulseBatch = pulseAccumulator(std::move(labelsBatch.first)).first;

    for (uint32_t laneIdx = 0; laneIdx < pulseBatch.Dims().lanesPerBatch; ++laneIdx)
    {
        const auto& lanePulses = pulseBatch.Pulses().LaneView(laneIdx);
        for (uint32_t zmwIdx = 0; zmwIdx < laneSize; ++zmwIdx)
        {
            EXPECT_EQ(0, lanePulses.size(zmwIdx));
        }
    }

    PulseAccumulator::Finalize();
}

TEST(TestHostSimulatedPulseAccumulator, Run)
{
    TestConfig config{1};
    const auto framesPerChunk = config.layout.framesPerChunk;

    HostSimulatedPulseAccumulator::Configure(config.pulses);

    auto cameraBatchFactory = std::make_unique<Data::CameraBatchFactory>(
            Cuda::Memory::SyncDirection::HostWriteDeviceRead);

    auto labelsBatchFactory = std::make_unique<Data::LabelsBatchFactory>(
            16u,    // NOTE: Viterbi frame latency lookback, eventually this should not be hard-coded.
            Cuda::Memory::SyncDirection::HostWriteDeviceRead);

    uint32_t poolId = 0;
    auto cameraBatch = cameraBatchFactory->NewBatch(Data::BatchMetadata(0, 0, framesPerChunk, 0), config.Dims());
    auto labelsBatch = labelsBatchFactory->NewBatch(std::move(cameraBatch.first));

    HostSimulatedPulseAccumulator pulseAccumulator(poolId);

    auto pulseBatch = pulseAccumulator(std::move(labelsBatch.first)).first;

    using NucleotideLabel = Data::Pulse::NucleotideLabel;

    // Repeating sequence of ACGT.
    const NucleotideLabel labels[] =
            { NucleotideLabel::A, NucleotideLabel::C, NucleotideLabel::G, NucleotideLabel::T };

    for (uint32_t laneIdx = 0; laneIdx < pulseBatch.Dims().lanesPerBatch; ++laneIdx)
    {
        const auto& lanePulses = pulseBatch.Pulses().LaneView(laneIdx);
        for (uint32_t zmwIdx = 0; zmwIdx < laneSize; ++zmwIdx)
        {
            EXPECT_EQ(config.pulses.maxCallsPerZmw, lanePulses.size(zmwIdx));
            for (uint32_t pulseNum = 0; pulseNum < config.pulses.maxCallsPerZmw; ++pulseNum)
            {
                const auto& pulse = lanePulses.ZmwData(zmwIdx)[pulseNum];
                EXPECT_EQ(labels[pulseNum % 4], pulse.Label());
                EXPECT_EQ(1, pulse.Start());
                EXPECT_EQ(3, pulse.Width());
            }
        }
    }

    HostSimulatedPulseAccumulator::Finalize();
}

namespace {

template <typename PulseAccumulatorToTest>
void TestPulseAccumulator()
{
    TestConfig config{1};
    const auto framesPerChunk = config.layout.framesPerChunk;

    Data::MovieConfig movieConfig;
    movieConfig.analogs[0].baseLabel = 'A';
    movieConfig.analogs[1].baseLabel = 'C';
    movieConfig.analogs[2].baseLabel = 'G';
    movieConfig.analogs[3].baseLabel = 'T';
    PulseAccumulatorToTest::Configure(movieConfig, config.pulses);

    auto cameraBatchFactory = std::make_unique<Data::CameraBatchFactory>(
            Cuda::Memory::SyncDirection::HostWriteDeviceRead);

    auto labelsBatchFactory = std::make_unique<Data::LabelsBatchFactory>(
            16u,    // NOTE: Viterbi frame latency lookback, eventually this should not be hard-coded.
            Cuda::Memory::SyncDirection::HostWriteDeviceRead);

    uint32_t poolId = 0;
    auto cameraBatch = cameraBatchFactory->NewBatch(Data::BatchMetadata(0, 0, framesPerChunk, 0), config.Dims());
    // Discard metrics:
    auto labelsBatch = labelsBatchFactory->NewBatch(std::move(cameraBatch.first)).first;

    // Simulate out labels batch accordingly fixed pattern of baseline + pulse frames.
    const size_t ipd = 6;
    const size_t pw = 10;
    assert(config.Dims().framesPerBatch % (ipd + pw) == 0);
    std::vector<Data::LabelsBatch::ElementType> simLabels;
    std::vector<Data::BaselinedTraceElement> simTrc;

    // Generate normally distributed baseline with given mean and variance.
    const short baselineMean = 0;
    const short baselineStd = 10;
    std::mt19937 gen;
    std::normal_distribution<> d{baselineMean, baselineStd};

    // Fixed signal values for pulses.
    const short latTraceVal = 40;
    const short curTraceVal = 50;

    size_t baselineFrames = 0;
    {
        size_t frameNum = 0;
        size_t base = 0;
        while (frameNum < framesPerChunk)
        {
            for (size_t b = 0; b < ipd; b++)
            {
                simTrc.push_back(std::round(d(gen)));
            }
            simLabels.insert(simLabels.end(), ipd, 0);
            if (frameNum < framesPerChunk - 16u)
            {
                baselineFrames += ipd;
            }
            frameNum += ipd;

            // Insert pulse down states and final pulse up state to complete pulse.
            assert(pw > 2);
            simLabels.insert(simLabels.end(), 1, (base % 4) + 5);
            simLabels.insert(simLabels.end(), pw-1, (base % 4 ) + 9);

            // Hardcode latency for now.
            simTrc.insert(simTrc.end(), pw, frameNum < 16u ? latTraceVal : curTraceVal);

            base++;
            frameNum += pw;
        }
    }

    for (uint32_t laneIdx = 0; laneIdx < labelsBatch.LanesPerBatch(); ++laneIdx)
    {
        auto blockLabels = labelsBatch.GetBlockView(laneIdx);
        auto latTrace = labelsBatch.LatentTrace().GetBlockView(laneIdx);
        auto curTrace = labelsBatch.TraceData().GetBlockView(laneIdx);
        for (size_t frameNum = 0; frameNum < simLabels.size(); ++frameNum)
        {
            std::vector<Data::LabelsBatch::ElementType> simll(blockLabels.LaneWidth(), simLabels[frameNum]);
            std::memcpy(blockLabels.Data() + (frameNum * blockLabels.LaneWidth()),
                        simll.data(), sizeof(Data::LabelsBatch::ElementType) * simll.size());

            std::vector<Data::BaselinedTraceElement> trcVal(curTrace.LaneWidth(), simTrc[frameNum]);
            if (frameNum < latTrace.NumFrames())
            {
                std::memcpy(latTrace.Data() + (frameNum * latTrace.LaneWidth()),
                            trcVal.data(), sizeof(Data::BaselinedTraceElement) * trcVal.size());
            }
            else
            {
                std::memcpy(curTrace.Data() + ((frameNum - latTrace.NumFrames()) * curTrace.LaneWidth()),
                            trcVal.data(), sizeof(Data::BaselinedTraceElement) * trcVal.size());
            }
        }
    }

    labelsBatch.TraceData().SetFrameLimit(labelsBatch.NumFrames() - labelsBatch.LatentTrace().NumFrames());

    assert(labelsBatch.NumFrames() == labelsBatch.TraceData().NumFrames() + labelsBatch.LatentTrace().NumFrames());

    PulseAccumulatorToTest pulseAccumulator(poolId, config.Dims().lanesPerBatch);

    const auto& pulseRet = pulseAccumulator(std::move(labelsBatch));
    const auto& pulseBatch = pulseRet.first;
    const auto& pulseMetrics = pulseRet.second;

    using NucleotideLabel = Data::Pulse::NucleotideLabel;

    // Repeating sequence of ACGT.
    const NucleotideLabel labels[] =
            { NucleotideLabel::A, NucleotideLabel::C, NucleotideLabel::G, NucleotideLabel::T };

    for (uint32_t laneIdx = 0; laneIdx < pulseBatch.Dims().lanesPerBatch; ++laneIdx)
    {
        const auto& lanePulses = pulseBatch.Pulses().LaneView(laneIdx);
        const auto& lanePulsesMetrics = pulseMetrics.baselineStats.GetHostView()[laneIdx];
        StatAccumulator<LaneArray<float>> stats{LaneArray<float>(lanePulsesMetrics.moment0),
                                                LaneArray<float>(lanePulsesMetrics.moment1),
                                                LaneArray<float>(lanePulsesMetrics.moment2)};
        const auto count = MakeUnion(stats.Count());
        const auto mean = MakeUnion(stats.Mean());
        const auto var = MakeUnion(stats.Variance());
        for (uint32_t zmwIdx = 0; zmwIdx < laneSize; ++zmwIdx)
        {
            EXPECT_EQ(baselineFrames-1, count[zmwIdx]);
            EXPECT_NEAR(baselineMean, mean[zmwIdx], 2*baselineStd);
            // Variance of sample variance for normally distributed random variable should be (2*sigma^4)/(n-1)
            EXPECT_NEAR(baselineStd*baselineStd, var[zmwIdx],2*std::sqrt((2*pow(baselineStd,4))/(baselineFrames-1)));
            for (uint32_t pulseNum = 0; pulseNum < lanePulses.size(zmwIdx); ++pulseNum)
            {
                const auto& pulse = lanePulses.ZmwData(zmwIdx)[pulseNum];
                EXPECT_EQ(labels[pulseNum % 4], pulse.Label());
                EXPECT_EQ(pulseNum * (ipd + pw) + ipd, pulse.Start());
                EXPECT_EQ(pw, pulse.Width());
                if (pulse.Start() < 16u)
                {
                    EXPECT_EQ(latTraceVal, pulse.MidSignal());
                    EXPECT_EQ(latTraceVal, pulse.MeanSignal());
                    EXPECT_EQ(latTraceVal, pulse.MaxSignal());
                    EXPECT_NEAR((latTraceVal * latTraceVal) * (pw-2), pulse.SignalM2(), latTraceVal);
                }
                else
                {
                    EXPECT_EQ(curTraceVal, pulse.MidSignal());
                    EXPECT_EQ(curTraceVal, pulse.MeanSignal());
                    EXPECT_EQ(curTraceVal, pulse.MaxSignal());
                    EXPECT_NEAR((curTraceVal * curTraceVal) * (pw-2), pulse.SignalM2(), curTraceVal);
                }
            }
        }
    }

    PulseAccumulatorToTest::Finalize();
}

}

TEST(TestHostPulseAccumulator, Run)
{
    using PulseAccumulator = HostPulseAccumulator<SubframeLabelManager>;
    TestPulseAccumulator<PulseAccumulator>();
}

TEST(TestDevicePulseAccumulator, Run)
{
    using PulseAccumulator = DevicePulseAccumulator<SubframeLabelManager>;
    TestPulseAccumulator<PulseAccumulator>();
}

}}} // namespace PacBio::Mongo::Basecaller

