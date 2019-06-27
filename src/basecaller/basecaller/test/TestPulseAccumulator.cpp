
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
#include <basecaller/traceAnalysis/HostSimulatedPulseAccumulator.h>

#include <common/DataGenerators/BatchGenerator.h>

#include <dataTypes/BasecallerConfig.h>
#include <dataTypes/LabelsBatch.h>

#include <gtest/gtest.h>

namespace PacBio {
namespace Mongo {
namespace Basecaller {

TEST(TestNoOpPulseAccumulator, Run)
{
    Data::BasecallerAlgorithmConfig bcConfig;
    PulseAccumulator::Configure(bcConfig.pulseAccumConfig.maxCallsPerZmw);

    // Simulate a single lane of data.
    Data::GetPrimaryConfig().lanesPerPool = 1;
    const auto framesPerChunk = Data::GetPrimaryConfig().framesPerChunk;
    const auto lanesPerPool = Data::GetPrimaryConfig().lanesPerPool;

    auto cameraBatchFactory = std::make_unique<Data::CameraBatchFactory>(
            framesPerChunk,
            lanesPerPool,
            Cuda::Memory::SyncDirection::HostWriteDeviceRead,
            true);

    auto labelsBatchFactory = std::make_unique<Data::LabelsBatchFactory>(
            framesPerChunk,
            lanesPerPool,
            16u,
            Cuda::Memory::SyncDirection::HostWriteDeviceRead,
            true);

    uint32_t poolId = 0;
    auto cameraBatch = cameraBatchFactory->NewBatch(
            Data::BatchMetadata(0, 0, 128),
            Data::BatchDimensions{Data::GetPrimaryConfig().zmwsPerLane, framesPerChunk, lanesPerPool});

    auto labelsBatch = labelsBatchFactory->NewBatch(std::move(cameraBatch));

    PulseAccumulator pulseAccumulator(poolId);

    auto pulseBatch = pulseAccumulator(std::move(labelsBatch));

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
    Data::BasecallerAlgorithmConfig bcConfig;
    HostSimulatedPulseAccumulator::Configure(bcConfig.pulseAccumConfig.maxCallsPerZmw);

    // Simulate a single lane of data.
    Data::GetPrimaryConfig().lanesPerPool = 1;
    const auto framesPerChunk = Data::GetPrimaryConfig().framesPerChunk;
    const auto lanesPerPool = Data::GetPrimaryConfig().lanesPerPool;

    auto cameraBatchFactory = std::make_unique<Data::CameraBatchFactory>(
            framesPerChunk,
            lanesPerPool,
            Cuda::Memory::SyncDirection::HostWriteDeviceRead,
            true);

    auto labelsBatchFactory = std::make_unique<Data::LabelsBatchFactory>(
            framesPerChunk,
            lanesPerPool,
            16u,
            Cuda::Memory::SyncDirection::HostWriteDeviceRead,
            true);

    uint32_t poolId = 0;
    auto cameraBatch = cameraBatchFactory->NewBatch(
            Data::BatchMetadata(0, 0, 128),
            Data::BatchDimensions{Data::GetPrimaryConfig().zmwsPerLane, framesPerChunk, lanesPerPool});

    auto labelsBatch = labelsBatchFactory->NewBatch(std::move(cameraBatch));

    HostSimulatedPulseAccumulator pulseAccumulator(poolId);

    auto pulseBatch = pulseAccumulator(std::move(labelsBatch));

    using NucleotideLabel = Data::Pulse::NucleotideLabel;

    // Repeating sequence of ACGT.
    const NucleotideLabel labels[] =
            { NucleotideLabel::A, NucleotideLabel::C, NucleotideLabel::G, NucleotideLabel::T };

    for (uint32_t laneIdx = 0; laneIdx < pulseBatch.Dims().lanesPerBatch; ++laneIdx)
    {
        const auto& lanePulses = pulseBatch.Pulses().LaneView(laneIdx);
        for (uint32_t zmwIdx = 0; zmwIdx < laneSize; ++zmwIdx)
        {
            EXPECT_EQ(bcConfig.pulseAccumConfig.maxCallsPerZmw, lanePulses.size(zmwIdx));
            for (uint32_t pulseNum = 0; pulseNum < bcConfig.pulseAccumConfig.maxCallsPerZmw; ++pulseNum)
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


}}} // namespace PacBio::Mongo::Basecaller

