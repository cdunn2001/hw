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

Data::BasecallBatch GenerateBases(BaseSimConfig config, size_t batchNo=0)
{
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
    Data::BasecallBatchFactory batchFactory(
        basecallerConfig.pulseAccumConfig.maxCallsPerZmw,
        dims,
        Cuda::Memory::SyncDirection::HostWriteDeviceRead,
        true);

    auto bases = batchFactory.NewBatch(chunk.front().Metadata());

    auto LabelConv = [](size_t index) {
        switch (index % 4)
        {
        case 0:
            return SmrtData::NucleotideLabel::A;
        case 1:
            return SmrtData::NucleotideLabel::C;
        case 2:
            return SmrtData::NucleotideLabel::G;
        case 3:
            return SmrtData::NucleotideLabel::T;
        case 4:
            return SmrtData::NucleotideLabel::N;
        default:
            assert(label == Data::Pulse::NucleotideLabel::NONE);
            return SmrtData::NucleotideLabel::NONE;
        }
    };
    for (size_t lane = 0; lane < bases.Dims().lanesPerBatch; ++lane)
    {
        auto baseView = bases.Basecalls().LaneView(lane);

        baseView.Reset();
        for (size_t zmw = 0; zmw < laneSize; ++zmw)
        {
            for (size_t b = 0; b < config.numBases; ++b)
            {
                PacBio::SmrtData::Basecall bc;

                static constexpr int8_t qvDefault_ = 0;

                auto label = LabelConv(b);

                // Populate pulse data
                bc.GetPulse().Start(b * config.baseWidth + b * config.ipd
                                    + batchNo * chunkSize)
                             .Width(config.baseWidth);
                bc.GetPulse().MeanSignal(config.meanSignal)
                             .MidSignal(config.midSignal)
                             .MaxSignal(config.maxSignal);
                bc.GetPulse().Label(label).LabelQV(qvDefault_);
                bc.GetPulse().AltLabel(label).AltLabelQV(qvDefault_);
                bc.GetPulse().MergeQV(qvDefault_);

                // Populate base data.
                bc.Base(label).InsertionQV(qvDefault_);
                bc.DeletionTag(SmrtData::NucleotideLabel::N).DeletionQV(qvDefault_);
                bc.SubstitutionTag(SmrtData::NucleotideLabel::N).SubstitutionQV(qvDefault_);

                baseView.push_back(zmw, bc);
            }
        }
    }
    return bases;
}
}

/*
TEST(TestHFMetricsFilter, End2End)
{
    int poolId = 0;
    Data::MovieConfig movConfig;
    Data::BasecallerAlgorithmConfig basecallerConfig;
    basecallerConfig.baselinerConfig.Method = Data::BasecallerBaselinerConfig::MethodName::MultiScaleLarge;

    Data::GetPrimaryConfig().lanesPerPool = 1;

    PBLOG_INFO << "Configuring stages";
    DeviceMultiScaleBaseliner::Configure(basecallerConfig.baselinerConfig, movConfig);
    DeviceSGCFrameLabeler::Configure(Data::GetPrimaryConfig().lanesPerPool,
                                     Data::GetPrimaryConfig().framesPerChunk);
    HostHFMetricsFilter::Configure(basecallerConfig.Metrics);

    unsigned int poolSize = Data::GetPrimaryConfig().lanesPerPool;
    PBLOG_INFO << poolSize << " lanes per pool.";

    unsigned int chunkSize = Data::GetPrimaryConfig().framesPerChunk;
    PBLOG_INFO << chunkSize << " frames per chunk.";

    Data::BatchDimensions dims;
    dims.framesPerBatch = chunkSize;
    dims.laneWidth = laneSize;
    dims.lanesPerBatch = poolSize;

    PBLOG_INFO << "Building baseliner";
    DeviceMultiScaleBaseliner baseliner(poolId,
                                        Data::GetPrimaryConfig().lanesPerPool);

    PBLOG_INFO << "Building framelabeler";
    DeviceSGCFrameLabeler frameLabeler(poolId);
    PBLOG_INFO << "Building pulseaccumulator";
    PulseAccumulator pulseAccumulator(poolId);

    PBLOG_INFO << "Building batchfactory";
    Data::BasecallBatchFactory batchFactory(
        basecallerConfig.pulseAccumConfig.maxCallsPerZmw,
        dims,
        Cuda::Memory::SyncDirection::HostWriteDeviceRead,
        true);

    PBLOG_INFO << "Building hfmetricsfilter";
    HostHFMetricsFilter hfMetrics(poolId);
    PBLOG_INFO << "Building models";
    Cuda::Memory::UnifiedCudaArray<Data::LaneModelParameters<Cuda::PBHalf, laneSize>> models(
        Data::GetPrimaryConfig().lanesPerPool, Cuda::Memory::SyncDirection::Symmetric, true);

    PBLOG_INFO << "Building batchgenerator";
    Cuda::Data::BatchGenerator batchGenerator(Data::GetPrimaryConfig().framesPerChunk,
                                              Data::GetPrimaryConfig().zmwsPerLane,
                                              Data::GetPrimaryConfig().lanesPerPool,
                                              8192,
                                              Data::GetPrimaryConfig().lanesPerPool);
    PBLOG_INFO << "Running stages";
    while (!batchGenerator.Finished())
    {
        PBLOG_INFO << 0;
        auto chunk = batchGenerator.PopulateChunk();
        PBLOG_INFO << 1;
        Data::CameraTraceBatch ctb = baseliner(std::move(chunk.front()));
        PBLOG_INFO << 2;
        auto labels = frameLabeler(std::move(ctb), models);
        PBLOG_INFO << 3;
        auto pulses = pulseAccumulator(std::move(labels));
        PBLOG_INFO << 4;
        auto bases = batchFactory.NewBatch(chunk.front().Metadata());
        PBLOG_INFO << 5;
        Temporary::ConvertPulsesToBases(pulses, bases);
        PBLOG_INFO << 6;
        auto basecallingMetrics = hfMetrics(bases);
        PBLOG_INFO << 7;
    }

    PBLOG_INFO << "Killing stages";
    DeviceMultiScaleBaseliner::Finalize();
}
*/

TEST(TestHFMetricsFilter, Populated)
{
    int poolId = 0;
    Data::BasecallerAlgorithmConfig basecallerConfig;
    HostHFMetricsFilter::Configure(basecallerConfig.Metrics);
    HostHFMetricsFilter hfMetrics(poolId);

    // TODO: test that the last block is finalized regardless of condition?

    size_t numFramesPerBatch = 128;
    size_t numBatchesPerHFMB = Data::GetPrimaryConfig().framesPerHFMetricBlock
                             / numFramesPerBatch; // = 32, for 4096 frame HFMBs

    BaseSimConfig config;
    config.ipd = 0;

    for (size_t i = 0; i < numBatchesPerHFMB; ++i)
    {
        auto bases = GenerateBases(config, i);
        auto basecallingMetrics = hfMetrics(bases);
        if (basecallingMetrics)
        {
            ASSERT_EQ(i, numBatchesPerHFMB - 1); // = 31, HFMB is complete
            for (uint32_t l = 0; l < bases.Dims().lanesPerBatch; l++)
            {
                const auto& mb = basecallingMetrics->GetHostView()[l];
                /*
                std::cout << "lane " << l << ":" << std::endl;
                for (const auto& analog : mb.NumPulsesByAnalog())
                {
                    for (const auto& zmw : analog)
                        std::cout << zmw << ", ";
                    std::cout << std::endl;
                }
                std::cout << "lane " << l << ":" << std::endl;
                for (const auto& analog : mb.NumBasesByAnalog())
                {
                    for (const auto& zmw : analog)
                        std::cout << zmw << ", ";
                    std::cout << std::endl;
                }
                */
                for (uint32_t z = 0; z < laneSize; ++z)
                {
                    EXPECT_EQ(numBatchesPerHFMB
                                * config.numBases
                                * config.baseWidth,
                              mb.NumPulseFrames()[z]);
                    EXPECT_EQ(numBatchesPerHFMB
                                * config.numBases
                                * config.baseWidth,
                              mb.NumBaseFrames()[z]);
                    // The pulses don't run to the end of each block, so all
                    // but one pulse is abutted
                    ASSERT_EQ((numBatchesPerHFMB) * (config.numBases - 1),
                              mb.NumHalfSandwiches()[z]);
                    // If numBases isn't evenly divisible by numAnalogs, the
                    // first analogs will be padded by the remainder
                    // e.g. 10 pulses per chunk means 2 for each analog, then
                    // three for the first two analogs:
                    EXPECT_EQ(numBatchesPerHFMB
                                * (config.numBases/numAnalogs
                                   + (config.numBases % numAnalogs > 0 ? 1 : 0)),
                              mb.NumPulsesByAnalog()[0][z]);
                    EXPECT_EQ(numBatchesPerHFMB
                                * (config.numBases/numAnalogs
                                   + (config.numBases % numAnalogs > 1 ? 1 : 0)),
                              mb.NumPulsesByAnalog()[1][z]);
                    EXPECT_EQ(numBatchesPerHFMB
                                * (config.numBases/numAnalogs
                                   + (config.numBases % numAnalogs > 2 ? 1 : 0)),
                              mb.NumPulsesByAnalog()[2][z]);
                    EXPECT_EQ(numBatchesPerHFMB
                                * (config.numBases/numAnalogs
                                   + (config.numBases % numAnalogs > 3 ? 1 : 0)),
                              mb.NumPulsesByAnalog()[3][z]);
                    ASSERT_EQ(numBatchesPerHFMB * config.numBases,
                              mb.NumPulses()[z]);
                    ASSERT_EQ(numBatchesPerHFMB * config.numBases,
                              mb.NumBases()[z]);
                }
            }
        }
    }
}

}}} // PacBio::Mongo::Basecaller
