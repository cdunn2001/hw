// Copyright (c) 2020, Pacific Biosciences of California, Inc.
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
//  Defines unit tests for detection model estimation performed on the host
//  compute system.

#include <algorithm>
#include <memory>
#include <random>
#include <sstream>

#include <gtest/gtest.h>
#include <boost/numeric/conversion/cast.hpp>

#include <basecaller/traceAnalysis/ComputeDevices.h>
#include <basecaller/traceAnalysis/DmeEmHost.h>
#include <basecaller/traceAnalysis/DmeEmDevice.h>
#include <basecaller/traceAnalysis/TraceHistogramAccumHost.h>
#include <common/cuda/PBCudaSimd.h>
#include <common/simd/SimdConvTraits.h>

#include <dataTypes/CameraTraceBatch.h>
#include <dataTypes/LaneDetectionModel.h>
#include <dataTypes/configs/BasecallerDmeConfig.h>
#include <dataTypes/configs/BasecallerTraceHistogramConfig.h>
#include <dataTypes/configs/AnalysisConfig.h>
#include <dataTypes/configs/StaticDetModelConfig.h>

using boost::numeric_cast;

namespace PacBio {
namespace Mongo {
namespace Basecaller {

namespace {

// Constants
static constexpr unsigned int nAnalogs = 4;
static constexpr unsigned int nModes = nAnalogs + 1;

using FrameIntervalSizeType = Data::FrameIntervalType::SizeType;

// Testing parameters
struct TestDmeEmParam
{
    float simSnr;
    std::array<FrameIntervalSizeType, nModes> nFrames;
    float initModelConf;
    float pulseAmpReg;
};

struct TestConfig : public Configuration::PBConfig<TestConfig>
{
    PB_CONFIG(TestConfig);

    PB_CONFIG_OBJECT(Data::BasecallerTraceHistogramConfig, histConfig);
    PB_CONFIG_OBJECT(Data::BasecallerDmeConfig, dmeConfig);

    PB_CONFIG_PARAM(ComputeDevices, analyzerHardware, ComputeDevices::Host);
};

} // anonymous namespace

// Test fixture that simulates trace data for the trace histogram with the
// number of frames for each component background and four analogs drawn
// from a baseline distribution and analog distribution.
template <typename T>
struct TestDmeEm : public ::testing::TestWithParam<TestDmeEmParam>
{
public: // Types and static constants
    using DmeImpl = T;
    using LaneDetectionModel = Data::LaneDetectionModel<Cuda::PBHalf>;
    using LaneDetectionModelHost = DmeEmHost::LaneDetModelHost;
public:
    struct CompleteData
    {
        // The simulated trace data histogram
        std::unique_ptr<TraceHistogramAccumHost> traceHistAccum;

        // The simulated trace baseliner metrics
        Data::BaselinerMetrics blMetrics;

        // The complete-data estimates of the model
        std::vector<std::unique_ptr<LaneDetectionModelHost>> detectionModels;

        // The mode simulated for each frame.
        std::vector<std::vector<unsigned short>> frameMode;

        // Constructors
        CompleteData() = default;

        CompleteData(TraceHistogramAccumHost&& tha,
                     Data::BaselinerMetrics&& metrics,
                     std::vector<std::unique_ptr<LaneDetectionModelHost>>&& dms,
                     std::vector<std::vector<unsigned short>>&& fm)
            : traceHistAccum{new TraceHistogramAccumHost(std::move(tha))}
            , blMetrics{std::move(metrics)}
            , detectionModels{std::move(dms)}
            , frameMode{std::move(fm)}
        { assert(traceHistAccum->FramesAdded() == frameMode.front().size()); }

        CompleteData(CompleteData&&) = default;
        CompleteData& operator=(CompleteData&&) = default;
    };

public: // Structors
    TestDmeEm()
    {
        Json::Value json;
        json["dmeConfig"]["PulseAmpRegularization"] = GetParam().pulseAmpReg;
        json["dmeConfig"]["ModelUpdateMethod"]   = 0;
        testConfig = TestConfig(json);
    }

public:
    void Assert(const DmeEmHost::FloatVec& expected,
                const DmeEmHost::FloatVec& actual,
                const DmeEmHost::FloatVec& absErrTol,
                const std::string& message)
    {
        auto exp = MakeUnion(expected);
        auto act = MakeUnion(actual);
        auto tol = MakeUnion(absErrTol);

        for (uint32_t i = 0; i < laneSize; ++i)
        {
            EXPECT_NEAR(exp[i], act[i], tol[i]) << message << " (i = " << i << ')';
        }
    }

    // The main test.  Ideally this should just be in the usual TEST_P macro, but we
    // need to parameterize both over type as well as value, which gtest does not make
    // easy to do.
    void RunTest()
    {
        DmeImpl::Configure(testConfig.dmeConfig, analysisConfig);

        // Based on MockMovieInfo data in analysisConfig.movieInfo
        EXPECT_FLOAT_EQ(CoreDMEstimator::Config().shotVarCoeff, 1.2168207);

        std::unique_ptr<CoreDMEstimator> dme = std::make_unique<DmeImpl>(poolId, poolSize);
        Data::DetectionModelPool<Cuda::PBHalf> models(poolSize,
                                                      Cuda::Memory::SyncDirection::Symmetric,
                                                      SOURCE_MARKER());

        // Initialize the model
        models.frameInterval = detModelStart->FrameInterval();
        for (unsigned int l = 0; l < poolSize; ++l)
        {
            detModelStart->ExportTo(&models.data.GetHostView()[l]);
        }

        // TODO: The Estimate() method currently doesn't return the DME diagnostics
        // with which further unit tests can be written to verify its contents.
        dme->Estimate(completeData->traceHistAccum->Histogram(), completeData->blMetrics, &models);

        const auto& nFrames = GetParam().nFrames;
        const auto numFrames = std::accumulate(nFrames.cbegin(), nFrames.cend(),
                                               FrameIntervalSizeType(0));

        using PacBio::Simd::MakeUnion;
        // Check the pool detection models that should be updated for each lane.
        for (unsigned int l = 0; l < poolSize; ++l)
        {
            LaneDetectionModelHost resultModel {models, l};
            LaneDetectionModelHost completeDataModel{*completeData->detectionModels[l]};

            // Not much point in continuing if the estimation effort failed.
            ASSERT_TRUE(all(resultModel.Confidence() > 0.0f)) << "Estimation failed.";

            {
                // Check baseline modes and mixture fractions.
                const auto& rbm = resultModel.BaselineMode();
                const auto& cdbm = completeDataModel.BaselineMode();
                DmeEmHost::FloatVec result = rbm.Weight();
                DmeEmHost::FloatVec expected = cdbm.Weight();
                DmeEmHost::FloatVec absErrTol = 5.0f * sqrt(expected * (1.0f - expected) / numFrames);
                Assert(expected, result, absErrTol, "Bad mixing fraction for baseline");

                // Check baseline estimated means.
                result = rbm.SignalMean();
                expected = cdbm.SignalMean();
                absErrTol = 5.0f * sqrt(cdbm.SignalCovar() / numFrames / cdbm.Weight());
                Assert(expected, result, absErrTol, "Bad baseline mean for baseline");

                // Check baseline estimated variance.
                result = rbm.SignalCovar();
                expected = cdbm.SignalCovar();
                absErrTol = 5.0f * sqrt(2.0f / (numFrames * cdbm.Weight() - 1.0f)) * expected;
                Assert(expected, result, absErrTol, "Bad baseline variance for baseline");
            }

            // Check detection modes.
            for (unsigned int a = 0; a < 4; ++a)
            {
                // Check mixture fraction.
                const auto& rbm = resultModel.DetectionModes()[a];
                const auto& cdbm = completeDataModel.DetectionModes()[a];
                DmeEmHost::FloatVec result = rbm.Weight();
                DmeEmHost::FloatVec expected = cdbm.Weight();
                DmeEmHost::FloatVec absErrTol = max(0.015f, 5.0f * sqrt(expected * (1.0f - expected) / numFrames));
                const auto astr = std::to_string(a);
                Assert(expected, result, absErrTol, "Bad mixing fraction for analog " + astr);

                // Check estimated means.
                result = rbm.SignalMean();
                expected = cdbm.SignalMean();
                absErrTol = 5.0f * sqrt(cdbm.SignalCovar() / numFrames / cdbm.Weight());
                Assert(expected, result, absErrTol, "Bad mean for analog " + astr);

                // Check estimated variances.
                result = rbm.SignalCovar();
                expected = cdbm.SignalCovar();
                absErrTol = 5.0f * sqrt(2.0f / (numFrames * cdbm.Weight() - 1.0f)) * expected;
                Assert(expected, result, absErrTol, "Bad variance for analog " + astr);
            }
        }
    }

public: // Data
    unsigned int poolId = 0;
    unsigned int poolSize = 1;

    std::unique_ptr<LaneDetectionModelHost> detModelStart;
    std::unique_ptr<LaneDetectionModelHost> detModelSim;

    Data::AnalysisConfig analysisConfig;
    TestConfig testConfig;

    std::unique_ptr<CompleteData> completeData;

    std::unique_ptr<Data::CameraBatchFactory> ctbFactory;

    Logging::LogSeverityContext logLevel = Logging::LogLevel::WARN;

private:
    // Constructs and initializes a fixed detection model used as an initial model for the DME.
    LaneDetectionModel MakeInitialModel(void)
    {
        LaneDetectionModel ldm;
        DmeEmHost::InitLaneDetModel(0.5f, 0.0f, 100.0f, &ldm);
        return ldm;
    }

    CompleteData SimulateCompleteData(const std::array<FrameIntervalSizeType, nModes>& nFrames,
                                      const LaneDetectionModelHost& simModel)
    {
        const size_t totalFrames = std::accumulate(nFrames.begin(), nFrames.end(),
                                                   FrameIntervalSizeType(0));

        // Random number generator.
        std::mt19937 rng (42);
        // Standard normal distribution.
        std::normal_distribution<float> normDist;

        // Construct camera trace block and trace histogram.
        Data::BatchDimensions bd{poolSize, static_cast<uint32_t>(totalFrames)};
        Data::BatchMetadata batchMeta{poolId, 0, static_cast<int32_t>(totalFrames), poolId};
        ctbFactory = std::make_unique<Data::CameraBatchFactory>(Cuda::Memory::SyncDirection::Symmetric);
        auto [traces, stats] = ctbFactory->NewBatch(batchMeta, bd);
        stats.frameInterval = {0, numeric_cast<Data::FrameIndexType>(totalFrames)};

        std::vector<std::vector<unsigned short>> frameMode(poolSize);
        std::vector<std::unique_ptr<LaneDetectionModelHost>> detectionModels;

        for (unsigned int l = 0; l < poolSize; ++l)
        {
            auto tr = traces.GetBlockView(l);
            auto frame = tr.Begin();
            Data::BaselinerStatAccumulator<Data::BaselinedTraceElement> bsa;

            // Mode each frame belongs to.
            auto& mode = frameMode[l];

            // Add frames for each analog to the camera trace block while
            // calculating the complete-data mean of each mode.
            AlignedVector<DmeEmHost::FloatVec> cdMean(nModes, 0.0f);
            for (unsigned int a = 0; a < nAnalogs; ++a)
            {
                const unsigned int m = a + 1;
                const auto& dm = simModel.DetectionModes()[a];
                const auto& mean = dm.SignalMean();
                const auto& stdev = sqrt(dm.SignalCovar());
                const auto nFramesM = nFrames[m];
                for (unsigned int t = 0; t < nFramesM; ++t)
                {
                    // TODO: The rounding and casting to int seems completely
                    // unnecessary.  Moreover, the rounding method is biased.
                    DmeEmHost::FloatVec x = floorCastInt(stdev * normDist(rng) + mean);
                    frame.Store(x);
                    // The traces have zero baseline so the raw trace and baselined trace are the same.
                    bsa.AddSample(frame.Extract(), frame.Extract(), false);
                    cdMean.at(m) += x;
                    mode.push_back(m);
                    frame++;
                }
                cdMean.at(m) /= boost::numeric_cast<float>(nFramesM);
            }

            // Add baseline frames to the camera trace block.
            const auto& nFramesBg = nFrames.front();
            {
                const auto& bm = simModel.BaselineMode();
                const auto& mean = bm.SignalMean();
                const auto& stdev = sqrt(bm.SignalCovar());
                for (unsigned int t = 0; t < nFramesBg; ++t)
                {
                    DmeEmHost::FloatVec x = floorCastInt(stdev * normDist(rng) + mean);
                    frame.Store(x);
                    bsa.AddSample(frame.Extract(), frame.Extract(), true);
                    cdMean.front() += x;
                    mode.push_back(0);
                    frame++;
                }
            }
            cdMean.front() /= numeric_cast<float>(nFramesBg);

            // Calculate the variances of the simulated data using "complete data".
            AlignedVector<DmeEmHost::FloatVec> cdVar(nModes, 0.0f);
            frame = tr.Begin();
            for (unsigned int t = 0; t < totalFrames; ++t)
            {
                const auto m = mode.at(t);
                cdVar.at(m) += pow2(frame.Extract() - cdMean.at(m));
                frame++;
            }
            cdVar.front() /= numeric_cast<float>(nFramesBg - 1);
            for (unsigned int m = 1; m < nModes; ++m)
            {
                cdVar.at(m) = (nFrames[m] < 2)
                        ? NAN
                        : cdVar.at(m) / numeric_cast<float>(nFrames[m] - 1);
            }

            // Set the baseline stats.
            Data::BaselinerStatAccumState& bls = stats.baselinerStats.GetHostView()[l];
            bls = bsa.GetState();

            // Package the complete-data estimates as a LaneDetectionModelHost.
            auto detModelCD = std::make_unique<LaneDetectionModelHost>(simModel);
            {
                auto& bgMode = detModelCD->BaselineMode();
                bgMode.Weight(numeric_cast<float>(nFramesBg) / numeric_cast<float>(totalFrames));
                bgMode.SignalMean(cdMean.front());
                bgMode.SignalCovar(cdVar.front());
                for (unsigned int a = 0; a < nAnalogs; ++a)
                {
                    const auto m = a + 1;
                    auto& dma = detModelCD->DetectionModes().at(a);
                    dma.Weight(numeric_cast<float>(nFrames.at(m)) / numeric_cast<float>(totalFrames));
                    dma.SignalMean(cdMean.at(m));
                    dma.SignalCovar(cdVar.at(m));
                }
            }

            detectionModels.push_back(std::move(detModelCD));
        }

        // Set up the detection models for the pool.
        TraceHistogramAccumulator::PoolDetModel pdm {poolSize,
                                                     Cuda::Memory::SyncDirection::Symmetric,
                                                     SOURCE_MARKER()};
        {
            pdm.frameInterval = detModelStart->FrameInterval();
            LaneDetectionModel laneDetModel;
            detModelStart->ExportTo(&laneDetModel);
            auto pdmv = pdm.data.GetHostView();
            std::fill(pdmv.begin(), pdmv.end(), laneDetModel);
        }

        // Configure and fill the histogram.
        TraceHistogramAccumHost::Configure(testConfig.histConfig, analysisConfig);
        TraceHistogramAccumHost tha{poolId, poolSize};
        tha.Reset(stats);
        tha.AddBatch(traces, pdm);
        return CompleteData{std::move(tha), std::move(stats), std::move(detectionModels), std::move(frameMode)};
    }

    void SetUp()
    {
        analysisConfig.movieInfo = PacBio::DataSource::MockMovieInfo();
        analysisConfig.movieInfo.refSnr = 5.0;
        analysisConfig.movieInfo.frameRate = 80.0f;
        analysisConfig.movieInfo.photoelectronSensitivity = 2.0f;
        analysisConfig.movieInfo.analogs[0].relAmplitude = 3.70f;
        analysisConfig.movieInfo.analogs[1].relAmplitude = 2.55f;
        analysisConfig.movieInfo.analogs[2].relAmplitude = 1.68f;
        analysisConfig.movieInfo.analogs[3].relAmplitude = 1.0f;

        // Need to configure DmeEmHost regardless of whether we are testing the
        // CPU or GPU implementation because we are using a couple static
        // functions of that class to set up the model parameters used for trace
        // simulation.
        // TODO: Find a way for host and gpu to share the same Configure?
        DmeEmHost::Configure(testConfig.dmeConfig, analysisConfig);

        const auto& nFrames = GetParam().nFrames;
        const auto totalFrameCount = numeric_cast<int>(std::accumulate(nFrames.begin(), nFrames.end(), 0u));
        detModelStart = std::make_unique<LaneDetectionModelHost>(MakeInitialModel(),
                                                                 Data::FrameIntervalType{0, totalFrameCount});
        detModelStart->Confidence(GetParam().initModelConf);
        const float simRefSnr = GetParam().simSnr;
        const auto startRefSnr = analysisConfig.movieInfo.refSnr;
        detModelSim = std::make_unique<LaneDetectionModelHost>(*detModelStart);
        DmeEmHost::ScaleModelSnr(simRefSnr/startRefSnr, detModelSim.get());
        completeData = std::make_unique<CompleteData>(SimulateCompleteData(GetParam().nFrames, *detModelSim));
    }
};
using EmHost = TestDmeEm<DmeEmHost>;
using EmDevice = TestDmeEm<DmeEmDevice>;

TEST_P(EmHost, EstimateFiniteMixture)
{
    RunTest();
}
TEST_P(EmDevice, EstimateFiniteMixture)
{
    RunTest();
}

const std::array<unsigned int,nModes> frameCountsBalanced = {500, 125, 125, 125, 125};
const auto snrSweep = ::testing::Values(
        TestDmeEmParam{3.6f,  frameCountsBalanced,      0.0f,       0.0f},
        TestDmeEmParam{4.0f,  frameCountsBalanced,      0.0f,       0.0f},
        TestDmeEmParam{5.0f,  frameCountsBalanced,      0.0f,       0.0f},
        TestDmeEmParam{6.0f,  frameCountsBalanced,      0.0f,       0.0f}
        // TestDmeEmParam{8.0f,  frameCountsBalanced,      0.0f,       0.0f},
        // TestDmeEmParam{10.0f,  frameCountsBalanced,      0.0f,       0.0f}
        );
INSTANTIATE_TEST_SUITE_P(SnrSweep, EmHost, snrSweep);
INSTANTIATE_TEST_SUITE_P(SnrSweep, EmDevice, snrSweep);

#if 0
// This #if/#endif block is necessary to avoid compiler warnings about unused variables (snrSweep2)
const std::array<unsigned int,nModes> frameCountsUnBalanced = {600, 200, 125, 300, 75};
const auto snrSweep2 = ::testing::Values(
        TestDmeEmParam{14.4f,  frameCountsUnBalanced,      0.0f,       0.0f},
        TestDmeEmParam{16.0f,  frameCountsUnBalanced,      0.0f,       0.0f},
        TestDmeEmParam{20.0f,  frameCountsUnBalanced,      0.0f,       0.0f},
        TestDmeEmParam{24.0f,  frameCountsUnBalanced,      0.0f,       0.0f},
        TestDmeEmParam{32.0f,  frameCountsUnBalanced,      0.0f,       0.0f},
        TestDmeEmParam{40.0f,  frameCountsUnBalanced,      0.0f,       0.0f}
);
// TODO: Both the test with unbalanced frame counts and missing
// analog currently fail and need to be further investigated.
INSTANTIATE_TEST_SUITE_P(SnrSweep2, EmHost, snrSweep2);
INSTANTIATE_TEST_SUITE_P(SnrSweep2, EmDevice, snrSweep2);
#endif


#if 0
// This #if/#endif block is necessary to avoid compiler warnings about unused variables (noAnalog0)
const std::array<unsigned int,nModes> frameCountsNoBrightest = {600, 0, 200, 200, 200};
const auto noAnalog0 = ::testing::Values(
        TestDmeEmParam{18.0f,  frameCountsNoBrightest,   1.0f,       0.1f},
        TestDmeEmParam{20.0f,  frameCountsNoBrightest,   1.0f,       0.1f},
        TestDmeEmParam{21.0f,  frameCountsNoBrightest,   1.0f,       0.1f}
);
INSTANTIATE_TEST_SUITE_P(NoBrightestAnalog, EmHost, noAnalog0);
INSTANTIATE_TEST_SUITE_P(NoBrightestAnalog, EmDevice, noAnalog0);
#endif

}}} // namespace PacBio::Mongo::Basecaller
