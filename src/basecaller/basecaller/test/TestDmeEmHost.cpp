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

#include <memory>
#include <random>
#include <gtest/gtest.h>

#include <basecaller/traceAnalysis/TraceHistogramAccumHost.h>
#include <basecaller/traceAnalysis/DmeEmHost.h>
#include <common/cuda/PBCudaSimd.h>

#include <dataTypes/CameraTraceBatch.h>
#include <dataTypes/LaneDetectionModel.h>
#include <dataTypes/MovieConfig.h>
#include <dataTypes/StaticDetModelConfig.h>
#include <dataTypes/BasecallerConfig.h>

namespace PacBio {
namespace Mongo {
namespace Basecaller {

namespace {

// Constants
static constexpr unsigned int nAnalogs = 4;
static constexpr unsigned int nModes = nAnalogs + 1;

} // anonymous namespace

// Testing parameters
struct TestDmeEmHostParam
{
    float simSnr;
    std::array<Data::FrameIntervalSizeType,nModes> nFrames;
    float initModelConf;
    float pulseAmpReg;
};

// Test fixture that simulates trace data for the trace histogram with the
// number of frames for each component background and four analogs drawn
// from a baseline distribution and analog distribution.
struct TestDmeEmHost : public ::testing::TestWithParam<TestDmeEmHostParam>
{
public: // Types and static constants
    using LaneDetectionModel = Data::LaneDetectionModel<Cuda::PBHalf>;
    using LaneDetectionModelHost = DmeEmHost::LaneDetModelHost;
    using TraceHistogramAccumulator = TraceHistogramAccumHost;
public:
    struct CompleteData
    {
        // The simulated trace data histogram
        std::unique_ptr<TraceHistogramAccumulator> traceHistAccum;

        // The complete-data estimates of the model
        std::vector<std::unique_ptr<LaneDetectionModelHost>> detectionModels;

        // The mode simulated for each frame.
        std::vector<std::vector<unsigned short>> frameMode;

        // Constructors
        CompleteData() = default;

        CompleteData(TraceHistogramAccumulator&& tha,
                     std::vector<std::unique_ptr<LaneDetectionModelHost>>&& dms,
                     std::vector<std::vector<unsigned short>>&& fm)
            : traceHistAccum{new TraceHistogramAccumulator(std::move(tha))}
            , detectionModels{std::move(dms)}
            , frameMode{std::move(fm)}
        { assert(tha.HistogramFrameCount() == frameMode.front().size()); }

        CompleteData(CompleteData&&) = default;
        CompleteData& operator=(CompleteData&&) = default;
    };

public: // Structors
    TestDmeEmHost() = default;

public: // Data
    unsigned int poolId = 0;
    unsigned int poolSize = 1;

    std::unique_ptr<LaneDetectionModelHost> detModelStart;
    std::unique_ptr<LaneDetectionModelHost> detModelSim;

    Data::MovieConfig movieConfig;
    Data::BasecallerTraceHistogramConfig histConfig;

    std::unique_ptr<CompleteData> completeData;

    std::unique_ptr<Data::CameraBatchFactory> ctbFactory;
private:
    // Constructs and initializes a fixed detection model used as an initial model for the DME.
    LaneDetectionModel MakeInitialModel(void)
    {
        movieConfig.frameRate = 100.0f;
        movieConfig.photoelectronSensitivity = 0.5f;
        movieConfig.refSnr = 5.0f;

        // Construct analog set.
        const float excessNoiseCV = 0.1f;
        movieConfig.analogs[0] = Data::AnalogMode{'C', 3.70f, excessNoiseCV};
        movieConfig.analogs[1] = Data::AnalogMode{'A', 2.55f, excessNoiseCV};
        movieConfig.analogs[2] = Data::AnalogMode{'T', 1.68f, excessNoiseCV};
        movieConfig.analogs[3] = Data::AnalogMode{'G', 1.00f, excessNoiseCV};

        // Construct and initialize detection model provided to DmeEmHost as the initial model.
        const float bgSigma{10.0f};
        Data::StaticDetModelConfig staticDetModel;
        staticDetModel.baselineVariance = bgSigma * bgSigma;
        const auto& analogs = staticDetModel.SetupAnalogs(movieConfig);

        LaneDetectionModel ldm;
        ldm.BaselineMode().SetAllMeans(staticDetModel.baselineMean);
        ldm.BaselineMode().SetAllVars(staticDetModel.baselineVariance);
        for (size_t i = 0; i < analogs.size(); i++)
        {
            ldm.AnalogMode(i).SetAllMeans(analogs[i].mean);
            ldm.AnalogMode(i).SetAllVars(analogs[i].var);
        }

        return ldm;
    }

    CompleteData SimulateCompleteData(const std::array<Data::FrameIntervalSizeType,nModes>& nFrames, const LaneDetectionModelHost& simModel)
    {
        const size_t totalFrames = std::accumulate(nFrames.begin(), nFrames.end(),Data::FrameIntervalSizeType(0));

        // Random number generator.
        std::mt19937 rng (42);
        // Standard normal distribution.
        std::normal_distribution<float> normDist;

        // Construct camera track block and trace histogram.
        ctbFactory = std::make_unique<Data::CameraBatchFactory>(totalFrames, poolSize,Cuda::Memory::SyncDirection::Symmetric);
        auto ctb = ctbFactory->NewBatch(Data::BatchMetadata{poolId, 42, static_cast<uint32_t>(totalFrames)});
        auto& traces = ctb.first;
        auto& stats = ctb.second;

        histConfig.NumFramesPreAccumStats = 100;
        TraceHistogramAccumulator::Configure(histConfig, movieConfig);

        TraceHistogramAccumulator tha{poolId, poolSize};
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
                const auto& vars = dm.SignalCovar();
                const auto nFramesM = nFrames[m];
                for (unsigned int t = 0; t < nFramesM; ++t)
                {
                    DmeEmHost::FloatVec x;
                    for (unsigned int i = 0; i < laneSize; ++i)
                    {
                        x[i] = sqrt(vars[i]) * normDist(rng) + mean[i];
                        (*frame)[i] = round_cast<Data::BaselinedTraceElement>(x[i]);
                    }

                    // The traces have zero baseline so the raw trace and baselined trace are the same.
                    bsa.AddSample(*frame, *frame, false);
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
                const auto& vars = bm.SignalCovar();
                for (unsigned int t = 0; t < nFramesBg; ++t)
                {
                    DmeEmHost::FloatVec x;
                    for (unsigned int i = 0; i < laneSize; ++i)
                    {
                        x[i] = sqrt(vars[i]) * normDist(rng) + mean[i];
                        (*frame)[i] = round_cast<Data::BaselinedTraceElement>(x[i]);
                    }

                    bsa.AddSample(*frame, *frame, true);
                    cdMean.front() += x;
                    mode.push_back(0);
                    frame++;
                }
            }
            cdMean.front() /= boost::numeric_cast<float>(nFramesBg);

            // Calculate the variances of the simulated data using "complete data".
            AlignedVector<DmeEmHost::FloatVec> cdVar(nModes, 0.0f);
            frame = tr.Begin();
            for (unsigned int t = 0; t < totalFrames; ++t)
            {
                const auto m = mode.at(t);
                DmeEmHost::FloatVec vf;
                for (unsigned int i = 0; i < laneSize; ++i)
                {
                    vf[i] = (*frame)[i];
                }
                cdVar.at(m) += pow2(vf - cdMean.at(m));
                frame++;
            }
            cdVar.front() /= boost::numeric_cast<float>(nFramesBg - 1);
            for (unsigned int m = 1; m < nModes; ++m)
            {
                cdVar.at(m) = (nFrames[m] < 2)
                        ? NAN
                        : cdVar.at(m) / boost::numeric_cast<float>(nFrames[m] - 1);
            }

            // Set the baseline stats.
            Data::BaselinerStatAccumState& bls = stats.baselinerStats.GetHostView()[l];
            bls = bsa.GetState();

            // Package the complete-data estimates as a lane detection model host.
            auto detModelCD = std::make_unique<LaneDetectionModelHost>(simModel);
            {
                auto& bgMode = detModelCD->BaselineMode();
                bgMode.Weight(boost::numeric_cast<float>(nFramesBg)/boost::numeric_cast<float>(totalFrames));
                bgMode.SignalMean(cdMean.front());
                bgMode.SignalCovar(cdVar.front());
                for (unsigned int a = 0; a < nAnalogs; ++a)
                {
                    const auto m = a + 1;
                    auto& dma = detModelCD->DetectionModes().at(a);
                    dma.SignalMean(cdMean.at(m));
                    dma.SignalCovar(cdVar.at(m));
                }
            }

            detectionModels.push_back(std::move(detModelCD));
        }
        tha.AddBatch(ctb.first, ctb.second.baselinerStats);

        return CompleteData{std::move(tha), std::move(detectionModels), std::move(frameMode)};
    }

    void SetUp()
    {
        detModelStart = std::make_unique<LaneDetectionModelHost>(MakeInitialModel());
        detModelStart->Confidence(DmeEmHost::FloatVec(GetParam().initModelConf));
        const float simRefSnr = GetParam().simSnr;
        const auto startRefSnr = movieConfig.refSnr;
        detModelSim = std::make_unique<LaneDetectionModelHost>(*detModelStart);
        DmeEmHost::ScaleModelSnr(DmeEmHost::FloatVec{simRefSnr/startRefSnr}, &(*detModelSim));
        completeData = std::make_unique<CompleteData>(SimulateCompleteData(GetParam().nFrames,*detModelSim));
    }
};

TEST_P(TestDmeEmHost, EstimateFiniteMixture)
{
    Data::BasecallerDmeConfig dmeConfig;
    dmeConfig.PulseAmpRegularization = GetParam().pulseAmpReg;
    DmeEmHost::Configure(dmeConfig, movieConfig);

    std::unique_ptr<DetectionModelEstimator> dme = std::make_unique<DmeEmHost>(poolId, poolSize);
    Cuda::Memory::UnifiedCudaArray<LaneDetectionModel> models(poolSize,
                                                              Cuda::Memory::SyncDirection::Symmetric,
                                                              SOURCE_MARKER());

    // Initialize the model using the initial fixed detection model host.
    for (unsigned int l = 0; l < poolSize; ++l)
    {
        detModelStart->ExportTo(&models.GetHostView()[l]);
    }

    //models = dme->InitDetectionModels(completeData->traceHistAccum->TraceStats());

    // TODO: The Estimate() method currently doesn't return the DME diagnostics
    // with which further unit tests can be written to verify its contents.
    dme->Estimate(completeData->traceHistAccum->Histogram(), &models);

    const auto& nFrames = GetParam().nFrames;
    const auto numFrames = std::accumulate(nFrames.cbegin(), nFrames.cend(),
                                    Data::FrameIntervalSizeType(0));
    const auto numFramesF = boost::numeric_cast<float>(numFrames);

    // Check the pool detection models that should be updated for each lane.
    for (unsigned int l = 0; l < poolSize; ++l)
    {
        LaneDetectionModelHost resultModel{models.GetHostView()[l]};
        LaneDetectionModelHost completeDataModel{*completeData->detectionModels[l]};

        {
            // Check baseline modes and mixture fractions.
            const auto& rbm = resultModel.BaselineMode();
            const auto& cdbm = completeDataModel.BaselineMode();
            DmeEmHost::FloatVec result = rbm.Weight();
            DmeEmHost::FloatVec expected = cdbm.Weight();
            DmeEmHost::FloatVec absErrTol = DmeEmHost::FloatVec{5} *
                    sqrt(expected * (1.0f - expected) / numFramesF);
            for (unsigned int i = 0; i < laneSize; ++i)
            {
                EXPECT_NEAR(expected[i], result[i], absErrTol[i])
                                    << "Bad mixing fraction for i = " << i << '.';
            }

            // Check baseline estimated means.
            result = rbm.SignalMean();
            expected = cdbm.SignalMean();
            absErrTol = DmeEmHost::FloatVec{5} * sqrt(cdbm.SignalCovar() / numFramesF / cdbm.Weight());
            for (unsigned int i = 0; i < laneSize; ++i)
            {
                EXPECT_NEAR(expected[i], result[i], absErrTol[i])
                                    << "Bad baseline mean for i = " << i << '.';
            }

            // Check baseline estimated variance.
            result = rbm.SignalCovar();
            expected = cdbm.SignalCovar();
            absErrTol = DmeEmHost::FloatVec{5} * sqrt(2.0f / (numFramesF * cdbm.Weight() - 1.0f)) * cdbm.SignalCovar();
            for (unsigned int i = 0; i < laneSize; ++i)
            {
                EXPECT_NEAR(expected[i], result[i], absErrTol[i])
                                    << "Bad baseline variance for i = " << i << '.';
            }
        }

        // Check detection modes.
        for (unsigned int a = 0; a < 4; ++a)
        {
            // Check mixture fraction.
            const auto& rbm = resultModel.DetectionModes()[a];
            const auto& cdbm = completeDataModel.DetectionModes()[a];
            DmeEmHost::FloatVec result = rbm.Weight();
            DmeEmHost::FloatVec expected = cdbm.Weight();
            DmeEmHost::FloatVec absErrTol = max(0.015f, DmeEmHost::FloatVec{5} * sqrt(expected * (1.0f - expected) / numFramesF));
            for (unsigned int i = 0; i < laneSize; ++i)
            {
                EXPECT_NEAR(expected[i], result[i], absErrTol[i])
                                    << "Bad mixing fraction for analog " << a
                                    << " for i = " << i << '.';
            }

            // Check estimated means.
            result = rbm.SignalMean();
            expected = cdbm.SignalMean();
            absErrTol = DmeEmHost::FloatVec{5} * sqrt(cdbm.SignalCovar() / numFramesF / cdbm.Weight());
            for (unsigned int i = 0; i < laneSize; ++i)
            {
                EXPECT_NEAR(expected[i], result[i], absErrTol[i])
                                    << "Bad pulse mean for analog " << a
                                    << " for i = " << i << ".";
            }

            // Check estimated variances.
            result = rbm.SignalCovar();
            expected = cdbm.SignalCovar();
            absErrTol = DmeEmHost::FloatVec{5} * sqrt(2.0f / (numFramesF * cdbm.Weight() - 1.0f)) * cdbm.SignalCovar();
            for (unsigned int i = 0; i < laneSize; ++i)
            {
                if (isnan(expected[i])) continue;
                EXPECT_NEAR(expected[i], result[i], absErrTol[i])
                                    << "Bad pulse variance for analog " << a
                                    << " for i = " << i << ".";
            }
        }
    }
}

const std::array<unsigned int,nModes> frameCountsBalanced = {500, 125, 125, 125, 125};
const auto snrSweep = ::testing::Values(
        TestDmeEmHostParam{3.6f,  frameCountsBalanced,      0.0f,       0.0f},
        //TestDmeEmHostParam{4.0f,  frameCountsBalanced,      0.0f,       0.0f},
        TestDmeEmHostParam{5.0f,  frameCountsBalanced,      0.0f,       0.0f},
        TestDmeEmHostParam{6.0f,  frameCountsBalanced,      0.0f,       0.0f},
        TestDmeEmHostParam{8.0f,  frameCountsBalanced,      0.0f,       0.0f},
        TestDmeEmHostParam{10.0f, frameCountsBalanced,      0.0f,       0.0f}
        );
INSTANTIATE_TEST_SUITE_P(SnrSweep, TestDmeEmHost, snrSweep);

const std::array<unsigned int,nModes> frameCountsNoBrightest = {550, 0, 150, 150, 150};
const auto noAnalog0 = ::testing::Values(
        TestDmeEmHostParam{4.5f,  frameCountsNoBrightest,   1.0f,       0.1f},
        TestDmeEmHostParam{5.0f,  frameCountsNoBrightest,   1.0f,       0.1f},
        TestDmeEmHostParam{5.25f,  frameCountsNoBrightest,   1.0f,       0.1f}
);
INSTANTIATE_TEST_SUITE_P(NoBrightestAnalog, TestDmeEmHost, noAnalog0);

}}} // namespace PacBio::Mongo::Basecaller
