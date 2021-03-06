#ifndef mongo_basecaller_traceAnalysis_DmeEmHost_H_
#define mongo_basecaller_traceAnalysis_DmeEmHost_H_

// Copyright (c) 2019-2021, Pacific Biosciences of California, Inc.
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
//  Defines class DmeEmHost.

#include <array>

#include <pacbio/auxdata/AnalogMode.h>
#include <common/LaneArray.h>
#include <dataTypes/UHistogramSimd.h>
#include <dataTypes/BaselinerStatAccumulator.h>

#include "CoreDMEstimator.h"
#include "DetectionModelHost.h"
#include "DmeDiagnostics.h"

namespace PacBio {
namespace Mongo {
namespace Basecaller {

/// Implements DetectionModelEstimator for host side compute.
/// Primary implementation uses a Expectation-Maximization (EM)
/// approach for model estimation, but a trivial mode is also
/// available that initializes the model off of baseline statistics
/// and then never updates it
class DmeEmHost : public CoreDMEstimator
{
public:     // Types
    using FloatVec = LaneArray<float>;
    using IntVec = LaneArray<int>;
    using BoolVec = LaneMask<>;
    using CountVec = LaneArray<LaneHist::CountType>;
    using UHistType = Data::UHistogramSimd<FloatVec, CountVec>;
    using LaneDetModelHost = Data::DetectionModelHost<FloatVec>;
    using BlStatAccState = Data::BaselinerStatAccumState;

public:     // Static functions
    static void Configure(const Data::BasecallerDmeConfig &dmeConfig,
                          const Data::AnalysisConfig &analysisConfig);

    static const AuxData::AnalogMode& Analog(unsigned int i)
    { return analogs_[i]; }

    // If mask[i], a[i] |= bits.
    static void SetBits(const BoolVec& mask, int32_t bits, IntVec* a)
    {
        // TODO: There is probably a more efficient way to implement this.
        const IntVec b = *a | IntVec(bits);
        *a = Blend(mask, b, *a);
    }

    /// Updates *detModel by increasing the amplitude of all detection modes by
    /// \a scale. Also updates all detection mode covariances according
    /// to the standard noise model. Ratios of amplitudes among detection modes
    /// and properties of the background mode are preserved.
    static void ScaleModelSnr(const FloatVec& scale, LaneDetModelHost* detModel);

    // A convenient overload that just extracts the necessary information from
    // blStats.
    // This overload doesn't really need to be public, but mixing
    // accessibility levels within an overload set seems a little awkward.
    static void InitLaneDetModel(const Data::BaselinerStatAccumState& blStats,
                                 LaneDetModel& ldm);

    // Initialize detection models to specified baseline statistics and
    // reference SNR.  Uses model to define pulse signal variances.
    // Made this public purely to support unit tests.  Not ideal, but an easy
    // near-term solution for properly initializing models in the unit tests.
    static void InitLaneDetModel(const FloatVec& blWeight,
                                 const FloatVec& blMean,
                                 const FloatVec& blVariance,
                                 LaneDetModel* ldm);

public:
    DmeEmHost(uint32_t poolId, unsigned int poolSize);

    PoolDetModel InitDetectionModels(const PoolBaselineStats& blStats) const override;

private:    // Types
    using LaneHistSimd = Data::UHistogramSimd<typename LaneHist::DataType, typename LaneHist::CountType>;
    using BaselinerStats = Data::BaselinerStatAccumulator<Data::RawTraceElement>;

private:    // Customized implementation
    void EstimateImpl(const PoolHist& hist,
                      const Data::BaselinerMetrics& metrics,
                      PoolDetModel* detModel) const override;

private:    // Static data
    static Cuda::Utility::CudaArray<AuxData::AnalogMode, numAnalogs> analogs_;
    static float refSnr_;       // Expected SNR for analog with relative amplitude of 1.
    static bool fixedModel_;
    static bool fixedBaselineParams_;
    static float fixedBaselineMean_;
    static float fixedBaselineVar_;

    static float analogMixFracThresh0_;
    static float analogMixFracThresh1_;
    static std::array<float, 2> confidHalfLife_;
    static float scaleSnrConfTol_;
    static unsigned short emIterLimit_;
    static float gTestFactor_;
    static bool iterToLimit_;
    static float pulseAmpRegCoeff_;
    static float snrDropThresh_;
    static float snrThresh0_, snrThresh1_;
    static float successConfThresh_;

private:    // Static functions
    // Compute a preliminary scaling factor based on a fractile statistic.
    static FloatVec PrelimScaleFactor(const LaneDetModelHost& model,
                                      const UHistType& hist);

    // Apply a G-test significance test to assess goodness of fit of the model
    // to the trace histogram.
    static GoodnessOfFitTest<FloatVec> Gtest(const UHistType& histogram,
                                             const LaneDetModelHost& model);

    // Compute the confidence factors of a model estimate, given the
    // diagnostics of the estimation, a reference model.
    static Cuda::Utility::CudaArray<FloatVec, ConfFactor::NUM_CONF_FACTORS>
    ComputeConfidence(const DmeDiagnostics<FloatVec>& dmeDx,
                      const LaneDetModelHost& refModel,
                      const LaneDetModelHost& modelEst);

    static void EvolveModel(const FrameIntervalType estimationFI,
                            const BaselinerStats& blStats,
                            LaneDetModelHost* model);

private:    // Functions
    void PrelimEstimate(const BaselinerStats& baselinerStats,
                        LaneDetModelHost *model) const;

    // Use the trace histogram and the input detection model to compute a new
    // estimate for the detection model. Mix the new estimate with the input
    // model, weighted by confidence scores. That result is returned in detModel.
    // estFrameInterval is the frame interval associated with the data (trace
    // histogram and baseliner statistics) used for the estimation.
    void EstimateLaneDetModel(FrameIntervalType estFrameInterval,
                              const LaneHist& blHist,
                              const BlStatAccState& blAccState,
                              LaneDetModelHost* model) const;
};

}}}     // namespace PacBio::Mongo::Basecaller

#endif // mongo_basecaller_traceAnalysis_DmeEmHost_H_
