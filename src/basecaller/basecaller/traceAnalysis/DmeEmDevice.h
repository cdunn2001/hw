#ifndef mongo_basecaller_traceAnalysis_DmeEmDevice_H_
#define mongo_basecaller_traceAnalysis_DmeEmDevice_H_

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

#include "CoreDMEstimator.h"
#include "DmeDiagnostics.h"

namespace PacBio {
namespace Mongo {
namespace Basecaller {

struct ZmwAnalogMode
{
    template <int low>
    CUDA_ENABLED void Assign(const Data::LaneAnalogMode<Cuda::PBHalf2, 32>& mode, int idx)
    {
        mean = mode.means[idx].Get<low>();
        var = mode.vars[idx].Get<low>();
        weight = mode.weights[idx].Get<low>();
    }

    float mean;
    float var;
    float weight;
};

struct ZmwDetectionModel
{
    static constexpr int numAnalogs = 4;

    template <int low>
    CUDA_ENABLED void Assign(const Data::LaneModelParameters<Cuda::PBHalf2, 32>& model, int idx)
    {
        for (int i = 0; i < numAnalogs; ++i)
        {
            analogs[i].Assign<low>(model.AnalogMode(i), idx);
        }
        baseline.Assign<low>(model.BaselineMode(), idx);
    }

    Cuda::Utility::CudaArray<ZmwAnalogMode, numAnalogs> analogs;
    ZmwAnalogMode baseline;

    // BENTODO not sure this is appropriate.  This doesn't make it into the full model
    float confidence = 0;
};

/// Implements DetectionModelEstimator using a Expectation-Maximization (EM)
/// approach for model estimation that runs on the CPU (as opposed to the GPU).
class DmeEmDevice : public CoreDMEstimator
{
public:     // Types
    using LaneDetModel = Data::LaneModelParameters<Cuda::PBHalf2, laneSize/2>;

    enum ZmwStatus : uint16_t
    {
        OK = 0,
        NO_CONVERGE     = 1u << 0,
        INSUF_DATA      = 1u << 1,
        VLOW_SIGNAL     = 1u << 2
    };

public:     // Static constants
    /// Number of free model parameters.
    /// Five mixture fractions, background mean, background variance, and
    /// pulse amplitude scale.
    static constexpr unsigned short nModelParams = 8;

    /// Minimum number of frames required for parameter estimation.
    static constexpr unsigned int nFramesMin = 20 * nModelParams;

public:     // Static functions
    static void Configure(const Data::BasecallerDmeConfig &dmeConfig,
                          const Data::MovieConfig &movConfig);

    // If mask[i], a[i] |= bits.
    // BENTODO eliminate?  Create a bit vector?
    __device__ static void SetBits(const bool mask, int32_t bits, int32_t* a)
    {
        if (mask) *a |= bits;
        // TODO: There is probably a more efficient way to implement this.
        //const IntVec b = *a | IntVec(bits);
        //*a = Blend(mask, b, *a);
    }

    PoolDetModel InitDetectionModels(const PoolBaselineStats& blStats) const override;

public:
    DmeEmDevice(uint32_t poolId, unsigned int poolSize);

private:    // Types

    enum ConfFactor
    {
        CONVERGED = 0,
        BL_FRACTION,
        BL_CV,
        BL_VAR_STABLE,
        ANALOG_REP,
        SNR_SUFFICIENT,
        SNR_DROP,
        G_TEST,
        NUM_CONF_FACTORS
    };

private:    // Customized implementation
    void EstimateImpl(const PoolHist& hist,
                      PoolDetModel* detModel) const override;

private:    // Static functions
    // Compute a preliminary scaling factor based on a fractile statistic.

    __device__ static float PrelimScaleFactor(const ZmwDetectionModel& model,
                                              const LaneHist& hist);

    // Apply a G-test significance test to assess goodness of fit of the model
    // to the trace histogram.
    __device__ static GoodnessOfFitTest<float> Gtest(const LaneHist& histogram,
                                                     const ZmwDetectionModel& model);

    // Compute the confidence factors of a model estimate, given the
    // diagnostics of the estimation, a reference model.
    static Cuda::Utility::CudaArray<float, NUM_CONF_FACTORS>
    __device__ ComputeConfidence(const DmeDiagnostics<float>& dmeDx,
                                 const ZmwDetectionModel& refModel,
                                 const ZmwDetectionModel& modelEst);

    // BENTODO rethink access
public:    // Functions
    // Use the trace histogram and the input detection model to compute a new
    // estimate for the detection model. Mix the new estimate with the input
    // model, weighted by confidence scores. That result is returned in detModel.
    __device__ static void EstimateLaneDetModel(const LaneHist& hist,
                                         LaneDetModel* detModel);
private:


    /// Updates *detModel by increasing the amplitude of all detection modes by
    /// \a scale. Also updates all detection mode covariances according
    /// to the standard noise model. Ratios of amplitudes among detection modes
    /// and properties of the background mode are preserved.
    __device__ static void ScaleModelSnr(const float& scale, ZmwDetectionModel* detModel);
};

}}}     // namespace PacBio::Mongo::Basecaller

#endif // mongo_basecaller_traceAnalysis_DmeEmDevice_H_
