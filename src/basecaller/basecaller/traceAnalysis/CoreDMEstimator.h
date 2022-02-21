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
//  Defines class CoreDMEstimator.h.

#ifndef mongo_basecaller_traceAnalysis_CoreDMEstimator_H_
#define mongo_basecaller_traceAnalysis_CoreDMEstimator_H_

#include <common/cuda/PBCudaSimd.h>
#include <common/NumericUtil.h>

#include <dataTypes/BatchMetrics.h>
#include <dataTypes/BaselinerStatAccumState.h>
#include <dataTypes/configs/ConfigForward.h>
#include <dataTypes/LaneDetectionModel.h>
#include <dataTypes/PoolHistogram.h>

namespace PacBio {
namespace Mongo {
namespace Basecaller {

/// Defines the interface for estimation of detection model parameters.
class CoreDMEstimator
{
public:     // Types
    using DetModelElementType = Cuda::PBHalf;
    // TODO: Definitions of LaneDetModel and PoolDetModel should be fused with
    // similar definitions in DetectionModelEstimator.
    using LaneDetModel = Data::LaneDetectionModel<DetModelElementType>;
    using PoolDetModel = Data::DetectionModelPool<DetModelElementType>;
    using PoolBaselineStats = Data::BaselinerMetrics;
    using PoolHist = Data::PoolHistogram<float, unsigned short>;
    using LaneHist = Data::LaneHistogram<float, unsigned short>;
    using FrameIntervalType = PoolHist::FrameIntervalType;

    enum ZmwStatus : uint16_t
    {
        OK = 0,
        NO_CONVERGE     = 1u << 0,
        INSUF_DATA      = 1u << 1,
        VLOW_SIGNAL     = 1u << 2
    };

    struct Configuration
    {
        float fallbackBaselineMean;
        float fallbackBaselineVariance;
        float signalScaler;     // photoelectronSensitivity (e-/DN)

        CUDA_ENABLED float BaselineVarianceMin() const
        {
            constexpr auto oneTwelfth = 1.0f / 12.0f;
            const auto ss = std::max(signalScaler, 1.0f);
            return ss * ss * oneTwelfth;
        }

    };

public:     // Static constants
    /// Number of free model parameters.
    /// Five mixture fractions, background mean, background variance, and
    /// pulse amplitude scale.
    static constexpr unsigned short nModelParams = 8;

    /// Minimum number of frames required for parameter estimation.
    static constexpr unsigned int nFramesMin = 20 * nModelParams;

    /// Cross-talk correction value. This value was pre-computed in
    /// Matlab using data read from an arbitrary Spider trace file
    static constexpr float shotVarCoeff = 1.2171f;

public:     // Static functions
    static void Configure(const Data::AnalysisConfig& analysisConfig);

    /// The current configuration.
    static const Configuration& Config()
    { return config_; }

    /// Mean used for the baseline signal distribution when sample statistics
    /// are insufficient.
    static float FallbackBaselineMean()
    { return config_.fallbackBaselineMean; }

    /// Variance used for the baseline signal distribution when sample
    /// statistics are insufficient.
    static float FallbackBaselineVariance()
    { return config_.fallbackBaselineVariance; }

    /// Baseline variance is constrained to be no smaller than this.
    static float BaselineVarianceMin()
    { return config_.BaselineVarianceMin(); }

    /// The scale factor applied to the raw trace signal values.
    /// In current practice, the conversion from DN to photoelectron units
    /// (a.k.a. photoelectron sensitivity)
    static float SignalScaler()
    { return config_.signalScaler; }

public:     // Structors and assignment
    CoreDMEstimator(uint32_t poolId, unsigned int poolSize);
    virtual ~CoreDMEstimator() = default;

public:     // Functions
    /// Initialize detection models based soley on baseline variance and
    /// reference SNR.
    virtual PoolDetModel InitDetectionModels(const PoolBaselineStats& blStats) const = 0;

    /// Estimate detection model parameters based on existing values and
    /// trace histogram.
    void Estimate(const PoolHist& hist, const Data::BaselinerMetrics& metrics, PoolDetModel* detModel) const
    {
        assert(hist.poolId == poolId_);
        assert(detModel);
        assert(hist.data.Size() == poolSize_);
        assert(detModel->data.Size() == poolSize_);
        EstimateImpl(hist, metrics, detModel);
    }

    unsigned int PoolSize() const
    { return poolSize_; }

protected:
    static PacBio::Logging::PBLogger logger_;

private:    // Static data
    static Configuration config_;

private:
    uint32_t poolId_;
    unsigned int poolSize_;

private:    // Customization functions
    virtual void EstimateImpl(const PoolHist&, const Data::BaselinerMetrics&, PoolDetModel*) const
    {
        // Do nothing.
        // Derived implementation class should update detModel.
    }
};


}}}     // namespace PacBio::Mongo::Basecaller

#endif //mongo_basecaller_traceAnalysis_CoreDMEstimator.h_H_
