#ifndef mongo_basecaller_traceAnalysis_DetectionModelEstimator_H_
#define mongo_basecaller_traceAnalysis_DetectionModelEstimator_H_

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
//  Defines class DetectionModelEstimator.

#include <stdint.h>

#include <common/cuda/memory/UnifiedCudaArray.h>
#include <common/cuda/PBCudaSimd.h>
#include <common/LaneArray.h>

#include <dataTypes/AnalogMode.h>
#include <dataTypes/BaselinerStatAccumState.h>
#include <dataTypes/configs/ConfigForward.h>
#include <dataTypes/LaneDetectionModel.h>
#include <dataTypes/PoolHistogram.h>

namespace PacBio {
namespace Mongo {
namespace Basecaller {

/// Defines the interface and trivial implementation for estimation of
/// detection model parameters.
class DetectionModelEstimator
{
public:     // Types
    using DetModelElementType = Cuda::PBHalf;
    using LaneDetModel = Data::LaneDetectionModel<DetModelElementType>;
    using PoolDetModel = Cuda::Memory::UnifiedCudaArray<LaneDetModel>;
    using PoolBaselineStats = Cuda::Memory::UnifiedCudaArray<Data::BaselinerStatAccumState>;
    using PoolHist = Data::PoolHistogram<float, unsigned short>;
    using LaneHist = Data::LaneHistogram<float, unsigned short>;

public:     // Static functions
    static void Configure(const Data::BasecallerDmeConfig& dmeConfig,
                          const Data::MovieConfig& movConfig);

    static const Data::AnalogMode& Analog(unsigned int i)
    { return analogs_[i]; }

    /// Minimum number of frames added to trace histograms before we estimate
    /// model parameters.
    static uint32_t MinFramesForEstimate()
    { return minFramesForEstimate_; }

    /// The variance for \analog signal based on model including Poisson and
    /// "excess" noise.
    static LaneArray<float> ModelSignalCovar(const Data::AnalogMode& analog,
                                             const LaneArray<float>& signalMean,
                                             const LaneArray<float>& baselineVar);

public:     // Structors and assignment
    DetectionModelEstimator(uint32_t poolId, unsigned int poolSize);

public:     // Functions
    /// Initialize detection models based soley on baseline variance and
    /// reference SNR.
    PoolDetModel InitDetectionModels(const PoolBaselineStats& blStats) const;

    /// Estimate detection model parameters based on existing values and
    /// trace histogram.
    void Estimate(const PoolHist& hist, PoolDetModel* detModel) const
    {
        assert(detModel);
        assert(hist.data.Size() == poolSize_);
        assert(detModel->Size() == poolSize_);
        EstimateImpl(hist, detModel);
    }

    unsigned int PoolSize() const
    { return poolSize_; }

protected:
    static PacBio::Logging::PBLogger logger_;

private:    // Static data
    static Cuda::Utility::CudaArray<Data::AnalogMode, numAnalogs> analogs_;
    static float refSnr_;   // Expected SNR for analog with relative amplitude of 1.
    static uint32_t minFramesForEstimate_;
    static bool fixedBaselineParams_;
    static float fixedBaselineMean_;
    static float fixedBaselineVar_;

private:
    uint32_t poolId_;
    unsigned int poolSize_;

private:    // Customization functions
    virtual void EstimateImpl(const PoolHist&, PoolDetModel*) const
    {
        // Do nothing.
        // Derived implementation class should update detModel.
    }

private:    // Functions
    void InitLaneDetModel(const Data::BaselinerStatAccumState& blStats, LaneDetModel& ldm) const;
};

}}}     // namespace PacBio::Mongo::Basecaller

#endif // mongo_basecaller_traceAnalysis_DetectionModelEstimator_H_
