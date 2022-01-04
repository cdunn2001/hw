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
    using LaneDetModel = Data::LaneDetectionModel<DetModelElementType>;
    using PoolDetModel = Cuda::Memory::UnifiedCudaArray<LaneDetModel>;
    using PoolBaselineStats = Cuda::Memory::UnifiedCudaArray<Data::BaselinerStatAccumState>;
    using PoolHist = Data::PoolHistogram<float, unsigned short>;
    using LaneHist = Data::LaneHistogram<float, unsigned short>;

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

    /// Cross-talk correction value. This value was pre-computed in
    /// Matlab using data read from an arbitrary Spider trace file
    static constexpr float shotVarCoeff = 1.2171f;

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
        assert(detModel->Size() == poolSize_);
        EstimateImpl(hist, metrics, detModel);
    }

    unsigned int PoolSize() const
    { return poolSize_; }

protected:
    static PacBio::Logging::PBLogger logger_;

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
