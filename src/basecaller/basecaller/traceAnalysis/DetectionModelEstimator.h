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

#include <dataTypes/AnalogMode.h>
#include <dataTypes/BaselineStats.h>
#include <dataTypes/ConfigForward.h>
#include <dataTypes/PoolDetectionModel.h>
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
    using PoolDetModel = Data::PoolDetectionModel<DetModelElementType>;
    using LaneDetModel = Data::LaneDetectionModel<DetModelElementType>;

public:     // Static functions
    static void Configure(const Data::BasecallerDmeConfig& dmeConfig,
                          const Data::MovieConfig& movConfig);

    /// Minimum number of frames added to trace histograms before we estimate
    /// model parameters.
    static uint32_t MinFramesForEstimate()
    { return minFramesForEstimate_; }

public:     // Structors and assignment
    DetectionModelEstimator(uint32_t poolId, unsigned int poolSize);

    PoolDetModel operator()(const Data::PoolHistogram<float, unsigned short>& hist,
                            const Cuda::Memory::UnifiedCudaArray<Data::BaselineStats<laneSize>>& blStats)
    {
        assert (hist.poolId == poolId_);

        PoolDetModel pdm (poolId_, poolSize_, Cuda::Memory::SyncDirection::HostWriteDeviceRead);

        auto pdmHost = pdm.laneModels.GetHostView();
        const auto& blStatsHost = blStats.GetHostView();
        for (unsigned int lane = 0; lane < poolSize_; ++lane)
        {
            InitDetModel(blStatsHost[lane], pdmHost[lane]);
        }

        // TODO

        return pdm;
    }

private:    // Static data
    static Cuda::Utility::CudaArray<Data::AnalogMode, numAnalogs> analogs_;
    static float refSnr_;   // Expected SNR for analog with relative amplitude of 1.
    static uint32_t minFramesForEstimate_;

private:
    uint32_t poolId_;
    unsigned int poolSize_;

private:    // Functions
    void InitDetModel(const Data::BaselineStats<laneSize>& blStats, LaneDetModel& ldm);
};

}}}     // namespace PacBio::Mongo::Basecaller

#endif // mongo_basecaller_traceAnalysis_DetectionModelEstimator_H_
