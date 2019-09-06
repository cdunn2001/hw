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
/// \file   DeviceHFMetricsFilter.cpp
/// \brief  A filter for computing or aggregating trace- and pulse-metrics
///         on a time scale equal to or greater than the standard block size.

#include "DeviceHFMetricsFilter.h"
#include <dataTypes/BatchData.cuh>
#include <dataTypes/BatchVectors.cuh>
#include <dataTypes/LaneDetectionModel.h>
#include <common/cuda/streams/LaunchManager.cuh>
#include <common/cuda/PBCudaSimd.cuh>
#include <common/cuda/memory/DeviceOnlyArray.cuh>

namespace PacBio {
namespace Mongo {
namespace Basecaller {


DeviceHFMetricsFilter::~DeviceHFMetricsFilter() = default;

namespace {

__global__ void InitializeMetrics(
        Cuda::Memory::DeviceView<BasecallingMetricsAccumulatorDevice> metrics)
{
    // TODO
}

__global__ void FinalizeMetrics(
        bool realtimeActivityLabels,
        float frameRate,
        Cuda::Memory::DeviceView<BasecallingMetricsAccumulatorDevice> metrics)
{
    // TODO
}

// TODO: figure out how many threads per block (as a function of laneSize...)
__global__ void ProcessChunk(
        const Cuda::Memory::DeviceView<const DeviceHFMetricsFilter::BaselinerStatsT> baselinerStats,
        const Cuda::Memory::DeviceView<const Data::LaneModelParameters<Cuda::PBHalf2, 32>> models,
        const Data::GpuBatchVectors<const Data::Pulse> pulses,
        const Cuda::Memory::DeviceView<const PacBio::Cuda::Utility::CudaArray<float, laneSize>> flMetrics,
        const Cuda::Memory::DeviceView<const PacBio::Mongo::StatAccumState> pdMetrics,
        Cuda::Memory::DeviceView<BasecallingMetricsAccumulatorDevice> metrics)
{
    // TODO
}

__global__ void PopulateBasecallingMetrics(
        const
        Cuda::Memory::DeviceView<BasecallingMetricsAccumulatorDevice> metrics,
        Cuda::Memory::DeviceView<Data::BasecallingMetrics> outMetrics)
{
    // TODO
}

} // anonymous

class DeviceHFMetricsFilter::AccumImpl
{
public:
    AccumImpl(size_t lanesPerPool)
        : metrics_(SOURCE_MARKER(), lanesPerPool)
        , framesSeen_(0)
        , lanesPerBatch_(lanesPerPool)
    { };

    std::unique_ptr<Cuda::Memory::UnifiedCudaArray<Data::BasecallingMetrics>>
    Process(const Data::PulseBatch& pulseBatch,
            const Data::BaselinerMetrics& baselinerMetrics,
            const
            Cuda::Memory::UnifiedCudaArray<Data::LaneModelParameters<Cuda::PBHalf, laneSize>>& models,
            const Data::FrameLabelerMetrics& flMetrics,
            const Data::PulseDetectorMetrics& pdMetrics)
    {
        if (framesSeen_ == 0)
        {
            const auto& initLauncher = Cuda::PBLauncher(InitializeMetrics,
                                                        lanesPerBatch_,
                                                        threadsPerBlock_);
            initLauncher(metrics_);
        }
        const auto& processLauncher = Cuda::PBLauncher(ProcessChunk,
                                                       lanesPerBatch_,
                                                       threadsPerBlock_);
        processLauncher(baselinerMetrics.baselinerStats,
                        models,
                        pulseBatch.Pulses(),
                        flMetrics.viterbiScore,
                        pdMetrics.baselineStats,
                        metrics_);
        framesSeen_ += pulseBatch.Dims().framesPerBatch;

        if (framesSeen_ >= framesPerHFMetricBlock_)
        {
            const auto& finalizeLauncher = Cuda::PBLauncher(
                    FinalizeMetrics,
                    lanesPerBatch_,
                    threadsPerBlock_);
            finalizeLauncher(realtimeActivityLabels_, frameRate_, metrics_);
            framesSeen_ = 0;
            auto ret = metricsFactory_->NewBatch();
            const auto& outputLauncher = Cuda::PBLauncher(PopulateBasecallingMetrics,
                                                          lanesPerBatch_,
                                                          threadsPerBlock_);
            outputLauncher(metrics_,
                           *(ret.get()));
            return ret;
        }
        return std::unique_ptr<Cuda::Memory::UnifiedCudaArray<Data::BasecallingMetrics>>();
    }

private:
    static constexpr size_t threadsPerBlock_ = laneSize / 2;

private:
    Cuda::Memory::DeviceOnlyArray<BasecallingMetricsAccumulatorDevice> metrics_;
    uint32_t framesSeen_;
    uint32_t lanesPerBatch_;

};

constexpr size_t DeviceHFMetricsFilter::AccumImpl::threadsPerBlock_;

DeviceHFMetricsFilter::DeviceHFMetricsFilter(uint32_t poolId,
                                             uint32_t lanesPerPool)
    : HFMetricsFilter(poolId)
    , impl_(std::make_unique<AccumImpl>(lanesPerPool))
{ };

std::unique_ptr<Cuda::Memory::UnifiedCudaArray<Data::BasecallingMetrics>>
DeviceHFMetricsFilter::Process(
        const Data::PulseBatch& pulseBatch,
        const Data::BaselinerMetrics& baselinerMetrics,
        const Cuda::Memory::UnifiedCudaArray<Data::LaneModelParameters<Cuda::PBHalf, 64>>& models,
        const Data::FrameLabelerMetrics& flMetrics,
        const Data::PulseDetectorMetrics& pdMetrics)
{
    return impl_->Process(pulseBatch, baselinerMetrics, models, flMetrics,
            pdMetrics);
}



}}} // PacBio::Mongo::Basecaller
