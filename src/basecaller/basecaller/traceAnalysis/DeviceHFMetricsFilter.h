#ifndef Mongo_BaseCaller_TraceAnalysis_DeviceHFMetricsFilter_H_

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
/// \file   DeviceHFMetricsFilter.h
/// \brief  A filter for computing or aggregating trace- and pulse-metrics
///         on a time scale  equal to or greater than the standard block size.

#include "HFMetricsFilter.h"
#include <dataTypes/BasecallingMetrics.h>
#include <dataTypes/BatchVectors.h>
#include <dataTypes/HQRFPhysicalStates.h>
#include <dataTypes/Pulse.h>
#include <dataTypes/PulseDetectionMetrics.h>
#include <common/StatAccumState.h>
#include <common/AutocorrAccumState.h>
#include <common/cuda/memory/DeviceOnlyArray.cuh>

namespace PacBio {
namespace Mongo {
namespace Basecaller {

template <unsigned int LaneWidth>
struct alignas(64) BasecallingMetricsAccumulatorDevice
{
public: // types
    using InputPulses = Data::LaneVectorView<const Data::Pulse>;
    using InputBaselineStats = Data::BaselinerStatAccumState;
    using InputModelsT = Data::LaneModelParameters<Cuda::PBHalf, LaneWidth>;

    using UnsignedInt = int16_t;
    using Int = uint16_t;
    using Flt = float;
    using SingleUnsignedIntegerMetric = Cuda::Utility::CudaArray<
        UnsignedInt, LaneWidth>;
    using SingleIntegerMetric = Cuda::Utility::CudaArray<Int, LaneWidth>;
    using SingleFloatMetric = Cuda::Utility::CudaArray<Flt, LaneWidth>;
    using AnalogUnsignedIntegerMetric = Cuda::Utility::CudaArray<
        Cuda::Utility::CudaArray<UnsignedInt, LaneWidth>, numAnalogs>;
    using AnalogFloatMetric = Cuda::Utility::CudaArray<
        Cuda::Utility::CudaArray<Flt, LaneWidth>, numAnalogs>;

    using BasecallingMetricsT = Data::BasecallingMetrics<LaneWidth>;
    using BasecallingMetricsBatchT = Cuda::Memory::UnifiedCudaArray<
        BasecallingMetricsT>;

private: // metrics
    SingleUnsignedIntegerMetric numPulseFrames_;
    SingleUnsignedIntegerMetric numBaseFrames_;
    SingleUnsignedIntegerMetric numSandwiches_;
    SingleUnsignedIntegerMetric numHalfSandwiches_;
    SingleUnsignedIntegerMetric numPulseLabelStutters_;
    Cuda::Utility::CudaArray<Data::HQRFPhysicalStates, LaneWidth> activityLabel_;
    AnalogFloatMetric pkMidSignal_;
    AnalogFloatMetric bpZvar_;
    AnalogFloatMetric pkZvar_;
    AnalogFloatMetric pkMax_;
    AnalogFloatMetric modelVariance_;
    AnalogFloatMetric modelMean_;
    AnalogUnsignedIntegerMetric pkMidNumFrames_;
    AnalogUnsignedIntegerMetric numPkMidBasesByAnalog_;
    AnalogUnsignedIntegerMetric numBasesByAnalog_;
    AnalogUnsignedIntegerMetric numPulsesByAnalog_;

    // From TraceAnalysisMetrics.h:

    SingleUnsignedIntegerMetric startFrame_;
    SingleUnsignedIntegerMetric numFrames_;
    SingleIntegerMetric pixelChecksum_;
    SingleFloatMetric pulseDetectionScore_;
    // TODO: replace these with states:
    StatAccumState baselineStatAccum_;
    AutocorrAccumState autocorrAccum_;

private: // state trackers
    Cuda::Utility::CudaArray<Data::Pulse, LaneWidth> prevBasecallCache_;
    Cuda::Utility::CudaArray<Data::Pulse, LaneWidth> prevprevBasecallCache_;

};

class DeviceHFMetricsFilter : public HFMetricsFilter
{
public:
    using BasecallingMetricsAccumulatorT = BasecallingMetricsAccumulatorDevice<laneSize>;
    using BasecallingMetricsAccumulatorBatchT = Cuda::Memory::DeviceOnlyArray<BasecallingMetricsAccumulatorT>;

public:
    static size_t threadsPerBlock_;

public:
    DeviceHFMetricsFilter(uint32_t poolId)
        : HFMetricsFilter(poolId)
        , metrics_(lanesPerBatch_)
    { };
    DeviceHFMetricsFilter(const DeviceHFMetricsFilter&) = delete;
    DeviceHFMetricsFilter(DeviceHFMetricsFilter&&) = default;
    DeviceHFMetricsFilter& operator=(const DeviceHFMetricsFilter&) = delete;
    DeviceHFMetricsFilter& operator=(DeviceHFMetricsFilter&&) = default;
    ~DeviceHFMetricsFilter() override;

private:
    std::unique_ptr<BasecallingMetricsBatchT> Process(
            const PulseBatchT& pulseBatch,
            const BaselinerStatsBatchT& baselineStats,
            const ModelsBatchT& models) override;

private: // Block management
    void FinalizeBlock() override;

private: // members
    BasecallingMetricsAccumulatorBatchT metrics_;
};

}}} // PacBio::Mongo::Basecaller

#endif // Mongo_BaseCaller_TraceAnalysis_DeviceHFMetricsFilter_H_
