#ifndef mongo_dataTypes_BasecallingMetrics_H_
#define mongo_dataTypes_BasecallingMetrics_H_

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
//  Defines class BasecallingMetrics

#include <numeric>
#include <pacbio/logging/Logger.h>

#include <common/cuda/utility/CudaArray.h>
#include <common/cuda/memory/UnifiedCudaArray.h>

#include <dataTypes/LaneDetectionModel.h>
#include <dataTypes/BaselinerStatAccumState.h>

#include "HQRFPhysicalStates.h"
#include "BatchMetadata.h"
#include "BatchData.h"
#include "BatchVectors.h"
#include "TraceAnalysisMetrics.h"
#include "Pulse.h"

namespace PacBio {
namespace Mongo {
namespace Data {

template <unsigned int LaneWidth>
class BasecallingMetrics
{

public: // types
    using UnsignedInt = uint16_t;
    using Int = int16_t;
    using Flt = float;
    using SingleUnsignedIntegerMetric = Cuda::Utility::CudaArray<UnsignedInt,
                                                                 LaneWidth>;
    using SingleIntegerMetric = Cuda::Utility::CudaArray<Int,
                                                                 LaneWidth>;
    using SingleFloatMetric = Cuda::Utility::CudaArray<Flt, LaneWidth>;
    using AnalogUnsignedIntegerMetric = Cuda::Utility::CudaArray<
        Cuda::Utility::CudaArray<UnsignedInt, LaneWidth>,
        numAnalogs>;
    using AnalogFloatMetric = Cuda::Utility::CudaArray<
        Cuda::Utility::CudaArray<Flt, LaneWidth>,
        numAnalogs>;

public: // metrics retained from accumulator (more can be pulled through if necessary)
    // TODO: remove anything that isn't consumed outside of HFMetricsFilter...
    Cuda::Utility::CudaArray<HQRFPhysicalStates,
                             LaneWidth> activityLabel;
    SingleUnsignedIntegerMetric numPulseFrames;
    SingleUnsignedIntegerMetric numBaseFrames;
    SingleUnsignedIntegerMetric numSandwiches;
    SingleUnsignedIntegerMetric numHalfSandwiches;
    SingleUnsignedIntegerMetric numPulseLabelStutters;
    AnalogFloatMetric pkMidSignal;
    AnalogFloatMetric bpZvar;
    AnalogFloatMetric pkZvar;
    AnalogFloatMetric pkMax;
    AnalogUnsignedIntegerMetric pkMidNumFrames;
    AnalogUnsignedIntegerMetric numPkMidBasesByAnalog;
    AnalogUnsignedIntegerMetric numBasesByAnalog;
    AnalogUnsignedIntegerMetric numPulsesByAnalog;
    SingleUnsignedIntegerMetric numBases;
    SingleUnsignedIntegerMetric numPulses;

    // TODO Add useful tracemetrics members here (there are others in the
    // accumulator member..., not sure if they are used):
    SingleUnsignedIntegerMetric startFrame;
    SingleUnsignedIntegerMetric stopFrame;
    SingleUnsignedIntegerMetric numFrames;
    SingleFloatMetric autocorrelation;
    SingleFloatMetric pulseDetectionScore;
    SingleIntegerMetric pixelChecksum;
};

static_assert(sizeof(BasecallingMetrics<laneSize>) == 127 * laneSize, "sizeof(BasecallingMetrics) is 127 bytes per zmw");


template <unsigned int LaneWidth>
class BasecallingMetricsAccumulator
{
public: // types
    using InputPulses = LaneVectorView<const Pulse>;
    using InputBaselineStats = Data::BaselinerStatAccumState;
    using InputModelsT = LaneModelParameters<Cuda::PBHalf, LaneWidth>;

    using UnsignedInt = uint16_t;
    using Flt = float;
    using SingleUnsignedIntegerMetric = Cuda::Utility::CudaArray<UnsignedInt,
                                                                 LaneWidth>;
    using SingleFloatMetric = Cuda::Utility::CudaArray<Flt, LaneWidth>;
    using AnalogUnsignedIntegerMetric = Cuda::Utility::CudaArray<
        Cuda::Utility::CudaArray<UnsignedInt, LaneWidth>,
        numAnalogs>;
    using AnalogFloatMetric = Cuda::Utility::CudaArray<
        Cuda::Utility::CudaArray<Flt, LaneWidth>,
        numAnalogs>;

    using BasecallingMetricsT = BasecallingMetrics<LaneWidth>;
    using BasecallingMetricsBatchT = Cuda::Memory::UnifiedCudaArray<BasecallingMetricsT>;

public:
    void Initialize();

    void Count(const InputPulses& pulses, uint32_t numFrames);

    void AddBaselineStats(const InputBaselineStats& baselineStats);

    void AddModels(const InputModelsT& models);

    void FinalizeMetrics(bool realtimeActivityLabels, float frameRate);

    void PopulateBasecallingMetrics(BasecallingMetricsT& metrics);

    void Reset()
    { Initialize(); };

private:
    void LabelBlock(float frameRate);

public: // complex accessors

    AnalogFloatMetric PkmidMean() const;

    SingleUnsignedIntegerMetric NumBases() const;

    SingleUnsignedIntegerMetric NumPulses() const;

    SingleFloatMetric PulseWidth() const;

private: // metrics
    SingleUnsignedIntegerMetric numPulseFrames_;
    SingleUnsignedIntegerMetric numBaseFrames_;
    SingleUnsignedIntegerMetric numSandwiches_;
    SingleUnsignedIntegerMetric numHalfSandwiches_;
    SingleUnsignedIntegerMetric numPulseLabelStutters_;
    Cuda::Utility::CudaArray<HQRFPhysicalStates,
                             LaneWidth> activityLabel_;
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

    TraceAnalysisMetrics<LaneWidth> traceMetrics_;

private: // state trackers
    Cuda::Utility::CudaArray<Pulse, LaneWidth> prevBasecallCache_;
    Cuda::Utility::CudaArray<Pulse, LaneWidth> prevprevBasecallCache_;

};

template <unsigned int LaneWidth>
class BasecallingMetricsAccumulatorFactory
{
    using AccumulatorT = BasecallingMetricsAccumulator<LaneWidth>;
    using AccumulatorBatchT = Cuda::Memory::UnifiedCudaArray<AccumulatorT>;

public:
    BasecallingMetricsAccumulatorFactory(
            const Data::BatchDimensions& batchDims,
            Cuda::Memory::SyncDirection syncDir)
        : batchDims_(batchDims)
        , syncDir_(syncDir)
    {}

    std::unique_ptr<AccumulatorBatchT> NewBatch()
    {
        return std::make_unique<AccumulatorBatchT>(
            batchDims_.lanesPerBatch,
            syncDir_,
            SOURCE_MARKER());
    }

private:
    Data::BatchDimensions batchDims_;
    Cuda::Memory::SyncDirection syncDir_;
};

template <unsigned int LaneWidth>
class BasecallingMetricsFactory
{
public: // types
    using BasecallingMetricsT = BasecallingMetrics<LaneWidth>;
    using BasecallingMetricsBatchT = Cuda::Memory::UnifiedCudaArray<BasecallingMetricsT>;

public: // methods:
    BasecallingMetricsFactory(const Data::BatchDimensions& batchDims,
                              Cuda::Memory::SyncDirection syncDir)
        : batchDims_(batchDims)
        , syncDir_(syncDir)
    {}

    std::unique_ptr<BasecallingMetricsBatchT> NewBatch()
    {
        return std::make_unique<BasecallingMetricsBatchT>(
            batchDims_.lanesPerBatch,
            syncDir_,
            SOURCE_MARKER());
    }

private: // members:
    Data::BatchDimensions batchDims_;
    Cuda::Memory::SyncDirection syncDir_;
};

}}}     // namespace PacBio::Mongo::Data

#endif // mongo_dataTypes_BasecallingMetrics_H_
