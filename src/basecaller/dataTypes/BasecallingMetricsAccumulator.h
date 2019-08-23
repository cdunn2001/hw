#ifndef mongo_dataTypes_BasecallingMetricsAccumulator_H_
#define mongo_dataTypes_BasecallingMetricsAccumulator_H_

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
//  Defines class BasecallingMetricsAccumulator

#include <numeric>
#include <pacbio/logging/Logger.h>

#include <common/cuda/utility/CudaArray.h>
#include <common/cuda/memory/UnifiedCudaArray.h>
#include <common/LaneArray.h>

#include "BaselinerStatAccumState.h"
#include "BatchMetadata.h"
#include "BatchData.h"
#include "BatchVectors.h"
#include "HQRFPhysicalStates.h"
#include "LaneDetectionModel.h"
#include "Pulse.h"
#include "PulseDetectionMetrics.h"
#include "TraceAnalysisMetrics.h"
#include "BasecallingMetrics.h"

namespace PacBio {
namespace Mongo {
namespace Data {

template <unsigned int LaneWidth>
class BasecallingMetricsAccumulator
{
public: // types
    using InputPulses = LaneVectorView<const Pulse>;
    using InputBaselineStats = Data::BaselinerStatAccumState;
    using InputModelsT = LaneModelParameters<Cuda::PBHalf, LaneWidth>;

    using UnsignedInt = uint16_t;
    using Flt = float;
    using SingleUnsignedIntegerMetric = LaneArray<UnsignedInt>;
    using SingleFloatMetric = LaneArray<Flt>;
    using AnalogUnsignedIntegerMetric = std::array<LaneArray<UnsignedInt>, numAnalogs>;
    using AnalogFloatMetric = std::array<LaneArray<Flt>, numAnalogs>;

    using BasecallingMetricsT = BasecallingMetrics<LaneWidth>;
    using BasecallingMetricsBatchT = Cuda::Memory::UnifiedCudaArray<BasecallingMetricsT>;

public:
    BasecallingMetricsAccumulator()
    {
        // Instead of using the initializer list to value-initialize a
        // bunch of lane arrays, we'll let them default initialize and set the
        // values of all members in Reset. More than half of the members are
        // std::arrays, and would still need to be filled anyway. This
        // constructor shouldn't be called after initial HFMetricsFilter
        // construction anwyay.
        Reset();
    };

public:
    void Count(const InputPulses& pulses, uint32_t numFrames);

    void AddPulseDetectionMetrics(const PulseDetectionMetrics& pdMetrics);

    void AddBaselinerStats(const InputBaselineStats& baselinerStats);

    void AddModels(const InputModelsT& models);

    void FinalizeMetrics(bool realtimeActivityLabels, float frameRate);

    void PopulateBasecallingMetrics(BasecallingMetricsT& metrics);

    void Reset();

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
    LaneArray<HQRFPhysicalStates> activityLabel_;
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
    std::array<Pulse, LaneWidth> prevBasecallCache_;
    std::array<Pulse, LaneWidth> prevprevBasecallCache_;

};

}}}     // namespace PacBio::Mongo::Data

#endif // mongo_dataTypes_BasecallingMetricsAccumulator_H_
