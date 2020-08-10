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
//  Defines class BasecallingMetricsAccumulator, which is Host-only and should
//  be named as such if a Device version is introduced. TODO

#include <numeric>
#include <pacbio/logging/Logger.h>

#include <common/cuda/utility/CudaArray.h>
#include <common/cuda/memory/UnifiedCudaArray.h>
#include <common/LaneArray.h>

#include "BaselinerStatAccumState.h"
#include "BatchData.h"
#include "BatchMetadata.h"
#include "BatchMetrics.h"
#include "BatchVectors.h"
#include "HQRFPhysicalStates.h"
#include "LaneDetectionModel.h"
#include "Pulse.h"
#include "TraceAnalysisMetrics.h"
#include "BasecallingMetrics.h"

namespace PacBio {
namespace Mongo {
namespace Data {

class BasecallingMetricsAccumulator
{
    // TODO rename
    template <typename T>
    using Array_t = PacBio::Cuda::Utility::CudaArray<T, 64>;
public: // types
    template <typename T>
    using SingleMetric = Array_t<T>;
    template <typename T>
    using AnalogMetric = std::array<Array_t<T>, numAnalogs>;

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
    void Count(const LaneVectorView<const Pulse>& pulses, uint32_t numFrames);

    void AddBatchMetrics(
            const BaselinerStatAccumState& baselinerStats,
            const Cuda::Utility::CudaArray<float, laneSize>& viterbiScore,
            const StatAccumState& pdBaselineStats);

    void AddModels(const LaneModelParameters<Cuda::PBHalf, laneSize>& models);

    void FinalizeMetrics(bool realtimeActivityLabels, float frameRate);

    void PopulateBasecallingMetrics(BasecallingMetrics& metrics);

    void Reset();

private:
    void LabelBlock(float frameRate);

public: // complex accessors

    AnalogMetric<float> PkmidMean() const;

    SingleMetric<uint16_t> NumBases() const;

    SingleMetric<uint16_t> NumPulses() const;

    SingleMetric<float> PulseWidth() const;

private: // metrics
    SingleMetric<uint16_t> numPulseFrames_;
    SingleMetric<uint16_t> numBaseFrames_;
    SingleMetric<uint16_t> numSandwiches_;
    SingleMetric<uint16_t> numHalfSandwiches_;
    SingleMetric<uint16_t> numPulseLabelStutters_;
    SingleMetric<uint16_t> activityLabel_;
    AnalogMetric<float> pkMidSignal_;
    AnalogMetric<float> bpZvar_;
    AnalogMetric<float> pkZvar_;
    AnalogMetric<float> pkMax_;
    AnalogMetric<float> modelVariance_;
    AnalogMetric<float> modelMean_;
    AnalogMetric<uint16_t> numPkMidFrames_;
    AnalogMetric<uint16_t> numPkMidBasesByAnalog_;
    AnalogMetric<uint16_t> numBasesByAnalog_;
    AnalogMetric<uint16_t> numPulsesByAnalog_;

    TraceAnalysisMetrics traceMetrics_;

private: // state trackers
    std::array<Pulse, laneSize> prevBasecallCache_;
    std::array<Pulse, laneSize> prevprevBasecallCache_;

};

}}}     // namespace PacBio::Mongo::Data

#endif // mongo_dataTypes_BasecallingMetricsAccumulator_H_
