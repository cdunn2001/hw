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
//  Defines class BasecallingMetrics, a simple POD class for conveying
//  basecalling metrics.

#include <numeric>
#include <pacbio/logging/Logger.h>

#include <common/cuda/utility/CudaArray.h>
#include <common/cuda/memory/UnifiedCudaArray.h>

#include "BaselinerStatAccumState.h"
#include "BatchData.h"
#include "HQRFPhysicalStates.h"

namespace PacBio {
namespace Mongo {
namespace Data {

/*
class HalfBasecallingMetrics
{

public: // types
    template <typename T>
    using SingleMetric = Cuda::Utility::CudaArray<T, laneSize/2>;
    template <typename T>
    using AnalogMetric = Cuda::Utility::CudaArray<SingleMetric<T>,
                                                  numAnalogs>;

public: // metrics retained from accumulator (more can be pulled through if necessary)
    // TODO: remove anything that isn't consumed outside of HFMetricsFilter...
    SingleMetric<PBShort2> activityLabel;
    SingleMetric<PBShort2> numPulseFrames;
    SingleMetric<PBShort2> numBaseFrames;
    SingleMetric<PBShort2> numSandwiches;
    SingleMetric<PBShort2> numHalfSandwiches;
    SingleMetric<PBShort2> numPulseLabelStutters;
    SingleMetric<PBShort2> numBases;
    SingleMetric<PBShort2> numPulses;
    AnalogMetric<PBHalf2> pkMidSignal;
    AnalogMetric<PBHalf2> bpZvar;
    AnalogMetric<PBHalf2> pkZvar;
    AnalogMetric<PBHalf2> pkMax;
    AnalogMetric<PBShort2> numPkMidFrames;
    AnalogMetric<PBShort2> numPkMidBasesByAnalog;
    AnalogMetric<PBShort2> numBasesByAnalog;
    AnalogMetric<PBShort2> numPulsesByAnalog;

    // TODO Add useful tracemetrics members here (there are others in the
    // accumulator member..., not sure if they are used):
    SingleMetric<PBHalf2> autocorrelation;
    SingleMetric<PBHalf2> pulseDetectionScore;
    SingleMetric<PBShort2> pixelChecksum;

    // These don't really match the Half pattern. They'll be a bit more
    // expensive, but nothing major
    SingleMetric<uint32_t> startFrame;
    SingleMetric<uint32_t> numFrames;
};

static_assert(sizeof(HalfBasecallingMetrics) == 65 * laneSize, "sizeof(BasecallingMetrics) is 65 bytes per zmw");

*/


class BasecallingMetrics
{

public: // types
    template <typename T>
    using SingleMetric = Cuda::Utility::CudaArray<T, laneSize>;
    template <typename T>
    using AnalogMetric = Cuda::Utility::CudaArray<SingleMetric<T>,
                                                  numAnalogs>;

public: // metrics retained from accumulator (more can be pulled through if necessary)
    // TODO: remove anything that isn't consumed outside of HFMetricsFilter...
    SingleMetric<HQRFPhysicalStates> activityLabel;
    SingleMetric<uint16_t> numPulseFrames;
    SingleMetric<uint16_t> numBaseFrames;
    SingleMetric<uint16_t> numSandwiches;
    SingleMetric<uint16_t> numHalfSandwiches;
    SingleMetric<uint16_t> numPulseLabelStutters;
    SingleMetric<uint16_t> numBases;
    SingleMetric<uint16_t> numPulses;
    AnalogMetric<float> pkMidSignal;
    AnalogMetric<float> bpZvar;
    AnalogMetric<float> pkZvar;
    AnalogMetric<float> pkMax;
    AnalogMetric<uint16_t> numPkMidFrames;
    AnalogMetric<uint16_t> numPkMidBasesByAnalog;
    AnalogMetric<uint16_t> numBasesByAnalog;
    AnalogMetric<uint16_t> numPulsesByAnalog;

    // TODO Add useful tracemetrics members here (there are others in the
    // accumulator member..., not sure if they are used):
    SingleMetric<uint32_t> startFrame;
    SingleMetric<uint32_t> numFrames;
    SingleMetric<float> autocorrelation;
    SingleMetric<float> pulseDetectionScore;
    SingleMetric<int16_t> pixelChecksum;
};

static_assert(sizeof(BasecallingMetrics) == 130 * laneSize, "sizeof(BasecallingMetrics) is 130 bytes per zmw");

class BasecallingMetricsFactory
{
public: // methods:
    BasecallingMetricsFactory(const Data::BatchDimensions& batchDims,
                              Cuda::Memory::SyncDirection syncDir)
        : batchDims_(batchDims)
        , syncDir_(syncDir)
    {}

    std::unique_ptr<Cuda::Memory::UnifiedCudaArray<BasecallingMetrics>> NewBatch()
    {
        return std::make_unique<Cuda::Memory::UnifiedCudaArray<BasecallingMetrics>>(
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
