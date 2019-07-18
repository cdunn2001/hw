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

#include <array>
#include <numeric>
#include <vector>
#include <pacbio/smrtdata/Basecall.h>
#include <pacbio/logging/Logger.h>

#include <common/cuda/utility/CudaArray.h>
#include <common/cuda/memory/UnifiedCudaArray.h>

#include "BatchMetadata.h"
#include "BatchData.h"
#include "BatchVectors.h"
#include "TraceAnalysisMetrics.h"

namespace PacBio {
namespace Mongo {
namespace Data {


template <unsigned int LaneWidth>
class BasecallingMetrics
{
public: // types
    using Basecall = PacBio::SmrtData::Basecall;

    static constexpr uint16_t NumAnalogs = 4;
    using UnsignedInt = uint16_t;
    using Flt = float;
    using SingleUnsignedIntegerMetric = Cuda::Utility::CudaArray<UnsignedInt, LaneWidth>;
    using SingleFloatMetric = Cuda::Utility::CudaArray<Flt, LaneWidth>;
    using AnalogUnsignedIntegerMetric = Cuda::Utility::CudaArray<
        Cuda::Utility::CudaArray<UnsignedInt, LaneWidth>,
        NumAnalogs>;
    using AnalogFloatMetric = Cuda::Utility::CudaArray<
        Cuda::Utility::CudaArray<Flt, LaneWidth>,
        NumAnalogs>;

    using InputBasecalls = LaneVectorView<const Basecall>;

public:

    BasecallingMetrics() = default;

    void Initialize();

    void Count(const InputBasecalls& bases);

    void FinalizeMetrics();

public:
    const SingleUnsignedIntegerMetric& NumPulseFrames() const
    { return numPulseFrames_; }

    const SingleUnsignedIntegerMetric& NumBaseFrames() const
    { return numBaseFrames_; }

    const SingleUnsignedIntegerMetric& NumSandwiches() const
    { return numSandwiches_; }

    const SingleUnsignedIntegerMetric& NumHalfSandwiches() const
    { return numHalfSandwiches_; }

    const SingleUnsignedIntegerMetric& NumPulseLabelStutters() const
    { return numPulseLabelStutters_; }

    const AnalogFloatMetric& PkMidSignal() const
    { return pkMidSignal_; }

    const AnalogFloatMetric& BpZvar() const
    { return bpZvar_; }

    const AnalogFloatMetric& PkZvar() const
    { return pkZvar_; }

    const AnalogUnsignedIntegerMetric& PkMidNumFrames() const
    { return pkMidNumFrames_; }

    const AnalogFloatMetric& PkMax() const
    { return pkMax_; }

    const AnalogUnsignedIntegerMetric& NumPkMidBasesByAnalog() const
    { return numPkMidBasesByAnalog_; }

    const AnalogUnsignedIntegerMetric& NumBasesByAnalog() const
    { return numBasesByAnalog_; }

    const AnalogUnsignedIntegerMetric& NumPulsesByAnalog() const
    { return numPulsesByAnalog_; }

    SingleUnsignedIntegerMetric NumBases() const
    {
        SingleUnsignedIntegerMetric tbr;
        for (size_t z = 0; z < LaneWidth; ++z)
            tbr[z] = 0;
        for (size_t a = 0; a < NumAnalogs; ++a)
        {
            for (size_t z = 0; z < LaneWidth; ++z)
            {
                tbr[z] += numBasesByAnalog_[a][z];
            }
        }
        return tbr;
    }

    SingleUnsignedIntegerMetric NumPulses() const
    {
        SingleUnsignedIntegerMetric tbr;
        for (size_t z = 0; z < LaneWidth; ++z)
            tbr[z] = 0;
        for (size_t a = 0; a < NumAnalogs; ++a)
        {
            for (size_t z = 0; z < LaneWidth; ++z)
            {
                tbr[z] += numPulsesByAnalog_[a][z];
            }
        }
        return tbr;
    }

/*
public:
    Cuda::Utility::CudaArray<uint16_t,4>& NumBasesByAnalog()
    { return numBasesByAnalog_; }

    Cuda::Utility::CudaArray<uint16_t,4>& NumPulsesByAnalog()
    { return numPulsesByAnalog_; }
*/

private: // metrics
    SingleUnsignedIntegerMetric numPulseFrames_;
    SingleUnsignedIntegerMetric numBaseFrames_;
    SingleUnsignedIntegerMetric numSandwiches_;
    SingleUnsignedIntegerMetric numHalfSandwiches_;
    SingleUnsignedIntegerMetric numPulseLabelStutters_;
    //ActivityLabeler::HQRFPhysicalState activityLabel_;
    AnalogFloatMetric pkMidSignal_;
    AnalogFloatMetric bpZvar_;
    AnalogFloatMetric pkZvar_;
    AnalogFloatMetric pkMax_;
    AnalogUnsignedIntegerMetric pkMidNumFrames_;
    AnalogUnsignedIntegerMetric numPkMidBasesByAnalog_;
    AnalogUnsignedIntegerMetric numBasesByAnalog_;
    AnalogUnsignedIntegerMetric numPulsesByAnalog_;

    TraceAnalysisMetrics<LaneWidth> traceMetrics_;

private: // state helpers
    Cuda::Utility::CudaArray<Basecall, LaneWidth> prevBasecallCache;
    Cuda::Utility::CudaArray<Basecall, LaneWidth> prevprevBasecallCache;

};




class SingleBasecallingMetrics
{
public:
    using Basecall = PacBio::SmrtData::Basecall;

    SingleBasecallingMetrics() = default;

    SingleBasecallingMetrics& Count(const Basecall& base);

    void FinalizeVariance();

public:
    const uint16_t NumPulseFrames() const
    { return numPulseFrames_; }

    const uint16_t NumBaseFrames() const
    { return numBaseFrames_; }

    const uint16_t NumSandwiches() const
    { return numSandwiches_; }

    const uint16_t NumHalfSandwiches() const
    { return numHalfSandwiches_; }

    const uint16_t NumPulseLabelStutters() const
    { return numPulseLabelStutters_; }

    const Cuda::Utility::CudaArray<uint16_t,4> PkMidSignal() const
    { return pkMidSignal_; }

    const Cuda::Utility::CudaArray<uint16_t,4> BpZvar() const
    { return bpZvar_; }

    const Cuda::Utility::CudaArray<uint16_t,4> PkZvar() const
    { return pkZvar_; }

    const Cuda::Utility::CudaArray<uint16_t,4> PkMidNumFrames() const
    { return pkMidNumFrames_; }

    const Cuda::Utility::CudaArray<uint16_t,4> PkMax() const
    { return pkMax_; }

    const Cuda::Utility::CudaArray<uint16_t,4> NumPkMidBasesByAnalog() const
    { return numPkMidBasesByAnalog_; }

    const Cuda::Utility::CudaArray<uint16_t,4> NumBasesByAnalog() const
    { return numBasesByAnalog_; }

    const Cuda::Utility::CudaArray<uint16_t,4> NumPulsesByAnalog() const
    { return numPulsesByAnalog_; }

    uint16_t NumBases() const
    { return std::accumulate(numBasesByAnalog_.begin(), numBasesByAnalog_.end(), 0); }

    uint16_t NumPulses() const
    { return std::accumulate(numPulsesByAnalog_.begin(), numPulsesByAnalog_.end(), 0); }

/*
public:
    Cuda::Utility::CudaArray<uint16_t,4>& NumBasesByAnalog()
    { return numBasesByAnalog_; }

    Cuda::Utility::CudaArray<uint16_t,4>& NumPulsesByAnalog()
    { return numPulsesByAnalog_; }
*/

private:
    uint16_t numPulseFrames_;
    uint16_t numBaseFrames_;
    uint16_t numSandwiches_;
    uint16_t numHalfSandwiches_;
    uint16_t numPulseLabelStutters_;
    //ActivityLabeler::HQRFPhysicalState activityLabel_;
    Cuda::Utility::CudaArray<uint16_t,4> pkMidSignal_;
    Cuda::Utility::CudaArray<uint16_t,4> bpZvar_;
    Cuda::Utility::CudaArray<uint16_t,4> pkZvar_;
    Cuda::Utility::CudaArray<uint16_t,4> pkMidNumFrames_;
    Cuda::Utility::CudaArray<uint16_t,4> pkMax_;
    Cuda::Utility::CudaArray<uint16_t,4> numPkMidBasesByAnalog_;
    Cuda::Utility::CudaArray<uint16_t,4> numBasesByAnalog_;
    Cuda::Utility::CudaArray<uint16_t,4> numPulsesByAnalog_;
};

class MinimalBasecallingMetrics
{
    // TODO: Fill in the details of class BasecallingMetrics.
public:
    using Basecall = PacBio::SmrtData::Basecall;

    MinimalBasecallingMetrics() = default;

    MinimalBasecallingMetrics& Count(const Basecall& base);

public:
    const Cuda::Utility::CudaArray<uint8_t,4> NumBasesByAnalog() const
    { return numBasesByAnalog_; }

    const Cuda::Utility::CudaArray<uint8_t,4> NumPulsesByAnalog() const
    { return numPulsesByAnalog_; }

    uint16_t NumBases() const
    { return std::accumulate(numBasesByAnalog_.begin(), numBasesByAnalog_.end(), 0); }

    uint16_t NumPulses() const
    { return std::accumulate(numPulsesByAnalog_.begin(), numPulsesByAnalog_.end(), 0); }

public:
    Cuda::Utility::CudaArray<uint8_t,4>& NumBasesByAnalog()
    { return numBasesByAnalog_; }

    Cuda::Utility::CudaArray<uint8_t,4>& NumPulsesByAnalog()
    { return numPulsesByAnalog_; }

private:
    Cuda::Utility::CudaArray<uint8_t,4> numBasesByAnalog_;
    Cuda::Utility::CudaArray<uint8_t,4> numPulsesByAnalog_;
};


template <unsigned int LaneWidth>
class BasecallingMetricsFactory
{
    // Switch back for DualAllocationPools:
    //using Pools = Cuda::Memory::DualAllocationPools;
    using Pools = Cuda::Memory::HostAllocationPool;
    using BasecallingMetricsT = BasecallingMetrics<LaneWidth>;
    using MetricsBlock = Cuda::Memory::UnifiedCudaArray<BasecallingMetricsT>;

public:
    BasecallingMetricsFactory(const Data::BatchDimensions& batchDims,
                              Cuda::Memory::SyncDirection syncDir,
                              bool pinned)
        : batchDims_(batchDims)
        , syncDir_(syncDir)
        , pinned_(pinned)
        , metricsPool_(std::make_shared<Pools>(
                Cuda::Memory::HostAllocator(
                    batchDims.lanesPerBatch * sizeof(BasecallingMetricsT),
                    pinned)))
    // Switch back for DualAllocationPools:
        //, metricsPool_(std::make_shared<Pools>(
        //        batchDims.lanesPerBatch * sizeof(BasecallingMetricsT),
        //        pinned))
    {}

    std::unique_ptr<MetricsBlock> NewBatch()
    {
        return std::make_unique<MetricsBlock>(
            batchDims_.lanesPerBatch,
            syncDir_,
            pinned_);
            // Switch back for DualAllocationPools:
            //metricsPool_);
    }

private:
    Data::BatchDimensions batchDims_;
    Cuda::Memory::SyncDirection syncDir_;
    bool pinned_;

    std::shared_ptr<Pools> metricsPool_;
};


//static_assert(sizeof(BasecallingMetrics<laneSize>) == 212, "sizeof(BasecallingMetrics) is 212 bytes");

}}}     // namespace PacBio::Mongo::Data

#endif // mongo_dataTypes_BasecallingMetrics_H_
