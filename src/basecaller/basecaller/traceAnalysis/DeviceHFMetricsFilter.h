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
#include <dataTypes/BatchMetrics.h>
#include <dataTypes/BatchVectors.h>
#include <dataTypes/HQRFPhysicalStates.h>
#include <dataTypes/Pulse.h>
#include <dataTypes/TrainedCartDevice.h>
#include <common/StatAccumState.h>
#include <common/AutocorrAccumState.h>
#include <common/cuda/memory/DeviceAllocationStash.h>

namespace PacBio {
namespace Mongo {
namespace Basecaller {
class DeviceHFMetricsFilter : public HFMetricsFilter
{
public: // static methods
    static void Configure(uint32_t sandwichTolerance,
                          uint32_t framesPerHFMetricBlock,
                          double frameRate,
                          bool realtimeActivityLabels);

public:
    class AccumImpl;

public:
    DeviceHFMetricsFilter(uint32_t poolId, uint32_t lanesPerPool,
                          Cuda::Memory::StashableAllocRegistrar* registrar = nullptr);
    DeviceHFMetricsFilter(const DeviceHFMetricsFilter&) = delete;
    DeviceHFMetricsFilter(DeviceHFMetricsFilter&&) = default;
    DeviceHFMetricsFilter& operator=(const DeviceHFMetricsFilter&) = delete;
    DeviceHFMetricsFilter& operator=(DeviceHFMetricsFilter&&) = default;
    ~DeviceHFMetricsFilter() override;

private:
    std::unique_ptr<Cuda::Memory::UnifiedCudaArray<Data::BasecallingMetrics>>
    Process(const Data::PulseBatch& basecallBatch,
            const Data::BaselinerMetrics& baselinerStats,
            const ModelsBatchT& models,
            const Data::FrameLabelerMetrics& flMetrics,
            const Data::PulseDetectorMetrics& pdMetrics) override;

private: // members
    std::unique_ptr<AccumImpl> impl_;
};

}}} // PacBio::Mongo::Basecaller

#endif // Mongo_BaseCaller_TraceAnalysis_DeviceHFMetricsFilter_H_
