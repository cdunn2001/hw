// Copyright (c) 2022, Pacific Biosciences of California, Inc.
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

#ifndef Mongo_BaseCaller_TraceAnalysis_HFMetricsFilterHybrid_H_
#define Mongo_BaseCaller_TraceAnalysis_HFMetricsFilterHybrid_H_

#include <common/AlignedVector.h>
#include <dataTypes/BasecallingMetricsAccumulator.h>
#include <dataTypes/BatchMetrics.h>

#include "HFMetricsFilterDevice.h"
#include "HFMetricsFilterHost.h"

namespace PacBio::Mongo::Basecaller
{
    
class HFMetricsFilterHybrid : public HFMetricsFilter
{
public:
    static void Configure(uint32_t sandwichTolerance,
                          uint32_t framesPerHFMetricBlock,
                          double frameRate,
                          bool realtimeActivityLabels);

public:
    HFMetricsFilterHybrid(uint32_t poolId, uint32_t lanesPerBatch,
                          Cuda::Memory::StashableAllocRegistrar* registrar = nullptr)
        : HFMetricsFilter(poolId)
        , device_(std::make_unique<HFMetricsFilterDevice>(poolId, lanesPerBatch, registrar))
        , host_(std::make_unique<HFMetricsFilterHost>(poolId, lanesPerBatch))
    { }

    HFMetricsFilterHybrid(const HFMetricsFilterHybrid&) = delete;
    HFMetricsFilterHybrid(HFMetricsFilterHybrid&&) = default;
    HFMetricsFilterHybrid& operator=(const HFMetricsFilterHybrid&) = delete;
    HFMetricsFilterHybrid& operator=(HFMetricsFilterHybrid&&) = default;
    ~HFMetricsFilterHybrid() override;

private:
    std::unique_ptr<BasecallingMetricsBatchT> Process(
            const PulseBatchT& pulseBatch,
            const Data::BaselinerMetrics& baselinerMetrics,
            const ModelsBatchT& models,
            const Data::FrameLabelerMetrics& flMetrics,
            const Data::PulseDetectorMetrics& pdMetrics) override;

private:

    static void DiffMetrics(const BasecallingMetricsBatchT& gpu, const BasecallingMetricsBatchT& cpu);

    std::unique_ptr<HFMetricsFilterDevice> device_;
    std::unique_ptr<HFMetricsFilterHost> host_;
};
        
} // PacBio::Mongo::Basecaller

#endif // Mongo_BaseCaller_TraceAnalysis_HFMetricsFilterHybrid_H_
