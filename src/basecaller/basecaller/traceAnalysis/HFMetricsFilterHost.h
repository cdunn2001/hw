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
/// \file   HFMetricsFilterHost.h
/// \brief  A filter for computing or aggregating trace- and pulse-metrics
///         on a time scale  equal to or greater than the standard block size.

#ifndef Mongo_BaseCaller_TraceAnalysis_HFMetricsFilterHost_H_
#define Mongo_BaseCaller_TraceAnalysis_HFMetricsFilterHost_H_

#include "HFMetricsFilter.h"
#include <common/AlignedVector.h>
#include <dataTypes/BasecallingMetricsAccumulator.h>
#include <dataTypes/BatchMetrics.h>

namespace PacBio {
namespace Mongo {
namespace Basecaller {

class HFMetricsFilterHost : public HFMetricsFilter
{
public:
    HFMetricsFilterHost(uint32_t poolId, uint32_t lanesPerBatch)
        : HFMetricsFilter(poolId)
        , metrics_(lanesPerBatch)
    { }
    HFMetricsFilterHost(const HFMetricsFilterHost&) = delete;
    HFMetricsFilterHost(HFMetricsFilterHost&&) = default;
    HFMetricsFilterHost& operator=(const HFMetricsFilterHost&) = delete;
    HFMetricsFilterHost& operator=(HFMetricsFilterHost&&) = default;
    ~HFMetricsFilterHost() override;

private:
    std::unique_ptr<BasecallingMetricsBatchT> Process(
            const PulseBatchT& pulseBatch,
            const Data::BaselinerMetrics& baselinerMetrics,
            const ModelsBatchT& models,
            const Data::FrameLabelerMetrics& flMetrics,
            const Data::PulseDetectorMetrics& pdMetrics) override;

private: // Block management
    void FinalizeBlock();

    void AddPulses(const PulseBatchT& batch);

    void AddModels(const ModelsBatchT& models);

    void AddMetrics(const Data::BaselinerMetrics& baselinerMetrics,
                    const Data::FrameLabelerMetrics& frameLabelerMetrics,
                    const Data::PulseDetectorMetrics& pdMetrics);

private: // members
    AlignedVector<Data::BasecallingMetricsAccumulator> metrics_;
};

}}} // PacBio::Mongo::Basecaller

#endif // Mongo_BaseCaller_TraceAnalysis_HFMetricsFilterHost_H_
