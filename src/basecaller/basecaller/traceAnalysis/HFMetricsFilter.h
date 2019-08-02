#ifndef Mongo_BaseCaller_TraceAnalysis_HFMetricsFilter_H_
#define Mongo_BaseCaller_TraceAnalysis_HFMetricsFilter_H_

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
/// \file   HFMetricsFilter.h
/// \brief  A filter for computing or aggregating trace- and pulse-metrics
///         on a time scale  equal to or greater than the standard block size.

#include <atomic>
#include <tbb/cache_aligned_allocator.h>

#include <dataTypes/ConfigForward.h>
#include <common/MongoConstants.h>
#include <dataTypes/PulseBatch.h>
#include <dataTypes/BasecallerConfig.h>
#include <dataTypes/BasecallingMetrics.h>
#include <dataTypes/LaneDetectionModel.h>


namespace PacBio {
namespace Mongo {
namespace Basecaller {

class HFMetricsFilter
{
public:     // Types
    using PulseBatchT = Data::PulseBatch;
    using BaselineStatsT = Cuda::Memory::UnifiedCudaArray<Data::BaselineStats<laneSize>>;
    using BasecallingMetricsT = Data::BasecallingMetrics<laneSize>;
    using BasecallingMetricsBatchT = Cuda::Memory::UnifiedCudaArray<BasecallingMetricsT>;
    using BasecallingMetricsAccumulatorT = Data::BasecallingMetricsAccumulator<laneSize>;
    using BasecallingMetricsBatchAccumulatorT = Cuda::Memory::UnifiedCudaArray<BasecallingMetricsAccumulatorT>;
    using ModelsT = Cuda::Memory::UnifiedCudaArray<Data::LaneModelParameters<Cuda::PBHalf, laneSize>>;

public: // Static functions
    static void Configure(const Data::BasecallerMetricsConfig&);
    static void Finalize();
    static void InitAllocationPools(bool hostExecution);
    static void DestroyAllocationPools();

protected: // Static members
    static std::unique_ptr<Data::BasecallingMetricsFactory<laneSize>> metricsFactory_;
    static std::unique_ptr<Data::BasecallingMetricsAccumulatorFactory<laneSize>> metricsAccumulatorFactory_;
    static uint32_t sandwichTolerance_;
    static uint32_t framesPerHFMetricBlock_;
    static double frameRate_;
    static bool realtimeActivityLabels_;
    static uint32_t lanesPerBatch_;
    static uint32_t zmwsPerBatch_;


public: // Structors
    HFMetricsFilter(uint32_t poolId)
        : poolId_(poolId)
        , framesSeen_(0)
    {
        metrics_ = std::move(metricsAccumulatorFactory_->NewBatch());
        for (size_t l = 0; l < lanesPerBatch_; ++l)
        {
            metrics_->GetHostView()[l].Initialize();
        }
    };
    HFMetricsFilter(const HFMetricsFilter&) = delete;
    HFMetricsFilter(HFMetricsFilter&&) = default;
    virtual ~HFMetricsFilter() = default;

public: // Filter API
    std::unique_ptr<BasecallingMetricsBatchT> operator()(
            const PulseBatchT& basecallBatch,
            const BaselineStatsT& baselineStats,
            const ModelsT& models)
    {
        assert(basecallBatch.GetMeta().PoolId() == poolId_);
        return Process(basecallBatch, baselineStats, models);
    }

protected:    // Block management
    virtual void FinalizeBlock() = 0;

    void AddPulses(const PulseBatchT& batch);

private:    // Block management
    virtual std::unique_ptr<BasecallingMetricsBatchT> Process(
            const PulseBatchT& basecallBatch,
            const BaselineStatsT& baselineStats,
            const ModelsT& models) = 0;

private:    // State
    uint32_t poolId_;

protected: // State
    uint32_t framesSeen_;
    std::unique_ptr<BasecallingMetricsBatchAccumulatorT> metrics_;

};

class HostHFMetricsFilter : public HFMetricsFilter
{
public:
    using HFMetricsFilter::HFMetricsFilter;
    ~HostHFMetricsFilter() override;

private:    // Block management
    void FinalizeBlock() override;

    void AddBaselineStats(const BaselineStatsT& baselineStats);

    void AddModels(const ModelsT& models);

    std::unique_ptr<BasecallingMetricsBatchT> Process(
            const PulseBatchT& basecallBatch,
            const BaselineStatsT& baselineStats,
            const ModelsT& models) override;

};


class NoHFMetricsFilter : public HFMetricsFilter
{
public:
    NoHFMetricsFilter(uint32_t poolId)
        : HFMetricsFilter(poolId)
    { };

    NoHFMetricsFilter(const NoHFMetricsFilter&) = delete;
    NoHFMetricsFilter(NoHFMetricsFilter&&) = default;
    ~NoHFMetricsFilter() override;

private:
    uint32_t poolId_;

private:    // Block management
    std::unique_ptr<BasecallingMetricsBatchT> Process(
            const PulseBatchT& basecallBatch,
            const BaselineStatsT& baselineStats,
            const ModelsT& models) override
    { return std::unique_ptr<BasecallingMetricsBatchT>(); };

    void FinalizeBlock() override
    { };
};

}}} // PacBio::Mongo::Basecaller

#endif // Mongo_BaseCaller_TraceAnalysis_HFMetricsFilter_H_
