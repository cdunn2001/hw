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

#include <common/MongoConstants.h>
#include <dataTypes/PulseBatch.h>
#include <dataTypes/BasecallingMetrics.h>
#include <dataTypes/BaselinerStatAccumState.h>
#include <dataTypes/LaneDetectionModel.h>


namespace PacBio {
namespace Mongo {
namespace Basecaller {

class HFMetricsFilter
{
public:     // Types
    using PulseBatchT = Data::PulseBatch;
    using BaselinerStatsT = Cuda::Memory::UnifiedCudaArray<Data::BaselinerStatAccumState>;
    using BasecallingMetricsT = Data::BasecallingMetrics<laneSize>;
    using BasecallingMetricsBatchT = Cuda::Memory::UnifiedCudaArray<BasecallingMetricsT>;
    using BasecallingMetricsAccumulatorT = Data::BasecallingMetricsAccumulator<laneSize>;
    using BasecallingMetricsAccumulatorBatchT = std::array<BasecallingMetricsAccumulatorT, laneSize>;
    using ModelsT = Cuda::Memory::UnifiedCudaArray<Data::LaneModelParameters<Cuda::PBHalf, laneSize>>;

public: // Static functions
    static void Configure(uint32_t sandwichTolerance,
                          uint32_t framesPerHFMetricBlock,
                          uint32_t framesPerChunk,
                          double frameRate,
                          bool realtimeActivityLabels,
                          uint32_t lanesPerBatch);
    static void Finalize();
    static void InitAllocationPools(bool hostExecution);
    static void DestroyAllocationPools();

protected: // Static members
    static std::unique_ptr<Data::BasecallingMetricsFactory<laneSize>> metricsFactory_;
    static uint32_t sandwichTolerance_;
    static uint32_t framesPerHFMetricBlock_;
    static double frameRate_;
    static bool realtimeActivityLabels_;
    static uint32_t lanesPerBatch_;
    static uint32_t zmwsPerBatch_;
    static uint32_t framesPerChunk_;


public: // Structors
    HFMetricsFilter(uint32_t poolId)
        : poolId_(poolId)
        , framesSeen_(0)
    { };
    HFMetricsFilter(const HFMetricsFilter&) = delete;
    HFMetricsFilter(HFMetricsFilter&&) = default;
    HFMetricsFilter& operator=(const HFMetricsFilter&) = delete;
    HFMetricsFilter& operator=(HFMetricsFilter&&) = default;
    virtual ~HFMetricsFilter() = default;

public: // Filter API
    std::unique_ptr<BasecallingMetricsBatchT> operator()(
            const PulseBatchT& basecallBatch,
            const BaselinerStatsT& baselinerStats,
            const ModelsT& models)
    {
        assert(basecallBatch.GetMeta().PoolId() == poolId_);
        return Process(basecallBatch, baselinerStats, models);
    }

private:
    virtual std::unique_ptr<BasecallingMetricsBatchT> Process(
            const PulseBatchT& pulses,
            const BaselinerStatsT& baselinerStats,
            const ModelsT& models) = 0;

protected:    // Block management
    virtual void FinalizeBlock() = 0;

protected: // State
    uint32_t poolId_;

protected: // State
    uint32_t framesSeen_;

};

class HostHFMetricsFilter : public HFMetricsFilter
{
public:
    HostHFMetricsFilter(uint32_t poolId)
        : HFMetricsFilter(poolId)
    { };
    HostHFMetricsFilter(const HostHFMetricsFilter&) = delete;
    HostHFMetricsFilter(HostHFMetricsFilter&&) = default;
    HostHFMetricsFilter& operator=(const HostHFMetricsFilter&) = delete;
    HostHFMetricsFilter& operator=(HostHFMetricsFilter&&) = default;
    ~HostHFMetricsFilter() override;

private:
    std::unique_ptr<BasecallingMetricsBatchT> Process(
            const PulseBatchT& pulseBatch,
            const BaselinerStatsT& baselineStats,
            const ModelsT& models) override;

private: // Block management
    void FinalizeBlock() override;

    void AddPulses(const PulseBatchT& batch);

    void AddBaselinerStats(const BaselinerStatsT& baselinerStats);

    void AddModels(const ModelsT& models);

private: // members
    BasecallingMetricsAccumulatorBatchT metrics_;
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
    std::unique_ptr<BasecallingMetricsBatchT> Process(
            const PulseBatchT& pulseBatch,
            const BaselinerStatsT& baselinerStats,
            const ModelsT& models) override
    { return std::unique_ptr<BasecallingMetricsBatchT>(); };

private:    // Block management
    void FinalizeBlock() override
    { };
};

}}} // PacBio::Mongo::Basecaller

#endif // Mongo_BaseCaller_TraceAnalysis_HFMetricsFilter_H_
