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
#include <dataTypes/BasecallBatch.h>
#include <dataTypes/BasecallerConfig.h>
#include <dataTypes/BasecallingMetrics.h>

namespace PacBio {
namespace Mongo {
namespace Basecaller {

class HFMetricsFilter
{
public:     // Types, static constants
    using ElementTypeIn = Data::BasecallBatch;
    using ElementTypeOut = Cuda::Memory::UnifiedCudaArray<Data::BasecallingMetrics<laneSize>>;

public: // Static functions
    static void Configure(const Data::BasecallerMetricsConfig&);
    static void Finalize();
    static void InitAllocationPools(bool hostExecution);
    static void DestroyAllocationPools();

protected: // Static members
    static std::unique_ptr<Data::BasecallingMetricsFactory<laneSize>> metricsFactory_;
    static uint32_t sandwichTolerance_;
    static uint32_t framesPerHFMetricBlock_;
    static double frameRate_;
    static bool realtimeActivityLabels_;
    static uint32_t zmwsPerBatch_;


public: // Structors
    HFMetricsFilter(uint32_t poolId);
    HFMetricsFilter(const HFMetricsFilter&) = delete;
    HFMetricsFilter(HFMetricsFilter&&) = default;
    virtual ~HFMetricsFilter() = default;

public: // Filter API
    std::unique_ptr<ElementTypeOut> operator()(const ElementTypeIn& batch)
    {
        assert(batch.GetMeta().PoolId() == poolId_);
        return Process(batch);
    }

private:    // Block management
    virtual void FinalizeBlock() = 0;

private:    // Block management
    virtual std::unique_ptr<ElementTypeOut> Process(const ElementTypeIn& batch) = 0;

private:    // State
    uint32_t poolId_;

protected: // State
    uint32_t framesSeen_;
    std::unique_ptr<ElementTypeOut> metrics_;

};

class HostHFMetricsFilter : public HFMetricsFilter
{
public:

    using HFMetricsFilter::HFMetricsFilter;

    ~HostHFMetricsFilter() override;

    void FinalizeBlock() override;

private:    // Block management
    void AddBatch(const ElementTypeIn& batch);

    std::unique_ptr<ElementTypeOut> Process(const ElementTypeIn& batch) override;

};

/*
class MinimalHFMetricsFilter
{
private: // static
    static std::unique_ptr<Data::BasecallingMetricsFactory<laneSize>> metricsFactory_;
    static uint32_t sandwichTolerance_;
    static uint32_t framesPerHFMetricBlock_;
    static double frameRate_;
    static bool realtimeActivityLabels_;
    static uint32_t zmwsPerBatch_;

public:     // Types, static constants
    using ElementTypeIn = Data::BasecallBatch;
    using ElementTypeOut = Cuda::Memory::UnifiedCudaArray<Data::MinimalBasecallingMetrics>;

public:

    MinimalHFMetricsFilter(uint32_t poolId);
    MinimalHFMetricsFilter(const MinimalHFMetricsFilter&) = delete;
    MinimalHFMetricsFilter(MinimalHFMetricsFilter&&) = default;
    ~MinimalHFMetricsFilter() = default;

private:
    uint32_t poolId_;
    uint32_t framesSeen_;
    std::unique_ptr<ElementTypeOut> metrics_;

private:    // Block management
    void AddBatch(const ElementTypeIn& batch);

    std::unique_ptr<ElementTypeOut> Process(const ElementTypeIn& batch);
};

*/

}}} // PacBio::Mongo::Basecaller

#endif // Mongo_BaseCaller_TraceAnalysis_HFMetricsFilter_H_
