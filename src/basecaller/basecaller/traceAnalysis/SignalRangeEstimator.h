#ifndef mongo_basecaller_traceAnalysis_SignalRangeEstimator_H_
#define mongo_basecaller_traceAnalysis_SignalRangeEstimator_H_

// Copyright (c) 2020-2021, Pacific Biosciences of California, Inc.
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
//  Defines abstract class SignalRangeEstimator.

#include <dataTypes/BaselinerStatAccumState.h>
#include <dataTypes/BatchMetrics.h>
#include <dataTypes/configs/ConfigForward.h>

#include <basecaller/traceAnalysis/TraceHistogramAccumulator.h>

namespace PacBio {
namespace Mongo {
namespace Basecaller {

// Used to aggregate baseline statistics over multiple chunks and
// predict a range of future data suitable for use as histogram bounds
class SignalRangeEstimator
{
public:     // Types
    using DataType = Data::BaselinedTraceElement;

public:
    static void Configure(const Data::BasecallerSignalRangeEstimatorConfig& sigConfig);

    static unsigned int NumFramesPreAccumStats()
    { return numFramesPreAccumStats_; }

    static float BinSizeCoeff()
    { return binSizeCoeff_; }

    static unsigned int BaselineStatMinFrameCount()
    { return baselineStatMinFrameCount_; }

    static float FallBackBaselineSigma()
    { return fallBackBaselineSigma_; }

public:
    SignalRangeEstimator(uint32_t poolId, unsigned int poolSize);
    virtual ~SignalRangeEstimator() = default;

    /// The ZMW pool associated with this instance.
    uint32_t PoolId() const
    { return poolId_; }

    /// The number of lanes in the pool associated with this instance.
    unsigned int PoolSize() const
    { return poolSize_; }

    /// Returns a copy of the accumulated trace statistics.
    Data::BaselinerMetrics TraceStats() const
    {
        return TraceStatsImpl();
    }

    void AddMetrics(const Data::BaselinerMetrics& stats)
    {
        AddMetricsImpl(stats);
    }

    Cuda::Memory::UnifiedCudaArray<LaneHistBounds> EstimateRangeAndReset()
    {
        return EstimateRangeAndResetImpl();
    }

private:    // Static data
    // Number of frames to accumulate baseliner statistics before initializing
    // histograms.
    static unsigned int numFramesPreAccumStats_;
    static float binSizeCoeff_;
    static unsigned int baselineStatMinFrameCount_;
    static float fallBackBaselineSigma_;

private:
    virtual void AddMetricsImpl(const Data::BaselinerMetrics& stats) = 0;

    virtual Data::BaselinerMetrics TraceStatsImpl() const = 0;

    virtual Cuda::Memory::UnifiedCudaArray<LaneHistBounds> EstimateRangeAndResetImpl() = 0;

private:
    uint32_t poolId_;
    unsigned int poolSize_;  // Number of lanes in this pool.
};

}}}     // namespace PacBio::Mongo::Basecaller

#endif // mongo_basecaller_traceAnalysis_SignalRangeEstimator_H_
