#ifndef mongo_basecaller_traceAnalysis_BaselineStatsAggregator_H_
#define mongo_basecaller_traceAnalysis_BaselineStatsAggregator_H_

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
//  Defines abstract class BaselineStatsAggregator.

#include <common/IntInterval.h>
#include <dataTypes/BaselinerStatAccumState.h>
#include <dataTypes/BatchMetrics.h>
#include <dataTypes/configs/ConfigForward.h>

#include <basecaller/traceAnalysis/TraceHistogramAccumulator.h>

namespace PacBio {
namespace Mongo {
namespace Basecaller {

// Used to aggregate baseline statistics over multiple chunks and
// predict a range of future data suitable for use as histogram bounds
class BaselineStatsAggregator
{
public:     // Types
    using DataType = Data::BaselinedTraceElement;
    using FrameIntervalType = IntInterval<Data::FrameIndexType>;

public:
    BaselineStatsAggregator(uint32_t poolId, unsigned int poolSize);
    virtual ~BaselineStatsAggregator() = default;

    /// The ZMW pool associated with this instance.
    uint32_t PoolId() const
    { return poolId_; }

    /// The number of lanes in the pool associated with this instance.
    unsigned int PoolSize() const
    { return poolSize_; }

    FrameIntervalType FrameInterval() const
    { return frameInterval_; }

    /// Returns a copy of the accumulated trace statistics.
    Data::BaselinerMetrics TraceStats() const
    {
        auto result = TraceStatsImpl();
        result.frameInterval = frameInterval_;
        return result;
    }

    void AddMetrics(const Data::BaselinerMetrics& stats)
    {
        AddMetricsImpl(stats);

        // Assume that stats are received in contiguous frame intervals.
        assert(AreOrderedAdjacent(frameInterval_, stats.frameInterval));
        frameInterval_ = Hull(frameInterval_, stats.frameInterval);
    }

    void Reset()
    {
        ResetImpl();
        frameInterval_.Clear();
    }

private:
    virtual void AddMetricsImpl(const Data::BaselinerMetrics& stats) = 0;

    virtual Data::BaselinerMetrics TraceStatsImpl() const = 0;

    virtual void ResetImpl() = 0;

private:
    uint32_t poolId_;
    unsigned int poolSize_;  // Number of lanes in this pool.
    FrameIntervalType frameInterval_ {};
};

}}}     // namespace PacBio::Mongo::Basecaller

#endif // mongo_basecaller_traceAnalysis_BaselineStatsAggregator_H_
