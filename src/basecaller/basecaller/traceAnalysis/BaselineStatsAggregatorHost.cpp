// Copyright (c) 2020-2021 Pacific Biosciences of California, Inc.
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
//  Defines some members of class BaselineStatsAggregatorHost.

#include "BaselineStatsAggregatorHost.h"

#include <algorithm>

#include <tbb/task_arena.h>
#include <tbb/parallel_for.h>

namespace PacBio {
namespace Mongo {
namespace Basecaller {

void BaselineStatsAggregatorHost::AddMetricsImpl(const Data::BaselinerMetrics& metrics)
{
    assert(stats_.size() == metrics.baselinerStats.Size());

    const auto& statsView = metrics.baselinerStats.GetHostView();
    tbb::task_arena().execute([&] {
        tbb::parallel_for((size_t) {0}, stats_.size(), [&](size_t lane) {
            // Accumulate baseliner stats.
            stats_[lane].Merge(Data::BaselinerStatAccumulator<Data::RawTraceElement>(statsView[lane]));
        });
    });
}

Data::BaselinerMetrics BaselineStatsAggregatorHost::TraceStatsImpl() const
{
    Data::BaselinerMetrics poolTraceStats(PoolSize(),
                                          Cuda::Memory::SyncDirection::HostWriteDeviceRead,
                                          SOURCE_MARKER());
    auto ptsv = poolTraceStats.baselinerStats.GetHostView();
    for (unsigned int lane = 0; lane < PoolSize(); ++lane)
    {
        ptsv[lane] = stats_[lane].GetState();
    }
    return poolTraceStats;
}

void BaselineStatsAggregatorHost::ResetImpl()
{
    // Reset for the next aggregation
    stats_.clear();
    stats_.resize(PoolSize());
}

}}}     // namespace PacBio::Mongo::Basecaller
