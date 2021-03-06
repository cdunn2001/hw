#ifndef mongo_basecaller_traceAnalysis_BaselineStatsAggregatorHost_H_
#define mongo_basecaller_traceAnalysis_BaselineStatsAggregatorHost_H_

// Copyright (c) 2019-2021, Pacific Biosciences of California, Inc.
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
//  Defines class BaselineStatsAggregatorHost, which customizes
//  BaselineStatsAggregator.


#include <common/AlignedVector.h>
#include <common/LaneArray.h>
#include <dataTypes/BaselinerStatAccumulator.h>
#include <dataTypes/UHistogramSimd.h>

#include "BaselineStatsAggregator.h"

namespace PacBio {
namespace Mongo {
namespace Basecaller {

class BaselineStatsAggregatorHost : public BaselineStatsAggregator
{
public:
    BaselineStatsAggregatorHost(uint32_t poolId, unsigned int poolSize)
        : BaselineStatsAggregator(poolId, poolSize)
        , stats_(poolSize)
    {}

public:     // Const access (extensions to BaselineStatsAggregatorHost interface)

    const AlignedVector<Data::BaselinerStatAccumulator<DataType>>&
    TraceStatsHost() const
    { return stats_; }

private:    // BaselineStatsAggregatorHost implementation.
    void AddMetricsImpl(const Data::BaselinerMetrics& metrics) override;

    Data::BaselinerMetrics TraceStatsImpl() const override;

    void ResetImpl() override;

private:    // Data
    AlignedVector<Data::BaselinerStatAccumulator<DataType>> stats_;
};

}}}     // namespace PacBio::Mongo::Basecaller

#endif // mongo_basecaller_traceAnalysis_BaselineStatsAggregatorHost_H_
