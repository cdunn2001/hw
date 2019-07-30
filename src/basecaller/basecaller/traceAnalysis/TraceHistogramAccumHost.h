#ifndef mongo_basecaller_traceAnalysis_TraceHistogramAccumHost_H_
#define mongo_basecaller_traceAnalysis_TraceHistogramAccumHost_H_

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
//  Defines class TraceHistogramAccumHost, which customizes
//  TraceHistogramAccumulator.


#include <common/AlignedVector.h>
#include <common/LaneArray.h>
#include <dataTypes/BaselinerStatAccumulator.h>
#include <dataTypes/UHistogramSimd.h>

#include "TraceHistogramAccumulator.h"

namespace PacBio {
namespace Mongo {
namespace Basecaller {

class TraceHistogramAccumHost : public TraceHistogramAccumulator
{
public:     // Types
    using TraceElementType = Data::CameraTraceBatch::ElementType;

public:     // Structors and assignment.
    TraceHistogramAccumHost(unsigned int poolId,
                            unsigned int poolSize,
                            bool pinnedAlloc = true);

private:    // TraceHistogramAccumulator implementation.
    void AddBatchImpl(const Data::CameraTraceBatch& ctb) override;

    const PoolHistType& HistogramImpl() const override;

    const PoolTraceStatsType& TraceStatsImpl() const override;

private:    // Data
    AlignedVector<Data::UHistogramSimd<LaneArray<HistDataType>, LaneArray<HistCountType>>> hist_;
    AlignedVector<Data::BaselinerStatAccumulator<DataType>> stats_;
    mutable PoolHistType poolHist_;
    mutable PoolTraceStatsType poolTraceStats_;
    bool isHistInitialized_ {false};

private:    // Functions
    // Compute histogram parameters (e.g., bin size, lower bound, etc.),
    // construct empty histogram, and add it to hist_.
    void InitHistogram(unsigned int lane);

    // Allocates, if necessary, and initializes all the baseliner statistics
    // accumulators contained in stats_.
    void InitStats(unsigned int numLanes);

    // Add the frames of one trace block (i.e., lane-chunk) into the appropriate histogram.
    void AddBlock(const Data::CameraTraceBatch& ctb, unsigned int lane);
};

}}}     // namespace PacBio::Mongo::Basecaller

#endif // mongo_basecaller_traceAnalysis_TraceHistogramAccumHost_H_
