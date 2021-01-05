#ifndef mongo_basecaller_traceAnalysis_TraceHistogramAccumHost_H_
#define mongo_basecaller_traceAnalysis_TraceHistogramAccumHost_H_

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
    using TraceElementType = Data::BaselinedTraceElement;

public:     // Structors and assignment.
    TraceHistogramAccumHost(unsigned int poolId,
                            unsigned int poolSize);

public:     // Const access (extensions to TraceHistogramAccumulator interface)
    const AlignedVector<Data::UHistogramSimd<LaneArray<HistDataType>, LaneArray<HistCountType>>>&
    HistogramHost() const
    { return hist_; }

private:    // TraceHistogramAccumulator implementation.
    void AddBatchImpl(const Data::TraceBatch<TraceElementType>& ctb) override;

    void ResetImpl(const Cuda::Memory::UnifiedCudaArray<LaneHistBounds>& bounds) override;

    PoolHistType HistogramImpl() const override;
private:    // Data
    AlignedVector<Data::UHistogramSimd<LaneArray<HistDataType>, LaneArray<HistCountType>>> hist_;

private:    // Functions
    // Add the frames of one trace block (i.e., lane-chunk) into the appropriate histogram.
    void AddBlock(const Data::TraceBatch<TraceElementType>& ctb, unsigned int lane);
};

}}}     // namespace PacBio::Mongo::Basecaller

#endif // mongo_basecaller_traceAnalysis_TraceHistogramAccumHost_H_
