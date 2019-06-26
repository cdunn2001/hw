
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
//  Defines some members of class TraceHistogramAccumHost.


#include "TraceHistogramAccumHost.h"

namespace PacBio {
namespace Mongo {
namespace Basecaller {

TraceHistogramAccumHost::TraceHistogramAccumHost(unsigned int poolId, unsigned int poolSize)
    : TraceHistogramAccumulator(poolId, poolSize)
{
    // TODO
}


void TraceHistogramAccumHost::AddBatchImpl(const Data::CameraTraceBatch& ctb)
{
    // TODO: Can some of this logic be lifted into the base class's AddBatch
    // method (template method pattern)?

    const auto numLanes = ctb.Dimensions().lanesPerBatch;

    if (FramesAdded() == ctb.Dimensions().framesPerBatch)
    {
        // This is the first trace batch.

        // Reset all the histograms.
        hist_.clear();
        hist_.reserve(ctb.Dimensions().lanesPerBatch);
        isHistInitialized_ = false;

        // Reset baseline stat accumulators.
        InitStats(numLanes);
    }

    const bool doInitHist = !isHistInitialized_
            && FramesAdded() >= NumFramesPreAccumStats();

    // For each lane/block in the batch ...
    for (unsigned int i = 0; i < numLanes; ++i)
    {
        // Get views to the trace data and the statistics.
        const auto traceBlock = ctb.GetBlockView(i);
        const auto stats = ctb.Stats(i);

        // TODO: Accumulate baseliner stats.

        if (doInitHist)
        {
            // Define histogram parameters and construct empty histogram.
            InitHistogram(i);
        }

        // TODO: Map trace data to LaneArray and feed to UHistogramSimd.
    }
}


void TraceHistogramAccumHost::InitHistogram(unsigned int lane)
{
    assert (hist_.size() == lane);

    // Determine histogram parameters.

// Code from Sequel DmeMonochrome.
//    unsigned int nBins;
//    {
//        auto minSignal = dtbs[0]->MinSignal().first();
//        auto maxSignal = dtbs[0]->MaxSignal().first();
//        for (size_t i = 1; i < dtbs.size(); ++i)
//        {
//            minSignal = min(minSignal, dtbs[i]->MinSignal().first());
//            maxSignal = max(maxSignal, dtbs[i]->MaxSignal().first());
//        }

//        // Nominally define the bin size as a fraction of the baseline sigma.
//        const auto binSize = binSizeCoeff_ * bgSigma;

//        // Push lower bound down by a small amount to avoid any subtle boundary
//        // issues in histogram.
//        lowerBound = minSignal - 0.1f * bgSigma;

//        // Scale up just a bit to avoid boundary problems in histogram.
//        // Does not assume that data > 0.
//        const auto nudge = max(0.1f * binSize, 10.0f * abs(maxSignal) * numeric_limits<float>::epsilon());
//        upperBound = maxSignal + nudge;

//        nBins = round_cast<unsigned int>(reduceMax((upperBound - lowerBound) / binSize));
//        nBins = std::max(nBins, 20u);

//        // Cap the bins at 250 if necessary
//        if (nBins > binCap)
//        {
//            const auto newBinSize = (upperBound - lowerBound) / binCap;
//            nBins = binCap;

//            binClampCount_++;
//            if (any(newBinSize > bgSigma))
//                majorBinClampCount_++;
//            for (const auto& dtb : dtbs)
//            {
//                if (any(sqrt(dtb->BaselineCovar()[0]) < 1.0f))
//                {
//                    tinyBaselineSigmaCount_++;
//                }
//            }
//        }
//    }

//    UHistogramSimd<FloatVec> hist(nBins, lowerBound, upperBound);
//    ScrubEdgeFrames(dtbs, *workModel, &hist);
}


void TraceHistogramAccumHost::InitStats(unsigned int numLanes)
{
    stats_.clear();
    stats_.resize(numLanes);
}


TraceHistogramAccumHost::PoolHistType
TraceHistogramAccumHost::HistogramImpl() const
{
    // TODO
    return PoolHistType(PoolId(), PoolSize());
}

TraceHistogramAccumHost::PoolTraceStatsType
TraceHistogramAccumHost::TraceStatsImpl() const
{
    // TODO
    return PoolTraceStatsType(PoolSize(),
                              Cuda::Memory::SyncDirection::Symmetric);
}

}}}     // namespace PacBio::Mongo::Basecaller
