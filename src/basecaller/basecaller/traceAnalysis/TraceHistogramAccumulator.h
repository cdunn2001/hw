#ifndef mongo_basecaller_traceAnalysis_TraceHistogramAccumulator_H_
#define mongo_basecaller_traceAnalysis_TraceHistogramAccumulator_H_

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
//  Defines abstract class TraceHistogramAccumulator.

#include <dataTypes/BaselineStats.h>
#include <dataTypes/CameraTraceBatch.h>
#include <dataTypes/ConfigForward.h>
#include <dataTypes/PoolHistogram.h>

namespace PacBio {
namespace Mongo {
namespace Basecaller {

/// A non-virtual interface for accumulation of trace histograms and statistics
/// produced by the baseline estimator.
class TraceHistogramAccumulator
{
public:     // Types
    using DataType = Data::CameraTraceBatch::ElementType;
    using HistDataType = float;
    using HistCountType = unsigned short;
    using PoolHistType = Data::PoolHistogram<HistDataType, HistCountType>;
    using PoolTraceStatsType = Cuda::Memory::UnifiedCudaArray<Data::BaselineStats<laneSize>>;

public:     // Static functions
    static void Configure(const Data::BasecallerTraceHistogramConfig& histConfig,
                          const Data::MovieConfig& movConfig);

    static unsigned int NumFramesPreAccumStats()
    { return numFramesPreAccumStats_; }

    static float BinSizeCoeff()
    { return binSizeCoeff_; }

    static unsigned int BaselineStatMinFrameCount()
    { return baselineStatMinFrameCount_; }

    static float FallBackBaselineSigma()
    { return fallBackBaselineSigma_; }

public:     // Structors and assignment
    TraceHistogramAccumulator(uint32_t poolId, unsigned int poolSize);

public:     // Const functions
    /// Total pool frames added.
    /// Total of counts in each zmw histogram will typically be somewhat
    /// smaller due to filtering of edge frames.
    size_t FramesAdded() const
    { return frameCount_; }

    // TODO: Maybe Histogram() and TraceStats() should return const &.

    /// The accumulated histogram.
    PoolHistType Histogram() const
    {
        return HistogramImpl();
    }

    /// The accumulated trace statistics.
    PoolTraceStatsType TraceStats() const
    {
        return TraceStatsImpl();
    }

    /// The ZMW pool associated with this instance.
    uint32_t PoolId() const
    { return poolId_; }

    /// The number of lanes in the pool associated with this instance.
    unsigned int PoolSize() const
    { return poolSize_; }

public:     // Non-const functions
    /// Adds data to histograms for a pool.
    /// May include filtering of edge frames.
    void AddBatch(const Data::CameraTraceBatch& ctb)
    {
        assert (ctb.GetMeta().PoolId() == poolId_);
        frameCount_ += ctb.numFrames();
        AddBatchImpl(ctb);
    }

private:    // Static data
    // Number of frames to accumulate baseliner statistics before initializing
    // histograms.
    static unsigned int numFramesPreAccumStats_;
    static float binSizeCoeff_;
    static unsigned int baselineStatMinFrameCount_;
    static float fallBackBaselineSigma_;

private:    // Data
    size_t frameCount_ = 0;
    uint32_t poolId_;
    unsigned int poolSize_;

private:    // Customizable implementation.
    // Bins frames in ctb and updates poolHist_.
    virtual void AddBatchImpl(const Data::CameraTraceBatch& ctb) = 0;

    virtual PoolHistType HistogramImpl() const = 0;

    virtual PoolTraceStatsType TraceStatsImpl() const = 0;
};

}}}     // namespace PacBio::Mongo::Basecaller

#endif // mongo_basecaller_traceAnalysis_TraceHistogramAccumulator_H_
