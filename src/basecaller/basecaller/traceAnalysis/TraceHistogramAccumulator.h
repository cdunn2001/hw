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
    using LaneHistType = Data::LaneHistogram<HistDataType, HistCountType>;
    using PoolHistType = Data::PoolHistogram<HistDataType, HistCountType>;
    using PoolTraceStatsType = Cuda::Memory::UnifiedCudaArray<Data::BaselinerStatAccumState>;
    // TODO: Switch from BaselineStats to BaselinerStatAccumState.

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
    /// Total pool frames added via AddBatch.
    size_t FramesAdded() const
    { return frameCount_; }

    /// Number of frames added to histograms.
    /// Total of counts in each zmw histogram will typically be somewhat
    /// smaller due to filtering of edge frames.
    size_t HistogramFrameCount() const
    { return histFrameCount_; }

    /// The accumulated histogram.
    /// \note Calls to AddBatch will modify the referenced value.
    /// \note Calling this function is not necessarily cheap.
    const PoolHistType& Histogram() const
    {
        return HistogramImpl();
    }

    /// The accumulated trace statistics.
    const PoolTraceStatsType& TraceStats() const
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
        AddBatchImpl(ctb);
        frameCount_ += ctb.NumFrames();
    }

protected:  // Data
    // Number of frames added to histograms.
    size_t histFrameCount_ = 0;

private:    // Static data
    // Number of frames to accumulate baseliner statistics before initializing
    // histograms.
    static unsigned int numFramesPreAccumStats_;
    static float binSizeCoeff_;
    static unsigned int baselineStatMinFrameCount_;
    static float fallBackBaselineSigma_;

private:    // Data
    size_t frameCount_ = 0;  // Total number of frames added via AddBatch.
    uint32_t poolId_;
    unsigned int poolSize_;  // Number of lanes in this pool.

private:    // Customizable implementation.
    // Bins frames in ctb and updates poolHist_.
    virtual void AddBatchImpl(const Data::CameraTraceBatch& ctb) = 0;

    virtual const PoolHistType& HistogramImpl() const = 0;

    virtual const PoolTraceStatsType& TraceStatsImpl() const = 0;
};

}}}     // namespace PacBio::Mongo::Basecaller

#endif // mongo_basecaller_traceAnalysis_TraceHistogramAccumulator_H_
