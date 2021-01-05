#ifndef mongo_basecaller_traceAnalysis_TraceHistogramAccumulator_H_
#define mongo_basecaller_traceAnalysis_TraceHistogramAccumulator_H_

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
//  Defines abstract class TraceHistogramAccumulator.

#include <dataTypes/BasicTypes.h>
#include <dataTypes/configs/ConfigForward.h>
#include <dataTypes/PoolHistogram.h>
#include <dataTypes/TraceBatch.h>

namespace PacBio {
namespace Mongo {
namespace Basecaller {

struct alignas(64) LaneHistBounds
{
    Cuda::Utility::CudaArray<float, laneSize> upperBounds;
    Cuda::Utility::CudaArray<float, laneSize> lowerBounds;
};

// Handles the aggregation of trace data into histograms.  This class
// is not robustly initialized upon construction, and need to be supplied
// with histogram bounds before aggregating any data
class TraceHistogramAccumulator
{
public:     // Types
    using DataType = Data::BaselinedTraceElement;
    using HistDataType = float;
    using HistCountType = unsigned short;
    using LaneHistType = Data::LaneHistogram<HistDataType, HistCountType>;
    using PoolHistType = Data::PoolHistogram<HistDataType, HistCountType>;

public:     // Structors and assignment
    TraceHistogramAccumulator(uint32_t poolId, unsigned int poolSize);
    virtual ~TraceHistogramAccumulator() = default;

public:     // Const functions
    /// Total pool frames added via AddBatch.
    size_t FramesAdded() const
    { return frameCount_; }

    /// Returns a copy of the accumulated trace histogram
    PoolHistType Histogram() const
    {
        return HistogramImpl();
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
    void AddBatch(const Data::TraceBatch<DataType>& traces)
    {
        if (!initialized_)
            throw PBException("Cannot aggregate trace data into histograms "
                              "before calling Reset with the desired histogram bounds");
        assert (traces.GetMeta().PoolId() == poolId_);
        AddBatchImpl(traces);
        frameCount_ += traces.NumFrames();
    }

    // Clears out the bins and resets the histogram bounds
    void Reset(Cuda::Memory::UnifiedCudaArray<LaneHistBounds> bounds)
    {
        initialized_ = true;
        frameCount_ = 0;
        ResetImpl(bounds);
    }

private:    // Data
    size_t frameCount_ = 0;  // Total number of frames added via AddBatch.
    uint32_t poolId_;
    unsigned int poolSize_;  // Number of lanes in this pool.
    bool initialized_ = false; // Do we have histogram bounds yet?

private:    // Customizable implementation.
    // Bins frames in traces and updates poolHist_.
    virtual void AddBatchImpl(const Data::TraceBatch<DataType>& traces) = 0;

    // Clears out current histogram data and resets histogram bounds
    virtual void ResetImpl(const Cuda::Memory::UnifiedCudaArray<LaneHistBounds>& bounds) = 0;

    virtual PoolHistType HistogramImpl() const = 0;
};

}}}     // namespace PacBio::Mongo::Basecaller

#endif // mongo_basecaller_traceAnalysis_TraceHistogramAccumulator_H_
