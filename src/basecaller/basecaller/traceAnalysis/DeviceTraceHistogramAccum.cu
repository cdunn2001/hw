// Copyright (c) 2021, Pacific Biosciences of California, Inc.
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

#include <cuda_runtime.h>

#include <cstdint>

#include <basecaller/traceAnalysis/DeviceTraceHistogramAccum.h>

#include <common/cuda/PBCudaSimd.cuh>
#include <common/cuda/memory/DeviceOnlyArray.cuh>
#include <common/cuda/memory/UnifiedCudaArray.h>
#include <common/cuda/streams/LaunchManager.cuh>

#include <dataTypes/BatchData.cuh>
#include <dataTypes/configs/BasecallerTraceHistogramConfig.h>

using namespace PacBio::Cuda::Memory;
using namespace PacBio::Cuda::Utility;
using namespace PacBio::Cuda;
using namespace PacBio::Mongo::Data;

namespace PacBio {
namespace Mongo {
namespace Basecaller {

namespace {

// static/global data is annoying to get to the GPU.  Bundle it in a
// struct so we can have a single transfer.
struct StaticConfig
{
    float binSizeCoeff_;
    unsigned int baselineStatMinFrameCount_;
    float fallBackBaselineSigma_;
};

__constant__ StaticConfig staticConfig;

using LaneModel = Data::LaneModelParameters<PBHalf2, laneSize/2>;

// A specialized implementation of a length 2 circular buffer,
// for use in keeping track of if the previous two frames were
// baseline or not.  This is very heavily tied to our use of
// PBShort2 and certain cuda intrinsics, it is not meant for
// general use.
class BaselineHistory
{
public:
    __device__ PBShort2 PeakBack() const
    {
        // Grabs the back value of our "circular buffer".
        // 0x3322 means that our low two bytes will be a replication
        // of the third byte of history_, and the high two bytes
        // will be a replication of the fourth byte of history_.
        // Since bytes in history_ are either 0xFF or 0x00,
        // the replication turns things into the true and false values
        // for PBShort2.
        return PBShort2::FromRaw(__byte_perm(history_, 0, 0x3322));
    }

    __device__ void PushFront(PBShort2 isBaseline)
    {
        // Now do a "push front" to our "circular buffer".
        // 0x1064 means that the low two bytes from history_
        // are moved to the high two bytes, and the new low two
        // bytes in history_ are first and third byte from
        // `isBaseline`.  Again since true and false for PBShort2
        // are 0xFFFF and 0x0000 respectively, grabbing the first
        // and third byte is the same as grabbing the second and
        // fourth, and either gives us complete information.
        history_ = __byte_perm(history_, isBaseline.data(), 0x1064);
    }

private:
    // history_ is an array of bytes, where 0x00 indicates false and
    // 0xFF indicates true.  The low two bytes represent whether the
    // previous frame was baseline for a pair of ZMW, and the high
    // two bytes indicate the same for the frame prior to that
    uint32_t history_ = 0;
};

// Raw state data for a whole lane of EdgeFinder instances,
// for preserving state data between chunks.
struct EdgeScrubbingState
{
    static constexpr uint32_t Len = laneSize/2;

    __device__ EdgeScrubbingState()
    {
        for (uint32_t i = 0; i < Len; ++i)
        {
            prev[i] = PBShort2{0};
        }
    }

    Cuda::Utility::CudaArray<PBShort2, Len> prev;
    Cuda::Utility::CudaArray<BaselineHistory, Len> baselineHistory;
};

class EdgeFinder
{
    __device__ PBShort2 Computethreshold(const LaneModel& laneModel,
                                         uint32_t idx)
    {
        // The threshold below which the sample is most likely full-frame baseline.
        // TODO: This value should be configurable, depend on the SNR of
        // the dimmest pulse component of the detection model, or both.
        static constexpr float threshSigma = 2.0f;

        const auto& laneBg = laneModel.BaselineMode();
        return ToShort(threshSigma * sqrt(laneBg.vars[idx]) + laneBg.means[idx]);
    }

public:
    // Constructs by loading data from an EdgeScrubbingState
    __device__ EdgeFinder(const EdgeScrubbingState& l,
                          const LaneModel& laneModel,
                          uint32_t idx)
        : prev_{l.prev[idx]}
        , baselineHistory_{l.baselineHistory[idx]}
    {
        threshold_ = Computethreshold(laneModel, idx);
    }

    // Constructs by using two explicit data inputs to prime
    // the EdgeFinder
    __device__ EdgeFinder(PBShort2 frame1, PBShort2 frame2,
                          const LaneModel& laneModel,
                          uint32_t idx = threadIdx.x)
    {
        threshold_ = Computethreshold(laneModel, idx);

        // Intentional discard of return values, we just want to prime
        // the state data
        IsEdgeFrame(frame1);
        IsEdgeFrame(frame2);
    }

    // Stores the current data back into an EdgeScrubbingState
    __device__ void Store(EdgeScrubbingState& l,
                          uint32_t idx = threadIdx.x)
    {
        l.prev[idx] = prev_;
        l.baselineHistory[idx] = baselineHistory_;
    }

    // Used to classify edge frames.  The return is the previously
    // added frame, along with a PBShort2 indicating if that was an
    // edge frame or not.
    __device__ std::pair<PBShort2, PBShort2> IsEdgeFrame(PBShort2 frame)
    {
        const auto isBaseline = (frame < threshold_);
        const auto edge = baselineHistory_.PeakBack() ^ isBaseline;
        baselineHistory_.PushFront(isBaseline);

        auto candidate = prev_;
        prev_ = frame;
        return {edge, candidate};
    }
private:
    PBShort2 prev_;
    BaselineHistory baselineHistory_;
    PBShort2 threshold_;
};

}

// This is essentially just a copy of LaneHistogram with the storage
// order of binCount swapped.  Could conceivably unify the two with
// a little template work, but it's probably not worth doing.  This
// is only used in implementations that are not the fastest, and are
// only kept around for reference, to be kept around as a reminder
// of what has been tried, and for re-evaluation if we get new hardware
// with different characteristics.
struct LaneHistogramTrans
{
    using DataT = float;
    using CountT = uint16_t;
    template <typename T>
    using Array = Cuda::Utility::CudaArray<T, laneSize>;

    using DataType = DataT;
    using CountType = CountT;

    // This constant must be large enough to accomodate high SNR data.
    // Ideally, it would be a function of BinSizeCoeff and the SNR and excess-
    // noise CV of the brightest analog.
    // nBins(snr, bsc, xsn) = (snr + 4(1 + sqrt(1 + snr + xsn*snr^2))) / bsc,
    // where bsc = binSizeCoeff, and xsn = excess noise CV.
    // In practice, the value will probably be determined somewhat empirically.
    static constexpr unsigned int numBins = Data::LaneHistogram<float, uint16_t>::numBins;

    /// The lower bound of the lowest bin.
    Array<DataT> lowBound;

    /// The size of all bins for each ZMW.
    Array<DataT> binSize;

    /// The number of data less than lowBound.
    Array<CountT> outlierCountLow;

    /// The number of data >= the high bound = lowBound + numBins*binSize.
    Array<CountT> outlierCountHigh;

    /// The number of data in each bin.
    Array<CudaArray<CountT, numBins>> binCount;
};

// Simple initial attempt, with data kept in global memory and the histogram
// laid out in the usual fashion where each bin has all zmw stored
// contiguously (e.g. the histograms for each zmw are interleaved)
__global__ void BinningGlobalInterleaved(Data::GpuBatchData<const PBShort2> traces,
                                         DeviceView<Data::LaneHistogram<float, uint16_t>> hists,
                                         DeviceView<const LaneModel> models,
                                         DeviceView<EdgeScrubbingState> edgeState,
                                         bool initEdgeDetection)
{
    assert(blockDim.x == 64);
    auto& hist = hists[blockIdx.x];

    auto zmw = traces.ZmwData(blockIdx.x, threadIdx.x/2);

    // I'm going to be lazy and have two threads do the same edge finding.
    // Scrubbing is so cheap compared to the rest of this filter
    // that I don't care
    const auto start = initEdgeDetection ? 2 : 0;
    const auto stop = traces.NumFrames();
    auto edgeFinder = (start == 0)
        ? EdgeFinder(edgeState[blockIdx.x], models[blockIdx.x], threadIdx.x/2)
        : EdgeFinder(zmw[0], zmw[1], models[blockIdx.x], threadIdx.x/2);

    float lowBound = hist.lowBound[threadIdx.x];
    float binSize = hist.binSize[threadIdx.x];
    ushort lowOutlier = hist.outlierCountLow[threadIdx.x];
    ushort highOutlier = hist.outlierCountHigh[threadIdx.x];

    constexpr int16_t numBins = LaneHistogramTrans::numBins;

    for (int i = start; i < stop; ++i)
    {
        auto [isEdge, frame] = edgeFinder.IsEdgeFrame(zmw[i]);
        bool zmwEdge = threadIdx.x % 2 == 0 ? isEdge.X() : isEdge.Y();
        if (zmwEdge) continue;

        // We're doing one thread per zmw, which means we have to do
        // a little dance here since traces automatically come over as
        // a paired PBShort2
        auto val = (threadIdx.x % 2 == 0 ? frame.X() : frame.Y());
        int bin = (val - lowBound) / binSize;
        if (bin < 0) lowOutlier++;
        else if (bin >= numBins) highOutlier++;
        else hist.binCount[bin][threadIdx.x]++;
    }

    hist.outlierCountLow[threadIdx.x] = lowOutlier;
    hist.outlierCountHigh[threadIdx.x] = highOutlier;
    if (threadIdx.x % 2 == 0)
        edgeFinder.Store(edgeState[blockIdx.x], threadIdx.x/2);
}


// Switches storage order of the histograms, so that all data for a given ZMW is contiguous.
// The different access pattern will interact with the caches differently.
//
// Notes: I had been hoping that un-coallesced reads would still trigger a whole cacheline
//        load into L2 (and maybe even L1).  Thus by having bins contiguous I was hoping I'd
//        have better cache hits since the series is generally time correlated.
//
//        That said, things run slower.  Cases of 1 signal are drastically slower, which makes
//        sense as we gave up very good warp convergence.  1 signal of course is not a good
//        and representative case.  Even a mix of different signals however shows a performance
//        degredation.  Eithegiving up the minor read-coalescing we randomly get isn't worth it,
//        or the L2 doesn't work as I'd hoped.
//
//        Need to profile more and do some reasearch, to test the many assumptions in the original
//        theory
__global__ void BinningGlobalContig(Data::GpuBatchData<const PBShort2> traces,
                                    DeviceView<LaneHistogramTrans> hists,
                                    DeviceView<const LaneModel> models,
                                    DeviceView<EdgeScrubbingState> edgeState,
                                    bool initEdgeDetection)
{
    auto& hist = hists[blockIdx.x];
    auto zmw = traces.ZmwData(blockIdx.x, threadIdx.x/2);

    // I'm going to be lazy and have two threads do the same edge finding.
    // Scrubbing is so cheap compared to the rest of this filter
    // that I don't care
    const auto start = initEdgeDetection ? 2 : 0;
    const auto stop = traces.NumFrames();
    auto edgeFinder = (start == 0)
        ? EdgeFinder(edgeState[blockIdx.x], models[blockIdx.x], threadIdx.x/2)
        : EdgeFinder(zmw[0], zmw[1], models[blockIdx.x], threadIdx.x/2);

    float lowBound = hist.lowBound[threadIdx.x];
    float binSize = hist.binSize[threadIdx.x];
    ushort lowOutlier = hist.outlierCountLow[threadIdx.x];
    ushort highOutlier = hist.outlierCountHigh[threadIdx.x];

    constexpr int16_t numBins = LaneHistogramTrans::numBins;

    for (int i = start; i < stop; ++i)
    {
        auto [isEdge, frame] = edgeFinder.IsEdgeFrame(zmw[i]);
        bool zmwEdge = threadIdx.x % 2 == 0 ? isEdge.X() : isEdge.Y();
        if (zmwEdge) continue;

        auto val = (threadIdx.x % 2 == 0 ? frame.X() : frame.Y());
        int bin = (val - lowBound) / binSize;
        if (bin < 0) lowOutlier++;
        else if (bin >= numBins) highOutlier++;
        else hist.binCount[threadIdx.x][bin]++;
    }

    hist.outlierCountLow[threadIdx.x] = lowOutlier;
    hist.outlierCountHigh[threadIdx.x] = highOutlier;
    if (threadIdx.x % 2 == 0)
        edgeFinder.Store(edgeState[blockIdx.x], threadIdx.x/2);
}

// Still doing contiguous historgrams, but this time having the whole warp work on one ZMW at a time.
// The intent is to lower the footprint of our hot memory at any one point in time, to minimize
// cache misses.
//
// Notes: This function isn't too hard to tweak so that multiple warps participate, each warp working
//        on a disjoint subset of zmw.
//
//        There is some slight evidece that having 2-4 warps active in the same block have a marginal
//        benefit in some cases and a marginal detrement in others.  Having too many warps definitely
//        does hurt performance, presumably because we're back to having a large enough portion of the
//        histogram being accessed, causing more cache trashing.
__global__ void BinningGlobalContigCoopWarps(Data::GpuBatchData<const PBShort2> traces,
                                             DeviceView<LaneHistogramTrans> hists,
                                             DeviceView<const LaneModel> models,
                                             DeviceView<EdgeScrubbingState> edgeState,
                                             bool initEdgeDetection)
{
    // Some magic values for certain cases
    constexpr int highOutlier = LaneHistogramTrans::numBins;
    constexpr int lowOutlier = -1;
    constexpr int scrubbedFrame = -2;

    assert(traces.NumFrames() % blockDim.x == 0);
    auto& hist = hists[blockIdx.x];

    constexpr int16_t numBins = LaneHistogramTrans::numBins;

    auto threadFrames = traces.NumFrames() / blockDim.x;
    auto start = threadIdx.x * threadFrames;
    auto stop = start + threadFrames;

    for (int zmw = 0; zmw < laneSize/2; ++zmw)
    {
        float2 lowBound = {hist.lowBound[2*zmw], hist.lowBound[2*zmw+1]};
        float2 binSize = {hist.binSize[2*zmw], hist.binSize[2*zmw+1]};

        auto dat = traces.ZmwData(blockIdx.x, zmw);
        auto edgeFinder = (start == 0)
            ? EdgeFinder(edgeState[blockIdx.x], models[blockIdx.x], zmw)
            : EdgeFinder(dat[start-2], dat[start-1], models[blockIdx.x], zmw);

        for (int i = start; i < stop; ++i)
        {
            auto [isEdge, val] = edgeFinder.IsEdgeFrame(dat[i]);
            isEdge = isEdge || (initEdgeDetection && i < 2);

            int bin = (val.X() - lowBound.x) / binSize.x;
            if (bin < 0) bin = lowOutlier;
            else if (bin > numBins) bin = highOutlier;
            if (isEdge.X()) bin = scrubbedFrame;

            // Get bit flag with each thread that has the same bin as us.
            auto same = __match_any_sync(0xFFFFFFFF, bin);
            // Number of threads with the same bin
            auto count = __popc(same);
            // Thread with the most significant bit gets to own the update
            bool owner = (32 - __clz(same) -1 ) == threadIdx.x;

            // Intentional skip over any edge frame counts
            if (owner && bin != scrubbedFrame)
            {
                if (bin  == lowOutlier) hist.outlierCountLow[2*zmw] += count;
                else if (bin == highOutlier) hist.outlierCountHigh[2*zmw] += count;
                else hist.binCount[2*zmw][bin] += count;
            }

            bin = (val.Y() - lowBound.y) / binSize.y;
            if (bin < 0) bin = lowOutlier;
            else if (bin > numBins) bin = highOutlier;
            if (isEdge.Y()) bin = scrubbedFrame;

            // Get bit flag with each thread that has the same bin as us.
            same = __match_any_sync(0xFFFFFFFF, bin);
            // Number of threads with the same bin
            count = __popc(same);
            // Thread with the most significant bit gets to own the update
            owner = (32 - __clz(same) -1 ) == threadIdx.x;

            // Intentional skip over any edge frame counts
            if (owner && bin != scrubbedFrame)
            {
                if (bin == lowOutlier) hist.outlierCountLow[2*zmw+1] +=count;
                else if (bin == highOutlier) hist.outlierCountHigh[2*zmw+1] += count;
                else hist.binCount[2*zmw+1][bin] += count;
            }
        }

        if (threadIdx.x == blockDim.x-1)
            edgeFinder.Store(edgeState[blockIdx.x], zmw);
    }
}


// Same strategy as the last attempt, just moving the active histogram to shared memory
// for the faster data access speeds
__global__ void BinningSharedContigCoopWarps(Data::GpuBatchData<const PBShort2> traces,
                                             DeviceView<LaneHistogramTrans> hists,
                                             DeviceView<const LaneModel> models,
                                             DeviceView<EdgeScrubbingState> edgeState,
                                             bool initEdgeDetection)
{
    constexpr int16_t numBins = LaneHistogramTrans::numBins;
    constexpr int16_t lowOutlier = numBins;
    constexpr int16_t highOutlier = numBins+1;
    constexpr int16_t scrubbedFrame = numBins+2;
    __shared__ uint16_t localHist[numBins+3][2];

    assert(traces.NumFrames() % blockDim.x == 0);
    auto& hist = hists[blockIdx.x];

    auto threadFrames = traces.NumFrames() / blockDim.x;
    auto start = threadIdx.x * threadFrames;
    auto stop = start + threadFrames;

    for (int zmw = 0; zmw < laneSize/2; ++zmw)
    {
        float2 lowBound = {hist.lowBound[2*zmw], hist.lowBound[2*zmw+1]};
        float2 binSize = {hist.binSize[2*zmw], hist.binSize[2*zmw+1]};

        for (int i = threadIdx.x; i < numBins+2; i+=blockDim.x)
        {
            localHist[i][0] = 0;
            localHist[i][1] = 0;
        }
        __syncwarp(0xFFFFFFFF);

        auto dat = traces.ZmwData(blockIdx.x, zmw);
        auto edgeFinder = (start == 0)
            ? EdgeFinder(edgeState[blockIdx.x], models[blockIdx.x], zmw)
            : EdgeFinder(dat[start-2], dat[start-1], models[blockIdx.x], zmw);

        for (int i = start; i < stop; ++i)
        {
            auto [isEdge, val] = edgeFinder.IsEdgeFrame(dat[i]);
            isEdge = isEdge || (initEdgeDetection && i < 2);

            int bin = (val.X() - lowBound.x) / binSize.x;
            if (bin >= numBins) bin = highOutlier;
            else if (bin < 0) bin = lowOutlier;
            if (isEdge.X()) bin = scrubbedFrame;

            // Get bit flag with each thread that has the same bin as us.
            auto same = __match_any_sync(0xFFFFFFFF, bin);
            // Number of threads with the same bin
            auto count = __popc(same);
            // Thread with the most significant bit gets to own the update
            bool owner = (32 - __clz(same) -1 ) == threadIdx.x;

            if (owner)
            {
                localHist[bin][0] += count;
            }

            bin = (val.Y() - lowBound.y) / binSize.y;
            if (bin >= numBins) bin = highOutlier;
            else if (bin < 0) bin = lowOutlier;
            if (isEdge.Y()) bin = scrubbedFrame;

            // Get bit flag with each thread that has the same bin as us.
            same = __match_any_sync(0xFFFFFFFF, bin);
            // Number of threads with the same bin
            count = __popc(same);
            // Thread with the most significant bit gets to own the update
            owner = (32 - __clz(same) -1 ) == threadIdx.x;

            if (owner)
            {
                localHist[bin][1] += count;
            }
            __syncwarp();
        }

        for (int i = threadIdx.x; i < numBins; i+=blockDim.x)
        {
            hist.binCount[2*zmw][i] += localHist[i][0];
            hist.binCount[2*zmw+1][i] += localHist[i][1];
        }
        if (threadIdx.x == 0)
        {
            hist.outlierCountHigh[2*zmw] += localHist[highOutlier][0];
            hist.outlierCountHigh[2*zmw+1] += localHist[highOutlier][1];
        }
        else if (threadIdx.x == 1)
        {
            hist.outlierCountLow[2*zmw] += localHist[lowOutlier][0];
            hist.outlierCountLow[2*zmw+1] += localHist[lowOutlier][1];
        }

        if (threadIdx.x == blockDim.x-1)
            edgeFinder.Store(edgeState[blockIdx.x], zmw);
        __syncwarp(0xFFFFFFFF);
    }
}

// Now trying 2D parallelism.  A weakness in the last attempt was that reads of the
// trace data were not coallesced.  Here we have 32 warps.  First each warp does
// a coallesced read of 32 frames of data.  Then they use shared memory to transpose
// the data, so a warp goes from holding 1 frame of all zmw, to 32 frames of a pair
// of zmw.  Finally once a warp has data from the same ZMW, we all back to the previous
// strategy for cooperative binning.
__global__ void BinningSharedContig2DBlock(Data::GpuBatchData<const PBShort2> traces,
                                           DeviceView<LaneHistogramTrans> hists,
                                           DeviceView<const LaneModel> models,
                                           DeviceView<EdgeScrubbingState> edgeState,
                                           bool initEdgeDetection)
{
    constexpr int16_t numBins = LaneHistogramTrans::numBins;
    constexpr int16_t lowOutlier = numBins;
    constexpr int16_t highOutlier = numBins+1;
    constexpr int16_t scrubbedFrame = numBins+2;

    struct SharedData
    {
        uint16_t localHist[32][numBins+3][2];
        // The 33 is intentional, to avoid bank conflicts during
        // the transpose
        PBShort2 trans[32][33];
    };

    assert(traces.NumFrames() % blockDim.x == 0);
    assert(traces.NumFrames() % blockDim.y == 0);
    assert(blockDim.x == 32);
    assert(blockDim.y == 32);
    assert(blockDim.z == 1);

    __shared__ SharedData shared;
    auto& hist = hists[blockIdx.x];

    const auto zmw = threadIdx.y;

    float2 lowBound = {hist.lowBound[2*zmw], hist.lowBound[2*zmw+1]};
    float2 binSize = {hist.binSize[2*zmw], hist.binSize[2*zmw+1]};

    for (int i = threadIdx.x; i < numBins+2; i+=blockDim.x)
    {
        shared.localHist[zmw][i][0] = 0;
        shared.localHist[zmw][i][1] = 0;
    }
    __syncwarp(0xFFFFFFFF);

    auto threadFrames = traces.NumFrames() / blockDim.y;
    auto start = threadIdx.y * threadFrames;
    auto stop = start + threadFrames;

    auto dat = traces.ZmwData(blockIdx.x, threadIdx.x);

    auto edgeFinder = (start == 0)
        ? EdgeFinder(edgeState[blockIdx.x], models[blockIdx.x], threadIdx.x)
        : EdgeFinder(dat[start-2], dat[start-1], models[blockIdx.x], threadIdx.x);

    auto scrubbed = std::numeric_limits<int16_t>::lowest();
    for (int i = start; i < stop; ++i)
    {
        auto [isEdge, val] = edgeFinder.IsEdgeFrame(dat[i]);
        isEdge = isEdge || (initEdgeDetection && i < 2);
        val = Blend(isEdge, scrubbed, val);

        __syncthreads();
        shared.trans[threadIdx.y][threadIdx.x] = val;
        __syncthreads();
        val = shared.trans[threadIdx.x][threadIdx.y];

        int bin = (val.X() - lowBound.x) / binSize.x;
        if (bin >= numBins) bin = highOutlier;
        else if (bin < 0) bin = lowOutlier;
        if (val.X() == scrubbed) bin = scrubbedFrame;

        // Get bit flag with each thread that has the same bin as us.
        auto same = __match_any_sync(0xFFFFFFFF, bin);
        // Number of threads with the same bin
        auto count = __popc(same);
        // Thread with the most significant bit gets to own the update
        bool owner = (32 - __clz(same) -1 ) == threadIdx.x;

        if (owner)
        {
            shared.localHist[zmw][bin][0] += count;
        }

        bin = (val.Y() - lowBound.y) / binSize.y;
        if (bin >= numBins) bin = highOutlier;
        else if (bin < 0) bin = lowOutlier;
        if (val.Y() == scrubbed) bin = scrubbedFrame;

        // Get bit flag with each thread that has the same bin as us.
        same = __match_any_sync(0xFFFFFFFF, bin);
        // Number of threads with the same bin
        count = __popc(same);
        // Thread with the most significant bit gets to own the update
        owner = (32 - __clz(same) -1 ) == threadIdx.x;

        if (owner)
        {
            shared.localHist[zmw][bin][1] += count;
        }
    }

    for (int i = threadIdx.x; i < numBins; i+=blockDim.x)
    {
        hist.binCount[2*zmw][i] += shared.localHist[zmw][i][0];
        hist.binCount[2*zmw+1][i] += shared.localHist[zmw][i][1];
    }
    if (threadIdx.x == 0)
    {
        hist.outlierCountHigh[2*zmw] += shared.localHist[zmw][highOutlier][0];
        hist.outlierCountHigh[2*zmw+1] += shared.localHist[zmw][highOutlier][1];
    }
    else if (threadIdx.x == 1)
    {
        hist.outlierCountLow[2*zmw] += shared.localHist[zmw][numBins][0];
        hist.outlierCountLow[2*zmw+1] += shared.localHist[zmw][numBins][1];
    }

    if (threadIdx.y == blockDim.y - 1)
        edgeFinder.Store(edgeState[blockIdx.x]);
}

// The last attempt was good, but this one is better.  We still use the same 2D
// block of threads, but now the histogram data is stored interleaved.  We no
// longer have to transpose the data after reading it, though we now have to rely
// on explicit atomic operations to prevent different warps from stomping on each
// other.  The use of atomics turns out to be a big win in this case.
__global__ void BinningSharedInterleaved2DBlock(Data::GpuBatchData<const PBShort2> traces,
                                                DeviceView<Data::LaneHistogram<float, uint16_t>> hists,
                                                DeviceView<const LaneModel> models,
                                                DeviceView<EdgeScrubbingState> edgeState,
                                                bool initEdgeDetection)
{
    assert(blockDim.x == 32);
    auto& ghist = hists[blockIdx.x];
    const auto zmw = threadIdx.x;

    PBHalf2 lowBound = {ghist.lowBound[2*zmw], ghist.lowBound[2*zmw+1]};
    PBHalf2 binSize = {ghist.binSize[2*zmw], ghist.binSize[2*zmw+1]};

    constexpr int16_t numBins = LaneHistogramTrans::numBins;
    constexpr int16_t lowOutlier = numBins;
    constexpr int16_t highOutlier = numBins+1;
    constexpr int16_t edgeFrame = numBins+2;
    __shared__ PBShort2 lhist[numBins+3][32];

    for (int i = threadIdx.y; i < numBins+2; i+=blockDim.y)
    {
        lhist[i][threadIdx.x] = 0;
    }

    __syncthreads();

    auto count = (traces.NumFrames() + blockDim.y - 1) / blockDim.y;
    auto start = threadIdx.y * count;
    auto stop = min(start + count, traces.NumFrames());
    if (threadIdx.y == 0 && initEdgeDetection) start = 2;

    auto trace = traces.ZmwData(blockIdx.x, threadIdx.x);

    auto edgeFinder = (start == 0)
        ? EdgeFinder(edgeState[blockIdx.x], models[blockIdx.x], threadIdx.x)
        : EdgeFinder(trace[start - 2], trace[start - 1], models[blockIdx.x], threadIdx.x);

    for (int i = start; i < stop; ++i)
    {
        auto [isEdge, val] = edgeFinder.IsEdgeFrame(trace[i]);

        auto bin = (val - lowBound) / binSize;
        bin = Blend(bin >= numBins, highOutlier, bin);
        bin = Blend(bin < 0, lowOutlier, bin);
        bin = Blend(isEdge, edgeFrame, bin);

        atomicAdd(&lhist[bin.IntX()][threadIdx.x].data(), 1);
        atomicAdd(&lhist[bin.IntY()][threadIdx.x].data(), 1<<16);
    }

    __syncthreads();

    for (int i = threadIdx.y; i < numBins; i+=blockDim.y)
    {
        ghist.binCount[i][2*threadIdx.x] += lhist[i][threadIdx.x].X();
        ghist.binCount[i][2*threadIdx.x+1] += lhist[i][threadIdx.x].Y();
    }

    if (threadIdx.y == 0)
    {
        ghist.outlierCountLow[2*threadIdx.x] += lhist[lowOutlier][threadIdx.x].X();
        ghist.outlierCountLow[2*threadIdx.x+1] += lhist[lowOutlier][threadIdx.x].Y();
    }
    else if (threadIdx.y == 1 % blockDim.y)
    {
        ghist.outlierCountHigh[2*threadIdx.x] += lhist[highOutlier][threadIdx.x].X();
        ghist.outlierCountHigh[2*threadIdx.x+1] += lhist[highOutlier][threadIdx.x].Y();
    }

    if (threadIdx.y == blockDim.y - 1)
        edgeFinder.Store(edgeState[blockIdx.x]);
}

// Small kernel to un-transpose LaneHistogramTrans when sharing it out to the rest
// of the world.
__global__ void CopyToContig(DeviceView<const LaneHistogramTrans> source,
                             DeviceView<Data::LaneHistogram<float, uint16_t>> dest)
{
    assert(blockDim.x == 64);
    auto& sBlock = source[blockIdx.x];
    auto& dBlock = dest[blockIdx.x];

    for (int i = 0; i < dBlock.numBins; ++i)
    {
        dBlock.binCount[i][threadIdx.x] = sBlock.binCount[threadIdx.x][i];
    }
    dBlock.outlierCountHigh[threadIdx.x] = sBlock.outlierCountHigh[threadIdx.x];
    dBlock.outlierCountLow[threadIdx.x] = sBlock.outlierCountLow[threadIdx.x];
    dBlock.lowBound[threadIdx.x] = sBlock.lowBound[threadIdx.x];
    dBlock.binSize[threadIdx.x] = sBlock.binSize[threadIdx.x];
}

struct ZmwBinsInfo
{
    float lowerBound;
    float binSize;
};

__device__ ZmwBinsInfo ComputeBounds(const StatAccumState& laneBlStats)
{
    const auto blCount = laneBlStats.moment0[threadIdx.x];
    const auto mom1 = laneBlStats.moment1[threadIdx.x];
    auto blMean = mom1 / blCount + laneBlStats.offset[threadIdx.x];
    auto blSigma = laneBlStats.moment2[threadIdx.x] - blMean * mom1;
    blSigma = sqrt(blSigma / (blCount - 1));

    if (blCount < staticConfig.baselineStatMinFrameCount_)
    {
        blMean = 0.0f;
        blSigma = staticConfig.fallBackBaselineSigma_;
    }

    const auto lower = blMean - 4.0f*blSigma;
    const auto binSize = staticConfig.binSizeCoeff_ * blSigma;
    return {lower, binSize};
}

// These two functions really only differ in the typeof histogram they accept, and the
// order of indexing for the binCount member.
__device__ void ResetHist(Data::LaneHistogram<float, uint16_t>* hist, const ZmwBinsInfo& binInfo)
{
    assert(blockDim.x == 64);

    for (int i = 0; i < hist->numBins; ++i)
    {
        hist->binCount[i][threadIdx.x] = 0;
    }
    hist->outlierCountHigh[threadIdx.x] = 0;
    hist->outlierCountLow[threadIdx.x] = 0;
    hist->lowBound[threadIdx.x] = binInfo.lowerBound;
    hist->binSize[threadIdx.x] = binInfo.binSize;
}

__device__ void ResetHist(LaneHistogramTrans* hist, const ZmwBinsInfo& binInfo)
{
    assert(blockDim.x == 64);

    for (int i = 0; i < hist->numBins; ++i)
    {
        hist->binCount[threadIdx.x][i] = 0;
    }
    hist->outlierCountHigh[threadIdx.x] = 0;
    hist->outlierCountLow[threadIdx.x] = 0;
    hist->lowBound[threadIdx.x] = binInfo.lowerBound;
    hist->binSize[threadIdx.x] = binInfo.binSize;
}


template <typename Hist>
__global__ void ResetHistsBounds(DeviceView<Hist> hists,
                                 DeviceView<const LaneHistBounds> bounds)
{
    assert(blockDim.x == 64);
    auto& hist = hists[blockIdx.x];
    const auto& bound = bounds[blockIdx.x];
    ZmwBinsInfo binInfo {
        bound.lowerBounds[threadIdx.x],
        (bound.upperBounds[threadIdx.x] - bound.lowerBounds[threadIdx.x])
            / static_cast<float>(hist.numBins)
    };
    ResetHist(&hists[blockIdx.x], binInfo);
}

template <typename Hist>
__global__ void ResetHistsStats(DeviceView<Hist> hists,
                                DeviceView<const Data::BaselinerStatAccumState> stats)
{
    const auto& binInfo = ComputeBounds(stats[blockIdx.x].baselineStats);
    ResetHist(&hists[blockIdx.x], binInfo);
}

class DeviceTraceHistogramAccum::ImplBase
{
public:
    using HistDataType = TraceHistogramAccumulator::HistDataType;
    using HistCountType = TraceHistogramAccumulator::HistCountType;

    ImplBase(uint32_t poolId, uint32_t poolSize, DeviceHistogramTypes type)
        : poolId_(poolId)
        , poolSize_(poolSize)
        , type_(type)
    {}

    virtual ~ImplBase() = default;

    virtual void AddBatchImpl(const Data::TraceBatch<DataType>& traces,
                              const TraceHistogramAccumulator::PoolDetModel& detModel) = 0;

    virtual void ResetImpl(const Cuda::Memory::UnifiedCudaArray<LaneHistBounds>& bounds) = 0;

    virtual void ResetImpl(const Data::BaselinerMetrics& metrics) = 0;

    virtual Data::PoolHistogram<HistDataType, HistCountType> HistogramImpl() const = 0;

    uint32_t PoolSize() const { return poolSize_; }
    uint32_t PoolId() const { return poolId_; }
    DeviceHistogramTypes Type() const { return type_; }
private:
    uint32_t poolId_;
    uint32_t poolSize_;
    DeviceHistogramTypes type_;
};

// Handles trace histograms for strategies that interleave zmw data
// e.g. all data for a given bin is contiguous in memory
class HistInterleavedZmw : public DeviceTraceHistogramAccum::ImplBase
{
public:
    HistInterleavedZmw(unsigned int poolId,
                       unsigned int poolSize,
                       Cuda::Memory::StashableAllocRegistrar* registrar,
                       DeviceHistogramTypes type)
        : DeviceTraceHistogramAccum::ImplBase(poolId, poolSize, type)
        , data_(registrar, SOURCE_MARKER(), poolSize)
        , edgeState_(registrar, SOURCE_MARKER(), poolSize)
    {}

    void AddBatchImpl(const Data::TraceBatch<int16_t>& traces,
                      const TraceHistogramAccumulator::PoolDetModel& detModel) override
    {
        switch (Type())
        {
        case DeviceHistogramTypes::GlobalInterleaved:
            {
                auto binning = PBLauncher(BinningGlobalInterleaved, PoolSize(), laneSize);
                binning(traces, data_, detModel, edgeState_, initEdgeDetection_);
                break;
            }
        case DeviceHistogramTypes::SharedInterleaved2DBlock:
            {
                auto binning = PBLauncher(BinningSharedInterleaved2DBlock, PoolSize(), dim3{laneSize/2, 32, 1});
                binning(traces, data_, detModel, edgeState_, initEdgeDetection_);
                break;
            }
        default:
            throw PBException("Unexpected device histogram type in HistInterleavedZmw");
        }
        initEdgeDetection_ = false;
    }

    void ResetImpl(const Cuda::Memory::UnifiedCudaArray<LaneHistBounds>& bounds) override
    {
        assert(bounds.Size() == PoolSize());
        PBLauncher(ResetHistsBounds<HistogramType>, PoolSize(), laneSize)(data_, bounds);
    }

    void ResetImpl(const Data::BaselinerMetrics& metrics) override
    {
        assert(metrics.baselinerStats.Size() == PoolSize());
        PBLauncher(ResetHistsStats<HistogramType>, PoolSize(), laneSize)(data_, metrics.baselinerStats);
    }

    Data::PoolHistogram<HistDataType, HistCountType> HistogramImpl() const override
    {
        auto rawHist = data_.CopyAsUnifiedCudaArray(SyncDirection::HostReadDeviceWrite, SOURCE_MARKER());
        return Data::PoolHistogram<HistDataType, HistCountType>(PoolId(), std::move(rawHist));
    }

private:
    using HistogramType = LaneHistogram<float, uint16_t>;
    DeviceOnlyArray<HistogramType> data_;
    DeviceOnlyArray<EdgeScrubbingState> edgeState_;
    bool initEdgeDetection_ = true;
};

// Handles trace histograms for strategies that have contiguous histograms
// e.g. all data for a given zmw is contiguous in memory
class HistContigZmw : public DeviceTraceHistogramAccum::ImplBase
{
public:
    HistContigZmw(unsigned int poolId,
                  unsigned int poolSize,
                  Cuda::Memory::StashableAllocRegistrar* registrar,
                  DeviceHistogramTypes type)
        : DeviceTraceHistogramAccum::ImplBase(poolId, poolSize, type)
        , data_(registrar, SOURCE_MARKER(), poolSize)
        , edgeState_(registrar, SOURCE_MARKER(), poolSize)
    {}

    void AddBatchImpl(const Data::TraceBatch<int16_t>& traces,
                      const TraceHistogramAccumulator::PoolDetModel& detModel) override
    {

        switch (Type())
        {
        case DeviceHistogramTypes::GlobalContig:
            {
                auto binning = PBLauncher(BinningGlobalContig, PoolSize(), laneSize);
                binning(traces, data_, detModel, edgeState_, initEdgeDetection_);
                break;
            }
        case DeviceHistogramTypes::GlobalContigCoopWarps:
            {
                auto binning = PBLauncher(BinningGlobalContigCoopWarps, PoolSize(), laneSize/2);
                binning(traces, data_, detModel, edgeState_, initEdgeDetection_);
                break;
            }
        case DeviceHistogramTypes::SharedContigCoopWarps:
            {
                auto binning = PBLauncher(BinningSharedContigCoopWarps, PoolSize(), laneSize/2);
                binning(traces, data_, detModel, edgeState_, initEdgeDetection_);
                break;
            }
        case DeviceHistogramTypes::SharedContig2DBlock:
            {
                auto binning = PBLauncher(BinningSharedContig2DBlock, PoolSize(), dim3{laneSize/2, laneSize/2,1});
                binning(traces, data_, detModel, edgeState_, initEdgeDetection_);
                break;
            }
        default:
            throw PBException("Unexpected device histogram type in HistInterleavedZmw");
        }
        initEdgeDetection_ = false;
    }

    void ResetImpl(const Cuda::Memory::UnifiedCudaArray<LaneHistBounds>& bounds) override
    {
        assert(bounds.Size() == PoolSize());
        PBLauncher(ResetHistsBounds<LaneHistogramTrans>, PoolSize(), laneSize)(data_, bounds);
    }

    void ResetImpl(const Data::BaselinerMetrics& metrics) override
    {
        assert(metrics.baselinerStats.Size() == PoolSize());
        PBLauncher(ResetHistsStats<LaneHistogramTrans>, PoolSize(), laneSize)(data_, metrics.baselinerStats);
    }

    Data::PoolHistogram<HistDataType, HistCountType> HistogramImpl() const override
    {
        Data::PoolHistogram<HistDataType, HistCountType> ret(PoolId(),
                                                             PoolSize(),
                                                             SyncDirection::HostReadDeviceWrite);
        PBLauncher(CopyToContig, PoolSize(), laneSize)(data_, ret.data);
        return ret;
    }

private:
    DeviceOnlyArray<LaneHistogramTrans> data_;
    DeviceOnlyArray<EdgeScrubbingState> edgeState_;
    bool initEdgeDetection_ = true;
};

void DeviceTraceHistogramAccum::Configure(const Data::BasecallerTraceHistogramConfig& traceConfig)
{
    StaticConfig config;

    config.binSizeCoeff_ = traceConfig.BinSizeCoeff;
    PBLOG_INFO << "TraceHistogramAccumulator: BinSizeCoeff = "
               << config.binSizeCoeff_ << '.';

    config.baselineStatMinFrameCount_ = traceConfig.BaselineStatMinFrameCount;
    PBLOG_INFO << "TraceHistogramAccumulator: BaselineStatMinFrameCount = "
               << config.baselineStatMinFrameCount_ << '.';

    config.fallBackBaselineSigma_ = traceConfig.FallBackBaselineSigma;
    PBLOG_INFO << "TraceHistogramAccumulator: FallBackBaselineSigma = "
               << config.fallBackBaselineSigma_ << '.';

    CudaRawCopyToSymbol(&staticConfig, &config, sizeof(StaticConfig));
}


void DeviceTraceHistogramAccum::AddBatchImpl(const Data::TraceBatch<DataType>& traces,
                                             const TraceHistogramAccumulator::PoolDetModel& detModel)
{
    impl_->AddBatchImpl(traces, detModel);
    CudaSynchronizeDefaultStream();
}

void DeviceTraceHistogramAccum::ResetImpl(const Cuda::Memory::UnifiedCudaArray<LaneHistBounds>& bounds)
{
    impl_->ResetImpl(bounds);
    CudaSynchronizeDefaultStream();
}

void DeviceTraceHistogramAccum::ResetImpl(const Data::BaselinerMetrics& metrics)
{
    impl_->ResetImpl(metrics);
    CudaSynchronizeDefaultStream();
}

DeviceTraceHistogramAccum::PoolHistType DeviceTraceHistogramAccum::HistogramImpl() const
{
    return impl_->HistogramImpl();
}

DeviceTraceHistogramAccum::DeviceTraceHistogramAccum(unsigned int poolId,
                                                     unsigned int poolSize,
                                                     Cuda::Memory::StashableAllocRegistrar* registrar,
                                                     DeviceHistogramTypes type)
    : TraceHistogramAccumulator(poolId, poolSize)
{
    switch (type)
    {
    case DeviceHistogramTypes::GlobalInterleaved:
    case DeviceHistogramTypes::SharedInterleaved2DBlock:
        {
            impl_ = std::make_unique<HistInterleavedZmw>(poolId, poolSize, registrar, type);
            break;
        }
    case DeviceHistogramTypes::GlobalContig:
    case DeviceHistogramTypes::GlobalContigCoopWarps:
    case DeviceHistogramTypes::SharedContigCoopWarps:
    case DeviceHistogramTypes::SharedContig2DBlock:
        {
            impl_ = std::make_unique<HistContigZmw>(poolId, poolSize, registrar, type);
            break;
        }
    default:
        throw PBException("Unexpected value for DeviceHistogramType");
    }
    CudaSynchronizeDefaultStream();
}

DeviceTraceHistogramAccum::~DeviceTraceHistogramAccum() = default;

}}}
