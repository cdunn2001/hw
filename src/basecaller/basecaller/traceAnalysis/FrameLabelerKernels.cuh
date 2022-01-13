// Copyright (c) 2019-2022, Pacific Biosciences of California, Inc.
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

#ifndef PACBIO_MONGO_BASECALLER_FRAMELABELER_KERNELS_H
#define PACBIO_MONGO_BASECALLER_FRAMELABELER_KERNELS_H

#include <common/cuda/PBCudaSimd.cuh>

#include <dataTypes/BatchData.cuh>

#include <basecaller/traceAnalysis/FrameLabelerDeviceDataStructures.cuh>
#include <basecaller/traceAnalysis/RoiDevice.cuh>
#include <basecaller/traceAnalysis/SubframeScorer.h>

namespace PacBio::Mongo::Basecaller {

using namespace Cuda;

// Make the transition matrix visible to all threads via constant memory,
// which for this use case, has performance benefits over generic device
// memory.
//
// Unfortunately __constant__ variables *have* to be global.  We will initialize
// this during the FrameLabeler::Configure function.
extern __constant__ Subframe::TransitionMatrix<half> trans;

using Subframe::numStates;

/// Transforms a loglikelihood to a normalized probability distribution.
///
/// \param[in, out] vec    A loglikilhood for each of the stats on input, and a
///                        normalized probability on output
/// \param[in]      isRoi  boolean (simd) value indicating if either of the two
///                        zmw are within the "region of interest".  If we are
///                        not in the roi then the result will have all
///                        probability in the background state.
__device__ inline void Normalize(Utility::CudaArray<PBHalf2, numStates>& vec, PBShort2 isRoi)
{
    auto maxVal = vec[0];
    #pragma unroll 1
    for (int i = 1; i < numStates; ++i)
    {
        maxVal = max(vec[i], maxVal);
    }
    PBHalf2 sum(0.0f);
    #pragma unroll 1
    for (int i = 0; i < numStates; ++i)
    {
        vec[i] = exp(vec[i] - maxVal);
        sum += vec[i];
    }
    vec[0] = Blend(isRoi, vec[0] / sum, 1);
    #pragma unroll 1
    for (int i = 1; i < numStates; ++i)
    {
        vec[i] = Blend(isRoi, vec[i] / sum, 0);
    }
}

__global__ static void InitLatent(Mongo::Data::GpuBatchData<PBShort2> latent)
{
    auto zmwData = latent.ZmwData(blockIdx.x, threadIdx.x);
    for (auto val : zmwData) val = PBShort2(0);
}

template <typename T2, typename Labels, typename LogLike, typename Scorer, typename TransT, class... Rows>
__device__ void Recursion(T2& maxVal, Labels& labels, LogLike& logLike, const Scorer& scorer,
                          const SparseMatrix<TransT, Rows...>& trans, PBShort2 data, int idx)
{
    Utility::CudaArray<PBHalf2, numStates> logAccum;
    PackedLabels packedLabels;

    // Note: The cuda optimizer seems to be finicky about what gets placed
    // inside a register and what gets pushed out to the stack in memory.
    // The packedLabels variable used to be a raw uint32_t that various
    // code sections did bit twiddles on.  Trying to encapsulate that into
    // a small class caused this filter to slow down by almost 2x.  The
    // combination of making it a struct, and the fact that AddRow used to
    // be a template function that accepted packedLabelsby reference, caused
    // the variable to be pushed to global memory and the increased memory
    // traffic killed performance.  For whatever reason, capturing this
    // by reference in a lambda doesn't cause any issues, though one would
    // think after inlining the two approaches would be identical.  One would
    // also think that the wrapping a uint32_t inside a light struct with a
    // few inline functions to handle the bit twiddles would also have no
    // effect.
    //
    // Moral of the story is that we're pressed up against a performance cliff
    // and subject to the whims of the optimizer.  If you tweak this lambda
    // be sure to check the overal runtime for performance regressions, as they
    // may be dramatic.
    auto AddRow = [&](const PBHalf2& score,
                      const half* rowData,
                      auto* row)
    {
        // row parameter just used to extract the Row type.
        // I wouldn't even give it a name save we need to use
        // decltype to extract the type information within a lambda.
        using Row = std::remove_pointer_t<decltype(row)>;
        constexpr auto firstIdx = Row::firstIdx;

        // Currently only handle Rows with a single Segment.  This can be
        // generalized to handle an arbitrary number of Segments, but it
        // came with a mild performance penalty, so not doing that unless/
        // until necessary
        using Segment = typename Row::Segment0;

        maxVal = score + PBHalf2(rowData[0]) + logLike[firstIdx];
        ushort2 maxIdx = make_ushort2(firstIdx,firstIdx);

        uint32_t dataIndex = Segment::dataIndex;
        #pragma unroll(numStates)
        for (int prevState = Segment::firstCol; prevState < Segment::lastCol; ++prevState, dataIndex++)
        {
            auto val = score + PBHalf2(rowData[dataIndex]) + logLike[prevState];

            auto cond = val >= maxVal;
            maxVal = Blend(cond, val, maxVal);
            if (cond.X()) maxIdx.x = prevState;
            if (cond.Y()) maxIdx.y = prevState;
        }
        constexpr auto nextState = Row::rowIdx;
        logAccum[nextState] = maxVal;

        // Always slot new entries into the back, and after 4 inserts it will be fully populated
        // and ready for storage.  This approach has empirically been observed to be faster than
        // trying to slot data directly into it's desired final location, as the current version
        // can be done without any runtime dependance on the value of nextState.
        packedLabels.PushBack(maxIdx);
        if ((nextState & 3) == 3)
        {
            labels(idx, nextState / 4) = packedLabels;
        }
    };


    const auto dat = PBHalf2(data);
    // Compile time loop, to loop over all the Rows in our sparse matrix (each of which have
    // a different type)
    auto loop = {(
        AddRow(scorer.StateScores(dat, Rows::rowIdx),
               trans.RowData(Rows::rowIdx),
               (Rows*){nullptr}),0
    )...};
    (void)loop;

    // Every time we handled a 4th state we wrote the result to
    // labels, but the 13th state is the odd man out.  Need to
    // manually shift things into the correct location and store it.
    static_assert(Subframe::numStates == 13, "Expected 13 states");
    packedLabels.PushBackZeroes<3>();
    labels(idx, 3) = packedLabels;
    logLike = logAccum;
}

// A helper to facilitate range based for loops on a subset of data.
// Use MakeRange below to loop between a pair of iterators.
template <typename T>
struct Range
{
    T b;
    T e;

    __device__ T begin() { return b; }
    __device__ T end() { return e; }
};

template <typename T>
__device__ Range<std::decay_t<T>> MakeRange(T&& b, T&&e)
{
    return Range<std::decay_t<T>>{std::forward<T>(b), std::forward<T>(e)};
}

template <size_t blockThreads, typename RoiFilter>
__global__ void FrameLabelerKernel(const Cuda::Memory::DeviceView<const Data::LaneModelParameters<PBHalf2, blockThreads>> models,
                                   const Mongo::Data::GpuBatchData<const PBShort2> input,
                                   Cuda::Memory::DeviceView<LatentViterbi<blockThreads, RoiFilter::lookBack>> latentData,
                                   ViterbiData<blockThreads> batchViterbiData,
                                   Mongo::Data::GpuBatchData<PBShort2> prevLat,
                                   Mongo::Data::GpuBatchData<PBShort2> nextLat,
                                   Mongo::Data::GpuBatchData<PBShort2> output,
                                   Mongo::Data::GpuBatchData<PBShort2> roiWorkspace,
                                   Cuda::Memory::DeviceView<Utility::CudaArray<float, laneSize>> viterbiScore)
{
    // When/if this changes, some of this kernel is going to have to be udpated or generalized
    static_assert(Subframe::numStates == 13,
                  "FrameLabelerKernel currently hard coded to only handle 13 states");
    using namespace Subframe;

    assert(blockDim.x == blockThreads);
    __shared__ BlockStateSubframeScorer<CudaLaneArray<PBHalf2, blockThreads>> scorer;

    // This optimization requires some notes, and may need to be revisited
    // periodically.  The BlockStateSubframeScorer above uses 26 32-bit words
    // of storage per thread.  In order to get 32 occupant blocks (the best we
    // can do when our block size is 32 threads) then there really are only 24
    // words available.  The best we can do with the above data structure is
    // roughly 29 blocks.  However in an experiment where I pushed two members from
    // `scorer` to local mem / registers, my throughput went down.  I did get the
    // desired increase in occupancy, and overall there were the same number of
    // memory requests so the new local variables were not causing new memory
    // traffic, but our cache hit rate went down.  The improved occupancy helped
    // us less than the new increase in memory latency hurt us.
    //
    // This is not entirely unexpected as more resident blocks means they effectively
    // each get less L1 space to use.  So I did a subsequent experiment adding this one
    // extra shared variable, to see if we could decrease our occpancy a little more
    // and get even better L1 usage.
    //
    // The result was a 4% increase in throughput, which for a single change is good
    // enough to want to keep.  However when profiling things, it did not appear that
    // we were actually benefiting from an increase in cache hits.  Instead the delta
    // between our theoretical occupancy (limited by our shared memory usage) and
    // our actual achieved occupancy went down.  In other words the added shared
    // memory usage decreased our maximum occupancy, but for whatever reason, the
    // occpancy we actually got stayed the same.
    //
    // I'm not sure what all affects the delta between theoretical and achieved.  If
    // other changes affect that balance, then this might become a less optimal choice,
    // and these two should be moved back to being per-thread automatic variables in
    // the `Recursion` function
    __shared__ CudaLaneArray<PBHalf2, blockThreads> sharedMaxVal;

    // Initial setup
    Utility::CudaArray<PBHalf2, numStates> scratch;
    auto& logLike = scratch;
    auto& latent = latentData[blockIdx.x];
    const int numFrames = input.NumFrames();
    auto bc = latent.GetLabelsBoundary();
    const PBHalf2 zero(0.0f);
    const PBHalf2 ninf(-std::numeric_limits<float>::infinity());
    for (int i = 0; i < numStates; ++i)
    {
        logLike[i] = Blend(bc == i, zero, ninf);
    }

    auto latZmw = prevLat.ZmwData(blockIdx.x, threadIdx.x);
    Roi::ForwardRecursion<RoiFilter> forwardRoi(latent.GetLatentTraces(), latZmw,
                                                     latent.GetRoiBoundary(),
                                                     1 / sqrt(latent.GetModel().BaselineMode().vars[threadIdx.x]),
                                                     roiWorkspace.ZmwData(blockIdx.x, threadIdx.x));

    auto labels = batchViterbiData.BlockData();

    // We're going to need to do the forward recursion in a few different
    // stages, mostly becuase we're stitching together two different
    // data sequences.  Throwing together a quick lambda here to avoid
    // duplication elsewhere, and make it a touch clearer what is different
    // in the various stages.
    size_t frame = 0;
    auto RecursionLoop = [&] (auto&& range) mutable
    {
        for (const auto& val : range)
        {
            Recursion(sharedMaxVal, labels,
                      logLike, scorer,
                      trans, forwardRoi.Process(val),
                      frame);
            frame++;
        }
    };

    // Run through all the latent data first
    scorer.Setup(latent.GetModel());
    RecursionLoop(MakeRange(latZmw.begin()+ RoiFilter::lookForward, latZmw.end()));
    assert(frame == ViterbiStitchLookback + RoiFilter::stitchFrames);
    assert(forwardRoi.NextIdx() == frame);

    // Take care to handle the latency of the roi filter.  We have
    // a few frames now where we input data from the new block into
    // the roi filter, but still get back data from the latent data
    // and so we need to continue using the old model
    const auto& inZmw = input.ZmwData(blockIdx.x, threadIdx.x);
    RecursionLoop(MakeRange(inZmw.begin(), inZmw.begin() + RoiFilter::lookForward));
    assert(frame == ViterbiStitchLookback + RoiFilter::stitchFrames + RoiFilter::lookForward);
    assert(forwardRoi.NextIdx() == frame);

    // We've been fudging things a bit, by using a singular baseline variance in
    // the roi computation, though really the roi computation spans a few frames
    // which means that near a block boundary not all frames have the same model.
    // With the current setting (4 points, with a latency of 2), updating things
    // here means we split the difference and never have more than 2 frames using
    // the wrong variance. (though the effect spans all four frames)
    forwardRoi.UpdateInvSigma(1 / sqrt(models[blockIdx.x].BaselineMode().vars[threadIdx.x]));

    // Now we can update things to use the new model, though we'll pause
    // at the anchor point to record the log likelihood at that point
    scorer.Setup(models[blockIdx.x]);
    const int anchor = numFrames - ViterbiStitchLookback - RoiFilter::stitchFrames;
    RecursionLoop(MakeRange(inZmw.begin()+RoiFilter::lookForward, inZmw.begin() + anchor));
    Utility::CudaArray<PBHalf2, numStates> anchorLogLike = logLike;
    assert(frame == numFrames);
    assert(forwardRoi.NextIdx() == frame);

    // Now we can do forward recursion on the up until the start of the roi lookback.
    // There's no point in doing recursion beyond this point, because we won't ever
    // have a valid roi value to use during traceback.
    RecursionLoop(MakeRange(inZmw.begin() + anchor, inZmw.end() - RoiFilter::stitchFrames));
    assert(frame == numFrames + ViterbiStitchLookback);
    assert(forwardRoi.NextIdx() == frame);

    // Continue doing the forward ROI computation on the rest of the data.
    for (const auto& val : MakeRange(inZmw.end() - RoiFilter::stitchFrames, inZmw.end()))
    {
        forwardRoi.Process(val);
    }
    assert(forwardRoi.NextIdx() == frame + RoiFilter::stitchFrames);

    // Applies the faux roi boundary condition for the LHS (currently `off`), and then
    // walk things back for a number of frames.  The more we walk things back the more
    // likely we'll have a robust roi determination, but the more we walk things back
    // the more latent data we have to pay to store.
    Roi::BackwardRecursion<RoiFilter> backRoi(roiWorkspace.ZmwData(blockIdx.x, threadIdx.x));
    assert(backRoi.FramesRemaining() == frame);

    // Now that we have a better ROI boundary condition, we can work on finding
    // the viterbi boundary contidion. We start just by computing the probabilities
    // of the possible end states.  We then propogate these probabilities backwards
    // by doing a traceback of all states simultaneously.  As we propogate backwards
    // some paths may converge, which effectively increases our certainty about the
    // true state at that frame.  However as before, the more we traverse back the
    // more latent data we have to pay to store.
    //
    // Note: There are two ways we can incorporate the roi information.  If we look
    //       at a non-roi classification for the *current* frame, then we can adjust
    //       the loglikelihood for the current frame to make every state but the
    //       baseline impossible.  If we look at a non-roi classification for the *previous*
    //       frame then we can adjust the traceback links to force all states to
    //       trace back to the baseline state.
    //
    //       We use both of those here.  We use the current frame's roi as part of
    //       operation converting the loglikelihood to a normalized probabilitiy.
    //       This also means that during the subsequent loops, each iteration is
    //       now looking at the previous frames ROI and we can adjust the traceback
    //       information
    Normalize(logLike, backRoi.PopRoiState());
    auto& prob = scratch;
    const int lookStart = numFrames + ViterbiStitchLookback - 1;
    const int lookStop = numFrames - 1;
    PBShort2 isRoi = 0;
    for (int i = lookStart; i > lookStop; --i)
    {
        Utility::CudaArray<PBHalf2, numStates> newProb;
        for (short state = 0; state < numStates; ++state)
        {
            newProb[state] = PBHalf2(0.0f);
        }
        isRoi = backRoi.PopRoiState();
        auto packedLabels = labels(i, 0);
        for (short state = 0; state < numStates; ++state)
        {
            auto prev = Blend(isRoi, packedLabels.PopFront(), 0);
            // a PackedLabels fits four x/y pairs, so after every 4th state we have exhausted
            // the current packedLabels and need to extract the next one.
            if ((state % 4) == 3)
            {
                packedLabels = labels(i, state / 4 + 1);
            }
            newProb[prev.X()] += Blend(PBBool2(true,false), prob[state], zero);
            newProb[prev.Y()] += Blend(PBBool2(false,true), prob[state], zero);
        }
        prob = newProb;
    }
    // We've started looking at the roi for the previous frame, so we need
    // the -1 here to account for that.
    assert(backRoi.FramesRemaining() == input.NumFrames()-1);

    latent.SetRoiBoundary(isRoi);
    PBHalf2 maxProb = prob[0];
    PBShort2 anchorState(0);
    #pragma unroll 1
    for (int i = 1; i < numStates; ++i)
    {
        auto cond = maxProb > prob[i];
        maxProb = Blend(cond, maxProb, prob[i]);
        anchorState = Blend(cond, anchorState, PBShort2(i));
    }

    // Now that we have an anchor state, save the associated viterbi score
    viterbiScore[blockIdx.x][2 * threadIdx.x]     = anchorLogLike[anchorState.X()].FloatX();
    viterbiScore[blockIdx.x][2 * threadIdx.x + 1] = anchorLogLike[anchorState.Y()].FloatY();

    // Traceback
    auto traceState = anchorState;
    auto outZmw = output.ZmwData(blockIdx.x, threadIdx.x);
    for (int frame = numFrames - 1; frame >= 0; --frame)
    {
        outZmw[frame] = traceState;
        // Break once we've output the labels for frame 0.  We don't need
        // to trace back to the boundary condition, and we don't have the
        // roi for frame negative 1 anyway.
        if (frame == 0) break;
        auto x = labels(frame, traceState.X() / 4).XAt(traceState.X() % 4);
        auto y = labels(frame, traceState.Y() / 4).YAt(traceState.Y() % 4);
        auto isRoi = backRoi.PopRoiState();
        traceState = Blend(isRoi, PBShort2(x,y), 0);
    }
    assert(backRoi.FramesRemaining() == 0);

    // Update latent data
    latent.SetLabelsBoundary(anchorState);
    latent.SetModel(models[blockIdx.x]);

    auto outLatZmw = nextLat.ZmwData(blockIdx.x, threadIdx.x);
    const auto numLatent = ViterbiStitchLookback + RoiFilter::lookForward + RoiFilter::stitchFrames;
    const auto offset = numFrames - numLatent;
    assert(outLatZmw.size() == numLatent);
    for (int i = 0; i < numLatent; ++i)
    {
        outLatZmw[i] = inZmw[i + offset];
    }

    // Store frames that have had their labels emitted already, but we also need
    // access to during the next batch for use in the roi filter
    latent.SetLatentTraces(inZmw.begin() + offset - 1);
}

}

#endif /*PACBIO_MONGO_BASECALLER_FRAMELABELER_KERNELS_H*/
