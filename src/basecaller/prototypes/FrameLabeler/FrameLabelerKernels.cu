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

#include "FrameLabelerKernels.cuh"

#include <type_traits>

#include <common/cuda/CudaLaneArray.cuh>
#include <common/cuda/streams/LaunchManager.cuh>

#include <dataTypes/configs/BasecallerFrameLabelerConfig.h>

using namespace PacBio::Cuda;
using namespace PacBio::Simd;
using namespace PacBio::Cuda::Memory;
using namespace PacBio::Cuda::Utility;
using namespace PacBio::Cuda::Subframe;
using namespace PacBio::Mongo::Data;

namespace PacBio {
namespace Cuda {

namespace
{

// Make the transition matrix visible to all threads via constant memory,
// which for this use case, has performance benefits over generic device
// memory.
//
// Unfortunately __constant__ variables *have* to be global.  We will initialize
// this during the FrameLabeler::Configure function.
__constant__ Subframe::TransitionMatrix<half> trans;

struct RoiThresholds
{
    float upperThreshold = 0;
    float lowerThreshold = 0;
};

__constant__ RoiThresholds roiThresh;

__device__ void Normalize(CudaArray<PBHalf2, numStates>& vec)
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
    #pragma unroll 1
    for (int i = 0; i < numStates; ++i)
    {
        vec[i] /= sum;
    }
}

__global__ void InitLatent(Mongo::Data::GpuBatchData<PBShort2> latent)
{
    auto zmwData = latent.ZmwData(blockIdx.x, threadIdx.x);
    for (auto val : zmwData) val = PBShort2(0);
}

template <typename T2, typename Labels, typename LogLike, typename Scorer, typename TransT, class... Rows>
__device__ void Recursion(T2& maxVal, Labels& labels, LogLike& logLike, const Scorer& scorer,
                          const SparseMatrix<TransT, Rows...>& trans, PBShort2 data, int idx)
{
    CudaArray<PBHalf2, numStates> logAccum;
    PackedLabels packedLabels;

    // Note: The cuda optimizer seems to be finicky about what gets placed
    // inside a register and what gets pushed out to the stack in memory.
    // The packedLabels variable used to be a raw uint32_t that various
    // code sections did bit twiddles on.  Trying to encapsulate that into
    // a small class caused this filter to slow down by almost 2x.  The
    // combination of making it a struct, and the fact that AddRow used to
    // be a template function that accepted packedLabels by reference, caused
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

template <size_t blockThreads>
__launch_bounds__(blockThreads, 32)
__global__ void FrameLabelerKernel(const Memory::DeviceView<const LaneModelParameters<PBHalf2, blockThreads>> models,
                                   const Mongo::Data::GpuBatchData<const PBShort2> input,
                                   Memory::DeviceView<LatentViterbi<blockThreads>> latentData,
                                   ViterbiData<blockThreads> batchViterbiData,
                                   Mongo::Data::GpuBatchData<PBShort2> prevLat,
                                   Mongo::Data::GpuBatchData<PBShort2> nextLat,
                                   Mongo::Data::GpuBatchData<PBShort2> output,
                                   Memory::DeviceView<CudaArray<float, laneSize>> viterbiScore)
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
    CudaArray<PBHalf2, numStates> scratch;
    auto& logLike = scratch;
    auto& latent = latentData[blockIdx.x];
    auto bc = latent.GetBoundary();
    const PBHalf2 zero(0.0f);
    const PBHalf2 ninf(-std::numeric_limits<float>::infinity());
    for (int i = 0; i < numStates; ++i)
    {
        logLike[i] = Blend(bc == i, zero, ninf);
    }

    auto labels = batchViterbiData.BlockData();

    // Forward recursion on latent data
    {
        auto latZmw = prevLat.ZmwData(blockIdx.x, threadIdx.x);
        scorer.Setup(latent.GetModel());
        for (int frame = 0; frame < ViterbiStitchLookback; ++frame)
        {
            Recursion(sharedMaxVal, labels, logLike, scorer, trans, latZmw[frame], frame);
        }
    }

    // Forward recursion on this block's data
    scorer.Setup(models[blockIdx.x]);
    const int numFrames = input.NumFrames();
    const auto& inZmw = input.ZmwData(blockIdx.x, threadIdx.x);
    const int anchor = numFrames - ViterbiStitchLookback;
    for (int frame = 0; frame < anchor; ++frame)
    {
        Recursion(sharedMaxVal, labels, logLike, scorer, trans, inZmw[frame], frame + ViterbiStitchLookback);
    }

    // Need to store the log likelihoods at the actual anchor point, so
    // once we choose a terminus state, we can retrieve it's proper
    // log likelihood
    CudaArray<PBHalf2, numStates> anchorLogLike = logLike;

    for (int frame = anchor; frame < numFrames; ++frame)
    {
        Recursion(sharedMaxVal, labels, logLike, scorer, trans, inZmw[frame], frame + ViterbiStitchLookback);
    }

    // Compute the probabilities of the possible end states.  Propagate
    // them backwards a few frames, as some paths may converge and we
    // can have a more certain estimate.
    Normalize(logLike);
    auto& prob = scratch;
    const int lookStart = numFrames + ViterbiStitchLookback - 1;
    const int lookStop = numFrames - 1;
    for (int i = lookStart; i > lookStop; --i)
    {
        CudaArray<PBHalf2, numStates> newProb;
        for (short state = 0; state < numStates; ++state)
        {
            newProb[state] = PBHalf2(0.0f);
        }
        auto packedLabels = labels(i, 0);
        for (short state = 0; state < numStates; ++state)
        {
            auto prev = packedLabels.PopFront();
            // a PackedLabels fits four x/y pairs, so after every 4th state we have exhausted
            // the current packedLabels and need to extract the next one.
            if ((state % 4) == 3)
            {
                packedLabels = labels(i, state / 4 + 1);
            }
            newProb[prev.x] += Blend(PBBool2(true,false), prob[state], zero);
            newProb[prev.y] += Blend(PBBool2(false,true), prob[state], zero);
        }
        prob = newProb;
    }

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
        auto x = labels(frame, traceState.X() / 4).XAt(traceState.X() % 4);
        auto y = labels(frame, traceState.Y() / 4).YAt(traceState.Y() % 4);
        traceState = PBShort2(x,y);
    }

    // Update latent data
    latent.SetBoundary(anchorState);
    latent.SetModel(models[blockIdx.x]);

    auto outLatZmw = nextLat.ZmwData(blockIdx.x, threadIdx.x);
    const auto offset = numFrames - ViterbiStitchLookback;
    for (int i = 0; i < ViterbiStitchLookback; ++i)
    {

        outLatZmw[i] = inZmw[i + offset];
    }
}

}

constexpr size_t FrameLabeler::BlockThreads;

void FrameLabeler::Configure(const std::array<PacBio::AuxData::AnalogMode,4>& analogs,
                             const BasecallerFrameLabelerConfig& config,
                             double frameRate)
{
    Subframe::TransitionMatrix<half> transHost(analogs, config.viterbi, frameRate);
    CudaCopyToSymbol(trans, &transHost);

    RoiThresholds thresh;
    thresh.upperThreshold = config.roi.upperThreshold;
    thresh.lowerThreshold = config.roi.lowerThreshold;
    CudaCopyToSymbol(roiThresh, &thresh);
}

void FrameLabeler::Finalize() {}

static BatchDimensions LatBatchDims(size_t lanesPerPool)
{
    BatchDimensions ret;
    ret.framesPerBatch = ViterbiStitchLookback;
    ret.laneWidth = laneSize;
    ret.lanesPerBatch = lanesPerPool;
    return ret;
}

FrameLabeler::FrameLabeler(size_t lanesPerPool, StashableAllocRegistrar* registrar)
    : latent_(registrar, SOURCE_MARKER(), lanesPerPool)
    , prevLat_(LatBatchDims(lanesPerPool), Memory::SyncDirection::HostReadDeviceWrite, SOURCE_MARKER())
{
    PBLauncher(InitLatent, lanesPerPool, BlockThreads)(prevLat_);
    CudaSynchronizeDefaultStream();
}

void FrameLabeler::ProcessBatch(const Memory::UnifiedCudaArray<LaneModelParameters<PBHalf, laneSize>>& models,
                                const Mongo::Data::BatchData<int16_t>& input,
                                Mongo::Data::BatchData<int16_t>& latOut,
                                Mongo::Data::BatchData<int16_t>& output,
                                Mongo::Data::FrameLabelerMetrics& metricsOutput)
{
    ViterbiDataHost<BlockThreads> labels(input.NumFrames() + ViterbiStitchLookback, input.LanesPerBatch());

    const auto& launcher = PBLauncher(FrameLabelerKernel<BlockThreads>, input.LanesPerBatch(), BlockThreads);
    launcher(models,
             input,
             latent_,
             labels,
             prevLat_,
             latOut,
             output,
             metricsOutput.viterbiScore);

    CudaSynchronizeDefaultStream();
    std::swap(prevLat_, latOut);
}

}}
