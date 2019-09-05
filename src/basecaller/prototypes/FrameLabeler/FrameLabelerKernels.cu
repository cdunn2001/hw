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

#include <common/cuda/streams/LaunchManager.cuh>

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
__constant__ Subframe::TransitionMatrix trans;

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

template <size_t blockThreads>
__launch_bounds__(blockThreads, 32)
__global__ void FrameLabelerKernel(const Memory::DeviceView<const LaneModelParameters<PBHalf2, blockThreads>> models,
                                   const Mongo::Data::GpuBatchData<const PBShort2> input,
                                   Memory::DeviceView<LatentViterbi<blockThreads>> latentData,
                                   ViterbiData<blockThreads> batchViterbiData,
                                   Mongo::Data::GpuBatchData<PBShort2> prevLat,
                                   Mongo::Data::GpuBatchData<PBShort2> nextLat,
                                   Mongo::Data::GpuBatchData<PBShort2> output,
                                   Memory::DeviceView<Cuda::Utility::CudaArray<float, laneSize>> viterbiScoreCache)
{
    // When/if this changes, some of the bit twiddle logic below is going to have to be udpated or generalized
    static_assert(Subframe::numStates == 13,
                  "FrameLabelerKernel currently hard coded to only handle 13 states");
    using namespace Subframe;

    assert(blockDim.x == blockThreads);
    __shared__ BlockStateSubframeScorer<blockThreads> scorer;

    // This optimization requires some notes, and may need to be revisited
    // periodically.  The BlockStateSubframeScorer above uses 26 32 bit words
    // of storage per thread.  In order to get 32 occupant blocks (the best we
    // can do when our block size is 32 threads) then there really are only 24
    // words available.  The best we can do with the above data structure is
    // roughly 29 blocks.  However in an experiment where I pushed two members from
    // `scorer` to local mem / registers, my throughput went down.  I did get the
    // desired increase in occupancy, and overal there were the same number of
    // memory requests so the new local variables were not causing new memory
    // traffic, but our cache hit rate went down.  The improved occupancy helped
    // us less than the new increase in memory latency hurt us.
    //
    // This is not entirely unexpected as more resident blocks means they effectively
    // each get less L1 space to use.  So I did a subsequent experiement adding these two
    // extra shared variables, to see if we could decrease our occpancy a little more
    // and get even better L1 usage.
    //
    // The result was a 4% increase in throughput, which for a single change is good
    // enough to want to keep.  However when profiling, things it did not appear that
    // we were actually benefiting from an increase in cache hits.  Instead the delta
    // between our theoretical occupancy (limited by our shared memory usage) and
    // our actual achieved occupancy went down.  In other words the added shared
    // memory usage decreased our maximum occupancy, but for whatever reason, the
    // occpancy we actually got stayed the same.
    //
    // I'm not sure what all affects the delta between theoretical and achieved.  If
    // other changes affect that balance, then this might become a less optimal choice,
    // and these two should be moved back to being per-thread automatic variables in
    // the `Recursion` lambda.
    //
    // As another note, these variables are obviously of a specific type.  If a need
    // arises to re-use this storage for multiple types, then it can in principle be
    // declared as a char array, and carved up manually via `reinterpret_casts`
    __shared__ CudaArray<PBHalf2, blockThreads> sharedMaxVal;
    __shared__ CudaArray<PBHalf2, blockThreads> sharedScore;

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

    auto Recursion = [&labels, &logLike](PBShort2 data, int idx)
    {
        auto& score = sharedScore[threadIdx.x];
        auto& maxVal = sharedMaxVal[threadIdx.x];
        CudaArray<PBHalf2, numStates> logAccum;

        const auto dat = PBHalf2(data);
        uint32_t packedLabels = 0;
        for (int nextState = 0; nextState < numStates; ++nextState)
        {
            score = scorer.StateScores(dat, nextState);
            maxVal = score + PBHalf2(trans.Entry(nextState, 0)) + logLike[0];
            ushort2 maxIdx = make_ushort2(0,0);

            #pragma unroll(numStates)
            for (int prevState = 1; prevState < numStates; ++prevState)
            {
                auto transScore = trans.Entry(nextState, prevState);
                if (__hisinf(transScore)) continue;

                auto val = score + PBHalf2(transScore + logLike[prevState]);

                auto cond = val >= maxVal;
                maxVal = Blend(cond, val, maxVal);
                if (cond.X()) maxIdx.x = prevState;
                if (cond.Y()) maxIdx.y = prevState;
            }
            // Always slot new entries into the most significant bits, and
            // before that always shift things to the right by a slot. This
            // makes all bit operations happen with compile time arguments,
            // which has proven faster than calculating on the fly what shifts
            // are necessary to populate the correct slot for each iteration.
            logAccum[nextState] = maxVal;
            packedLabels  = packedLabels >> 8;
            packedLabels |= (maxIdx.x << 24);
            packedLabels |= (maxIdx.y << 28);
            if ((nextState & 3) == 3)
            {
                labels(idx, nextState / 4) = packedLabels;
            }
        }
        // Every time we handled a 4th state we wrote the result to
        // labels, but the 13th state is the odd man out.  Need to
        // manually shift things into the correct location and store it.
        labels(idx, 3) = (packedLabels >> 24);
        logLike = logAccum;
    };

    // Forward recursion on latent data
    {
        auto latZmw = prevLat.ZmwData(blockIdx.x, threadIdx.x);
        scorer.Setup(latent.GetModel());
        for (int frame = 0; frame < ViterbiStitchLookback; ++frame)
        {
            Recursion(latZmw[frame], frame);
        }
    }

    // Forward recursion on this block's data
    scorer.Setup(models[blockIdx.x]);
    const int numFrames = input.NumFrames();
    const auto& inZmw = input.ZmwData(blockIdx.x, threadIdx.x);
    const int anchor = numFrames - ViterbiStitchLookback;
    for (int frame = 0; frame < anchor; ++frame)
    {
        Recursion(inZmw[frame], frame + ViterbiStitchLookback);
    }

    // Need to store the log likelihoods at the actual anchor point, so
    // once we choose a terminus state, we can retrieve it's proper
    // log likelihood
    CudaArray<PBHalf2, numStates> anchorLogLike = logLike;

    for (int frame = anchor; frame < numFrames; ++frame)
    {
        Recursion(inZmw[frame], frame + ViterbiStitchLookback);
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
            // We're accessing labels in order, so we can always strip off
            // the low bits to get the current labels, and then shift everything
            // to the right to prepare for the next iteration
            short prevx = (packedLabels & 0xF);
            short prevy = ((packedLabels >> 4) & 0xF);
            if ((state & 3) == 3)
            {
                packedLabels = labels(i, state / 4 + 1);
            } else {
                packedLabels = packedLabels >> 8;
            }
            newProb[prevx] += Blend(PBBool2(true,false), prob[state], zero);
            newProb[prevy] += Blend(PBBool2(false,true), prob[state], zero);
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
    viterbiScoreCache[blockIdx.x][2 * threadIdx.x]     = anchorLogLike[anchorState.X()].FloatX();
    viterbiScoreCache[blockIdx.x][2 * threadIdx.x + 1] = anchorLogLike[anchorState.Y()].FloatY();

    // Traceback
    auto traceState = anchorState;
    auto outZmw = output.ZmwData(blockIdx.x, threadIdx.x);
    for (int frame = numFrames - 1; frame >= 0; --frame)
    {
        outZmw[frame] = traceState;
        uint32_t x = labels(frame, traceState.X() / 4);
        x = x >> (8 * (traceState.X() % 4));
        x &= 0xF;
        uint32_t y = labels(frame, traceState.Y() / 4);
        y = y >> (8 * (traceState.Y() % 4) + 4);
        y &= 0xF;
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

int32_t FrameLabeler::framesPerChunk_ = 0;
int32_t FrameLabeler::lanesPerPool_ = 0;
ThreadSafeQueue<std::unique_ptr<ViterbiDataHost<FrameLabeler::BlockThreads>>> FrameLabeler::scratchData_;

void FrameLabeler::Configure(const std::array<Subframe::AnalogMeta, 4>& meta,
                             int32_t lanesPerPool, int32_t framesPerChunk)
{
    if (lanesPerPool <= 0) throw PBException("Invalid value for lanesPerPool");
    if (framesPerChunk <= 0) throw PBException("Invalid value for framesPerChunk");

    Subframe::TransitionMatrix transHost(CudaArray<Subframe::AnalogMeta, 4>{meta});
    CudaCopyToSymbol(trans, &transHost);

    framesPerChunk_ = framesPerChunk;
    lanesPerPool_ = lanesPerPool;
}

void FrameLabeler::Finalize()
{
    scratchData_.Clear();
}

std::unique_ptr<ViterbiDataHost<FrameLabeler::BlockThreads>> FrameLabeler::BorrowScratch()
{
    std::unique_ptr<ViterbiDataHost<BlockThreads>> ret;
    bool success = scratchData_.TryPop(ret);
    if (! success)
    {
        ret = std::make_unique<ViterbiDataHost<BlockThreads>>(framesPerChunk_ + ViterbiStitchLookback, lanesPerPool_);
    }
    assert(ret);
    return ret;
}

void FrameLabeler::ReturnScratch(std::unique_ptr<ViterbiDataHost<FrameLabeler::BlockThreads>> data)
{
    scratchData_.Push(std::move(data));
}

static BatchDimensions LatBatchDims(size_t lanesPerPool)
{
    BatchDimensions ret;
    ret.framesPerBatch = ViterbiStitchLookback;
    ret.laneWidth = laneSize;
    ret.lanesPerBatch = lanesPerPool;
    return ret;
}

FrameLabeler::FrameLabeler()
    : latent_(SOURCE_MARKER(), lanesPerPool_)
    , prevLat_(LatBatchDims(lanesPerPool_), Memory::SyncDirection::HostReadDeviceWrite, SOURCE_MARKER())
{
    if (framesPerChunk_ == 0 || lanesPerPool_ == 0)
    {
        throw PBException("Must call FrameLabeler::Configure before constructing FrameLabeler objects!");
    }

    PBLauncher(InitLatent, lanesPerPool_, BlockThreads)(prevLat_);
    CudaSynchronizeDefaultStream();
}

void FrameLabeler::ProcessBatch(const Memory::UnifiedCudaArray<LaneModelParameters<PBHalf, laneSize>>& models,
                                const Mongo::Data::BatchData<int16_t>& input,
                                Mongo::Data::BatchData<int16_t>& latOut,
                                Mongo::Data::BatchData<int16_t>& output,
                                Mongo::Data::FrameLabelerMetrics& metricsOutput)
{
    auto labels = BorrowScratch();

    const auto& launcher = PBLauncher(FrameLabelerKernel<BlockThreads>, lanesPerPool_, BlockThreads);
    launcher(models,
             input,
             latent_,
             *labels,
             prevLat_,
             latOut,
             output,
             metricsOutput.viterbiScore);

    Cuda::CudaSynchronizeDefaultStream();
    std::swap(prevLat_, latOut);
    ReturnScratch(std::move(labels));
}

}}
