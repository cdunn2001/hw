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

int32_t FrameLabeler::framesPerChunk_ = 0;
int32_t FrameLabeler::lanesPerPool_ = 0;
ThreadSafeQueue<std::unique_ptr<ViterbiDataHost<PBShort2, FrameLabeler::BlockThreads>>> FrameLabeler::scratchData_;
std::unique_ptr<Memory::DeviceOnlyObj<const Subframe::TransitionMatrix>> FrameLabeler::trans_;


void FrameLabeler::Configure(const std::array<Subframe::AnalogMeta, 4>& meta,
                             int32_t lanesPerPool, int32_t framesPerChunk)
{
    if (lanesPerPool <= 0) throw PBException("Invalid value for lanesPerPool");
    if (framesPerChunk <= 0) throw PBException("Invalid value for framesPerChunk");

    trans_ = std::make_unique<Memory::DeviceOnlyObj<const Subframe::TransitionMatrix>>(
            SOURCE_MARKER(),
            CudaArray<Subframe::AnalogMeta, 4>{meta});

    framesPerChunk_ = framesPerChunk;
    lanesPerPool_ = lanesPerPool;
}

void FrameLabeler::Finalize()
{
    scratchData_.Clear();
    trans_.release();
}

std::unique_ptr<ViterbiDataHost<PBShort2, FrameLabeler::BlockThreads>> FrameLabeler::BorrowScratch()
{
    std::unique_ptr<ViterbiDataHost<PBShort2, BlockThreads>> ret;
    bool success = scratchData_.TryPop(ret);
    if (! success)
    {
        ret = std::make_unique<ViterbiDataHost<PBShort2, BlockThreads>>(framesPerChunk_ + ViterbiStitchLookback, lanesPerPool_);
    }
    assert(ret);
    return ret;
}

void FrameLabeler::ReturnScratch(std::unique_ptr<ViterbiDataHost<PBShort2, FrameLabeler::BlockThreads>> data)
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

__global__ void InitLatent(Mongo::Data::GpuBatchData<PBShort2> latent)
{
    auto zmwData = latent.ZmwData(blockIdx.x, threadIdx.x);
    for (auto val : zmwData) val = PBShort2(0);
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


__launch_bounds__(32, 32)
__global__ void FrameLabelerKernel(const Memory::DevicePtr<const Subframe::TransitionMatrix> trans,
                                   const Memory::DeviceView<const LaneModelParameters<PBHalf2, 32>> models,
                                   const Mongo::Data::GpuBatchData<const PBShort2> input,
                                   Memory::DeviceView<LatentViterbi<32>> latentData,
                                   ViterbiData<PBShort2, 32> labels,
                                   Mongo::Data::GpuBatchData<PBShort2> prevLat,
                                   Mongo::Data::GpuBatchData<PBShort2> nextLat,
                                   Mongo::Data::GpuBatchData<PBShort2> output,
                                   Memory::DeviceView<Cuda::Utility::CudaArray<float, laneSize>> viterbiScoreCache)
{
    using namespace Subframe;

    static constexpr unsigned laneWidth = 32;
    assert(blockDim.x == laneWidth);
    __shared__ BlockStateSubframeScorer<laneWidth> scorer;

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

    auto Recursion = [&labels, &trans, &logLike](PBShort2 data, int idx)
    {
        CudaArray<PBHalf2, numStates> logAccum;

        //auto scores = scorer.StateScores(PBHalf2(data));
        const auto dat = PBHalf2(data);
        for (int nextState = 0; nextState < numStates; ++nextState)
        {
            auto score = scorer.StateScores(dat, nextState);
            auto maxVal = score + PBHalf2(trans->Entry(nextState, 0)) + logLike[0];
            auto maxIdx = PBShort2(0);
            for (int prevState = 1; prevState < numStates; ++prevState)
            {
                auto val = score + PBHalf2(trans->Entry(nextState, prevState)) + logLike[prevState];

                auto cond = maxVal > val;
                maxVal = Blend(cond, maxVal, val);
                maxIdx = Blend(cond, maxIdx, PBShort2(prevState));
            }
            logAccum[nextState] = maxVal;
            labels(idx, nextState) = maxIdx;
        }
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
        for (short state = 0; state < numStates; ++state)
        {
            auto prev = labels(i, state);
            newProb[prev.X()] += Blend(PBBool2(true,false), prob[state], zero);
            newProb[prev.Y()] += Blend(PBBool2(false,true), prob[state], zero);
        }
        prob = newProb;
    }

    PBHalf2 maxProb = prob[0];
    PBShort2 anchorState = {0,0};
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
        traceState = PBShort2(labels(frame, traceState.X()).X(),
                              labels(frame, traceState.Y()).Y());
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

void FrameLabeler::ProcessBatch(const Memory::UnifiedCudaArray<LaneModelParameters<PBHalf, 64>>& models,
                                const Mongo::Data::BatchData<int16_t>& input,
                                Mongo::Data::BatchData<int16_t>& latOut,
                                Mongo::Data::BatchData<int16_t>& output,
                                Mongo::Data::FrameLabelerMetrics& metricsOutput)
{
    auto labels = BorrowScratch();

    const auto& launcher = PBLauncher(FrameLabelerKernel, lanesPerPool_, BlockThreads);
    launcher(*trans_,
             models,
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

constexpr size_t FrameLabeler::BlockThreads;

}}
