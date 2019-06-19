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
ThreadSafeQueue<std::unique_ptr<ViterbiDataHost<short2, FrameLabeler::BlockThreads>>> FrameLabeler::scratchData_;
std::unique_ptr<Memory::DeviceOnlyObj<Subframe::TransitionMatrix>> FrameLabeler::trans_;


void FrameLabeler::Configure(const std::array<Subframe::AnalogMeta, 4>& meta,
                             int32_t lanesPerPool, int32_t framesPerChunk)
{
    if (lanesPerPool <= 0) throw PBException("Invalid value for lanesPerPool");
    if (framesPerChunk <= 0) throw PBException("Invalid value for framesPerChunk");

    trans_ = std::make_unique<Memory::DeviceOnlyObj<Subframe::TransitionMatrix>>(
            CudaArray<Subframe::AnalogMeta, 4>{meta});

    framesPerChunk_ = framesPerChunk;
    lanesPerPool_ = lanesPerPool;
}

void FrameLabeler::Finalize()
{
    scratchData_.Clear();
    trans_.release();
}

std::unique_ptr<ViterbiDataHost<short2, FrameLabeler::BlockThreads>> FrameLabeler::BorrowScratch()
{
    std::unique_ptr<ViterbiDataHost<short2, BlockThreads>> ret;
    bool success = scratchData_.TryPop(ret);
    if (! success)
    {
        ret = std::make_unique<ViterbiDataHost<short2, BlockThreads>>(framesPerChunk_ + ViterbiStitchLookback, lanesPerPool_);
    }
    assert(ret);
    return ret;
}

void FrameLabeler::ReturnScratch(std::unique_ptr<ViterbiDataHost<short2, FrameLabeler::BlockThreads>> data)
{
    scratchData_.Push(std::move(data));
}

FrameLabeler::FrameLabeler()
    : latent_(lanesPerPool_)
{
    if (framesPerChunk_ == 0 || lanesPerPool_ == 0)
    {
        throw PBException("Must call FrameLabeler::Configure before constructing FrameLabeler objects!");
    }
}


__launch_bounds__(32, 32)
__global__ void FrameLabelerKernel(const Memory::DevicePtr<const Subframe::TransitionMatrix> trans,
                                   const Memory::DeviceView<const LaneModelParameters<PBHalf2, 32>> models,
                                   const Mongo::Data::GpuBatchData<const short2> input,
                                   Memory::DeviceView<LatentViterbi<32>> latentData,
                                   ViterbiData<short2, 32> labels,
                                   Mongo::Data::GpuBatchData<short2> output)
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
        short2 cond = {bc.x == i, bc.y == i};
        logLike[i] = Blend(cond, zero, ninf);
    }

    auto Recursion = [&labels, &trans, &logLike](short2 data, int idx)
    {
        CudaArray<PBHalf2, numStates> logAccum;

        //auto scores = scorer.StateScores(PBHalf2(data));
        const auto dat = PBHalf2(data);
        for (int nextState = 0; nextState < numStates; ++nextState)
        {
            auto score = scorer.StateScores(dat, nextState);
            auto maxVal = score + PBHalf2(trans->Entry(nextState, 0)) + logLike[0];
            auto maxIdx = make_short2(0,0);
            for (int prevState = 1; prevState < numStates; ++prevState)
            {
                auto val = score + PBHalf2(trans->Entry(nextState, prevState)) + logLike[prevState];

                auto cond = maxVal > val;
                maxVal = Blend(cond, maxVal, val);
                maxIdx = Blend(cond, maxIdx, make_short2(prevState, prevState));
            }
            logAccum[nextState] = maxVal;
            labels(idx, nextState) = maxIdx;
        }
        logLike = logAccum;
    };

    // Forward recursion on latent data
    const int latentFrames = latent.NumFrames();
    scorer.Setup(latent.GetModel());
    for (int frame = 0; frame < latentFrames; ++frame)
    {
        Recursion(latent.FrameData(frame), frame);
    }

    // Forward recursion on this block's data
    scorer.Setup(models[blockIdx.x]);
    const int numFrames = input.Dims().framesPerBatch;
    const auto& inZmw = input.ZmwData(blockIdx.x, threadIdx.x);
    for (int frame = 0; frame < numFrames; ++frame)
    {
        Recursion(inZmw[frame], frame + latentFrames);
    }

    // Compute the probabilities of the possible end states.  Propagate
    // them backwards a few frames, as some paths may converge and we
    // can have a more certain estimate.
    Normalize(logLike);
    auto& prob = scratch;
    const int lookStart = numFrames + latentFrames - 1;
    const int lookStop = lookStart - ViterbiStitchLookback;
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
            newProb[prev.x] += Blend(make_short2(1,0), prob[state], zero);
            newProb[prev.y] += Blend(make_short2(0,1), prob[state], zero);
        }
        prob = newProb;
    }

    PBHalf2 maxProb = prob[0];
    short2 anchorState = {0,0};
    #pragma unroll 1
    for (int i = 1; i < numStates; ++i)
    {
        auto cond = maxProb > prob[i];
        maxProb = Blend(cond, maxProb, prob[i]);
        anchorState = Blend(cond, anchorState, make_short2(i, i));
    }

    // Traceback
    auto traceState = anchorState;
    auto outZmw = output.ZmwData(blockIdx.x, threadIdx.x);
    const int stopFrame = latentFrames == 0 ? ViterbiStitchLookback : 0;
    const int startFrame = numFrames - 1;
    for (int frame = startFrame; frame >= stopFrame; --frame)
    {
        outZmw[frame] = traceState;
        const auto lookbackIdx = frame + latentFrames - ViterbiStitchLookback;
        traceState.x = labels(lookbackIdx, traceState.x).x;
        traceState.y = labels(lookbackIdx, traceState.y).y;
    }

    // Update latent data
    latent.SetBoundary(anchorState);
    latent.SetModel(models[blockIdx.x]);
    latent.SetData(inZmw);
}

void FrameLabeler::ProcessBatch(const Memory::UnifiedCudaArray<LaneModelParameters<PBHalf, 64>>& models,
                                const Mongo::Data::BatchData<int16_t>& input,
                                Mongo::Data::BatchData<int16_t>& output)
{
    auto labels = BorrowScratch();

    FrameLabelerKernel<<<lanesPerPool_, BlockThreads>>>(trans_->GetDevicePtr(),
                                                        models.GetDeviceHandle(),
                                                        input,
                                                        latent_.GetDeviceView(),
                                                        *labels,
                                                        output);
    ReturnScratch(std::move(labels));
}

}}
