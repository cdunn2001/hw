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

namespace PacBio {
namespace Cuda {

__global__ void FrameLabelerKernel(Memory::DevicePtr<Subframe::TransitionMatrix> trans,
                                   Memory::DeviceView<LaneModelParameters<32>> models,
                                   ViterbiData<PBHalf2, 32> recur,
                                   ViterbiData<short2, 32> labels,
                                   Mongo::Data::GpuBatchData<short2> input,
                                   Mongo::Data::GpuBatchData<short2> output)
{
    using namespace Subframe;

    static constexpr unsigned laneWidth = 32;
    static constexpr int numStates = 12;
    assert(blockDim.x == laneWidth);
    __shared__ BlockStateScorer<laneWidth> scorer;
    scorer.Setup(models[blockIdx.x]);

    // Forward recursion
    const int numFrames = input.Dims().framesPerBatch;
    auto inZmw = input.ZmwData(blockIdx.x, threadIdx.x);
    for (int frame = 0; frame < numFrames; ++frame)
    {
        auto scores = scorer.StateScores(PBHalf2(inZmw[frame]));
        for (int nextState = 0; nextState < numStates; ++nextState)
        {
            auto score = scores[nextState];
            // Is this transposed?
            auto maxVal = score + PBHalf2(trans->Entry(0, nextState)) + recur(frame, 0);
            auto maxIdx = make_short2(0,0);
            for (int prevState = 1; prevState < numStates; ++prevState)
            {
                auto val = score + PBHalf2(trans->Entry(prevState, nextState)) + recur(frame, prevState);

                auto cond = maxVal > val;
                maxVal = Blend(cond, maxVal, val);
                maxIdx = Blend(cond, maxIdx, make_short2(prevState, prevState));
            }
            recur(frame+1, nextState) = maxVal;
            labels(frame, nextState) = maxIdx;
        }
    }

    // Cheat for now, find it for real later
    const int anchorIdx = numFrames-1;
    auto maxVal = recur(anchorIdx, 0);
    short2 traceState = make_short2(0,0);
    for (int state = 1; state < numStates; ++state)
    {
        auto val = recur(anchorIdx, state);
        auto cond = maxVal > val;

        maxVal = Blend(cond, maxVal, val);
        traceState = Blend(cond, traceState, make_short2(state, state));
    }

    // Traceback
    auto outZmw = output.ZmwData(blockIdx.x, threadIdx.x);
    for (int frame = numFrames-1; frame >= 0; --frame)
    {
        outZmw[frame] = traceState;
        traceState.x = labels(frame, traceState.x).x;
        traceState.y = labels(frame, traceState.x).y;
    }

    // Set the new boundary conditions for next block
    for (int state = 0; state < numStates; ++state)
    {
        recur(0, state) = PBHalf2(0);
    }

}

}}
