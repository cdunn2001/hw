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

#include <basecaller/traceAnalysis/FrameLabelerHost.h>

#include <common/AlignedVector.h>
#include <common/AlignedVector.h>
#include <common/cuda/memory/UnifiedCudaArray.h>
#include <common/LaneArray.h>
#include <dataTypes/configs/MovieConfig.h>

// BENTODO move from prototypes?
#include <prototypes/FrameLabeler/SubframeScorer.cuh>
#include <basecaller/traceAnalysis/SubframeLabelManager.h>
#include <tbb/parallel_for.h>

using namespace PacBio::Cuda;
using namespace PacBio::Cuda::Memory;
using namespace PacBio::Cuda::Utility;
using namespace PacBio::Mongo;
using namespace PacBio::Mongo::Data;

namespace {

template <size_t laneWidth>
struct __align__(128) LatentViterbi
{
    using ModelParams = LaneModelParameters<PBHalf, laneWidth>;
    using LaneLabel = LaneArray<int16_t, laneWidth>;
 public:
    LatentViterbi()
        : boundary_(0)
    {
        // I'm a little uncertain how to initialize this model.  I'm tryin to
        // make it work with latent data that we are guaranteed to be all zeroes,
        // and needs to produce baseline labels.
        //
        // I'm not even sure we need this `oldModel` it may be sufficient to use
        // the current block's model while processing the few frames of latent data
        //
        // In either case, any issues here are flushed out after only a few frames
        // (16) and we hit steady state, so I'm not hugely concerned.
        oldModel.BaselineMode().means = 0.f;
        oldModel.BaselineMode().vars = 50.f;

        for (uint32_t a = 0; a < numAnalogs; ++a)
        {
            oldModel.AnalogMode(a).means = 1000.f;
            oldModel.AnalogMode(a).vars = 10.f;
        }
    }

    void SetBoundary(const LaneLabel& boundary) { boundary_ = boundary; }
    const LaneLabel& GetBoundary() const { return boundary_; }

    const ModelParams& GetModel() const { return oldModel; }
    void SetModel(const ModelParams& model)
    {
        oldModel = model;
    }

private:
    ModelParams oldModel;
    LaneLabel boundary_;
};

// BENTODO comment
template <size_t laneWidth, size_t numLabels>
class PackedLabels
{
    using PackedLane = LaneArray<uint32_t, laneWidth>;
    using LabelLane = LaneArray<uint16_t, laneWidth>;
public:

    PackedLabels()
        : data_{}
    {}

    void SetSlot(int idx, const LabelLane& labels)
    {
        const auto largeIdx = idx / LabelsPerWord();
        const auto smallIdx = idx % LabelsPerWord();

        const PackedLane ZeroMask = ~(((1 << BitsPerLabel()) - 1) << (smallIdx * BitsPerLabel()));
        PackedLane tmp = labels;
        tmp = tmp << (smallIdx * BitsPerLabel());
        data_[largeIdx] &= ZeroMask;
        data_[largeIdx] |= tmp;
    }

    LabelLane GetSlot(const LabelLane& idx) const
    {
        const auto largeIdx = idx / LabelsPerWord();
        const auto smallIdx = idx & ((1 << BitsPerLabel()) - 1);

        PackedLane ExtractMask = ((1 << BitsPerLabel()) - 1);
        ExtractMask = ExtractMask << (smallIdx * BitsPerLabel());

        LabelLane ret;
        for (uint32_t i = 0; i < ArrayLen(); ++i)
        {
            LabelLane tmp = (data_[i] & ExtractMask) >> (smallIdx * BitsPerLabel());
            ret = Blend(i == largeIdx, tmp, ret);
        }
        return ret;
    }
private:
    static constexpr uint32_t BitsPerLabel()
    {
        int curr = numLabels-1;
        int ret = 0;
        while (curr > 0)
        {
            curr /= 2;
            ret += 1;
        }
        return ret;
    }
    static constexpr uint32_t LabelsPerWord()
    {
        return 32 / BitsPerLabel();
    }
    static constexpr size_t ArrayLen()
    {
        return (numLabels + LabelsPerWord() - 1) / LabelsPerWord();
    }
    // BENTODO kill
    static_assert(ArrayLen() == 2, "");
    std::array<PackedLane, ArrayLen()> data_;
};

template <typename VF, size_t numStates>
auto Normalize(CudaArray<VF, numStates> vec)
{
    auto maxVal = vec[0];
    for (uint32_t i = 1; i < numStates; ++i)
    {
        maxVal = max(vec[i], maxVal);
    }
    auto sum = VF(0.0f);
    for (uint32_t i = 0; i < numStates; ++i)
    {
        vec[i] = exp(vec[i] - maxVal);
        sum += vec[i];
    }
    for (uint32_t i = 0; i < numStates; ++i)
    {
        vec[i] /= sum;
    }
    return vec;
}

// BENTODO update comments, some are still cuda specific
template <typename PackedLabels, typename VF, typename VI, typename Scorer, typename TransT, class... Rows>
void Recursion(PackedLabels& labels, CudaArray<VF, Subframe::numStates>& logLike, const Scorer& scorer,
               const SparseMatrix<TransT, Rows...>& trans, VI data)
{
    CudaArray<VF, Subframe::numStates> logAccum;
    auto AddRow = [&](const VF& score,
                      const float* rowData,
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

        auto maxVal = score + VF(rowData[0]) + logLike[firstIdx];
        auto maxIdx = VI(firstIdx);

        uint32_t dataIndex = Segment::dataIndex;
        for (size_t prevState = Segment::firstCol; prevState < Segment::lastCol; ++prevState, dataIndex++)
        {
            auto val = score + VF(rowData[dataIndex]) + logLike[prevState];

            auto cond = val >= maxVal;
            maxVal = Blend(cond, val, maxVal);
            maxIdx = Blend(cond, prevState, maxIdx);
        }
        constexpr auto nextState = Row::rowIdx;
        logAccum[nextState] = maxVal;

        // Always slot new entries into the back, and after 4 inserts it will be fully populated
        // and ready for storage.  This approach has empirically been observed to be faster than
        // trying to slot data directly into it's desired final location, as the current version
        // can be done without any runtime dependance on the value of nextState.
        labels.SetSlot(nextState, maxIdx);
    };

    // Compile time loop, to loop over all the Rows in our sparse matrix (each of which have
    // a different type)
    auto loop = {(
        AddRow(scorer.StateScores(data, Rows::rowIdx),
               trans.RowData(Rows::rowIdx),
               (Rows*){nullptr}),0
    )...};
    (void)loop;
    logLike = logAccum;
}

LaneArray<float, laneSize>
LabelBlock(const LaneModelParameters<PBHalf, laneSize>& models,
           const Subframe::TransitionMatrix<float>& trans,
           const BlockView<const int16_t> input,
           LatentViterbi<laneSize>& latentData,
           BlockView<int16_t> prevLat,
           BlockView<int16_t> nextLat,
           BlockView<int16_t> output)
{
    using VF = LaneArray<float, laneSize>;
    using VI = LaneArray<uint16_t, laneSize>;
    // When/if this changes, some of this kernel is going to have to be udpated or generalized
    static_assert(Subframe::numStates == 13,
                  "LabelBlock currently hard coded to only handle 13 states");
    using namespace Subframe;

    BlockStateSubframeScorer<LaneArray<float, laneSize>> scorer;

    // Initial setup
    CudaArray<VF, numStates> logLike;
    auto bc = latentData.GetBoundary();
    const VF zero(0.0f);
    const VF ninf(-std::numeric_limits<float>::infinity());
    for (int i = 0; i < numStates; ++i)
    {
        logLike[i] = Blend(bc == i, zero, ninf);
    }

    auto labels = std::vector<PackedLabels<laneSize, numStates>>(input.NumFrames() + ViterbiStitchLookback);

    // Forward recursion on latent data
    auto frame = 0;
    {
        scorer.Setup(latentData.GetModel());
        for (auto itr = prevLat.Begin(); itr != prevLat.End(); ++itr, ++frame)
        {
            Recursion(labels[frame], logLike, scorer, trans, itr.Extract());
        }
    }

    // Forward recursion on this block's data
    scorer.Setup(models);
    const int numFrames = input.NumFrames();
    decltype(logLike) anchorLogLike;
    for (auto itr = input.CBegin(); itr != input.CEnd(); ++itr, ++frame)
    {
        Recursion(labels[frame], logLike, scorer, trans, itr.Extract());
        if (frame == numFrames-1) anchorLogLike = logLike;
    }

    // Compute the probabilities of the possible end states.  Propagate
    // them backwards a few frames, as some paths may converge and we
    // can have a more certain estimate.
    auto prob = Normalize(logLike);
    const int lookStart = numFrames + ViterbiStitchLookback - 1;
    const int lookStop = numFrames - 1;
    for (int i = lookStart; i > lookStop; --i)
    {
        CudaArray<VF, numStates> newProb;
        for (short state = 0; state < numStates; ++state)
        {
            newProb[state] = VF(0.0f);
        }
        const auto& packedLabels = labels[i];
        for (uint16_t from = 0; from < numStates; ++from)
        {
            auto prev = packedLabels.GetSlot(from);
            for (uint16_t to = 0; to < numStates; ++to)
            {
                newProb[to] += Blend(to == prev, prob[from], zero);
            }
        }
        prob = newProb;
    }

    VF maxProb = prob[0];
    VI anchorState(0u);
    for (int i = 1; i < numStates; ++i)
    {
        auto cond = maxProb > prob[i];
        maxProb = Blend(cond, maxProb, prob[i]);
        anchorState = Blend(cond, anchorState, VI(i));
    }

    // Traceback
    auto traceState = anchorState;
    auto outItr = output.End() - 1;
    for (int frame = numFrames - 1; frame >= 0; --frame, --outItr)
    {
        outItr.Store(traceState);
        traceState = labels[frame].GetSlot(traceState);
    }

    // Update latent data
    latentData.SetBoundary(anchorState);
    latentData.SetModel(models);

    auto latItr = nextLat.Begin();
    auto inItr = input.CEnd() - ViterbiStitchLookback;
    for (uint32_t i = 0; i < ViterbiStitchLookback; ++i, latItr++, inItr++)
    {
        latItr.Store(inItr.Extract());
    }

    VF ret;
    for (uint16_t i = 0; i < numStates; ++i)
    {
        ret = Blend(anchorState == i, anchorLogLike[i], ret);
    }

    return ret;
}


}

namespace PacBio {
namespace Mongo {
namespace Basecaller {

class FrameLabelerHost::Impl
{
    static BatchDimensions LatBatchDims(size_t lanesPerPool)
    {
        BatchDimensions ret;
        ret.framesPerBatch = ViterbiStitchLookback;
        ret.laneWidth = laneSize;
        ret.lanesPerBatch = lanesPerPool;
        return ret;
    }

public:

    // Helpers to provide scratch space data.  Used to pool allocations so we
    // only need enough to satisfy the current active batches, not one for
    // each possible pool.
    static void Configure(const std::array<Subframe::AnalogMeta, 4>& meta)
    {
        trans = Subframe::TransitionMatrix<float>(CudaArray<Subframe::AnalogMeta, 4>{meta});
    }

public:
    static void Finalize() {}

public:
    Impl(size_t lanesPerPool)
        : latent_(lanesPerPool, SyncDirection::HostWriteDeviceRead, SOURCE_MARKER())
        , prevLat_(LatBatchDims(lanesPerPool), SyncDirection::HostWriteDeviceRead, SOURCE_MARKER())
        , numLanes_(lanesPerPool)
    {

        tbb::parallel_for((uint32_t){0}, numLanes_, [&](unsigned int lane)
        {
            auto view = prevLat_.GetBlockView(lane);
            for (auto itr = view.Begin(); itr != view.End(); ++itr)
            {
                itr.Store(0);
            }
        });
    }

    Impl(const Impl&) = delete;
    Impl(Impl&&) = default;
    Impl& operator=(const Impl&) = delete;
    Impl& operator=(Impl&&) = default;


   void ProcessBatch(const UnifiedCudaArray<LaneModelParameters>& models,
                     const BatchData<int16_t>& input,
                     BatchData<int16_t>& latOut,
                     BatchData<int16_t>& output,
                     FrameLabelerMetrics& metricsOutput)
    {
        auto metricsView = metricsOutput.viterbiScore.GetHostView();
        auto modelView = models.GetHostView();

        tbb::parallel_for((uint32_t){0}, numLanes_, [&](unsigned int lane)
        {
            metricsView[lane] = LabelBlock(modelView[lane],
                                           trans,
                                           input.GetBlockView(lane),
                                           latent_.GetHostView()[lane],
                                           prevLat_.GetBlockView(lane),
                                           latOut.GetBlockView(lane),
                                           output.GetBlockView(lane));
        });
        std::swap(prevLat_, latOut);
    }

private:
    UnifiedCudaArray<LatentViterbi<laneSize>, true> latent_;
    BatchData<int16_t> prevLat_;

    static Subframe::TransitionMatrix<float> trans;
    uint32_t numLanes_;
};

Subframe::TransitionMatrix<float> FrameLabelerHost::Impl::trans;

void FrameLabelerHost::Configure(const Data::MovieConfig& movieConfig)
{
    const auto hostExecution = true;
    InitFactory(hostExecution, ViterbiStitchLookback);

    std::array<Subframe::AnalogMeta, 4> meta;
    for (size_t i = 0; i < meta.size(); i++)
    {
        meta[i].ipdSSRatio = movieConfig.analogs[i].ipd2SlowStepRatio;
        meta[i].ipd = movieConfig.frameRate * movieConfig.analogs[i].interPulseDistance;
        meta[i].pw = movieConfig.frameRate * movieConfig.analogs[i].pulseWidth;
        meta[i].pwSSRatio = movieConfig.analogs[i].pw2SlowStepRatio;
    }
    Impl::Configure(meta);
}

void FrameLabelerHost::Finalize()
{
    Impl::Finalize();
}

FrameLabelerHost::FrameLabelerHost(uint32_t poolId,
                                   uint32_t lanesPerPool)
    : FrameLabeler(poolId)
    , impl_(std::make_unique<Impl>(lanesPerPool))
{
}
FrameLabelerHost::~FrameLabelerHost() {}

std::pair<Data::LabelsBatch, Data::FrameLabelerMetrics>
FrameLabelerHost::Process(Data::TraceBatch<Data::BaselinedTraceElement> trace,
                          const PoolModelParameters& models)
{
    auto ret = batchFactory_->NewBatch(std::move(trace));

    impl_->ProcessBatch(models,
                        ret.first.TraceData(),
                        ret.first.LatentTrace(),
                        ret.first,
                        ret.second);

    // Update the trace data so downstream filters can't see the held back portion
    ret.first.TraceData().SetFrameLimit(ret.first.NumFrames() - ViterbiStitchLookback);

    return ret;
}

}}}

