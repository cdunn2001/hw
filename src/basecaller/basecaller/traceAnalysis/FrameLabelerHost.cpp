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

#include <tbb/task_arena.h>
#include <tbb/parallel_for.h>

#include <common/AlignedVector.h>
#include <common/cuda/memory/UnifiedCudaArray.h>
#include <common/LaneArray.h>

#include <dataTypes/configs/AnalysisConfig.h>
#include <dataTypes/configs/BasecallerFrameLabelerConfig.h>

#include <basecaller/traceAnalysis/SubframeLabelManager.h>
#include <basecaller/traceAnalysis/SubframeScorer.h>
#include <basecaller/traceAnalysis/RoiHost.h>


using namespace PacBio::Cuda;
using namespace PacBio::Cuda::Memory;
using namespace PacBio::Cuda::Utility;
using namespace PacBio::Mongo;
using namespace PacBio::Mongo::Basecaller;
using namespace PacBio::Mongo::Data;

namespace {

template <typename RFT>
struct __align__(128) LatentViterbi
{
    static_assert(std::is_base_of<IRoiFilter, RFT>::value, "RFT must inherit from IRoiFilter");

 public:
    LatentViterbi()
        : labelsBoundary_(0)
        , roiBoundary_(0)
    {
        for (auto& val : latentTrc_) val = 0;

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

    void SetLabelBoundary(const FrameLabelerHost::LabelArray& boundary) { labelsBoundary_ = boundary; }
    const FrameLabelerHost::LabelArray& GetLabelBoundary() const { return labelsBoundary_; }

    void SetRoiBoundary(const FrameLabelerHost::UShortArray& boundary) { roiBoundary_ = boundary; }
    const FrameLabelerHost::UShortArray& GetRoiBoundary() const { return roiBoundary_; }

    const FrameLabelerHost::LaneModelParameters& GetModel() const { return oldModel; }
    void SetModel(const FrameLabelerHost::LaneModelParameters& model)
    {
        oldModel = model;
    }

    const CudaArray<typename RFT::LabelArray, RFT::lookBack>& GetLatentTraces() const
    {
        return latentTrc_;
    }

    void SetLatentTraces(BlockView<const int16_t>::ConstIterator trIt)
    {
        for (int idx = RFT::lookBack - 1; idx >= 0; --idx, --trIt)
        {
            latentTrc_[idx] = trIt.Extract();
        }
    }

private:
    FrameLabelerHost::LaneModelParameters oldModel;
    FrameLabelerHost::LabelArray labelsBoundary_;
    FrameLabelerHost::UShortArray roiBoundary_;
    CudaArray<typename RFT::LabelArray, RFT::lookBack> latentTrc_;
};

// Class that stores compressed "traceback" information for use during
// the viterbi algorthm.  You can conceptually think of this as
// an array that is `numLabels` long.  It is only valid to store
// values in this array between [0-numLabels).
template <size_t laneWidth, size_t numLabels>
class PackedLabels
{
    // Give each ZMW a 32 bit work for storing compressed data
    using PackedWord = LaneArray<uint32_t, laneWidth>;
    using LabelLane = LaneArray<uint16_t, laneWidth>;
public:

    PackedLabels()
        : data_{}
    {}

    void SetSlot(int idx, const LabelLane& labels)
    {
        const auto largeIdx = idx / LabelsPerWord();
        const auto smallIdx = idx % LabelsPerWord();

        const PackedWord ZeroMask = ~(((1 << BitsPerLabel()) - 1) << (smallIdx * BitsPerLabel()));
        PackedWord payload = labels;
        payload = payload << (smallIdx * BitsPerLabel());
        data_[largeIdx] &= ZeroMask;
        data_[largeIdx] |= payload;
    }

    LabelLane GetSlot(const LabelLane& idx) const
    {
        const auto largeIdx = idx / LabelsPerWord();
        const auto smallIdx = idx & ((1 << BitsPerLabel()) - 1);

        PackedWord ExtractMask = ((1 << BitsPerLabel()) - 1);
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

    std::array<PackedWord, ArrayLen()> data_;
};

// Accepting by value because we need a copy anyway
template <typename VF, size_t numStates>
CudaArray<VF, numStates> Normalize(CudaArray<VF, numStates> vec)
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
    auto invSum = 1 / sum;
    for (uint32_t i = 0; i < numStates; ++i)
    {
        vec[i] *= invSum;
    }
    return vec;
}

// Note: This is a copy/adaptation from the GPU implementation, which itself is a copy/adaptation
//       from Sequel.
//
// Here we do some template magic to iterate over a sparse matrix where the sparsity pattern is
// known at compile time.  It's a bit complicated, but it does come with significant performance
// savings.
template <typename PackedLabels, typename VF, typename VI, typename VU, typename Scorer, typename TransT, class... Rows>
void Recursion(PackedLabels& labels, CudaArray<VF, Subframe::numStates>& logLike, const Scorer& scorer,
               const SparseMatrix<TransT, Rows...>& trans, VI data, VU isRoi)
{
    CudaArray<VF, Subframe::numStates> logAccum;
    auto AddRow = [&](const VF& score, const float* rowData, auto* row)
    {
        // row parameter just used to extract the Row type.
        // I wouldn't even give it a name save we need to use
        // decltype to extract the type information within a lambda.
        using Row = std::remove_pointer_t<decltype(row)>;
        constexpr auto firstIdx = Row::firstIdx;
        constexpr auto nextState = Row::rowIdx;

        // Currently only handle Rows with a single Segment.  This can be
        // generalized to handle an arbitrary number of Segments, but here
        // we've coppied from the GPU implementation, and the GPU implementation
        // had a performance penalty when doing so.  I'm keeping the
        // implementations in sync for now, but that may change.
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

        // If we're not in the ROI, then all non-baseline states suffer
        // a "penalty" that should make them effectively impossible
        auto penalty = Blend((nextState == 0) | (isRoi != VU(0)), VF(0), VF(std::numeric_limits<float>::infinity()));
        logAccum[nextState] = maxVal - penalty;

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

// Performs the algorithm that determins if each frame is within the
// ROI.  The main gist of things are as follows:
//  * For each frame, we look at some window of baseline sigma normalized
//    traces to determine a value for that frame.  The precise window and
//    function used is controlled by the RoiFilter template parameter
//  * If the produced value is above the "high" threshold, then it is
//    determined to be ROI
//  * If the produced value is between the "high" and "low" thresholds, then
//    it is determined to be ROI *IFF* one of its neighbors is also ROI
//  * Since we might need to know if the next point is ROI before we can tell
//    if the curent point is ROI, we have to do both a forwards and backwards
//    traversal over the data.  On the forward pass we determine who is ROI from
//    the high threshold, and who is maybe ROI from the low threshold.  On the
//    backwards pass we then have enough information to make the final ROI
//    determination.
//  * The effect is that all consecutive frames above the low threshold are
//    ROI, as long as at least one of those points is also above the high threshold
template <typename RFT>
static void ComputeRoi(const CudaArray<typename RFT::LabelArray, RFT::lookBack>& latentTraces,
                       const FrameLabelerHost::UShortArray& roiBC,
                       BlockView<int16_t>& traces1,
                       const BlockView<const int16_t>& traces2,
                       const FrameLabelerHost::FloatArray& invSigma1,
                       const FrameLabelerHost::FloatArray& invSigma2,
                       BlockView<uint16_t>& roi)
{
    assert(roi.NumFrames() == traces1.NumFrames() + traces2.NumFrames() - RFT::lookForward);

    auto trIt1 = traces1.Begin();
    Roi::ForwardRecursion<RFT> forward(latentTraces, traces1, invSigma1, roiBC, roi);
    for (size_t i = RFT::lookForward; i < traces1.NumFrames(); ++i)
    {
        forward.ProcessNextFrame((trIt1 + i).Extract() * invSigma1);
    }

    auto trIt2 = traces2.CBegin();
    for (size_t i = 0; i < traces2.NumFrames(); ++i)
    {
        forward.ProcessNextFrame((trIt2 + i).Extract() * invSigma2);
    }

    Roi::BackwardRecursion<RFT> backward(roi);
    while (backward.FramesRemaining() > 0) backward.PopRoiState();
}

// Serial function that produces lane labels for a block of data.
template <typename RFT>
FrameLabelerHost::FloatArray
LabelBlock(const FrameLabeler::LaneModelParameters& model,
           const Subframe::TransitionMatrix<float>& trans,
           const BlockView<const int16_t> traceData,
           LatentViterbi<RFT>& latentData,
           BlockView<int16_t> prevLat,
           BlockView<int16_t> nextLat,
           BlockView<uint16_t> roiWorkspace,
           BlockView<int16_t> output)
{
    using VF = FrameLabelerHost::FloatArray;
    using VI = FrameLabelerHost::UShortArray;
    // When/if this changes, some of this kernel is going to have to be udpated or generalized
    static_assert(Subframe::numStates == 13,
                  "LabelBlock currently hard coded to only handle 13 states");
    using namespace Subframe;

    BlockStateSubframeScorer<VF> scorer;

    // Initial setup
    CudaArray<VF, numStates> logLike;
    auto bc = latentData.GetLabelBoundary();
    const auto numLatent = prevLat.NumFrames();
    const VF zero(0.0f);
    const VF ninf(-std::numeric_limits<float>::infinity());
    for (int i = 0; i < numStates; ++i)
    {
        logLike[i] = Blend(bc == i, zero, ninf);
    }

    // We need to have enough space to compute the ROI for all frames we will emit in this block,
    // plus enough extra to cover the viterbi lookback process, *plus* yet a little more extdra
    // so handle the viterbi stitching process (which really is just having a few extra frames
    // to lessen the impact of an arbitrary RHS boundary condition)
    assert(roiWorkspace.NumFrames() == traceData.NumFrames() + ViterbiStitchLookback + RFT::stitchFrames);
    ComputeRoi<RFT>(latentData.GetLatentTraces(),
                    latentData.GetRoiBoundary(),
                    prevLat,
                    traceData,
                    1 / sqrt(VF::FromArray(latentData.GetModel().BaselineMode().vars)),
                    1 / sqrt(VF::FromArray(model.BaselineMode().vars)),
                    roiWorkspace
    );

    auto labels = std::vector<PackedLabels<laneSize, numStates>>(traceData.NumFrames() + ViterbiStitchLookback);

    // Forward recursion on latent data
    auto latIt = prevLat.Begin();
    auto roiIt = roiWorkspace.Begin();
    scorer.Setup(latentData.GetModel());
    for (size_t frame = 0; frame < numLatent; ++latIt, ++roiIt, ++frame)
    {
        Recursion(labels[frame], logLike, scorer, trans, latIt.Extract(), roiIt.Extract());
    }

    // Forward recursion on this block's data
    scorer.Setup(model);
    auto trIt = traceData.CBegin();
    const size_t numFrames = traceData.NumFrames();
    const size_t anchor = numFrames - numLatent;
    for (size_t frame = 0; frame < anchor; ++trIt, ++roiIt, ++frame)
    {
        Recursion(labels[frame + numLatent], logLike, scorer, trans, trIt.Extract(), roiIt.Extract());
    }

    auto anchorLogLike = logLike;

    for (size_t frame = anchor; frame < anchor + ViterbiStitchLookback; ++trIt, ++roiIt, ++frame)
    {
        Recursion(labels[frame + numLatent], logLike, scorer, trans, trIt.Extract(), roiIt.Extract());
    }

    // Compute the probabilities of the possible end states. Propagate
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
    for (int frame = int(numFrames - 1); frame >= 0; --frame, --outItr)
    {
        outItr.Store(traceState);
        traceState = labels[frame].GetSlot(traceState);
    }

    // Update latent data
    latentData.SetRoiBoundary((roiWorkspace.CBegin() + numFrames - 1).Extract());
    latentData.SetLabelBoundary(anchorState);
    latentData.SetModel(model);
    latentData.SetLatentTraces(traceData.CBegin() + anchor - 1);

    latIt = nextLat.Begin();
    trIt = traceData.CBegin() + anchor;
    for (size_t i = 0; i < numLatent; ++i, latIt++, trIt++)
    {
        latIt.Store(trIt.Extract());
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

struct FrameLabelerHost::ILabelImpl
{
    virtual void ProcessBatch(const PoolModelParameters& models,
                              const BatchData<int16_t>& traceData,
                              BatchData<int16_t>& latOut,
                              BatchData<int16_t>& output,
                              FrameLabelerMetrics& metricsOutput) = 0;

    virtual ~ILabelImpl() = default;
};

static Subframe::TransitionMatrix<float> transHost;
extern RoiThresholdsHost roiThreshHost;

template <typename RFT>
class LabelerImpl : public FrameLabelerHost::ILabelImpl
{
    static_assert(std::is_base_of<IRoiFilter, RFT>::value, "RFT must inherit from IRoiFilter");

    static BatchDimensions LatBatchDims(size_t lanesPerPool)
    {
        BatchDimensions ret;
        ret.framesPerBatch = ViterbiStitchLookback + RFT::lookForward + RFT::stitchFrames;
        ret.laneWidth = laneSize;
        ret.lanesPerBatch = lanesPerPool;
        return ret;
    }

public:
    LabelerImpl(size_t lanesPerPool)
        : latent_(lanesPerPool, SyncDirection::HostWriteDeviceRead, SOURCE_MARKER())
        , prevLat_(LatBatchDims(lanesPerPool), SyncDirection::HostWriteDeviceRead, SOURCE_MARKER())
        , numLanes_(lanesPerPool)
    {
        tbb::task_arena().execute([&] {
            tbb::parallel_for((uint32_t) {0}, numLanes_, [&](unsigned int lane) {
                auto view = prevLat_.GetBlockView(lane);
                for (auto itr = view.Begin(); itr != view.End(); ++itr)
                {
                    itr.Store(0);
                }
            });
        });
    }

    LabelerImpl(const LabelerImpl&) = delete;
    LabelerImpl(LabelerImpl&&) = default;
    LabelerImpl& operator=(const LabelerImpl&) = delete;
    LabelerImpl& operator=(LabelerImpl&&) = default;

    void ProcessBatch(const FrameLabelerHost::PoolModelParameters& models,
                      const BatchData<int16_t>& traceData,
                      BatchData<int16_t>& latOut,
                      BatchData<int16_t>& output,
                      FrameLabelerMetrics& metricsOutput)
    {
        auto modelView = models.GetHostView();
        auto metricsView = metricsOutput.viterbiScore.GetHostView();

        BatchDimensions roiDims;
        roiDims.lanesPerBatch  = traceData.LanesPerBatch();
        roiDims.framesPerBatch = traceData.NumFrames() + ViterbiStitchLookback + RFT::stitchFrames;

        Data::BatchData<uint16_t> roiWorkspace(roiDims, SyncDirection::HostWriteDeviceRead, SOURCE_MARKER());

        tbb::task_arena().execute([&] {
            tbb::parallel_for((uint32_t) {0}, numLanes_, [&](unsigned int lane) {
                metricsView[lane] = LabelBlock<RFT>(modelView[lane],
                                               transHost,
                                               traceData.GetBlockView(lane),
                                               latent_.GetHostView()[lane],
                                               prevLat_.GetBlockView(lane),
                                               latOut.GetBlockView(lane),
                                               roiWorkspace.GetBlockView(lane),
                                               output.GetBlockView(lane));
            });
        });
        std::swap(prevLat_, latOut);
    }

private:
    UnifiedCudaArray<LatentViterbi<RFT>, true> latent_;
    BatchData<int16_t> prevLat_;
    uint32_t numLanes_;
};

void FrameLabelerHost::Configure(const Data::AnalysisConfig& analysisConfig,
                                 const Data::BasecallerFrameLabelerConfig& labelerConfig)
{
    auto roiLatency = 0;
    roiType = labelerConfig.roi.filterType;
    switch (roiType)
    {
        case Data::BasecallerRoiConfig::RoiFilterType::RawEnum::Default:
        {
            roiLatency = RoiFilterDefault::lookForward
                       + RoiFilterDefault::stitchFrames;
            break;
        }
        case Data::BasecallerRoiConfig::RoiFilterType::RawEnum::NoOp:
        {
            roiLatency = RoiFilterNoOp::lookForward
                       + RoiFilterNoOp::stitchFrames;
            break;
        }
        default:
            throw PBException("Unsupported roi filter type");
    }
    const auto hostExecution = true;
    InitFactory(hostExecution, ViterbiStitchLookback + roiLatency);

    auto movieInfo = analysisConfig.movieInfo;
    transHost = Subframe::TransitionMatrix<float>(movieInfo.analogs, labelerConfig.viterbi, movieInfo.frameRate);
    roiThreshHost.upperThreshold = FloatArray(labelerConfig.roi.upperThreshold);
    roiThreshHost.lowerThreshold = FloatArray(labelerConfig.roi.lowerThreshold);
}

void FrameLabelerHost::Finalize()
{
}

FrameLabelerHost::FrameLabelerHost(uint32_t poolId,
                                   uint32_t lanesPerPool)
    : FrameLabeler(poolId)
{
    switch (roiType)
    {
        case Data::BasecallerRoiConfig::RoiFilterType::RawEnum::Default:
            labeler_ = std::make_unique<LabelerImpl<RoiFilterDefault>>(lanesPerPool);
            break;
        case Data::BasecallerRoiConfig::RoiFilterType::RawEnum::NoOp:
            labeler_ = std::make_unique<LabelerImpl<RoiFilterNoOp>>(lanesPerPool);
            break;
        default:
            throw PBException("Unsupported roi filter type");
    }
}

FrameLabelerHost::~FrameLabelerHost() {}

std::pair<Data::LabelsBatch, Data::FrameLabelerMetrics>
FrameLabelerHost::Process(Data::TraceBatch<Data::BaselinedTraceElement> trace,
                          const PoolModelParameters& models)
{
    auto ret = batchFactory_->NewBatch(std::move(trace));

    labeler_->ProcessBatch(models,
                        ret.first.TraceData(),
                        ret.first.LatentTrace(),
                        ret.first,
                        ret.second);

    // Update the trace data so downstream filters can't see the held back portion
    ret.first.TraceData().SetFrameLimit(ret.first.NumFrames() - ret.first.LatentTrace().NumFrames());

    return ret;
}

}}}

