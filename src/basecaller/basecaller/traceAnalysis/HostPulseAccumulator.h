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

#ifndef PACBIO_MONGO_BASECALLER_HOST_PULSE_ACCUMULATOR_H_
#define PACBIO_MONGO_BASECALLER_HOST_PULSE_ACCUMULATOR_H_

#include <vector>

#include <common/LaneArray.h>
#include <common/StatAccumulator.h>
#include <dataTypes/Pulse.h>
#include <dataTypes/BasicTypes.h>

#include <basecaller/traceAnalysis/PulseAccumulator.h>
#include <basecaller/traceAnalysis/SubframeLabelManager.h>

namespace PacBio {
namespace Mongo {
namespace Basecaller {

template <typename LabelManager>
class HostPulseAccumulator : public PulseAccumulator
{
    using LabelArray = LaneArray<Data::LabelsBatch::ElementType>;
    using SignalArray = LaneArray<Data::BaselinedTraceElement>;
    using SignalBlockView = Data::BlockView<Data::BaselinedTraceElement>;
    using ConstSignalBlockView = Data::BlockView<const Data::BaselinedTraceElement>;
    using FloatArray = LaneArray<float>;
    using BaselineStats = StatAccumulator<FloatArray>;

public:     // Static functions
    static void Configure(const Data::AnalysisConfig& analysisConfig,
                          const Data::BasecallerPulseAccumConfig& pulseConfig);
    static void Finalize();

public:
    HostPulseAccumulator(uint32_t poolId, uint32_t lanesPerBatch);
    ~HostPulseAccumulator() override;

public:
    class LabelsSegment
    {
    public:
        using FrameArray = LaneArray<uint32_t>;
        using FloatArray = LaneArray<float>;
        using SignalArray = LaneArray<Data::BaselinedTraceElement>;
        using LabelArray = LaneArray<Data::LabelsBatch::ElementType>;

    public:
        LabelsSegment(const FrameArray& startFrame, const LabelArray& label, const SignalArray& signal)
            : startFrame_(startFrame)
            , endFrame_(0)
            , signalFrstFrame_(signal)
            , signalLastFrame_(0)
            , signalMax_(signal)
            , signalTotal_(0)
            , signalM2_(0)
            , label_(label)
        { }

        LabelsSegment()
            : startFrame_(0)
            , endFrame_(0)
            , signalFrstFrame_(0)
            , signalLastFrame_(0)
            , signalMax_(0)
            , signalTotal_(0)
            , signalM2_(0)
            , label_(LabelManager::BaselineLabel())
        { }

    public:
        LaneMask<> IsNewSegment(const LabelArray& label) const
        {
            return LabelManager::IsNewSegment(this->label_, label);
        }

        LaneMask<> IsPulse() const
        {
            return label_ != LabelArray{LabelManager::BaselineLabel()};
        }

        Data::Pulse ToPulse(uint32_t frameIndex, uint32_t zmw, float meanAmp, const LabelManager& manager)
        {
            const float maxSignal = Data::Pulse::SignalMax();
            const float minSignal = 0.0f;

            endFrame_ = frameIndex;
            auto startFrameZmw  = MakeUnion(startFrame_)[zmw];
            auto signalTotalZmw = MakeUnion(signalTotal_)[zmw];
            auto sigFrstFrame   = MakeUnion(signalLastFrame_)[zmw];
            auto sigLastFrame   = MakeUnion(signalFrstFrame_)[zmw];
            int width = frameIndex - startFrameZmw;

            float raw_mean = float(signalTotalZmw + sigFrstFrame + sigLastFrame) / width;
            float raw_mid  = float(signalTotalZmw) / (width - 2);

            auto lowAmp = meanAmp;
            bool keep = (width >= widthThresh_) || (raw_mean * width >= ampThresh_ * lowAmp);

            using std::min;
            using std::max;

            Data::Pulse pls{};
            pls.Start(startFrameZmw)
                .Width(width)
                .MeanSignal(min(maxSignal, max(minSignal, raw_mean)))
                .MidSignal(width < 3 ? 0.0f : min(maxSignal, max(minSignal, raw_mid)))
                .MaxSignal(min(maxSignal, max(minSignal, static_cast<float>(MakeUnion(signalMax_)[zmw]))))
                .SignalM2(MakeUnion(signalM2_)[zmw])
                .Label(manager.Nucleotide(MakeUnion(label_)[zmw]))
                .IsReject(!keep);

            return pls;
        }

        void ResetSegment(const LaneMask<laneSize>& boundaryMask, uint32_t frameIndex,
                          const LabelArray& label, const SignalArray& signal)
        {
            startFrame_ = Blend(boundaryMask, FrameArray{frameIndex}, startFrame_);
            endFrame_ = Blend(boundaryMask, FrameArray{0}, endFrame_);
            signalFrstFrame_ = Blend(boundaryMask, signal, signalFrstFrame_);
            signalLastFrame_ = Blend(boundaryMask, SignalArray{0}, signalLastFrame_);
            signalMax_ = Blend(boundaryMask, signal, signalMax_);
            signalTotal_ = Blend(boundaryMask, SignalArray{0}, signalTotal_);
            signalM2_ = Blend(boundaryMask, FloatArray{0}, signalM2_);
            label_ = Blend(boundaryMask, label, label_);
        }

        void AddSignal(const LaneMask<laneSize>& update, const SignalArray& signal)
        {
            signalTotal_ = Blend(update, signalTotal_ + signalLastFrame_, signalTotal_);
            signalM2_ = Blend(update, signalM2_ + signalLastFrame_ * signalLastFrame_, signalM2_);
            signalLastFrame_ = Blend(update, signal, signalLastFrame_);
            signalMax_ = Blend(update, max(signalMax_, SignalArray{signal}), signalMax_);
        }

    private:
        FrameArray  startFrame_;        // Needed because of partial segments
        FrameArray  endFrame_;          // 1 + the last frame included in the segment

        SignalArray signalFrstFrame_;   // Signal of the most recent frame added
        SignalArray signalLastFrame_;   // Signal recorded for the last frame in the segment
        SignalArray signalMax_;         // Max signal over all frames in segment
        SignalArray signalTotal_;       // Signal total, excluding the first and last frame
        FloatArray  signalM2_;          // Sum of squared signals, excluding the first and last frame

        LabelArray  label_;             // // Internal label ID corresponding to detection modes
    };

private:
    std::pair<Data::PulseBatch, Data::PulseDetectorMetrics>
    Process(Data::LabelsBatch trace, const PoolModelParameters&) override;

    void EmitFrameLabels(LabelsSegment& currSegment, Data::LaneVectorView<Data::Pulse>& pulses,
                         BaselineStats& baselineStats,
                         const LabelArray& label, const SignalBlockView& blockLatTrace,
                         const SignalBlockView& currTrace, const FloatArray& meanAmp,
                         size_t relativeFrameIndex, uint32_t absFrameIndex);

    SignalArray Signal(size_t relativeFrameIndex, const SignalBlockView& latTrace,
                               const SignalBlockView& currTrace) const
    {
        auto itr = relativeFrameIndex < latTrace.NumFrames()
            ?  latTrace.CBegin() + relativeFrameIndex
            :  currTrace.CBegin() + (relativeFrameIndex - latTrace.NumFrames());

        return itr.Extract();
    }

private:
    std::vector<LabelsSegment> startSegmentByLane;
    static std::unique_ptr<LabelManager> manager_;
};

}}} // namespace PacBio::Mongo::Basecaller

#endif // PACBIO_MONGO_BASECALLER_HOST_PULSE_ACCUMULATOR_H_
