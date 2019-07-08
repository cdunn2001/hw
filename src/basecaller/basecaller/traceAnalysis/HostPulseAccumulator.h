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

#include <common/LaneArray.h>
#include <common/LaneArrayRef.h>
#include <common/LaneMask.h>
#include <dataTypes/Pulse.h>

#include <basecaller/traceAnalysis/PulseAccumulator.h>

namespace PacBio {
namespace Mongo {
namespace Basecaller {

class HostPulseAccumulator : public PulseAccumulator
{
    using ConstLabelArrayRef = ConstLaneArrayRef<Data::LabelsBatch::ElementType>;
    using ConstSignalArrayRef = ConstLaneArrayRef<Data::CameraTraceBatch::ElementType>;
    using SignalBlockView = Data::BlockView<Data::CameraTraceBatch::ElementType>;
    using ConstSignalBlockView = Data::BlockView<const Data::CameraTraceBatch::ElementType>;

public:     // Static functions
    static void Configure(size_t maxCallsPerZmw);
    static void Finalize();

public:
    HostPulseAccumulator(uint32_t poolId, uint32_t lanesPerBatch);
    ~HostPulseAccumulator() override;

public:
    class LabelsSegment
    {
    public:
        using FrameArray = LaneArray<uint32_t>;
        using SignalArray = LaneArray<Data::CameraTraceBatch::ElementType>;
        using LabelArray = LaneArray<Data::LabelsBatch::ElementType>;

    public:     // Static constants
        /// Number nucleotide analogs supported.
        static constexpr int numAnalogs = 4;

        /// Number of states in HMM.
        static constexpr int numStates = 1 + 3*numAnalogs;
    public:

        static LaneMask<laneSize> IsPulseUpState(const ConstLabelArrayRef& i)
        { return (i > numAnalogs) & (i <= 2*numAnalogs); }

        static LaneMask<laneSize> IsPulseDownState(const ConstLabelArrayRef& i)
        { return (i > 2*numAnalogs) & (i < numStates); }

        static LabelArray ToFullFrame(const ConstLabelArrayRef& label)
        {
            LabelArray ret = label;
            ret = Blend(IsPulseDownState(label), label - 2*numAnalogs, ret);
            ret = Blend(IsPulseUpState(label), label - numAnalogs, ret);
            return ret;
        }

    public:
        LabelsSegment(const FrameArray& startFrame, const ConstLabelArrayRef& label, const ConstSignalArrayRef& signal)
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
            , label_(0)
        { }

    public:
        LaneMask<laneSize> IsNewSegment(const ConstLabelArrayRef& label)
        {
            return IsPulseUpState(label) | ((label == 0) & (this->label_ != 0));
        }
        
        LaneMask<laneSize> IsPulse()
        {
            return label_ != 0;
        }

        void ResetSegment(const LaneMask<laneSize>& boundaryMask, uint32_t frameIndex,
                          const ConstLabelArrayRef& label, const ConstSignalArrayRef& signal)
        {
            startFrame_ = Blend(boundaryMask, frameIndex, startFrame_);
            endFrame_ = Blend(boundaryMask, 0, endFrame_);
            signalFrstFrame_ = Blend(boundaryMask, signal, signalFrstFrame_);
            signalLastFrame_ = Blend(boundaryMask, 0, signalLastFrame_);
            signalMax_ = Blend(boundaryMask, signal, signalMax_);
            signalTotal_ = Blend(boundaryMask, 0, signalTotal_);
            signalM2_ = Blend(boundaryMask, 0, signalM2_);
            label_ = Blend(boundaryMask, label, label_);
        }
        
        void AddSignal(const LaneMask<laneSize>& update, const ConstSignalArrayRef& signal)
        {
            signalTotal_ = Blend(update, signalTotal_ + signalLastFrame_, signalTotal_);
            signalM2_ = Blend(update, signalM2_ + (signalLastFrame_ * signalLastFrame_), signalM2_);
            signalLastFrame_ = Blend(update, signal, signalLastFrame_);
            signalMax_ = Blend(update, max(signalMax_, signal), signalMax_);
        }

    private:
        FrameArray  startFrame_;        // Needed because of partial segments
        FrameArray  endFrame_;          // 1 + the last frame included in the segment

        SignalArray signalFrstFrame_;   // Signal of the most recent frame added
        SignalArray signalLastFrame_;   // Signal recorded for the last frame in the segment
        SignalArray signalMax_;         // Max signal over all frames in segment
        SignalArray signalTotal_;       // Signal total, excluding the first and last frame
        SignalArray signalM2_;          // Sum of squared signals, excluding the first and last frame

        LabelArray  label_;             // // Internal label ID corresponding to detection modes
    };

private:
    Data::PulseBatch Process(Data::LabelsBatch trace) override;

    void EmitFrameLabels(LabelsSegment& currSegment, Data::LaneVectorView<Data::Pulse>& pulses,
                         const ConstLabelArrayRef& label,
                         const SignalBlockView& blockLatTrace, const SignalBlockView& currTrace,
                         size_t relativeFrameIndex, uint32_t absFrameIndex);

    ConstSignalArrayRef Signal(size_t relativeFrameIndex,
                               const SignalBlockView& latTrace, const SignalBlockView& currTrace)
    {
        if (relativeFrameIndex < latTrace.NumFrames())
        {
            return ConstSignalArrayRef(latTrace.Data() + (relativeFrameIndex * latTrace.LaneWidth()));
        }
        else
        {
            return ConstSignalArrayRef(currTrace.Data() +
                                       ((relativeFrameIndex - latTrace.NumFrames()) * currTrace.LaneWidth()));
        }
    }

private:
    std::vector<LabelsSegment> startSegmentByLane;
};

}}} // namespace PacBio::Mongo::Basecaller

#endif // PACBIO_MONGO_BASECALLER_HOST_PULSE_ACCUMULATOR_H_
