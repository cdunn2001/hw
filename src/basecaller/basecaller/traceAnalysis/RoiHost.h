// Copyright (c) 2022, Pacific Biosciences of California, Inc.
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

#ifndef PACBIO_MONGO_BASECALLER_FRAME_ROI_HOST_H
#define PACBIO_MONGO_BASECALLER_FRAME_ROI_HOST_H

#include "FrameLabelerHost.h"

#include <common/LaneArray.h>
#include <common/cuda/memory/UnifiedCudaArray.h>

#include <dataTypes/BasicTypes.h>
#include <dataTypes/BatchData.h>
#include <dataTypes/configs/BasecallerFrameLabelerConfig.h>


using namespace PacBio::Mongo;
using namespace PacBio::Mongo::Data;

namespace PacBio::Mongo::Basecaller {

using namespace Cuda;

// Some configuration data for the whole process. Needs to
// be initialized before using (currently handled by the
// FrameLabelerDevice configuration process)
struct RoiThresholds
{
    FrameLabelerHost::FloatArray upperThreshold = FrameLabelerHost::FloatArray(0);
    FrameLabelerHost::FloatArray lowerThreshold = FrameLabelerHost::FloatArray(0);
};
extern RoiThresholds roiThreshHost;

// Implementations must provide three constexpr definitions:
// lookForward  : The number of frames needed to be examined
//                *after* the current frame to be emitted
// lookBack     : The number of frames needed to be examined
//                *before* the current frame to be emitted
// stitchFrames : The number of frames used to stitch together
//                two blocks of data
//
// SmoothedVal is a function for actually computing a smoothed value
//

// Base ROI filter parameter interface
struct IRoiFilter
{
    typedef FrameLabelerHost::LabelArray LabelArray;
};

// Implementation which matches the original Sequel version.
struct RoiFilterDefault : public IRoiFilter
{
    static constexpr int stitchFrames = 6;
    static constexpr int lookForward = 2;
    static constexpr int lookBack = 1;
    static constexpr int lookFull = lookBack + lookForward;

    // Look at four frames, from -1 to +2, where the incoming `newVal` is the +2 frame and
    // the roi filter will emit the 0 frame.
    // We will sum the following intervals, normalized by the square root of the interval widths.
    // and return the max:
    //   * [0,  1)
    //   * [-1, 1)
    //   * [-1, 2)
    //   * [-1, 3)
    static FrameLabelerHost::FloatArray SmoothedVal(const Utility::CudaArray<FrameLabelerHost::FloatArray, lookFull>& oldVals,
                                          const FrameLabelerHost::FloatArray& newVal)
    {
        using VF = FrameLabelerHost::FloatArray;
        auto mval = oldVals[1];
        mval = max(mval, (oldVals[0] + oldVals[1]) / sqrt(2.0f));
        mval = max(mval, (oldVals[0] + oldVals[1] + oldVals[2]) / sqrt(3.0f));
        mval = max(mval, (oldVals[0] + oldVals[1] + oldVals[2] + newVal) / 2.0f);
        return mval;
    }
};

// Defines settings that effectively disable the roi filter
struct RoiFilterNoOp : public IRoiFilter
{
    static constexpr int stitchFrames = 0;
    static constexpr int lookForward = 0;
    static constexpr int lookBack = 0;
    static constexpr int lookFull = lookBack + lookForward;

    static FrameLabelerHost::FloatArray SmoothedVal(const Utility::CudaArray<FrameLabelerHost::FloatArray, lookFull>&, 
                                     const FrameLabelerHost::FloatArray& newVal)
    {
        // A bit of a hack, but return an arbitrarily high value
        // to try and make sure this frame gets labeled as roi.
        // This can obviously be bypassed by similarly setting the
        // roi thresholds to a silly high value
        return 10000;
    }
};

// This class (or really rather the two nested classes) are holdovers
// from an earlier implementation which took a (flawed) approach at
// doing the roi and viterbi algorithms at the same time.  The
// abstractions are probably a bit heavier than is necessary now that
// the roi computation is done upfront and in isolation.  This probably
// could be simplified a lot, I've just run out of time.  Future maintainers
// should feel free to re-org this, as long as they verify that the algorithm
// runtime is not affected by the changes.
class Roi
{
    static FrameLabelerHost::UShortArray roiBit;
    static FrameLabelerHost::UShortArray midBit;

    // Struct to be used for the forward pass of the roi computation, and it is templated
    // upon a specific ROI filter that will perform some initial smoothing over the data
    template <typename RFT>
    class ForwardRecursion
    {
    public:
        ForwardRecursion(const Utility::CudaArray<typename RFT::LabelArray, RFT::lookBack>& latentTraces,
                         const Mongo::Data::BlockView<int16_t>& inTraces,
                         const FrameLabelerHost::FloatArray& invSigma,
                         const FrameLabelerHost::UShortArray& roiBC,
                         const Mongo::Data::BlockView<uint16_t>& roi)
           : roi_(roi)
           , prevRoi_(roiBC)
        {
            assert(inTraces.NumFrames() >= RFT::lookForward);

            int datIdx = 0;
            auto trIt = inTraces.CBegin();
            for (; datIdx < RFT::lookBack; ++datIdx)
            {
                previousData_[datIdx] = latentTraces[datIdx] * invSigma;
            }
            for (int trcIdx = 0; trcIdx < RFT::lookForward; ++trcIdx, ++datIdx)
            {
                previousData_[datIdx] = (trIt + trcIdx).Extract() * invSigma;
            }
            assert(datIdx == RFT::lookFull);
        }

        // Makes incremental process on the ROI forward recursion step.  It accepts
        // a new sigma normalized baseline frame, applies the RoiFilter smoothing
        // process to it, and then compares the resulting value to the lower/upper
        // thresholds used in the ROI determination process
        void ProcessNextFrame(FrameLabelerHost::FloatArray sigmaNormalized)
        {
            FrameLabelerHost::FloatArray frameValue = RFT::SmoothedVal(previousData_, sigmaNormalized);

            FrameLabelerHost::UShortArray zero(0);
            FrameLabelerHost::FloatArray thresh = Blend((prevRoi_ & roiBit) != zero,
                                   roiThreshHost.lowerThreshold,
                                   roiThreshHost.upperThreshold);

            FrameLabelerHost::UShortArray roiVal1 = Blend(frameValue >= thresh, roiBit, zero);
            FrameLabelerHost::UShortArray roiVal2 = Blend(frameValue >= roiThreshHost.lowerThreshold, midBit, zero);
            FrameLabelerHost::UShortArray roiVal = roiVal1 | roiVal2;

            assert((0 <= roiIdx_) && roiIdx_ < (int16_t)roi_.NumFrames());

            (roi_.Begin() + roiIdx_).Store(roiVal);
            roiIdx_++;
            prevRoi_ = roiVal;

            if constexpr (RFT::lookFull < 2)
                return;

            for (int i = 0; i < RFT::lookFull-1; ++i)
            {
                previousData_[i] = previousData_[i+1];
            }
            previousData_[RFT::lookFull-1] = sigmaNormalized;
        }

    private:
        Utility::CudaArray<typename FrameLabelerHost::FloatArray, RFT::lookFull> previousData_;
        Mongo::Data::BlockView<uint16_t> roi_;
        FrameLabelerHost::UShortArray prevRoi_;
        int16_t roiIdx_ = 0;
    };

    template <typename RFT>
    struct BackwardRecursion
    {
        BackwardRecursion(const Mongo::Data::BlockView<uint16_t>& roi)
            : prevRoi_(0)
            , roiIdx_(roi.NumFrames())
            , roi_(roi)
        {
            // Burn a few frames, to move away from the
            // artificial boundary condition hopefully far
            // enough to get to a more reliable state.
            for (int32_t i = 0; i < RFT::stitchFrames; ++i)
            {
                PopRoiState();
            }
        }

        void PopRoiState()
        {
            --roiIdx_;
            assert((0 <= roiIdx_) && roiIdx_ < (int16_t)roi_.NumFrames());

            FrameLabelerHost::UShortArray zero(0);
            auto roiIt = (roi_.Begin() + roiIdx_);

            auto roi = roiIt.Extract();
            auto roiBool1 = (roi & roiBit) != zero;
            auto roiBool2 = (roi & midBit) != zero;
            auto roiBool3 = prevRoi_ != zero;
            auto roiBool = roiBool1 | (roiBool2 & roiBool3);
            prevRoi_ = Blend(roiBool, roiBit, zero);
            auto cudaConvArr = roiBool.ToArray();
            ArrayUnion<FrameLabelerHost::UShortArray> unionArr;
            for (uint32_t i = 0; i < laneSize; ++i)
            {
                unionArr[i] = cudaConvArr[i];
            }
            roiIt.Store(unionArr);
        }

        int16_t FramesRemaining() const { return roiIdx_; }

        FrameLabelerHost::UShortArray prevRoi_;
        int16_t roiIdx_;
        Mongo::Data::BlockView<uint16_t> roi_;
    };
};

}

#endif /*PACBIO_MONGO_BASECALLER_FRAME_ROI_HOST_H*/
