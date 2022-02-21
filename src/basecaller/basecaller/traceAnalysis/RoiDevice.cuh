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

#ifndef PACBIO_MONGO_BASECALLER_FRAME_ROI_DEVICE_H
#define PACBIO_MONGO_BASECALLER_FRAME_ROI_DEVICE_H

#include <common/cuda/PBCudaSimd.cuh>
#include <common/cuda/utility/CudaArray.h>
#include <dataTypes/BatchData.cuh>
#include <dataTypes/configs/BasecallerFrameLabelerConfig.h>

namespace PacBio::Mongo::Basecaller {

using namespace Cuda;

// Some configuration data for the whole process.  Needs to
// be initialized before using (currently handled by the
// FrameLabelerDevice configuration process)
struct RoiThresholdsDevice
{
    float upperThreshold = 0;
    float lowerThreshold = 0;
};
extern __constant__ RoiThresholdsDevice roiThresh;

using RoiFilterType = Data::BasecallerRoiConfig::RoiFilterType::RawEnum;

// Implementations must provide three constexpr definitions:
// lookForward  : The number of frames needed to be examined
//                *after* the current frame to be emitted
// lookBack     : The number of frames needed to be examined
//                *before* the current frame to be emitted
// stitchFrames : The number of frames used to stitch together
//                two blocks of data
//
// And one static function for actually computing a smoothed
// value:
//
//   __device__ static PBHalf2 SmoothedVal(const CudaArray<PBShort2, lookForward+lookBack>& oldVals,
//                                         PBShort2 newVal)
template <RoiFilterType type>
struct RoiFilter;

// Implementation which matches the original Sequel version.
template <>
struct RoiFilter<RoiFilterType::Default>
{
    static constexpr size_t stitchFrames = 6;
    static constexpr size_t lookForward = 2;
    static constexpr size_t lookBack = 1;
    static constexpr size_t lookFull = lookBack + lookForward;

    // Look at four frames, from -1 to +2, where the incoming `newVal` is the +2 frame and
    // the roi filter will emit the 0 frame.
    // We will sum the following intervals, normalized by the square root of the interval widths.
    // and return the max:
    //   * [0,  1)
    //   * [-1, 1)
    //   * [-1, 2)
    //   * [-1, 3)
    __device__ static PBHalf2 SmoothedVal(const Utility::CudaArray<PBHalf2, lookFull>& oldVals,
                                          PBHalf2 newVal)
    {
        auto mval = max((oldVals[0] + oldVals[1]) / sqrt(2), oldVals[1]);
        mval = max(mval, (oldVals[0] + oldVals[1] + oldVals[2]) / sqrt(3));
        return  max(mval, (oldVals[0] + oldVals[1] + oldVals[2] + newVal) / 2.f);
    }
};

// Defines settings that effectively disable the roi filter
template <>
struct RoiFilter<RoiFilterType::NoOp>
{
    static constexpr size_t stitchFrames = 0;
    static constexpr size_t lookForward = 0;
    static constexpr size_t lookBack = 0;

    __device__ static PBHalf2 SmoothedVal(const Utility::CudaArray<PBHalf2, 0>&,
                                          PBHalf2)
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
    static constexpr int16_t roiBit = 0x1;
    static constexpr int16_t midBit = 0x2;

public:
    // Struct to be used for the forward pass of the roi computation, and it is templated
    // upon a specific ROI filter that will perform some initial smoothing over the data
    template <typename RoiFilter>
    class ForwardRecursion
    {
    public:
        template <typename T>
        __device__ ForwardRecursion(const Utility::CudaArray<T, RoiFilter::lookBack>& latentTraces,
                                    const Mongo::Data::StridedBlockView<PBShort2>& inTraces,
                                    PBHalf2 invSigma,
                                    PBShort2 roiBC,
                                    const Mongo::Data::StridedBlockView<PBShort2>& roi)
           : roi_(roi)
           , prevRoi_(roiBC)
        {
            int datIdx = 0;
            assert(inTraces.size() >= RoiFilter::lookForward);
            for (const auto& val : latentTraces)
            {
                previousData_[datIdx] = PBShort2{val} * invSigma;
                datIdx++;
            }
            for (int trcIdx = 0; trcIdx < RoiFilter::lookForward; ++trcIdx, ++datIdx)
            {
                previousData_[datIdx] = inTraces[trcIdx] * invSigma;
            }
            assert(datIdx == arrSize);
        }

        // Makes incremental process on the ROI forward recursion step.  It accepts
        // a new sigma normalized baseline frame, applies the RoiFilter smoothing
        // process to it, and then compares the resulting value to the lower/upper
        // thresholds used in the ROI determination process
        __device__ void ProcessNextFrame(PBHalf2 sigmaNormalized)
        {
            const auto frameValue = RoiFilter::SmoothedVal(previousData_, sigmaNormalized);

            PBHalf2 thresh = Blend((prevRoi_ & roiBit) != 0,
                                   PBHalf2{roiThresh.lowerThreshold},
                                   PBHalf2{roiThresh.upperThreshold});

            PBShort2 roiVal = Blend(frameValue >= thresh, roiBit, PBShort2{0});
            roiVal = roiVal | Blend(frameValue >= roiThresh.lowerThreshold, midBit, PBShort2{0});

            assert(roiIdx_ < roi_.size());
            assert(roiIdx_ >= 0);
            roi_[roiIdx_] = roiVal;
            roiIdx_++;
            prevRoi_ = roiVal;

            if constexpr (arrSize > 1)
            {
                #pragma unroll
                for (size_t i = 0; i < arrSize-1; ++i)
                {
                    previousData_[i] = previousData_[i+1];
                }
                previousData_[arrSize-1] = sigmaNormalized;
            }
        }

        __device__ int16_t NextIdx() const { return roiIdx_; }

    private:
        static constexpr size_t arrSize = RoiFilter::lookForward + RoiFilter::lookBack;

        Utility::CudaArray<PBHalf2, arrSize> previousData_;
        Mongo::Data::StridedBlockView<PBShort2> roi_;
        PBShort2 prevRoi_;
        int16_t roiIdx_ = 0;
    };

    template <typename RoiFilter>
    struct BackwardRecursion
    {
        __device__ BackwardRecursion(const Mongo::Data::StridedBlockView<PBShort2>& roi)
            : prevRoi_(false)
            , roiIdx_(roi.size())
            , roi_(roi)
        {
            // Burn a few frames, to move away from the
            // artificial boundary condition hopefully far
            // enough to get to a more reliable state.
            for (int32_t i = 0; i < RoiFilter::stitchFrames; ++i)
            {
                PopRoiState();
            }
        }

        __device__ void PopRoiState()
        {
            --roiIdx_;
            assert(roiIdx_ < roi_.size());
            assert(roiIdx_ >= 0);
            auto roiBool = (roi_[roiIdx_] & roiBit) != 0;
            roiBool = roiBool || ((roi_[roiIdx_] & midBit) != 0 && prevRoi_ != 0);
            prevRoi_ = Blend(roiBool, roiBit, PBShort2{0});
            roi_[roiIdx_] = roiBool;
        }

        __device__ int16_t FramesRemaining() const { return roiIdx_; }

        PBShort2 prevRoi_;
        int16_t roiIdx_;
        Mongo::Data::StridedBlockView<PBShort2> roi_;
    };
};

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
template <typename RoiFilter, typename T, size_t len>
__device__ static void ComputeRoi(const Utility::CudaArray<T, len>& latentTraces,
                                  PBShort2 roiBC,
                                  const Mongo::Data::StridedBlockView<PBShort2>& traces1,
                                  const Mongo::Data::StridedBlockView<const PBShort2>& traces2,
                                  PBHalf2 invSigma1,
                                  PBHalf2 invSigma2,
                                  const Mongo::Data::StridedBlockView<PBShort2>& roi)
{
    assert(roi.size() == traces1.size() + traces2.size() - RoiFilter::lookForward);

    Roi::ForwardRecursion<RoiFilter> forward(latentTraces, traces1, invSigma1, roiBC, roi);
    for (size_t i = RoiFilter::lookForward; i < traces1.size(); ++i)
    {
        forward.ProcessNextFrame(traces1[i] * invSigma1);
    }

    for (size_t i = 0; i < traces2.size(); ++i)
    {
        forward.ProcessNextFrame(traces2[i] * invSigma2);
    }

    Roi::BackwardRecursion<RoiFilter> backward(roi);
    while (backward.FramesRemaining() > 0) backward.PopRoiState();
}

}

#endif /*PACBIO_MONGO_BASECALLER_FRAME_ROI_DEVICE_H*/
