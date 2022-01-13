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
struct RoiThresholds
{
    float upperThreshold = 0;
    float lowerThreshold = 0;
};
extern __constant__ RoiThresholds roiThresh;

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
    __device__ static PBHalf2 SmoothedVal(const Utility::CudaArray<PBShort2, lookFull>& oldVals,
                                          PBShort2 newVal)
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

    __device__ static PBHalf2 SmoothedVal(const Utility::CudaArray<PBShort2, 0>&,
                                          PBShort2)
    {
        // A bit of a hack, but return an arbitrarily high value
        // to try and make sure this frame gets labeled as roi.
        // This can obviously be bypassed by similarly setting the
        // roi thresholds to a silly high value
        return 10000;
    }
};

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
                       PBShort2 roiBC,
                       PBHalf2 invSigma,
                       const Mongo::Data::StridedBlockView<PBShort2>& roi)
           : roi_(roi)
           , prevRoi_(roiBC)
           , invSigma_(invSigma)
        {
            int datIdx = 0;
            assert(inTraces.size() > RoiFilter::lookForward);
            for (const auto& val : latentTraces)
            {
                data_[datIdx] = val;
                datIdx++;
            }
            for (int trcIdx = 0; trcIdx < RoiFilter::lookForward; ++trcIdx, ++datIdx)
            {
                data_[datIdx] = inTraces[trcIdx];
            }
            assert(datIdx == arrSize);
        }

        __device__ void UpdateInvSigma(PBHalf2 invSigma)
        {
            invSigma_ = invSigma;
        }

        __device__ PBShort2 Process(PBShort2 val)
        {
            // Note: The sigma used is only an approximation while we're crossing
            //       a block boundary.  Some subset of the points will be
            //       using a (hopefully slightly) wrong baseline variance
            //       here
            const auto sVal = RoiFilter::SmoothedVal(data_, val) * invSigma_;

            PBHalf2 thresh = Blend((prevRoi_ & roiBit) != 0,
                                   PBHalf2{roiThresh.lowerThreshold},
                                   PBHalf2{roiThresh.upperThreshold});

            PBShort2 roiVal = Blend(sVal >= thresh, roiBit, PBShort2{0});
            roiVal = roiVal | Blend(sVal >= roiThresh.lowerThreshold, midBit, PBShort2{0});

            assert(roiIdx_ < roi_.size());
            assert(roiIdx_ >= 0);
            roi_[roiIdx_] = roiVal;
            roiIdx_++;
            prevRoi_ = roiVal;

            PBShort2 ret = (RoiFilter::lookForward > 0)
                ? data_[arrSize - RoiFilter::lookForward]
                : val;

            if constexpr (arrSize > 1)
            {
                #pragma unroll
                for (size_t i = 0; i < arrSize-1; ++i)
                {
                    data_[i] = data_[i+1];
                }
                data_[arrSize-1] = val;
            }
            return ret;
        }

        __device__ int16_t NextIdx() const { return roiIdx_; }

    private:
        static constexpr size_t arrSize = RoiFilter::lookForward + RoiFilter::lookBack;

        Utility::CudaArray<PBShort2, arrSize> data_;
        Mongo::Data::StridedBlockView<PBShort2> roi_;
        PBShort2 prevRoi_;
        PBHalf2 invSigma_;
        int16_t roiIdx_ = 0;
    };

    template <typename RoiFilter>
    struct BackwardRecursion
    {
        __device__ BackwardRecursion(const Mongo::Data::StridedBlockView<PBShort2>& roi)
            : prevRoi_(0)
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

        __device__ PBShort2 PopRoiState()
        {
            --roiIdx_;
            assert(roiIdx_ < roi_.size());
            assert(roiIdx_ >= 0);
            auto roiBool = (roi_[roiIdx_] & roiBit) != 0;
            roiBool = roiBool || ((roi_[roiIdx_] & midBit) != 0 && prevRoi_ != 0);
            prevRoi_ = Blend(roiBool, roiBit, PBShort2{0});
            return roiBool;
        }

        __device__ int16_t FramesRemaining() const { return roiIdx_; }

        PBShort2 prevRoi_;
        int16_t roiIdx_;
        Mongo::Data::StridedBlockView<PBShort2> roi_;
    };
};

}

#endif /*PACBIO_MONGO_BASECALLER_FRAME_ROI_DEVICE_H*/
