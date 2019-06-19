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

#ifndef mongo_dataTypes_DetectionModel_H_
#define mongo_dataTypes_DetectionModel_H_

#include <dataTypes/TraceBatch.h>
#include <common/MongoConstants.h>
#include <common/cuda/PBCudaSimd.h>

using namespace PacBio::Cuda;

namespace PacBio {
namespace Mongo {
namespace Data {

// Helper functions, to prevent unify the indexing logic the various views
// of the detection model need to keep consistent
namespace DetectionModelIdx {

    static constexpr int BaselineMean()    { return 0; }
    static constexpr int BaselineVar()     { return 1; }
    static constexpr int AnalogMean(int i) { return 2 + i*4; }
    static constexpr int AnalogVar(int i)  { return 6 + i*4; }
}

template <typename T>
class LaneDetectionModel
{
public:
    LaneDetectionModel(BlockView<T> data, Memory::detail::DataManagerKey)
        : data_(data)
    {}

    template <typename U = T, std::enable_if_t<!std::is_const<U>::value, int> = 0>
    operator LaneDetectionModel<const U>()
    {
        return LaneDetectionModel<const T>(data_);
    }

    // Ugh, I wanted this to return a LaneArray of floats, but not only does that
    // require a LaneArrayRef implementation (hard to do without tons of work, to
    // handle all the operators correctly), but you can't include the LaneArray header
    // in this file without breaking any cuda files that include this
    T* BaselineMean()    { return ExtractLane(DetectionModelIdx::BaselineMean());}
    T* BaselineVar()     { return ExtractLane(DetectionModelIdx::BaselineVar()); }
    T* AnalogMean(int i) { return ExtractLane(DetectionModelIdx::AnalogMean(i)); }
    T* AnalogVar(int i)  { return ExtractLane(DetectionModelIdx::AnalogMean(i)); }

private:
    T* ExtractLane(int idx)
    {
        return &data_(idx, 0);
    }
    BlockView<T> data_;
};

/// A bundle of model parameters for a normal mixture representing the
/// baselined trace data for a pool of ZMWs.
class DetectionModel : private Memory::detail::DataManager
{
private:
    static BatchDimensions ConstructDims(int lanesPerPool)
    {
        BatchDimensions ret;
        ret.framesPerBatch = 10; // coopting this dimension to store mean/var for 4 analogs and baseline
        ret.laneWidth = laneSize;
        ret.lanesPerBatch = lanesPerPool;
        return ret;
    }
public:
    DetectionModel(int lanesPerPool,
                   Memory::SyncDirection syncDirection,
                   bool pinned = true)
        : data_(ConstructDims(lanesPerPool), syncDirection, nullptr, pinned)
    {}

    BatchData<PBHalf>& Data(Memory::detail::DataManagerKey) { return data_; }
    const BatchData<PBHalf>& Data(Memory::detail::DataManagerKey) const { return data_; }

    LaneDetectionModel<PBHalf> HostLaneModel(int lane)
    {
        return LaneDetectionModel<PBHalf>(data_.GetBlockView(lane), DataKey());
    }
    LaneDetectionModel<const PBHalf> HostLaneModel(int lane) const
    {
        return LaneDetectionModel<const PBHalf>(data_.GetBlockView(lane), DataKey());
    }

private:
    BatchData<PBHalf> data_;

};

}}}     // namespace PacBio::Mongo::Data

#endif // mongo_dataTypes_DetectionModel_H_
