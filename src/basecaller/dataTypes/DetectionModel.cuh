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

#ifndef mongo_dataTypes_DetectionModel_CUH_
#define mongo_dataTypes_DetectionModel_CUH_

#include "DetectionModel.h"
#include "BatchData.cuh"

namespace PacBio {
namespace Mongo {
namespace Data {

template <typename T>
class ZmwDetectionModel
{
public:
    __device__ ZmwDetectionModel(StridedBlockView<T> data, Memory::detail::DataManagerKey)
        : data_(data)
    {}

    template <typename U, std::enable_if_t<!std::is_const<U>::value, int> = 0>
    __device__ operator ZmwDetectionModel<const U>() const
    {
        return ZmwDetectionModel<const U>(data_);
    }

    __device__ T& BaselineMean() { return data_[DetectionModelIdx::BaselineMean()]; }
    __device__ T& BaselineVar()  { return data_[DetectionModelIdx::BaselineVar()]; }

    __device__ T& AnalogMean(int i) { return data_[DetectionModelIdx::AnalogMean(i)]; }
    __device__ T& AnalogVar(int i)  { return data_[DetectionModelIdx::AnalogVar(i)]; }
private:
    StridedBlockView<T> data_;
};

template <typename T>
class GpuDetectionModel : private Memory::detail::DataManager
{
    GpuDetectionModel(DetectionModel& model)
        : data_(model.Data(DataKey()))
    {}

    ZmwDetectionModel<T> GetZmwModel(int lane, int zmw)
    {
        return ZmwDetectionModel<T>(data_.ZmwData(lane, zmw));
    }
    ZmwDetectionModel<const T> GetZmwModel(int lane, int zmw) const
    {
        return ZmwDetectionModel<const T>(data_.ZmwData(lane, zmw));
    }
private:
    GpuBatchData<T> data_;
};

}}}     // namespace PacBio::Mongo::Data

#endif // mongo_dataTypes_DetectionModel_H_
