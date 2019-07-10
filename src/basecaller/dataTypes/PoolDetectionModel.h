#ifndef mongo_dataTypes_PoolDetectionModel_H_
#define mongo_dataTypes_PoolDetectionModel_H_

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
//  Description:
//  Defines types PoolDetectionModel and LaneDetectionModel.

#include <common/cuda/memory/UnifiedCudaArray.h>
#include <common/MongoConstants.h>
#include "LaneDetectionModel.h"

namespace PacBio {
namespace Mongo {
namespace Data {

/// A bundle of model parameters for a normal mixture representing the
/// baselined trace data for a lane of ZMWs.
/// \tparam T is the elemental data type (e.g., float).
template <typename T>
using LaneDetectionModel = LaneModelParameters<T, laneSize>;


/// A bundle of model parameters for a normal mixture representing the
/// baselined trace data for a pool of ZMWs.
/// \tparam T is the elemental data type (e.g., float).
template <typename T>
struct PoolDetectionModel
{
    using ElementType = T;

    Cuda::Memory::UnifiedCudaArray<LaneDetectionModel<T>> laneModels;
    uint32_t poolId;

    PoolDetectionModel(uint32_t aPoolId,
                       unsigned int lanesPerPool,
                       Cuda::Memory::SyncDirection syncDirection,
                       bool pinned = true)
        : laneModels (lanesPerPool, syncDirection, pinned, nullptr)
        , poolId (aPoolId)
    {}
};

}}}     // namespace PacBio::Mongo::Data

#endif  // mongo_dataTypes_PoolDetectionModel_H_
