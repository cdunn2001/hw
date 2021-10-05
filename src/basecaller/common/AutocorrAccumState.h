#ifndef mongo_common_AutocorrAccumState
#define mongo_common_AutocorrAccumState

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
//  Defines POD struct StatAccumState.

#include <common/cuda/utility/CudaArray.h>
#include <common/MongoConstants.h>

#include "StatAccumState.h"

namespace PacBio {
namespace Mongo {

/// A CUDA-friendly POD struct that represents the state of a AutocorrAccumulator.
/// The ability to add more data samples is not be preserved by this
/// representation.
struct AutocorrAccumState
{
    static constexpr unsigned int lag = 4u;
    using FloatArray = Cuda::Utility::CudaArray<float, laneSize>;

    StatAccumState basicStats;

    FloatArray moment1First;
    FloatArray moment1Last;
    FloatArray moment2;

    FloatArray lBuf[4u];
    FloatArray rBuf[4u];
    uint16_t meta[laneSize]; // lbi, rbi and canAddSample
};

}}      // namespace PacBio::Mongo

#endif // mongo_common_AutocorrAccumState
