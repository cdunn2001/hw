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

using PacBio::Cuda::Utility::CudaArray;

namespace PacBio {
namespace Mongo {

/// A CUDA-friendly POD struct that represents the state of a AutocorrAccumulator.
/// The ability to add more data samples is not be preserved by this
/// representation.
struct AutocorrAccumState
{
    static constexpr unsigned int lag = 4u;
    
    using FloatArray  = CudaArray<float, laneSize>;
    using FloatArrayLag = CudaArray<CudaArray<float, laneSize>, lag>;
    using UByteArray2 = CudaArray<CudaArray<uint8_t, laneSize>, 2>;

    StatAccumState basicStats;

    FloatArray moment2;

    FloatArrayLag fBuf; // front buffer
    FloatArrayLag bBuf; // back buffer
    UByteArray2 bIdx; // buffer indices for right and left positions
};

}}      // namespace PacBio::Mongo

#endif // mongo_common_AutocorrAccumState
