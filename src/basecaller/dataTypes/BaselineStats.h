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

// Extensions to UnifiedCudaArray available only in cuda compilation units.
// In particular provide array access to device data when in device code

#ifndef PACBIO_MONGO_BASELINE_STATS_H_
#define PACBIO_MONGO_BASELINE_STATS_H_

#include <common/cuda/utility/CudaArray.h>
#include <common/cuda/CudaFunctionDecorators.h>

namespace PacBio {
namespace Mongo {
namespace Data {

// Baseline stats for a lane of data
template <uint32_t LaneWidth>
class BaselineStats
{
public:     // Types
    using ElementType = float;

public:
    BaselineStats() = default;

    // NOTE: We make the member variables public as this is
    // meant to be data transport type to avoid the need to write
    // accessors/setters.

    // Represents statistics from all frames
    /// State of AutocorrAccumulator.
    // TODO: This appears incomplete.
    Cuda::Utility::CudaArray<float, LaneWidth> lagM1First_;
    Cuda::Utility::CudaArray<float, LaneWidth> lagM1Last_;
    Cuda::Utility::CudaArray<float, LaneWidth> lagM2_;

    /// Minimum and Maximum of all frames after baseline subtraction.
    Cuda::Utility::CudaArray<int16_t, LaneWidth> traceMin_;
    Cuda::Utility::CudaArray<int16_t, LaneWidth> traceMax_;

    // Represents statistics from baseline frames
    /// Number of baseline frames.
    Cuda::Utility::CudaArray<float, LaneWidth> m0_;

    /// Sum of baseline frames after baseline subtraction.
    Cuda::Utility::CudaArray<float, LaneWidth> m1_;

    /// Sum of squares of baseline frames after baseline subtraction.
    Cuda::Utility::CudaArray<float, LaneWidth> m2_;

    /// Sum of baseline frames before baseline subtraction.
    Cuda::Utility::CudaArray<int16_t, LaneWidth> rawBaselineSum_;
};

}}} // ::PacBio::Mongo::Data

#endif //PACBIO_MONGO_BASELINE_STATS_H_
