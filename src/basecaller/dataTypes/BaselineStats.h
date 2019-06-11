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

namespace PacBio {
namespace Mongo {
namespace Data {

// Baseline stats for a lane of data
template <uint32_t LaneWidth>
class BaselineStats
{
public:
    BaselineStats() = default;

public:     // Accesors
    const Cuda::Utility::CudaArray<int16_t, LaneWidth> TraceMin() const
    { return traceMin_; }

    const Cuda::Utility::CudaArray<int16_t, LaneWidth> TraceMax() const
    { return traceMax_; }

    const Cuda::Utility::CudaArray<int16_t, LaneWidth> RawBaselineSum() const
    { return rawBaselineSum_; }

    const Cuda::Utility::CudaArray<float, LaneWidth> BaselineCount() const
    { return m0_; }

    const Cuda::Utility::CudaArray<float, LaneWidth> BaselineMean() const
    { return m1_; }

    const Cuda::Utility::CudaArray<float, LaneWidth> BaselineVariance() const
    { return m2_; }

    const Cuda::Utility::CudaArray<float, LaneWidth> AutocorrLagM1First() const
    { return lagM1First_; }

    const Cuda::Utility::CudaArray<float, LaneWidth> AutocorrLagM1Last() const
    { return lagM1Last_; }

    const Cuda::Utility::CudaArray<float, LaneWidth> AutocorrLagM2() const
    { return lagM2_; }

public:     // Modifiers
    BaselineStats& TraceMin(const Cuda::Utility::CudaArray<int16_t, LaneWidth>& traceMin)
    {
        traceMin_ = traceMin;
        return *this;
    }

    BaselineStats& TraceMax(const Cuda::Utility::CudaArray<int16_t, LaneWidth>& traceMax)
    {
        traceMax_ = traceMax;
        return *this;
    }

    BaselineStats& RawBaselineSum(const Cuda::Utility::CudaArray<int16_t, LaneWidth>& rawBaselineSum)
    {
        rawBaselineSum_ = rawBaselineSum;
        return *this;
    }

    BaselineStats& BaselineMoments(const Cuda::Utility::CudaArray<float, LaneWidth>& count,
                                   const Cuda::Utility::CudaArray<float, LaneWidth>& mean,
                                   const Cuda::Utility::CudaArray<float, LaneWidth>& variance)
    {
        m0_ = count;
        m1_ = mean;
        m2_ = variance;
        return *this;
    }

    BaselineStats& AutocorrMoments(const Cuda::Utility::CudaArray<float, LaneWidth>& lagM1First,
                                   const Cuda::Utility::CudaArray<float, LaneWidth>& lagM1Last,
                                   const Cuda::Utility::CudaArray<float, LaneWidth>& lagM2)
    {
        lagM1First_ = lagM1First;
        lagM1Last_ = lagM1Last;
        lagM2_ = lagM2;
        return *this;
    }

private:

    // Represents statistics from all frames
    Cuda::Utility::CudaArray<float, LaneWidth> lagM1First_;
    Cuda::Utility::CudaArray<float, LaneWidth> lagM1Last_;
    Cuda::Utility::CudaArray<float, LaneWidth> lagM2_;
    Cuda::Utility::CudaArray<int16_t, LaneWidth> traceMin_;
    Cuda::Utility::CudaArray<int16_t, LaneWidth> traceMax_;

    // Represents statistics from baseline frames
    Cuda::Utility::CudaArray<float, LaneWidth> m0_;
    Cuda::Utility::CudaArray<float, LaneWidth> m1_;
    Cuda::Utility::CudaArray<float, LaneWidth> m2_;
    Cuda::Utility::CudaArray<int16_t, LaneWidth> rawBaselineSum_;
};

}}} // ::PacBio::Mongo::Data

#endif //PACBIO_MONGO_BASELINE_STATS_H_
