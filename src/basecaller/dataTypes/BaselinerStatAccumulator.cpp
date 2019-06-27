
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
//  Defines some members of class BaselinerStatAccumulator.

#include "BaselinerStatAccumulator.h"

#include <dataTypes/BasicTypes.h>

namespace PacBio {
namespace Mongo {
namespace Data {

// Deprecated
template <typename T>
BaselinerStatAccumulator<T>::BaselinerStatAccumulator(const BaselineStats<laneSize>& bs)
    : baselineSubtractedStats_ (/*TODO*/)
    , traceMin (ConstLaneArrayRef<RawTraceElement>(bs.traceMin_.data()))
    , traceMax (ConstLaneArrayRef<RawTraceElement>(bs.traceMax_.data()))
    , baselineStats_ (/*TODO*/)
    , rawBaselineSum_ (ConstLaneArrayRef<RawTraceElement>(bs.rawBaselineSum_.data()))
    // TODO: Should get elemental data types from BaselineStat, but it does not provide any.
{ }

template <typename T>
void BaselinerStatAccumulator<T>::AddSample(const LaneArray& rawTrace, const LaneArray& baselineSubtracted, const Mask& isBaseline)
{
    const auto& bs = baselineSubtracted.AsFloat();

    // Add frame to complete trace statistics.
    baselineSubtractedStats_.AddSample(bs);
    traceMin = min(bs, traceMin);
    traceMax = max(bs, traceMax);

    // Add frame to baseline statistics if so flagged.
    baselineStats_.AddSample(bs, isBaseline);
    rawBaselineSum_ += Blend(isBaseline, rawTrace, {0});
}

template <typename T>
const BaselineStats<laneSize> BaselinerStatAccumulator<T>::ToBaselineStats() const
{
    auto baselineStats = Data::BaselineStats<laneSize>{};

    std::copy(BaselineSubtractedStats().M1First().begin(), BaselineSubtractedStats().M1First().end(),
              baselineStats.lagM1First_.data());
    std::copy(BaselineSubtractedStats().M1Last().begin(), BaselineSubtractedStats().M1Last().end(),
              baselineStats.lagM1Last_.data());
    std::copy(BaselineSubtractedStats().M2().begin(), BaselineSubtractedStats().M2().end(),
              baselineStats.lagM2_.data());

    std::copy(TraceMin().begin(), TraceMin().end(), baselineStats.traceMin_.data());
    std::copy(TraceMax().begin(), TraceMax().end(), baselineStats.traceMax_.data());

    std::copy(BaselineFramesStats().Count().begin(), BaselineFramesStats().Count().end(),
              baselineStats.m0_.data());
    std::copy(BaselineFramesStats().M1().begin(), BaselineFramesStats().M1().end(),
              baselineStats.m1_.data());
    std::copy(BaselineFramesStats().M2().begin(), BaselineFramesStats().M2().end(),
              baselineStats.m2_.data());

    std::copy(RawBaselineSum().begin(), RawBaselineSum().end(), baselineStats.rawBaselineSum_.data());

    return baselineStats;
}


template <typename T>
BaselinerStatAccumulator<T>&
BaselinerStatAccumulator<T>::Merge(const BaselinerStatAccumulator& other)
{
    baselineSubtractedStats_.Merge(other.BaselineSubtractedStats());
    traceMin = min(traceMin, other.traceMin);
    traceMax = max(traceMax, other.traceMax);
    baselineStats_.Merge(other.BaselineFramesStats());
    rawBaselineSum_ += rawBaselineSum_;
    return *this;
}

//
// Explicit Instantiations
//

template class BaselinerStatAccumulator<short>;

}}}     // namespace PacBio::Mongo::Data
