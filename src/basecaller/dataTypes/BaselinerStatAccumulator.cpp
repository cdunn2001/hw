
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

template <typename T>
void BaselinerStatAccumulator<T>::AddSample(const LaneArray& rawTrace,
                                            const LaneArray& blSubtracted,
                                            const Mask& isBaseline)
{
    const FloatArray bs(blSubtracted);

    // Add frame to complete trace statistics
    baselineSubtractedCorr_.AddSample(bs);
    traceMin = min(blSubtracted, traceMin);
    traceMax = max(blSubtracted, traceMax);

    // Add frame to baseline statistics if so flagged
    baselineStats_.AddSample(bs, isBaseline);
    rawBaselineSum_ += Blend(isBaseline, rawTrace, LaneArray{0});
}

template <typename T>
BaselinerStatAccumulator<T>&
BaselinerStatAccumulator<T>::Merge(const BaselinerStatAccumulator& other)
{
    baselineSubtractedCorr_.Merge(other.BaselineSubtractedStats());
    traceMin = min(traceMin, other.traceMin);
    traceMax = max(traceMax, other.traceMax);
    baselineStats_.Merge(other.BaselineFramesStats());
    rawBaselineSum_ += other.rawBaselineSum_;
    return *this;
}

//
// Explicit Instantiations
//

template class BaselinerStatAccumulator<short>;

}}}     // namespace PacBio::Mongo::Data
