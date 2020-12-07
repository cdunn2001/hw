// Copyright (c) 2020 Pacific Biosciences of California, Inc.
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
//  Defines some members of class SignalRangeEstimator

#include "SignalRangeEstimator.h"

#include <sstream>

#include <pacbio/logging/Logger.h>
#include <pacbio/PBException.h>
#include <dataTypes/configs/BasecallerSignalRangeEstimatorConfig.h>

namespace PacBio {
namespace Mongo {
namespace Basecaller {

// static
unsigned int SignalRangeEstimator::numFramesPreAccumStats_;
float SignalRangeEstimator::binSizeCoeff_;
unsigned int SignalRangeEstimator::baselineStatMinFrameCount_;
float SignalRangeEstimator::fallBackBaselineSigma_;

// static
void SignalRangeEstimator::Configure(const Data::BasecallerSignalRangeEstimatorConfig& sigConfig)
{
    numFramesPreAccumStats_ = sigConfig.NumFramesPreAccumStats;
    PBLOG_INFO << "TraceHistogramAccumulator: NumFramesPreAccumStats = "
               << numFramesPreAccumStats_ << '.';

    binSizeCoeff_ = sigConfig.BinSizeCoeff;
    PBLOG_INFO << "TraceHistogramAccumulator: BinSizeCoeff = "
               << binSizeCoeff_ << '.';
    if (binSizeCoeff_ <= 0.0f)
    {
        std::ostringstream msg;
        msg << "BinSizeCoeff must be positive.";
        throw PBException(msg.str());
    }

    baselineStatMinFrameCount_ = sigConfig.BaselineStatMinFrameCount;
    PBLOG_INFO << "TraceHistogramAccumulator: BaselineStatMinFrameCount = "
               << baselineStatMinFrameCount_ << '.';

    fallBackBaselineSigma_ = sigConfig.FallBackBaselineSigma;
    PBLOG_INFO << "TraceHistogramAccumulator: FallBackBaselineSigma = "
               << fallBackBaselineSigma_ << '.';
    if (fallBackBaselineSigma_ <= 0.0f)
    {
        std::ostringstream msg;
        msg << "FallBackBaselineSigma must be positive.";
        throw PBException(msg.str());
    }
}

SignalRangeEstimator::SignalRangeEstimator(uint32_t poolId, unsigned int poolSize)
    : poolId_ (poolId)
    , poolSize_ (poolSize)
{

}

}}}     // namespace PacBio::Mongo::Basecaller
