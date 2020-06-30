
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
//  Defines some members of class TraceHistogramAccumulator.

#include <sstream>
#include <pacbio/logging/Logger.h>
#include <pacbio/PBException.h>
#include <dataTypes/configs/BasecallerTraceHistogramConfig.h>
#include "TraceHistogramAccumulator.h"

namespace PacBio {
namespace Mongo {
namespace Basecaller {

// static
unsigned int TraceHistogramAccumulator::numFramesPreAccumStats_;
float TraceHistogramAccumulator::binSizeCoeff_;
unsigned int TraceHistogramAccumulator::baselineStatMinFrameCount_;
float TraceHistogramAccumulator::fallBackBaselineSigma_;

// static
void TraceHistogramAccumulator::Configure(
        const Data::BasecallerTraceHistogramConfig& histConfig,
        const Data::MovieConfig&)
{
    numFramesPreAccumStats_ = histConfig.NumFramesPreAccumStats;
    PBLOG_INFO << "TraceHistogramAccumulator: NumFramesPreAccumStats = "
               << numFramesPreAccumStats_ << '.';

    binSizeCoeff_ = histConfig.BinSizeCoeff;
    PBLOG_INFO << "TraceHistogramAccumulator: BinSizeCoeff = "
               << binSizeCoeff_ << '.';
    if (binSizeCoeff_ <= 0.0f)
    {
        std::ostringstream msg;
        msg << "BinSizeCoeff must be positive.";
        throw PBException(msg.str());
    }

    baselineStatMinFrameCount_ = histConfig.BaselineStatMinFrameCount;
    PBLOG_INFO << "TraceHistogramAccumulator: BaselineStatMinFrameCount = "
               << baselineStatMinFrameCount_ << '.';

    fallBackBaselineSigma_ = histConfig.FallBackBaselineSigma;
    PBLOG_INFO << "TraceHistogramAccumulator: FallBackBaselineSigma = "
               << fallBackBaselineSigma_ << '.';
    if (fallBackBaselineSigma_ <= 0.0f)
    {
        std::ostringstream msg;
        msg << "FallBackBaselineSigma must be positive.";
        throw PBException(msg.str());
    }
}

TraceHistogramAccumulator::TraceHistogramAccumulator(uint32_t poolId, unsigned int poolSize)
    : poolId_ (poolId)
    , poolSize_ (poolSize)
{

}

}}}     // namespace PacBio::Mongo::Basecaller
