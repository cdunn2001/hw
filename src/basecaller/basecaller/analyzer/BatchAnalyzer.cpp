
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
//  Defines members of class BatchAnalyzer.

#include "BatchAnalyzer.h"

#include <dataTypes/BasecallBatch.h>
#include <dataTypes/TraceBatch.h>
#include <dataTypes/BasecallerConfig.h>

using namespace PacBio::Mongo::Data;

namespace PacBio {
namespace Mongo {
namespace Basecaller {

BatchAnalyzer::BatchAnalyzer(uint32_t batchId,
                             const BasecallerAlgorithmConfig& bcConfig,
                             const MovieConfig& movConfig)
    : batchId_ (batchId)
{ }

BasecallBatch BatchAnalyzer::operator()(TraceBatch<int16_t> tbatch)
{
    if (tbatch.Metadata().PoolId() != batchId_)
    {
        // TODO: Log error. Throw exception.
    }

    if (tbatch.Metadata().FirstFrame() != nextFrameId_)
    {
        // TODO: Log error. Throw exception.
    }

    // TODO: Define this so that it scales properly with block size, frame rate,
    // and max polymerization rate.
    const uint16_t maxCallsPerBlock = 96;

    // TODO: Implement the analysis logic!

    nextFrameId_ = tbatch.Metadata().LastFrame();

    return BasecallBatch(maxCallsPerBlock, tbatch.Dimensions(), tbatch.Metadata());
}

}}}     // namespace PacBio::Mongo::Basecaller
