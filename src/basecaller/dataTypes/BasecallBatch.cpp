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
//  Defines members of class BasecallBatch.

#include "BasecallBatch.h"

namespace PacBio {
namespace Mongo {
namespace Data {

BasecallingMetrics::BasecallingMetrics()
    : numBasesByAnalog_{ 0, 0, 0, 0}
    , numPulsesByAnalog_{0, 0, 0, 0}
{ }

BasecallingMetrics& BasecallingMetrics::PushBack(const PacBio::Mongo::Data::BasecallingMetrics::Basecall& base)
{
    uint8_t pulseLabel = static_cast<uint8_t>(base.GetPulse().Label());
    numPulsesByAnalog_[pulseLabel]++;

    if (!base.IsNoCall())
    {
        uint8_t baseLabel = static_cast<uint8_t>(base.Base());
        numBasesByAnalog_[baseLabel]++;
    }

    return *this;
}

BasecallBatch::BasecallBatch(const size_t maxCallsPerZmwChunk,
                             const BatchDimensions& batchDims,
                             const BatchMetadata& batchMetadata)
    : dims_ (batchDims)
    , maxCallsPerZmwChunk_ (maxCallsPerZmwChunk)
    , metaData_ (batchMetadata)
    , seqLengths_ (batchDims.zmwsPerBatch(), 0)
    , basecalls_ (batchDims.zmwsPerBatch() * maxCallsPerZmwChunk)
    , metrics_ (batchDims.zmwsPerBatch())
{ }

void BasecallBatch::PushBack(uint32_t z, Basecall bc)
{
    if (seqLengths_[z] < maxCallsPerZmwChunk_)
    {
        basecalls_[zmwOffset(z) + seqLengths_[z]] = bc;
        metrics_[z].PushBack(bc);
        seqLengths_[z]++;
    }
}

}}}     // namespace PacBio::Mongo::Data
