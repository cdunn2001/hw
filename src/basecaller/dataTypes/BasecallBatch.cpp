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

BasecallingMetrics& BasecallingMetrics::Count(const PacBio::Mongo::Data::BasecallingMetrics::Basecall& base)
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

BasecallBatch::BasecallBatch(
        const size_t maxCallsPerZmwChunk,
        const BatchDimensions& batchDims,
        const BatchMetadata& batchMetadata,
        Cuda::Memory::SyncDirection syncDir,
        bool pinned,
        std::shared_ptr<Cuda::Memory::DualAllocationPools> callsPool,
        std::shared_ptr<Cuda::Memory::DualAllocationPools> lenPool,
        std::shared_ptr<Cuda::Memory::DualAllocationPools> metricsPool)
    : dims_ (batchDims)
    , metaData_(batchMetadata)
    , basecalls_(batchDims.ZmwsPerBatch(),  maxCallsPerZmwChunk, syncDir, pinned, callsPool, lenPool)
    , metrics_(batchDims.ZmwsPerBatch(), syncDir, pinned, metricsPool)
{}

}}}     // namespace PacBio::Mongo::Data
