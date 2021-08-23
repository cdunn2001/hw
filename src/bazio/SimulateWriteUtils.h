// Copyright (c) 2021, Pacific Biosciences of California, Inc.
//
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted (subject to the limitations in the
// disclaimer below) provided that the following conditions are met:
//
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//
//  * Redistributions in binary form must reproduce the above
//    copyright notice, this list of conditions and the following
//    disclaimer in the documentation and/or other materials provided
//    with the distribution.
//
//  * Neither the name of Pacific Biosciences nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE
// GRANTED BY THIS LICENSE. THIS SOFTWARE IS PROVIDED BY PACIFIC
// BIOSCIENCES AND ITS CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
// OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL PACIFIC BIOSCIENCES OR ITS
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
// USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
// OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
// SUCH DAMAGE.

#ifndef PACBIO_BAZIO_WRITE_UTILS_H
#define PACBIO_BAZIO_WRITE_UTILS_H

#include <memory>

#include <bazio/encoding/Types.h>
#include <bazio/writing/BazAggregator.h>
#include <bazio/writing/BazWriter.h>
#include <bazio/encoding/PulseToBaz.h>
#include <bazio/encoding/test/TestingPulse.h>

namespace PacBio {
namespace BazIO {

// TODO All this Simulate code should be in postprimary, not bazio.
// Once that happens, these placeholders can potentially be removed
// and the simulation code define it's own versions, if it doesn't
// want to be coupled with bazio directly

using SimProductionPulseGroups = ProductionPulses;
using SimInternalPulseGroups = InternalPulses;
using SimPulse = Pulse;

class SimBazWriter
{
public:
    SimBazWriter(const std::string& fileName,
                 FileHeaderBuilder& fhb,
                 const PacBio::Primary::BazIOConfig& conf, bool);
    ~SimBazWriter();

    void AddZmwSlice(SimPulse* basecalls,
                     size_t numEvents,
                     std::vector<Primary::SpiderMetricBlock>&& metrics, size_t zmw);
    void Flush()
    {
        writer_->Flush(aggregator_->ProduceBazBuffer());
    }

    void WaitForTermination()
    {
        writer_->WaitForTermination();
    }
    void Summarize(std::ostream& out) { writer_->Summarize(out); }

    const FileHeaderBuilder& GetFileHeaderBuilder() const { return writer_->GetFileHeaderBuilder(); }
    size_t NumEvents() const { return totalEvents_; }
    size_t BytesWritten() const { return writer_->BytesWritten(); }
    std::string Summary() const { return writer_->Summary(); }
    const std::string& FilePath() const { return writer_->FilePath(); }
private:
    size_t totalEvents_ = 0;
    size_t numZmw_;
    bool internal_;

    std::unique_ptr<BazIO::BazWriter> writer_;
    std::unique_ptr<BazIO::BazAggregator> aggregator_;

    std::vector<SimInternalPulseGroups> internalSerializer_;
    std::vector<SimProductionPulseGroups> prodSerializer_;
};

}}

#endif //PACBIO_BAZIO_WRITE_UTILS_H
