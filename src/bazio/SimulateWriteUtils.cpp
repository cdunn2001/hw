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

#include <bazio/SimulateWriteUtils.h>

namespace PacBio {
namespace BazIO {

SimBazWriter::~SimBazWriter() = default;

SimBazWriter::SimBazWriter(const std::string& fileName,
                           BazIO::FileHeaderBuilder& fhb,
                           const PacBio::Primary::BazIOConfig& conf, bool)
    : numZmw_(fhb.MaxNumZmws())
    , writer_(std::make_unique<BazIO::BazWriter>(fileName, fhb, conf))
    , aggregator_(std::make_unique<BazAggregator>(numZmw_, 0, 300))
{
    const auto& fields = writer_->GetFileHeaderBuilder().PacketFields();
    internal_ = std::any_of(fields.begin(), fields.end(), [](const FieldParams& fp) { return fp.name == BazIO::PacketFieldName::IsBase; });
    if (internal_) internalSerializer_.resize(numZmw_);
    else prodSerializer_.resize(numZmw_);
}

void SimBazWriter::AddZmwSlice(SimPulse* basecalls, size_t numEvents,
                               std::vector<Primary::SpiderMetricBlock>&& metrics, size_t zmw)
{
    totalEvents_ += numEvents;
    if (!metrics.empty())
    {
        aggregator_->AddMetrics(zmw,
                            [&](BazIO::MemoryBufferView<Primary::SpiderMetricBlock>& dest)
                            {
                                for (size_t i = 0; i < metrics.size(); ++i)
                                {
                                    dest[i] = metrics[i];
                                }
                            },
                            metrics.size());
    }
    if (internal_)
        aggregator_->AddZmw(zmw, basecalls, basecalls + numEvents, [](const auto) { return true; }, internalSerializer_[zmw]);
    else
        aggregator_->AddZmw(zmw, basecalls, basecalls + numEvents, [](const SimPulse& p) { return !p.IsReject(); }, prodSerializer_[zmw]);
}



}}
