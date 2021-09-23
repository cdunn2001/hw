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

#include <dataTypes/BasecallingMetrics.h>

#include "SimulateWriteUtils.h"

namespace PacBio::Primary::Postprimary
{

namespace {

std::vector<SimMetricT> ConvertMetrics(const std::vector<Primary::SpiderMetricBlock>& sms)
{
    std::vector<SimMetricT> smts;
    for (const auto& sm : sms)
    {
        Mongo::Data::BasecallingMetrics bcm;

        bcm.activityLabel = static_cast<Mongo::Data::HQRFPhysicalStates>(sm.activityLabel_);
        bcm.numPulseFrames = sm.pulseWidth_;
        bcm.numBaseFrames = sm.baseWidth_;
        bcm.numSandwiches = sm.numSandwiches_;
        bcm.numHalfSandwiches = sm.numHalfSandwiches_;
        bcm.numPulseLabelStutters = sm.numPulseLabelStutters_;
        bcm.numBases = sm.numBasesA_ + sm.numBasesC_ + sm.numBasesG_ + sm.numBasesT_;
        bcm.numPulses = sm.numPulses_;

        bcm.pkMidSignal[0] = sm.pkmidA_ * sm.numpkmidFramesA_;
        bcm.pkMidSignal[1] = sm.pkmidC_ * sm.numpkmidFramesC_;
        bcm.pkMidSignal[2] = sm.pkmidG_ * sm.numpkmidFramesG_;
        bcm.pkMidSignal[3] = sm.pkmidT_ * sm.numpkmidFramesT_;

        bcm.bpZvar[0] = sm.bpzvarA_;
        bcm.bpZvar[1] = sm.bpzvarC_;
        bcm.bpZvar[2] = sm.bpzvarG_;
        bcm.bpZvar[3] = sm.bpzvarT_;

        bcm.pkZvar[0] = sm.pkzvarA_;
        bcm.pkZvar[1] = sm.pkzvarC_;
        bcm.pkZvar[2] = sm.pkzvarG_;
        bcm.pkZvar[3] = sm.pkzvarT_;

        bcm.pkMax[0] = sm.pkmaxA_;
        bcm.pkMax[1] = sm.pkmaxC_;
        bcm.pkMax[2] = sm.pkmaxG_;
        bcm.pkMax[3] = sm.pkmaxT_;

        bcm.numPkMidFrames[0] = sm.numpkmidFramesA_;
        bcm.numPkMidFrames[1] = sm.numpkmidFramesC_;
        bcm.numPkMidFrames[2] = sm.numpkmidFramesG_;
        bcm.numPkMidFrames[3] = sm.numpkmidFramesT_;

        bcm.numPkMidBasesByAnalog[0] = sm.numPkmidBasesA_;
        bcm.numPkMidBasesByAnalog[1] = sm.numPkmidBasesC_;
        bcm.numPkMidBasesByAnalog[2] = sm.numPkmidBasesG_;
        bcm.numPkMidBasesByAnalog[3] = sm.numPkmidBasesT_;

        bcm.numBasesByAnalog[0] = sm.numBasesA_;
        bcm.numBasesByAnalog[1] = sm.numBasesC_;
        bcm.numBasesByAnalog[2] = sm.numBasesG_;
        bcm.numBasesByAnalog[3] = sm.numBasesT_;

        bcm.numPulsesByAnalog[0] = 0;
        bcm.numPulsesByAnalog[1] = 0;
        bcm.numPulsesByAnalog[2] = 0;
        bcm.numPulsesByAnalog[3] = 0;

        bcm.frameBaselineDWS = sm.baselines_[0];
        bcm.frameBaselineVarianceDWS = sm.baselineSds_[0] * sm.baselineSds_[0];
        bcm.numFramesBaseline = sm.numBaselineFrames_[0];
        bcm.startFrame = 0;
        bcm.numFrames = sm.numFrames_;
        bcm.autocorrelation = sm.traceAutoCorr_;
        bcm.pulseDetectionScore = sm.pulseDetectionScore_;
        bcm.pixelChecksum = sm.pixelChecksum_;

        smts.emplace_back(SimMetricT{bcm, 0});
    }

    return smts;
}

}

SimBazWriter::~SimBazWriter() = default;

// Configure the buffer to expect 5 cache lines of data
// per ZMW.  This corresponds loosely to the expected
// Kestrel production configuration, but since this is
// code used just in test, it doesn't really matter here
static constexpr uint64_t bytesPerZmw = 320;

SimBazWriter::SimBazWriter(const std::string& fileName,
                           BazIO::FileHeaderBuilder& fhb,
                           const PacBio::Primary::BazIOConfig& conf, bool)
    : numZmw_(fhb.MaxNumZmws())
    , writer_(std::make_unique<BazIO::BazWriter>(fileName, fhb, conf))
    , aggregator_(std::make_unique<SimBazAggregatorT>(numZmw_, 0, bytesPerZmw))
{
    const auto& fields = writer_->GetFileHeaderBuilder().PacketFields();
    internal_ = std::any_of(fields.begin(), fields.end(), [](const BazIO::FieldParams<BazIO::PacketFieldName>& fp) { return fp.name == BazIO::PacketFieldName::IsBase; });
    if (internal_) internalSerializer_.resize(numZmw_);
    else prodSerializer_.resize(numZmw_);
}

void SimBazWriter::AddZmwSlice(SimPulse* basecalls, size_t numEvents,
                               std::vector<Primary::SpiderMetricBlock>&& metrics, size_t zmw)
{
    totalEvents_ += numEvents;
    if (!metrics.empty())
    {
        aggregator_->AddMetrics(zmw, ConvertMetrics(metrics));
    }
    if (internal_)
        aggregator_->AddPulses(zmw, basecalls, basecalls + numEvents, [](const auto) { return true; }, internalSerializer_[zmw]);
    else
        aggregator_->AddPulses(zmw, basecalls, basecalls + numEvents, [](const SimPulse& p) { return !p.IsReject(); }, prodSerializer_[zmw]);
}

}
