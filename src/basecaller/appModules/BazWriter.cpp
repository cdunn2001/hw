// Copyright (c) 2020, Pacific Biosciences of California, Inc.
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

#include <appModules/BazWriter.h>

#include <pacbio/logging/Logger.h>
#include <pacbio/primary/FileHeaderBuilder.h>
#include <pacbio/primary/ZmwResultBuffer.h>

#include <common/MongoConstants.h>

#include <dataTypes/Pulse.h>

using namespace PacBio::Primary;
using namespace PacBio::Mongo;
using namespace PacBio::Mongo::Data;

namespace PacBio {
namespace Application {

namespace {

void ConvertMetric(const std::unique_ptr<BatchResult::MetricsT>& metricsPtr,
                   SpiderMetricBlock& sm,
                   size_t laneIndex,
                   size_t zmwIndex)
{
    if (metricsPtr)
    {
        const auto& metrics = metricsPtr->GetHostView()[laneIndex];
        sm.numBasesA_ = metrics.numBasesByAnalog[0][zmwIndex];
        sm.numBasesC_ = metrics.numBasesByAnalog[1][zmwIndex];
        sm.numBasesG_ = metrics.numBasesByAnalog[2][zmwIndex];
        sm.numBasesT_ = metrics.numBasesByAnalog[3][zmwIndex];

        sm.numPulses_ = metrics.numBases[zmwIndex];
    }
}

auto ConvertMongoPulsesToSequelBasecalls(const Pulse* pulses, uint32_t numPulses)
{
    auto LabelConv = [](Pulse::NucleotideLabel label) {
        switch (label)
        {
            case Pulse::NucleotideLabel::A:
                return PacBio::SmrtData::NucleotideLabel::A;
            case Pulse::NucleotideLabel::C:
                return PacBio::SmrtData::NucleotideLabel::C;
            case Pulse::NucleotideLabel::G:
                return PacBio::SmrtData::NucleotideLabel::G;
            case Pulse::NucleotideLabel::T:
                return PacBio::SmrtData::NucleotideLabel::T;
            case Pulse::NucleotideLabel::N:
                return PacBio::SmrtData::NucleotideLabel::N;
            default:
                assert(label == Pulse::NucleotideLabel::NONE);
                return PacBio::SmrtData::NucleotideLabel::NONE;
        }
    };

    std::vector<Basecall> baseCalls(numPulses);
    for (size_t pulseNum = 0; pulseNum < numPulses; ++pulseNum)
    {
        const auto& pulse = pulses[pulseNum];
        auto& bc = baseCalls[pulseNum];

        static constexpr int8_t qvDefault_ = 0;

        auto label = LabelConv(pulse.Label());

        // Populate pulse data
        bc.GetPulse().Start(pulse.Start()).Width(pulse.Width());
        bc.GetPulse().MeanSignal(pulse.MeanSignal()).MidSignal(pulse.MidSignal()).MaxSignal(pulse.MaxSignal());
        bc.GetPulse().Label(label).LabelQV(qvDefault_);
        bc.GetPulse().AltLabel(label).AltLabelQV(qvDefault_);
        bc.GetPulse().MergeQV(qvDefault_);

        // Populate base data.
        bc.Base(label).InsertionQV(qvDefault_);
        bc.DeletionTag(PacBio::SmrtData::NucleotideLabel::N).DeletionQV(qvDefault_);
        bc.SubstitutionTag(PacBio::SmrtData::NucleotideLabel::N).SubstitutionQV(qvDefault_);
    }
    return baseCalls;
}

} // anon

BazWriterBody::BazWriterBody(
        const std::string& bazName,
        size_t expectedFrames,
        const std::vector<uint32_t>& zmwNumbers,
        const std::vector<uint32_t>& zmwFeatures,
        const BasecallerConfig& basecallerConfig,
        size_t outputStride)
    : bazName_(bazName)
    , zmwOutputStrideFactor_(outputStride)
{
    PBLOG_INFO << "Opening BAZ file for writing: " << bazName_ << " zmws: " << zmwNumbers.size();

    FileHeaderBuilder fh(bazName_,
                         100.0f,
                         expectedFrames,
                         Readout::BASES,
                         MetricsVerbosity::MINIMAL,
                         "",
                         basecallerConfig.Serialize().toStyledString(),
                         zmwNumbers,
                         zmwFeatures,
                         PacBio::Mongo::Data::GetPrimaryConfig().framesPerChunk,
                         PacBio::Mongo::Data::GetPrimaryConfig().framesPerChunk,
                         PacBio::Mongo::Data::GetPrimaryConfig().framesPerChunk,
                         false,
                         true,
                         true,
                         false);
    fh.BaseCallerVersion("0.1");

    bazWriter_.reset(new BazWriter(bazName_, fh, BazIOConfig{}, ReadBuffer::MaxNumSamplesPerBuffer()));
}

void BazWriterBody::Process(BatchResult in)
{
    static const std::string bazWriterError = "BazWriter has failed. Last error message was ";

    const auto& pulseBatch = in.pulses;
    const auto& metricsPtr = in.metrics;

    if (pulseBatch.GetMeta().FirstFrame() > currFrame_)
    {
        bazWriter_->Flush();
        currFrame_ = pulseBatch.GetMeta().FirstFrame();
    } else if (pulseBatch.GetMeta().FirstFrame() < currFrame_)
    {
        throw PBException("Data out of order, multiple chunks being processed simultaneously");
    }

    const auto& primaryConfig = PacBio::Mongo::Data::GetPrimaryConfig();
    // TODO this needs to change once we support sparse layout for trace re-analysis
    // TODO note, this maybe needs to be contiguous integers?  Unless we can guarantee order of inputs, we may need more robust bookkeeping
    size_t currentZmwIndex = pulseBatch.GetMeta().PoolId() * primaryConfig.lanesPerPool * primaryConfig.zmwsPerLane;
    for (uint32_t lane = 0; lane < pulseBatch.Dims().lanesPerBatch; ++lane)
    {
        const auto& lanePulses = pulseBatch.Pulses().LaneView(lane);

        for (uint32_t zmw = 0; zmw < laneSize; zmw++)
        {
            if (currentZmwIndex % zmwOutputStrideFactor_ == 0)
            {
                const auto& baseCalls = ConvertMongoPulsesToSequelBasecalls(lanePulses.ZmwData(zmw),
                                                                            lanePulses.size(zmw));
                if (!bazWriter_->AddZmwSlice(baseCalls.data(),
                                             baseCalls.size(),
                                             [&](MemoryBufferView<SpiderMetricBlock>& dest)
                                             {
                                                for (size_t i = 0; i < dest.size(); i++)
                                                {
                                                    ConvertMetric(metricsPtr, dest[i], lane, zmw);
                                                }
                                             },
                                             1,
                                             currentZmwIndex))
                {
                    throw PBException(bazWriterError + bazWriter_->ErrorMessage());
                }
            }
            else
            {
                if (!bazWriter_->AddZmwSlice(NULL, 0, [](auto&){}, 0, currentZmwIndex))
                {
                    throw PBException(bazWriterError + bazWriter_->ErrorMessage());
                }
            }
            currentZmwIndex++;
        }
    }
}

}}
