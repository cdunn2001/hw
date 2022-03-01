// Copyright (c) 2018,2021, Pacific Biosciences of California, Inc.
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

#include <limits>

#include <boost/numeric/conversion/converter.hpp>
#include <boost/numeric/conversion/cast.hpp>

#include <pacbio/logging/Logger.h>

#include <bazio/FieldType.h>
#include <postprimary/bam/EventData.h>

namespace PacBio {
namespace Primary {
namespace Postprimary {

namespace {

template <typename OutputType, typename InputType>
std::vector<OutputType> boundValues(const std::vector<InputType>& in,
                                    const std::vector<bool>& mask,
                                    size_t pulseBegin, size_t pulseEnd)
{
    const auto& resolvedMask = (mask.size() == 0) ? std::vector<bool>(in.size(), true)
                                                  : mask;

    InputType maxValue = boost::numeric_cast<InputType>(
            std::numeric_limits<OutputType>::max());
    InputType minValue = 0;

    std::vector<OutputType> res;
    res.reserve(pulseEnd - pulseBegin);
    for (size_t i = pulseBegin; i < pulseEnd; ++i)
    {
        if (resolvedMask[i])
        {
            res.push_back(static_cast<OutputType>(
                                  std::max(std::min(in[i], maxValue), minValue)));
        }
    }
    return res;
}

BAM::Tag convertData(const std::vector<uint32_t>& data,
                     const std::vector<bool>& mask,
                     const FieldType& type,
                     size_t pulseBegin, size_t pulseEnd)
{
    BAM::Tag tag;
    switch(type)
    {
        case FieldType::CHAR:
        {
            const auto& chars = boundValues<char>(data, mask,
                                                  pulseBegin, pulseEnd);
            tag = std::string(chars.begin(), chars.end());
            break;
        }
        case FieldType::UINT8:
        {
            tag = boundValues<uint8_t>(data, mask, pulseBegin, pulseEnd);
            break;
        }
        case FieldType::UINT16:
        {
            tag = boundValues<uint16_t>(data, mask, pulseBegin, pulseEnd);
            break;
        }
        case FieldType::UINT32:
        {
            tag = boundValues<uint32_t>(data, mask, pulseBegin, pulseEnd);
            break;
        }
        default: throw PBException("Not aware of that FileType!");
    }
    return tag;
}

} // anonymous namespace

std::string EventData::BaseQualityValues(size_t leftPulseIndex,
                                         size_t rightPulseIndex) const
{
    static const bool warnOnce = [](){
        PBLOG_WARN << "QualityValues not currently supported\n";
        return true;
    }();
    (void) warnOnce;
    //if (HasPacketField(PacketFieldName::OVERALL_QV))
    //    return convertData(PacketField(PacketFieldName::OVERALL_QV),
    //                       IsBase(), FieldType::CHAR, leftPulseIndex,
    //                       rightPulseIndex).ToString();
    //else if (HasPacketField(PacketFieldName::SUB_QV))
    //    return convertData(PacketField(PacketFieldName::SUB_QV),
    //                       IsBase(), FieldType::CHAR, leftPulseIndex,
    //                       rightPulseIndex).ToString();
    const auto numBases = std::count(IsBase().begin() + leftPulseIndex,
                                     IsBase().begin() + rightPulseIndex,
                                     true);
    return std::string(numBases, '!');
}

const std::vector<std::pair<std::string, BAM::Tag>>
EventData::AvailableTagData(size_t pulseBegin, size_t pulseEnd) const
{
    std::vector<std::pair<std::string, BAM::Tag>> ret;
    auto AddIntField = [&](auto& vec, const std::string& tag,
                        const std::vector<bool>& filter, FieldType type)
    {
        ret.push_back(std::make_pair(tag, convertData(vec, filter, type, pulseBegin, pulseEnd)));
    };
    auto AddFloatField = [&](auto& vec, const std::string& tag,
                             const std::vector<bool>& filter, FieldType type)
    {
        std::vector<uint32_t> intVec;
        intVec.reserve(vec.size());
        int bamFixedPointFactor = 10;
        // Need to be careful.  static_cast<uint32_t>(static_cast<float>(UINT_MAX));
        // is UB because UINT_MAX is not perfectly representable in single precision and gets
        // rounded up to a value outside the range of uint32_t.
        const float maxVal = std::nextafterf(static_cast<float>(std::numeric_limits<uint32_t>::max()), 0.0f);
        std::transform(vec.begin(), vec.end(), std::back_inserter(intVec),
                       [&](float val) {
                           auto fixedPoint = std::round(val * bamFixedPointFactor);

                           // Note: NaN values are being pegged to zero, since that keeps us compatible
                           //       with what we've traditionlly done in Sequel.  There baz serialization
                           //       converted NaN's to 0, but now that Kestrel serialization preserves
                           //       NaN/inf, then we need to handle that here before inserting into the
                           //       baz file.
                           // Note: negative numbers are *not* expected, but we'll peg them to zero as
                           //       well just in case, so we avoid any UB when casting the float.
                           assert(std::isnan(fixedPoint) || fixedPoint >= 0);
                           if (fixedPoint > maxVal) return std::numeric_limits<uint32_t>::max();
                           if (std::isnan(fixedPoint) || fixedPoint < 0) return 0u;
                           return static_cast<uint32_t>(fixedPoint);
                       });
        ret.push_back(std::make_pair(tag, convertData(intVec, filter, type, pulseBegin, pulseEnd)));
    };
    if (Internal())
    {
        AddFloatField(bazEvents_.PkMeans(), "pa", {}, FieldType::UINT16);
        AddFloatField(bazEvents_.PkMids(),  "pm", {}, FieldType::UINT16);
        AddIntField(PulseWidths(),  "px", {}, FieldType::UINT16);
        AddIntField(ipdsWithPulses_,  "pd", {}, FieldType::UINT16);

        // supported in Sequel but not Kestrel yet.  Keeping just so I don't
        // have to hunt down what the TAG string is supposed to be...
        // {PacketFieldName::LAB_QV,       std::make_pair("pq", FieldType::CHAR)},
        // {PacketFieldName::ALT_QV,       std::make_pair("pv", FieldType::CHAR)},
        // {PacketFieldName::ALT_LABEL,    std::make_pair("pt", FieldType::CHAR)},
        // {PacketFieldName::PKMEAN2_LL,   std::make_pair("ps", FieldType::UINT32)},
        // {PacketFieldName::PKMID2_LL,    std::make_pair("pi", FieldType::UINT32)},
        // {PacketFieldName::PULSE_MRG_QV, std::make_pair("pg", FieldType::CHAR)},
    }
    // Always prodution fields
    {
        // supported in Sequel but not Kestrel yet.  Keeping just so I don't
        // have to hunt down what the TAG string is supposed to be...
        // {PacketFieldName::DEL_TAG, std::make_pair("dt", FieldType::CHAR)},
        // {PacketFieldName::SUB_TAG, std::make_pair("st", FieldType::CHAR)},
        // {PacketFieldName::DEL_QV,  std::make_pair("dq", FieldType::CHAR)},
        // {PacketFieldName::SUB_QV,  std::make_pair("sq", FieldType::CHAR)},
        // {PacketFieldName::INS_QV,  std::make_pair("iq", FieldType::CHAR)},
        // {PacketFieldName::MRG_QV,  std::make_pair("mq", FieldType::CHAR)},

        // Note: need to sort out compression.  We want these lossless, but
        //       will gzip be enough for us or do we need to compress as we
        //       do in the baz file
        //       PTSD-568 is a followup story that should address these
        //       questions
        AddIntField(bazEvents_.StartFrames(), "sf", {}, FieldType::UINT32);
        AddIntField(Ipds(), "ip", IsBase(), FieldType::UINT16);
        AddIntField(PulseWidths(), "pw", IsBase(), FieldType::UINT16);
        // {PacketFieldName::IPD_V1,  std::make_pair("ip", FieldType::UINT8)},
        // {PacketFieldName::PW_V1,   std::make_pair("pw", FieldType::UINT8)}
    }
    return ret;
}

EventData::EventData(size_t zmwIdx,
                     size_t zmwNum,
                     bool truncated,
                     BazIO::BazEventData&& events,
                     std::vector<InsertState>&& states)
      : truncated_(truncated)
      , zmwIndex_(zmwIdx)
      , zmwNum_(zmwNum)
      , bazEvents_(std::move(events))
      , insertStates_(std::move(states))
{
    assert(insertStates_.size() == NumEvents());
    assert(StartFrames().size() == NumEvents());
    assert(PulseWidths().size() == NumEvents());

    isBase_ = std::vector<bool>(NumEvents(), true);
    for (size_t i = 0; i < NumEvents(); ++i)
    {
        if (insertStates_[i] == InsertState::BASE)
            assert(events.IsBase().empty() || events.IsBase(i));
        else isBase_[i] = false;
    }

    ipdsWithPulses_.resize(NumEvents());
    ipdsSkipPulses_.resize(NumEvents());
    excludedBases_.resize(NumEvents());
    size_t lastPulse = 0;
    size_t lastBase = 0;
    for (size_t i = 0; i < NumEvents(); ++i)
    {
        excludedBases_[i] = (insertStates_[i] == InsertState::BURST_PULSE);
        ipdsWithPulses_[i] = StartFrames(i) - lastPulse;
        lastPulse = StartFrames(i) + PulseWidths(i);
        if (isBase_[i])
        {
            ipdsSkipPulses_[i] = StartFrames(i) - lastBase;
            lastBase = lastPulse;
        }
        else
        {
            ipdsSkipPulses_[i] = ipdsWithPulses_[i];
        }
    }

    baseCalls_.reserve(NumEvents());
    baseToPulseIndex_.reserve(NumEvents());
    baseToLeftmostPulseIndex_.reserve(NumEvents());
    const auto& readouts = Readouts();

    numBases_ = 0;
    size_t pulseIndex = 0;
    int lastBaseIndex = -1;
    for ( ; pulseIndex < NumEvents(); ++pulseIndex)
    {
        if (isBase_[pulseIndex])
        {
            baseToPulseIndex_.push_back(pulseIndex);
            if (lastBaseIndex == -1)
                baseToLeftmostPulseIndex_.push_back(0);
            else if (pulseIndex - lastBaseIndex > 1)
                baseToLeftmostPulseIndex_.push_back(lastBaseIndex + 1);
            else
                baseToLeftmostPulseIndex_.push_back(pulseIndex);
            lastBaseIndex = pulseIndex;
            // baseToPulseIndexRight.push_back(i);
            baseCalls_.push_back(readouts[pulseIndex]);
            ++numBases_;
        }
    }
    // Add a dummy base that maps to the pulse right after the last base.
    baseToPulseIndex_.push_back(pulseIndex);
    baseToLeftmostPulseIndex_.push_back(NumEvents());
}

}}} // ::PacBio::Primary::Postprimary
