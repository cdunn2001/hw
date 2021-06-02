// Copyright (c) 2018, Pacific Biosciences of California, Inc.
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

#include <bazio/PacketFieldMap.h>

#include "EventData.h"

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

const std::string EventDataParent::BaseQualityValues(size_t leftPulseIndex,
                                                  size_t rightPulseIndex) const
{
    if (HasPacketField(PacketFieldName::OVERALL_QV))
        return convertData(PacketField(PacketFieldName::OVERALL_QV),
                           IsBase(), FieldType::CHAR, leftPulseIndex,
                           rightPulseIndex).ToString();
    else if (HasPacketField(PacketFieldName::SUB_QV))
        return convertData(PacketField(PacketFieldName::SUB_QV),
                           IsBase(), FieldType::CHAR, leftPulseIndex,
                           rightPulseIndex).ToString();
    return {};
}

const std::vector<std::pair<std::string, BAM::Tag>>
EventDataParent::AvailableTagData(size_t pulseBegin, size_t pulseEnd) const
{
    std::vector<std::pair<std::string, BAM::Tag>> ret;
    ret.reserve(PacketFieldMap::packetBaseFieldToBamID.size() +
                PacketFieldMap::packetPulseFieldToBamID.size());
    if (Internal())
    {
        for (const auto& kv : PacketFieldMap::packetPulseFieldToBamID)
        {
            if (HasPacketField(kv.first))
            {
                const auto& tag = kv.second.first;
                const auto& data = PacketField(kv.first);
                const auto& type = kv.second.second;
                ret.push_back(
                        std::make_pair(tag,
                                       convertData(data, {}, type,
                                                   pulseBegin, pulseEnd)));
            }
        }
    }
    for (const auto& kv : PacketFieldMap::packetBaseFieldToBamID)
    {
        if (HasPacketField(kv.first))
        {
            const auto& tag = kv.second.first;
            const auto& data = PacketField(kv.first);
            const auto& type = kv.second.second;
            ret.push_back(
                    std::make_pair(tag,
                                   convertData(data, IsBase(), type,
                                               pulseBegin, pulseEnd)));
        }
    }
    return ret;
}

EventData::EventData(size_t zmwIdx,
                     size_t zmwNum,
                     bool truncated,
                     EventDataParent&& events,
                     std::vector<InsertState>&& states)
      : EventDataParent(std::move(events))
      , truncated_(truncated)
      , zmwIndex_(zmwIdx)
      , zmwNum_(zmwNum)
      , insertStates_(std::move(states))
{
    assert(insertStates_.size() == NumEvents());

    excludedBases_.reserve(insertStates_.size());
    std::transform(insertStates_.begin(), insertStates_.end(),
                   std::back_inserter(excludedBases_),
                   [](InsertState state) -> bool
                   {
                       return state == InsertState::BURST_PULSE;
                   }
    );

    baseCalls_.reserve(NumEvents());
    baseToPulseIndex_.reserve(NumEvents());
    baseToLeftmostPulseIndex_.reserve(NumEvents());
    const auto& readouts = Readouts();

    numBases_ = 0;
    size_t pulseIndex = 0;
    int lastBaseIndex = -1;
    for ( ; pulseIndex < NumEvents(); ++pulseIndex)
    {
        if (IsBase(pulseIndex))
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
            baseCalls_.push_back(static_cast<char>(readouts[pulseIndex]));
            ++numBases_;
        }
    }
    // Add a dummy base that maps to the pulse right after the last base.
    baseToPulseIndex_.push_back(pulseIndex);
    baseToLeftmostPulseIndex_.push_back(NumEvents());
}



void EventDataParent::UpdateIPDs(const std::vector<InsertState>& states)
{
    // Distances from the end of the last original pulse/base to
    // the end of the last accepted base.
    size_t pulseRelIdx = 0;
    size_t baseRelIdx = 0;

    for (size_t i = 0; i < states.size(); ++i)
    {
        if (states[i] == InsertState::BASE)
        {
            // Check ipd of subsequent bases is not changed
            if (!isBase_[i]) throw PBException("Attempting to promote pulse to base!");

            // base stayed base.  But the previous base may not have, in which
            // case baseRelIdx is nonzero and our ipd gets an update.
            ipds_[i] = baseRelIdx + ipds_[i];

            baseRelIdx = 0;
            pulseRelIdx = 0;
        }
        else
        {
            if (!isBase_[i])
            {
                // pulse stayed pulse
                ipds_[i] = ipds_[i];
                pulseRelIdx += ipds_[i] + pws_[i];
            }
            else
            {
                // base became pulse
                ipds_[i] = baseRelIdx + ipds_[i] - pulseRelIdx;
                pulseRelIdx += ipds_[i] + pws_[i];
                baseRelIdx = pulseRelIdx;
            }
        }
    }

    for (size_t i = 0; i < isBase_.size(); ++i)
    {
        if (states[i] != InsertState::BASE) isBase_[i] = false;
    }

    auto& rawIsBase = PacketField(PacketFieldName::IS_BASE);
    if (rawIsBase.size() != NumEvents()) rawIsBase.resize(NumEvents());
    for (size_t i = 0; i < NumEvents(); ++i)
    {
        rawIsBase[i] = isBase_[i];
    }

    // We've modified the ipds and pws in EventData (as uncompressed values),
    // and we need to write those back to the PacketFields so they are
    // reflected in the BAM tags
    if (Internal())
    {
        PacketField(PacketFieldName::IPD_LL) = ipds_;
        PacketField(PacketFieldName::PW_LL) = pws_;
    }
    else
    {
        for (auto& val : ipds_)
            val = codec_.FrameToCode(static_cast<uint16_t>(val));
        for (auto& val : pws_)
            val = codec_.FrameToCode(static_cast<uint16_t>(val));
        PacketField(PacketFieldName::IPD_V1) = ipds_;
        PacketField(PacketFieldName::PW_V1) = pws_;
    }
}

void EventDataParent::ComputeAdditionalData()
{
    PacketField(PacketFieldName::PX_LL) = pws_;
    auto& pdLL  = PacketField(PacketFieldName::PD_LL);
    auto& sf    = PacketField(PacketFieldName::START_FRAME);

    uint32_t framesSinceLastBase = 0;
    uint32_t currentStartFrame = 0;
    sf.resize(ipds_.size());
    pdLL.resize(ipds_.size());
    for (size_t i = 0; i < ipds_.size(); ++i)
    {
        // This is a mess, but that is because IPD is has different contextual meanings.
        // IPD means number of frames to previous event IF the current event is a PULSE
        // IPD means number of frames to previous BASE if the current event is a BASE
        // We first convert to "PD" which number of frames to previous event, regardless of event type.
        uint32_t pd;

        if (!isBase_[i])
        {
            // PD is same as IPD
            framesSinceLastBase += ipds_[i] + pws_[i];
            pd = ipds_[i];
        }
        else
        {
            if (framesSinceLastBase > ipds_[i]) throw PBException("ASONAS " + std::to_string(framesSinceLastBase) + " " + std::to_string(ipds_[i]));
            // PD is not the same at IPD. Need to correct by subtracting frames accumulated from intervening pulses.
            pd =  ipds_[i] - framesSinceLastBase;
            framesSinceLastBase = 0;
        }
        pdLL[i] = pd;

        currentStartFrame += pd;
        sf[i] = currentStartFrame;
        currentStartFrame += pws_[i];
    }
}

}}} // ::PacBio::Primary::Postprimary
