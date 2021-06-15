// Copyright (c) 2014-2018, Pacific Biosciences of California, Inc.
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

#include <bazio/PacketFieldMap.h>
#include <bazio/RegionLabel.h>

#include <postprimary/stats/ProductivityClass.h>
#include <postprimary/stats/ProductivityMetrics.h>

#include "SubreadSplitter.h"

namespace PacBio {
namespace Primary {
namespace Postprimary {

ResultPacket SubreadSplitter::PartialReadToResultPacket(
    const ProductivityInfo& pinfo,
    const EventData& events,
    const SubreadContext& lc,
    bool nobam,
    const std::string& movieName,
    const std::unordered_map<size_t, size_t>& base2frame)
{
    auto result = PartialReadToResultPacket(pinfo, events, lc.begin, lc.end,
                                            lc.label,
                                            nobam, movieName, base2frame);
    // Encoding local context
    result.context = SubreadContext(lc);
    if (result.context.label == RegionLabelType::INSERT)
    {
        uint8_t cx = 0;
        if (result.context.adapterBefore)         cx |= 1;
        if (result.context.adapterAfter)          cx |= 2;
        if (result.context.barcodeIdBefore != -1) cx |= 4;
        if (result.context.barcodeIdAfter  != -1) cx |= 8;
        if (result.context.adapterAccuracyBefore == 0.0f
                && result.context.adapterFlankingScoreBefore == 0.0f
                && result.context.adapterBefore)
            cx |= 64;
        if (result.context.adapterAccuracyAfter == 0.0f
                && result.context.adapterFlankingScoreAfter == 0.0f
                && result.context.adapterAfter)
            cx |= 128;
        result.tags["cx"] = cx;
        result.cx = cx;
    }

    return result;
}

ResultPacket SubreadSplitter::PartialReadToResultPacket(
    const ProductivityInfo& pinfo,
    const EventData& events,
    const size_t begin,
    const size_t end,
    const RegionLabelType label,
    bool nobam,
    const std::string& movieName,
    const std::unordered_map<size_t, size_t>& base2frame)
{
    // We normally wouldn't allow labels for zero-length reads, but we'll
    // make an exception for adapters
    if (begin == end && label != RegionLabelType::ADAPTER)
    {
        return ResultPacket();
    }
    if (begin > end)
    {
        PBLOG_ERROR << "Problem 2 " << begin << "-" << end << " " << static_cast<char>(label);
        return ResultPacket();
        // exit(0);
    }

    size_t leftPulseIndex = events.BaseToLeftmostPulseIndex(begin);
    size_t rightPulseIndex = events.BaseToLeftmostPulseIndex(end);

    ResultPacket result;
    result.zmwNum = events.ZmwNumber();
    result.name = movieName
        + "/"
        + std::to_string(result.zmwNum)
        + "/"
        + std::to_string(begin)
        + "_"
        + std::to_string(end);
    result.startPos = begin;

    std::string bases(events.BaseCalls().begin() + begin,
                      events.BaseCalls().begin() + end);

    const auto& baseQVs = events.BaseQualityValues(leftPulseIndex, rightPulseIndex);
    if (baseQVs.size() != 0)
        result.overallQv = baseQVs;

    result.length = bases.size();
    result.label = label;

    // Encode data to BamAlignment
    result.bases = bases;

    if (nobam) return result;

    result.bamRecord.Impl().Name(result.name);
    result.tags["zm"] = static_cast<int32_t>(result.zmwNum);
    result.tags["qs"] = static_cast<int32_t>(begin);
    result.tags["qe"] = static_cast<int32_t>(end);
    result.tags["np"] = static_cast<int32_t>(1);
    result.bamRecord.Impl().SetSequenceAndQualities(bases.c_str(),
                                                    bases.size(),
                                                    result.overallQv.c_str());
    result.bamRecord.Impl().CigarData(std::string(""));
    result.bamRecord.Impl().ReferenceId(-1);
    result.bamRecord.Impl().MatePosition(-1);
    result.bamRecord.Impl().MateReferenceId(-1);
    result.bamRecord.Impl().Flag(BAM::BamRecordImpl::AlignmentFlag::UNMAPPED);

    if (base2frame.size() > 0)
    {
        result.tags["ws"] = static_cast<int32_t>(base2frame.at(begin));
        // End is exclusive, but we want the first frame of the last pulse, so
        // we subtract 1
        result.tags["we"] = static_cast<int32_t>(base2frame.at(std::max(begin, end - 1)));
    }

    const bool isNotSequencing = (pinfo.isSequencing == 0);

    const bool productivityOther = (pinfo.productivity == ProductivityClass::OTHER);

    if (pinfo.snr)
    {
        std::vector<float> snrs = {{
            pinfo.snr->A,
            pinfo.snr->C,
            pinfo.snr->G,
            pinfo.snr->T
                                   }};
        result.tags["sn"] = snrs;
    }
    else if (isNotSequencing || productivityOther)
    {
        // Non-sequencing ZMW where SNR isn't computed.
        std::vector<float> snrs = {{ 0.0f, 0.0f, 0.0f, 0.0f }};
        result.tags["sn"] = snrs;
    }
    else
        throw std::runtime_error("SNRs not available");

    float rq = pinfo.readAccuracy;

    result.tags["rq"] = rq;
    if (rq == 0 && (label == RegionLabelType::INSERT))
        result.label = RegionLabelType::FILTERED;

    if (events.Internal())
    {
        const auto& upperPulsecalls = events.Readouts();
        std::string pulsecalls;
        for (size_t i = leftPulseIndex; i < rightPulseIndex; ++i)
        {
            if (!events.IsBase(i))
                pulsecalls.push_back(static_cast<char>(
                        std::tolower(upperPulsecalls[i])));
            else
                pulsecalls.push_back(static_cast<char>(
                        upperPulsecalls[i]));

        }
        result.tags["pc"] = pulsecalls;

        std::vector<uint8_t> rejectReasons;
        std::transform(
            events.InsertStates().cbegin() + leftPulseIndex,
            events.InsertStates().cbegin() + rightPulseIndex,
            std::back_inserter(rejectReasons),
            [](const InsertState& ins) -> uint8_t {return static_cast<uint8_t>(ins);});
        result.tags["pe"] = rejectReasons;

    }

    for (const auto& tag : events.AvailableTagData(leftPulseIndex, rightPulseIndex))
    {
        result.tags[tag.first] = tag.second;
    }

    return result;
}

}}} // ::PacBio::Primary::Postprimary
