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

/*
 * The classes in this file are meant to represent different stages of
 * processing for the per pulse/base event data for a given zmw.  Each stage is
 * an individual class so as to make the data as immutable and encapsualted
 * as possible.
 *
 * * EventDataParent is the result of additional processing, where things like
 *   excluded bases have been accounted for, and derivative data (such as
 *   start_frame in internal mode) has been computed
 *
 * * EventData is the final form, which is included all of the prior information
 *   as well as extra information like the mapping from base index to pulse index
 */

#pragma once

#include <pbbam/Tag.h>

#include <bazio/Codec.h>
#include <bazio/DataParsing.h>
#include <bazio/FieldType.h>
#include <bazio/BazEventData.h>

#include <postprimary/insertfinder/InsertState.h>


namespace PacBio {
namespace Primary {
namespace Postprimary {

using namespace PacBio::Primary;

// Provides read-only access to event data.  It no longer matches what was
// in the original file, as it will handle input from the pulse exclusion
// classifier, as well as generate derivative data such as the "start frame"
//
// N.B. private inheritance is an unusual choice, but is intentional.  Not
// only do we need access to a protected member of the parent (which will be
// unnecessary after the refactor is complete), but forwarding part of the
// EventData API is easier via private inheritance than vanilla composition.
class EventDataParent : private BazEventData
{
public:
    EventDataParent(PacBio::Primary::BazEventData&& rawPackets, const std::vector<InsertState>& states)
      : BazEventData(std::move(rawPackets))
    {
        UpdateIPDs(states);
        if (Internal())
            ComputeAdditionalData();
    }

    // Constructor required for `bam2bam`, which needs to skip several steps
    // in the data processing (e.g. it may already have start_frame in the input)
    EventDataParent(PacBio::Primary::RawEventData&& rawData)
      : BazEventData(std::move(rawData))
    {}

    // Move only semantics
    EventDataParent(const EventDataParent&) = delete;
    EventDataParent(EventDataParent&&) = default;
    EventDataParent& operator=(const EventDataParent&) = delete;
    EventDataParent& operator=(EventDataParent&&) = default;

public:

    const std::string BaseQualityValues(size_t leftPulseIndex,
                                        size_t rightPulseIndex) const;

    const std::vector<std::pair<std::string, BAM::Tag>> AvailableTagData(size_t pulseBegin,
                                                                         size_t pulseEnd) const;

public:

    // Forward the parts of the parent API we want exposed.
    using BazEventData::NumEvents;
    using BazEventData::Internal;
    using BazEventData::IsBase;
    using BazEventData::Ipds;
    using BazEventData::PulseWidths;
    using BazEventData::Readouts;
    using BazEventData::StartFrames;

private:

    void UpdateIPDs(const std::vector<InsertState>& states);
    void ComputeAdditionalData();
};

// Provides read-only access to the full set of pulse/base information. All
// necessary data transformations (such as handling pulse exclusion) has been
// handled previously, and once constructed this data should remain entirely
// immutable.
class EventData : private EventDataParent
{
private:
    // Private delegating ctor, to unify the two separate exposed constructors
    // necessary for baz2bam and bam2bam
    EventData(size_t zmwIdx,
              size_t zmwNum,
              bool truncated,
              EventDataParent&& events,
              std::vector<InsertState>&& states);
public:
    EventData(const FileHeader& fh,
              size_t zmwIdx,
              bool truncated,
              Postprimary::BazEventData&& events,
              std::vector<InsertState>&& states)
        : EventData(zmwIdx,
                    fh.ZmwIdToNumber(zmwIdx),
                    truncated,
                    EventDataParent(std::move(events), states),
                    std::move(states))
    {}

    // Special constructor for Bam2Bam.  ZmwIndex() will be invalid (0), but
    // bam2bam doesn't need to know it anyway
    EventData(size_t zmwNum,
              EventDataParent&& events,
              std::vector<InsertState>&& states)
        : EventData(0, zmwNum, false, std::move(events), std::move(states))
    {}

    // Move only semantics
    EventData(EventData&& src) = default;
    EventData(const EventData&) = delete;
    EventData& operator=(EventData&& rhs) = default;
    EventData& operator=(const EventData&) = delete;

public:
    // Expose desired portions of base class API (this list will get longer
    // as the refactor proceeds.
    using EventDataParent::NumEvents;
    using EventDataParent::Internal;
    using EventDataParent::IsBase;
    using EventDataParent::Ipds;
    using EventDataParent::PulseWidths;
    using EventDataParent::Readouts;
    using EventDataParent::StartFrames;
    using EventDataParent::BaseQualityValues;
    using EventDataParent::AvailableTagData;

public: // const data accessors

    const std::vector<InsertState>& InsertStates() const
    { return insertStates_; }

    const std::vector<uint8_t>& BaseCalls() const
    { return baseCalls_; }

    const std::vector<bool>& ExcludedBases() const
    { return excludedBases_; }

    size_t BaseToLeftmostPulseIndex(size_t index) const
    { return baseToLeftmostPulseIndex_[index]; }

    size_t BaseToPulseIndex(size_t index) const
    { return baseToPulseIndex_[index]; }

    uint64_t NumBases() const
    { return numBases_; }

    uint64_t NumPulses() const
    { return EventDataParent::NumEvents(); }

    bool Truncated() const
    { return truncated_; }

    uint32_t ZmwIndex() const
    { return zmwIndex_; }

    uint32_t ZmwNumber() const
    { return zmwNum_; }

private:

    bool truncated_;

    uint32_t zmwIndex_;
    uint32_t zmwNum_;

    uint64_t numBases_;

    std::vector<uint8_t> baseCalls_;
    std::vector<size_t>  baseToLeftmostPulseIndex_;
    std::vector<size_t>  baseToPulseIndex_;

    std::vector<bool> excludedBases_;

    std::vector<InsertState> insertStates_;
};

}}} // ::PacBio::Primary::Postprimary
