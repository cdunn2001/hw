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

#ifndef PACBIO_POSTPRIMARY_BAM_EVENT_DATA_H
#define PACBIO_POSTPRIMARY_BAM_EVENT_DATA_H

#include <optional>

#include <pbbam/Tag.h>

#include <bazio/BazEventData.h>

#include <postprimary/insertfinder/InsertState.h>

namespace PacBio {
namespace Primary {
namespace Postprimary {

using namespace PacBio::Primary;

// Provides read-only access to the full set of pulse/base information. All
// necessary data transformations (such as handling pulse exclusion) has been
// handled previously, and once constructed this data should remain entirely
// immutable.
class EventData
{
public:

    // Using optional for most values, because this class gets used
    // for both baz2bam and bamb2bam, but in the later case not all
    // of this information is present.  Using an optional type lets
    // us be clearer about values that are missing, and we'll throw
    // an exception if we ever try to use them without them being
    // explicitly set
    struct Info
    {
        std::optional<uint32_t> zmwNum;
        std::optional<uint32_t> zmwIdx;
        std::optional<uint16_t> yPos;
        std::optional<uint16_t> xPos;
        std::optional<uint8_t> holeType;
        std::optional<uint32_t> features;

        static Info BamDefault(uint32_t zmwNum)
        {
            Info ret;
            ret.zmwNum = zmwNum;
            return ret;
        }
    };
    EventData(const Info& meta,
              bool truncates,
              BazIO::BazEventData&& events,
              std::vector<InsertState>&& states);

    // Special constructor for Bam2Bam.  ZmwIndex() will be invalid (0), but
    // bam2bam doesn't need to know it anyway
    EventData(size_t zmwNum,
              BazIO::BazEventData&& events,
              std::vector<InsertState>&& states)
        : EventData(Info::BamDefault(zmwNum), false, std::move(events), std::move(states))
    {}

    // Move only semantics
    EventData(EventData&& src) = default;
    EventData(const EventData&) = delete;
    EventData& operator=(EventData&& rhs) = default;
    EventData& operator=(const EventData&) = delete;

public:
    std::string BaseQualityValues(size_t leftPulseIndex,
                                  size_t rightPulseIndex) const;

    const std::vector<std::pair<std::string, BAM::Tag>> AvailableTagData(size_t pulseBegin,
                                                                         size_t pulseEnd) const;


public: // const data accessors

    size_t NumEvents() const { return bazEvents_.NumEvents(); }

    bool Internal() const { return bazEvents_.Internal(); }

    bool StartFramesAreExact() const { return bazEvents_.StartFramesAreExact(); }

    const std::vector<bool>& IsBase() const { return isBase_; }
    bool IsBase(size_t idx) const { return isBase_[idx]; }

    const std::vector<uint32_t>& Ipds() const { return ipdsSkipPulses_; }
    uint32_t Ipds(size_t idx) const { return ipdsSkipPulses_[idx]; }

    const std::vector<uint32_t>& PulseWidths() const { return bazEvents_.PulseWidths(); }
    uint32_t PulseWidths(size_t idx) const { return bazEvents_.PulseWidths()[idx]; }

    const std::vector<char>& Readouts() const { return bazEvents_.Readouts(); }
    char Readouts(size_t idx) const { return bazEvents_.Readouts()[idx]; }

    const std::vector<uint32_t>& StartFrames() const { return bazEvents_.StartFrames(); }
    uint32_t StartFrames(size_t idx) const { return bazEvents_.StartFrames()[idx]; }

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
    { return bazEvents_.NumEvents(); }

    bool Truncated() const
    { return truncated_; }

    uint32_t ZmwIndex() const
    {
        if (!info_.zmwIdx.has_value())
            throw PBException("Accessing ZmwIndex value that was not initialized");
        return *info_.zmwIdx;
    }

    uint32_t ZmwNumber() const
    {
        if (!info_.zmwNum.has_value())
            throw PBException("Accessing ZmwNumber value that was not initialized");
        return *info_.zmwNum;
    }

    uint16_t XCord() const
    {
        if (!info_.xPos.has_value())
            throw PBException("Accessing x coordinate that was not initialized");
        return *info_.xPos;
    }

    uint16_t YCord() const
    {
        if (!info_.yPos.has_value())
            throw PBException("Accessing y coordinate that was not initialized");
        return *info_.yPos;
    }

    uint8_t HoleType() const
    {
        if (!info_.holeType.has_value())
            throw PBException("Accessing HolType value that was not initialized");
        return *info_.holeType;
    }

    uint32_t UnitFeature() const
    {
        if (!info_.features.has_value())
            throw PBException("Accessing UnitFeature value that was not initialized");
        return *info_.features;
    }

private:
    Info info_;
    bool truncated_;

    uint64_t numBases_;

    BazIO::BazEventData bazEvents_;

    std::vector<bool> isBase_;
    std::vector<uint32_t> ipdsWithPulses_;
    std::vector<uint32_t> ipdsSkipPulses_;
    std::vector<uint8_t> baseCalls_;
    std::vector<size_t>  baseToLeftmostPulseIndex_;
    std::vector<size_t>  baseToPulseIndex_;

    std::vector<bool> excludedBases_;

    std::vector<InsertState> insertStates_;
};

}}} // ::PacBio::Primary::Postprimary

#endif //PACBIO_POSTPRIMARY_BAM_EVENT_DATA_H
