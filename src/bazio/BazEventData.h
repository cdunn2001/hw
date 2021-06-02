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
 * * Data starts as RawEventData which is returned directly by the data parsers
 *   in another file, and represents an encapsualted vector of vector view of
 *   the data, where entries in the the outermost vector corresponds to individual
 *   PacketFieldNames
 *
 * * BazEventData is the result of very light processing, where data is accessed
 *   via a more direct API, and subtlties like IPD_V1 and IPD_LL have been
 *   abstracted away
 *
 * * EventDataParent is the result of additional processing, where things like
 *   excluded bases have been accounted for, and derivative data (such as
 *   start_frame in internal mode) has been computed
 *
 * * EventData is the final form, which is included all of the prior information
 *   as well as extra information like the mapping from base index to pulse index
 */

#pragma once

#include <limits>

#include <bazio/Codec.h>
#include <bazio/DataParsing.h>
#include <bazio/FieldType.h>

namespace PacBio {
namespace Primary {

using namespace PacBio::Primary;

// Provides read-only access to event data extracted from the baz file.
// It has been lightly processed and abstracted, so that one does not have
// to worry about things like data compression of pulse widths used during binary
// storage.
//
// N.B. private inheritance is a somewhat unusual choice, but intentional.
// We need access to the protected members of RawEventData, but do *not*
// want to expose it's API as our own.  Once the refactor is finished and parent
// class' protected function is removed we could move to standard composition.
class BazEventData : public RawEventData
{
public:

    BazEventData(RawEventData&& packets)
            : RawEventData(std::move(packets))
    {

        const auto& rawIsBase = PacketField(PacketFieldName::IS_BASE);
        if (rawIsBase.empty())
        {
            isBase_ = std::vector<bool>(NumEvents(), true);
        }
        else
        {
            isBase_ = std::vector<bool>(rawIsBase.begin(), rawIsBase.end());
        }

        if(Internal())
        {
            ipds_ = PacketField(PacketFieldName::IPD_LL);
            pws_ = PacketField(PacketFieldName::PW_LL);
        }
        else
        {
            ipds_ = PacketField(PacketFieldName::IPD_V1);
            pws_ = PacketField(PacketFieldName::PW_V1);

            for (auto& val : ipds_) val = codec_.CodeToFrame(static_cast<uint8_t>(val));
            for (auto& val : pws_) val = codec_.CodeToFrame(static_cast<uint8_t>(val));
        }
    }

    // Move only semantics
    BazEventData(const BazEventData&) = delete;
    BazEventData(BazEventData&&) = default;
    BazEventData& operator=(const BazEventData&) = delete;
    BazEventData& operator=(BazEventData&&) = default;

public: // const data accesors
    const std::vector<bool>& IsBase() const { return isBase_; }
    bool IsBase(size_t idx) const { return isBase_[idx]; }

    const std::vector<uint32_t> Ipds() const { return ipds_; }
    uint32_t Ipds(size_t idx) const { return ipds_[idx]; }

    const std::vector<uint32_t> PulseWidths() const { return pws_; }
    uint32_t PulseWidths(size_t idx) const { return pws_[idx]; }

    const std::vector<uint32_t>& Readouts() const
    { return PacketField(PacketFieldName::READOUT); }

    const std::vector<uint32_t>& StartFrames() const
    { return PacketField(PacketFieldName::START_FRAME); }

    // Expose the few parts of the base class interface we do want to use.
    using RawEventData::NumEvents;
    using RawEventData::Internal;

protected:

    // Only present because our child class will need to modify the underlying
    // raw data structure.  We don't want this exposed as part of this class'
    // API, and once the refactor is complete this need will go away and
    // this is to be removed.
    using RawEventData::PacketField;

    std::vector<bool> isBase_;
    std::vector<uint32_t> ipds_;
    std::vector<uint32_t> pws_;

    Codec codec_;
};

}} // ::PacBio::Primary

